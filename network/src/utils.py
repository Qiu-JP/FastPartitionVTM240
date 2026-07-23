import os
import sys

import numpy as np
import pandas as pd
import paths
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from tqdm import tqdm


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ID_COLUMNS = ["sequence_name", "qp", "frame_id", "ctu_id"]
GRID_POSITIVE_THRESHOLD = 0.5
LUMA_LCU_SIZE = 64
CHROMA_LCU_SIZE = 32
CLASSIFIER_SUPPORTED_SIZES = {
    (16, 16),
    (8, 8),
    (8, 4),
    (4, 8),
    (8, 2),
    (2, 8),
    (8, 1),
    (1, 8),
    (4, 2),
    (2, 4),
    (4, 1),
    (1, 4),
    (4, 4),
    (2, 2),
    (2, 1),
    (1, 2),
}
CLASSIFIER_SUPPORTED_SIZE_ORDER = tuple(sorted(CLASSIFIER_SUPPORTED_SIZES))

BCE_PROB_MIN = 1e-7
BCE_PROB_MAX = 1.0 - BCE_PROB_MIN


def clamp_bce_probability(prediction):
    return prediction.clamp(min=BCE_PROB_MIN, max=BCE_PROB_MAX)


class ClampedBCELoss(nn.Module):
    def forward(self, prediction, target):
        return nn.functional.binary_cross_entropy(
            clamp_bce_probability(prediction),
            target,
        )


LOSS_FUNCTIONS = {
    "BCE": ClampedBCELoss(),
    "HUBER": nn.SmoothL1Loss(),
    "L1": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "CE": nn.CrossEntropyLoss(),
}


class WeightedBCELoss(nn.Module):
    def __init__(self, positive_weight=1.0, negative_weight=1.0):
        super().__init__()
        if positive_weight <= 0 or negative_weight <= 0:
            raise ValueError("Weighted BCE weights must be positive")
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)

    def forward(self, prediction, target):
        prediction = clamp_bce_probability(prediction)
        loss = -(
            self.positive_weight * target * torch.log(prediction)
            + self.negative_weight * (1.0 - target) * torch.log(1.0 - prediction)
        )
        return loss.mean()


class BCEL1Loss(nn.Module):
    def __init__(self, l1_weight=0.2):
        super().__init__()
        if l1_weight < 0:
            raise ValueError("BCE+L1 weight must be non-negative")
        self.l1_weight = float(l1_weight)
        self.bce = ClampedBCELoss()
        self.l1 = nn.L1Loss()

    def forward(self, prediction, target):
        return self.bce(prediction, target) + self.l1_weight * self.l1(prediction, target)


def get_loss_function(
    loss_name,
    positive_weight=1.0,
    negative_weight=1.0,
    l1_weight=0.2,
):
    if loss_name.upper() == "WBCE":
        return WeightedBCELoss(
            positive_weight=positive_weight,
            negative_weight=negative_weight,
        )
    if loss_name.upper() == "BCE_L1":
        return BCEL1Loss(l1_weight=l1_weight)
    try:
        return LOSS_FUNCTIONS[loss_name.upper()]
    except KeyError:
        raise ValueError(
            "Unsupported lossFunction '{}'. Available options: {}".format(
                loss_name,
                ", ".join(sorted(list(LOSS_FUNCTIONS.keys()) + ["BCE_L1", "WBCE"])),
            )
        )


def adjust_learning_rate(lr, optimizer, epoch, decay_rate):
    adj_lr = lr * (0.5 ** (epoch // decay_rate))
    if adj_lr > 1e-6:
        for param_group in optimizer.param_groups:
            param_group["lr"] = adj_lr


def split_dir_from_name(type):
    return type.lower()


def collate_gridmap_with_nodes(batch):
    inputs, qps, gridmaps, nodes = zip(*batch)
    return (
        torch.stack(inputs, dim=0),
        torch.stack(qps, dim=0),
        torch.stack(gridmaps, dim=0),
        list(nodes),
    )


def normalize_preview_image(image):
    image = image.astype(np.float32)
    if image.ndim == 3:
        if np.max(image) > 1.0:
            image = image / 255.0
        return np.clip(image, 0.0, 1.0)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val <= min_val:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def yuv_to_rgb(yuv):
    yuv = yuv.astype(np.float32)
    y = yuv[..., 0]
    u = yuv[..., 1] - 128.0
    v = yuv[..., 2] - 128.0
    rgb = np.stack(
        (
            y + 1.402 * v,
            y - 0.344136 * u - 0.714136 * v,
            y + 1.772 * u,
        ),
        axis=-1,
    )
    return np.clip(rgb, 0.0, 255.0)


def center_crop_array(image, crop_size):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {crop_size}x{crop_size} from image {image.shape}")
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return image[y0:y0 + crop_size, x0:x0 + crop_size, ...]


def upsample_nearest(channel, target_shape):
    y_scale = target_shape[0] // channel.shape[0]
    x_scale = target_shape[1] // channel.shape[1]
    if y_scale <= 0 or x_scale <= 0:
        raise ValueError(f"Cannot upsample {channel.shape} to {target_shape}")
    return np.repeat(np.repeat(channel, y_scale, axis=0), x_scale, axis=1)[:target_shape[0], :target_shape[1]]


def load_chroma_input_for_preview(dataset):
    if hasattr(dataset, "_preview_chroma_ids") and hasattr(dataset, "_preview_chroma_array"):
        return dataset._preview_chroma_ids, dataset._preview_chroma_array
    chroma_pkl_path = dataset.dataset_dir / "Chroma_Input.pkl"
    if not chroma_pkl_path.exists():
        raise FileNotFoundError(f"Chroma input metadata pkl not found: {chroma_pkl_path}")
    chroma_payload = pd.read_pickle(chroma_pkl_path)
    if not isinstance(chroma_payload, dict) or "array_file" not in chroma_payload:
        raise RuntimeError(f"{chroma_pkl_path} is not in the current metadata format")
    chroma_npy_path = dataset.dataset_dir / chroma_payload["array_file"]
    if not chroma_npy_path.exists():
        raise FileNotFoundError(f"Chroma input npy array not found: {chroma_npy_path}")
    dataset._preview_chroma_ids = chroma_payload["ids"]
    dataset._preview_chroma_array = np.load(chroma_npy_path, mmap_mode="r")
    return dataset._preview_chroma_ids, dataset._preview_chroma_array


def build_tensorboard_preview_background(dataset, sample_index):
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"tbImageSampleIndex {sample_index} out of range [0, {len(dataset)})")
    sample_id = dataset.common_ids[sample_index]
    luma = dataset.input_array[dataset.input_positions[sample_index]][0]
    luma_lcu = center_crop_array(luma, LUMA_LCU_SIZE)

    chroma_ids, chroma_array = load_chroma_input_for_preview(dataset)
    if sample_id not in chroma_ids.index:
        raise KeyError(f"Sample id not found in Chroma input: {sample_id}")
    chroma_pos = int(chroma_ids.loc[sample_id, "sample_index"])
    chroma = chroma_array[chroma_pos]
    chroma_lcu = np.stack(
        (
            center_crop_array(chroma[0], CHROMA_LCU_SIZE),
            center_crop_array(chroma[1], CHROMA_LCU_SIZE),
        ),
        axis=0,
    )
    u = upsample_nearest(chroma_lcu[0], luma_lcu.shape)
    v = upsample_nearest(chroma_lcu[1], luma_lcu.shape)
    return yuv_to_rgb(np.stack((luma_lcu, u, v), axis=-1))


def draw_gridmap_overlay(ax, image, gridmap, title=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    grid_h, grid_w = gridmap.shape[-2:]
    target_h, target_w = image.shape[:2] if image.ndim == 3 else image.shape[-2:]
    unit_y = target_h / float(grid_h)
    unit_x = target_w / float(grid_w)
    image = normalize_preview_image(image)

    if image.ndim == 3:
        ax.imshow(image, extent=[0, target_w, target_h, 0])
    else:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, extent=[0, target_w, target_h, 0])

    for y in range(grid_h):
        for x in range(grid_w):
            x_left = x * unit_x
            x_right = (x + 1) * unit_x
            y_top = y * unit_y
            y_bottom = (y + 1) * unit_y
            ax.add_line(Line2D([x_right, x_right], [y_top, y_bottom], color="black", linewidth=0.45, alpha=0.65, linestyle=":"))
            ax.add_line(Line2D([x_left, x_right], [y_bottom, y_bottom], color="black", linewidth=0.45, alpha=0.65, linestyle=":"))
            ax.add_line(Line2D([x_right, x_right], [y_top, y_bottom], color="orange", linewidth=2.0, alpha=float(np.clip(gridmap[0, y, x], 0.0, 1.0))))
            ax.add_line(Line2D([x_left, x_right], [y_bottom, y_bottom], color=(0 / 255, 160 / 255, 255 / 255), linewidth=2.0, alpha=float(np.clip(gridmap[1, y, x], 0.0, 1.0))))

    ax.add_patch(Rectangle((0, 0), target_w, target_h, fill=False, edgecolor="black", linewidth=1.0))
    if title:
        ax.set_title(title)
    ax.set_xlim(0, target_w)
    ax.set_ylim(target_h, 0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="both", direction="out", length=3)
    ax.set_xticks(list(range(0, int(target_w) + 1, 16)))
    ax.set_yticks(list(range(0, int(target_h) + 1, 16)))


def format_gridmap_values(gridmap):
    vertical = np.array2string(
        np.asarray(gridmap[0]),
        precision=2,
        suppress_small=True,
        max_line_width=120,
        separator=" ",
    )
    horizontal = np.array2string(
        np.asarray(gridmap[1]),
        precision=2,
        suppress_small=True,
        max_line_width=120,
        separator=" ",
    )
    return "gridmap[0] vertical\n{}\n\ngridmap[1] horizontal\n{}".format(vertical, horizontal)


def draw_gridmap_values(ax, gridmap, title=None):
    ax.axis("off")
    if title:
        ax.set_title(title)
    ax.text(
        0.0,
        1.0,
        format_gridmap_values(gridmap),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=5.6,
    )


def gridmap_comparison_image(background, label_gridmap, pred_gridmap, title=None):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.0, 3.8),
        gridspec_kw={"width_ratios": [1.0, 1.25, 1.0, 1.25]},
    )
    if title:
        fig.suptitle(title, fontsize=10)
    draw_gridmap_overlay(axes[0], background, label_gridmap, title="label")
    draw_gridmap_values(axes[1], label_gridmap, title="label values")
    draw_gridmap_overlay(axes[2], background, pred_gridmap, title="prediction")
    draw_gridmap_values(axes[3], pred_gridmap, title="prediction values")
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return image


def ensure_pillow_antialias_compat():
    try:
        from PIL import Image
    except ImportError:
        return
    if hasattr(Image, "ANTIALIAS"):
        return
    if hasattr(Image, "Resampling"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    elif hasattr(Image, "LANCZOS"):
        Image.ANTIALIAS = Image.LANCZOS


@torch.no_grad()
def add_gridmap_prediction_image(tb_writer, tag, model, dataset, sample_index, device, epoch):
    if tb_writer is None:
        return
    was_training = model.training
    model.eval()
    input_sample, qp_sample, label_gridmap, _ = dataset[sample_index]
    pred_gridmap = model(
        input_sample.unsqueeze(0).to(device),
        qp_sample.unsqueeze(0).to(device),
    )[0].detach().cpu().numpy()
    label_gridmap = label_gridmap.detach().cpu().numpy()
    background = build_tensorboard_preview_background(dataset, sample_index)
    sample_id = dataset.common_ids[sample_index]
    image = gridmap_comparison_image(
        background=background,
        label_gridmap=label_gridmap,
        pred_gridmap=pred_gridmap,
        title=f"{dataset.type} sample {sample_index} {sample_id}",
    )
    ensure_pillow_antialias_compat()
    tb_writer.add_image(tag, image, epoch, dataformats="HWC")
    if was_training:
        model.train()


class IdAlignedGridmapDataset(Dataset):
    def __init__(self, dataset_name, type, component="Luma", min_qp=0, max_qp=51):
        self.dataset_name = dataset_name
        self.type = type
        self.component = component
        self.min_qp = min_qp
        self.max_qp = max_qp
        if max_qp <= min_qp:
            raise ValueError(f"max_qp must be larger than min_qp, got {min_qp} and {max_qp}")
        split_dir = split_dir_from_name(type)
        dataset_dir = paths.dataset_root() / dataset_name / split_dir
        self.dataset_dir = dataset_dir
        self.input_path = dataset_dir / f"{component}_Input.pkl"
        self.gridmap_path = dataset_dir / f"{component}_Gridmap.pkl"
        self.input_npy_path = dataset_dir / f"{component}_Input.npy"
        self.gridmap_npy_path = dataset_dir / f"{component}_Gridmap.npy"

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input metadata pkl not found: {self.input_path}")
        if not self.gridmap_path.exists():
            raise FileNotFoundError(f"Gridmap metadata pkl not found: {self.gridmap_path}")
        if not self.input_npy_path.exists():
            raise FileNotFoundError(f"Input npy array not found: {self.input_npy_path}")
        if not self.gridmap_npy_path.exists():
            raise FileNotFoundError(f"Gridmap npy array not found: {self.gridmap_npy_path}")

        print(
            f"{component} {type}: using mmap arrays "
            f"{self.input_npy_path.name}, {self.gridmap_npy_path.name}"
        )
        input_payload = pd.read_pickle(self.input_path)
        gridmap_payload = pd.read_pickle(self.gridmap_path)
        self.input_array = np.load(self.input_npy_path, mmap_mode="r")
        self.gridmap_array = np.load(self.gridmap_npy_path, mmap_mode="r")
        self.input_ids = input_payload["ids"]
        self.gridmap_ids = gridmap_payload["ids"]

        if list(self.input_ids.index.names) != ID_COLUMNS:
            raise RuntimeError(
                f"input ids must be indexed by {ID_COLUMNS}. "
                "Regenerate the pkl files with the current createDataset script."
            )
        if list(self.gridmap_ids.index.names) != ID_COLUMNS:
            raise RuntimeError(
                f"gridmap ids must be indexed by {ID_COLUMNS}. "
                "Regenerate the pkl files with the current createDataset script."
            )

        common_ids = self.input_ids.index.intersection(self.gridmap_ids.index, sort=False)
        if common_ids.empty:
            raise RuntimeError(
                f"No common ids between {self.input_path} and {self.gridmap_path}"
            )
        self.common_ids = common_ids
        self.input_positions = self.input_ids.loc[common_ids, "sample_index"].to_numpy(dtype=np.int64)
        self.gridmap_positions = self.gridmap_ids.loc[common_ids, "sample_index"].to_numpy(dtype=np.int64)
        qp_values = common_ids.get_level_values("qp").to_numpy(dtype=np.float32)
        self.qp_values = (qp_values - float(min_qp)) / float(max_qp - min_qp)
        missing_input = len(self.gridmap_ids) - len(common_ids)
        missing_gridmap = len(self.input_ids) - len(common_ids)
        print(
            f"{component} {type}: {len(common_ids):,} aligned samples "
            f"from {self.input_path.name} and {self.gridmap_path.name} "
            f"(missing input: {missing_input:,}, missing gridmap: {missing_gridmap:,})"
        )
        print(
            f"{component} {type}: input shape {self.input_array.shape}, "
            f"gridmap shape {self.gridmap_array.shape}"
        )
        print(
            f"{component} {type}: normalized QP range "
            f"{self.qp_values.min():.4f} to {self.qp_values.max():.4f} "
            f"(min_qp={min_qp}, max_qp={max_qp})"
        )

    def __len__(self):
        return len(self.input_positions)

    def __getitem__(self, idx):
        input_idx = self.input_positions[idx]
        gridmap_idx = self.gridmap_positions[idx]
        input_sample = torch.from_numpy(self.input_array[input_idx].copy()).float()
        qp_sample = torch.tensor([self.qp_values[idx]], dtype=torch.float32)
        gridmap_sample = torch.from_numpy(self.gridmap_array[gridmap_idx].copy()).float()
        return input_sample, qp_sample, gridmap_sample


class IdAlignedGridmapCuTreeDataset(IdAlignedGridmapDataset):
    def __init__(self, dataset_name, type, component="Luma", min_qp=0, max_qp=51):
        super().__init__(
            dataset_name=dataset_name,
            type=type,
            component=component,
            min_qp=min_qp,
            max_qp=max_qp,
        )
        split_dir = split_dir_from_name(type)
        dataset_dir = paths.dataset_root() / dataset_name / split_dir
        self.cu_tree_path = dataset_dir / f"{component}_CU_Tree.pkl"
        if not self.cu_tree_path.exists():
            raise FileNotFoundError(
                f"CU tree labels not found: {self.cu_tree_path}. "
                "Generate them first with createDataset.py --action cu-tree."
            )

        payload = pd.read_pickle(self.cu_tree_path)
        samples = payload["samples"]
        if list(samples.index.names) != ID_COLUMNS:
            raise RuntimeError(f"CU tree samples must be indexed by {ID_COLUMNS}")
        label_format = payload.get("format")
        if label_format != "cu_tree_numpy":
            raise RuntimeError(
                f"Unsupported CU tree label format in {self.cu_tree_path}. "
                "Regenerate with the current createDataset.py --action cu-tree."
            )

        common_index = self.common_ids
        missing = common_index.difference(samples.index)
        if not missing.empty:
            raise RuntimeError(
                f"CU tree labels are missing {len(missing):,} aligned samples. "
                f"First missing key: {missing[0]}"
            )

        tree_sample_indices = samples.loc[common_index, "sample_index"].to_numpy(dtype=np.int64)
        self.tree_sample_indices = tree_sample_indices
        self.node_offsets = payload["offsets"]
        nodes_path = self.cu_tree_path.with_name(payload["array_file"])
        if not nodes_path.exists():
            raise FileNotFoundError(f"CU tree node array not found: {nodes_path}")
        self.node_memmap = np.load(nodes_path, mmap_mode="r")
        expected_shape = tuple(payload["array_shape"])
        if tuple(self.node_memmap.shape) != expected_shape:
            raise RuntimeError(
                f"CU tree node array shape mismatch: {self.node_memmap.shape} != {expected_shape}"
            )
        print(
            f"{component} {type}: loaded CU tree labels from {self.cu_tree_path.name}, "
            f"{len(self.tree_sample_indices):,} aligned samples, "
            f"{self.node_memmap.shape[0]:,} nodes"
        )

    def __getitem__(self, idx):
        input_sample, qp_sample, gridmap_sample = super().__getitem__(idx)
        sample_index = int(self.tree_sample_indices[idx])
        start = int(self.node_offsets[sample_index])
        end = int(self.node_offsets[sample_index + 1])
        node_sample = torch.from_numpy(np.asarray(self.node_memmap[start:end]).copy()).long()
        return input_sample, qp_sample, gridmap_sample, node_sample


def is_supported_classifier_size(h, w):
    return (h, w) in CLASSIFIER_SUPPORTED_SIZES


def classifier_shape_name(grid_h, grid_w):
    return "{}x{}".format(int(grid_h), int(grid_w))


def empty_classifier_shape_stats():
    return {}


def update_classifier_shape_stats(stats, shape_name, loss_value, correct, count):
    if shape_name not in stats:
        stats[shape_name] = {
            "loss_sum": 0.0,
            "correct": 0,
            "count": 0,
        }
    payload = stats[shape_name]
    payload["loss_sum"] = payload["loss_sum"] + loss_value * int(count)
    payload["correct"] = payload["correct"] + correct
    payload["count"] += int(count)


def finalize_classifier_shape_stats(stats):
    finalized = {}
    for shape_name, payload in sorted(stats.items()):
        count = int(payload["count"])
        if count <= 0:
            continue
        loss_sum = payload["loss_sum"]
        correct = payload["correct"]
        if torch.is_tensor(loss_sum):
            loss_sum = loss_sum.detach().item()
        if torch.is_tensor(correct):
            correct = correct.detach().item()
        finalized[shape_name] = {
            "loss": float(loss_sum) / float(count),
            "acc": float(correct) / float(count),
        }
    return finalized


def classifier_node_loss_legacy(
    classifier,
    pred_gridmap,
    node_batch,
    ce_loss,
):
    node_batches = {}

    for batch_idx, nodes in enumerate(node_batch):
        nodes_iter = nodes.tolist() if torch.is_tensor(nodes) else nodes
        for node in nodes_iter:
            grid_y, grid_x, grid_h, grid_w, label = [int(v) for v in node]
            if not is_supported_classifier_size(grid_h, grid_w):
                continue
            key = (grid_h, grid_w)
            if key not in node_batches:
                node_batches[key] = []
            node_batches[key].append((batch_idx, grid_y, grid_x, label))

    total_nodes = 0
    losses = []
    correct = pred_gridmap.new_zeros((), dtype=torch.long)
    shape_stats = empty_classifier_shape_stats()
    for (grid_h, grid_w), shape_nodes in node_batches.items():
        roi = torch.cat(
            [
                pred_gridmap[
                    batch_idx:batch_idx + 1,
                    :,
                    grid_y:grid_y + grid_h,
                    grid_x:grid_x + grid_w,
                ]
                for batch_idx, grid_y, grid_x, _ in shape_nodes
            ],
            dim=0,
        )
        labels = torch.tensor(
            [label for _, _, _, label in shape_nodes],
            dtype=torch.long,
            device=pred_gridmap.device,
        )
        total_nodes += labels.numel()

        logits = classifier(roi)
        shape_loss = ce_loss(logits, labels)
        losses.append(shape_loss)
        pred_labels = torch.argmax(logits, dim=1)
        shape_correct = torch.sum(pred_labels == labels)
        correct += shape_correct

        update_classifier_shape_stats(
            shape_stats,
            classifier_shape_name(grid_h, grid_w),
            shape_loss.detach(),
            shape_correct,
            labels.numel(),
        )

    if total_nodes == 0:
        zero = pred_gridmap.sum() * 0.0
        return zero, zero.detach(), 0, {}

    return torch.stack(losses).mean(), correct / float(total_nodes), total_nodes, shape_stats


def _pack_classifier_nodes(node_batch):
    """Group CPU node metadata by ROI shape without per-node Python conversion."""
    node_tensors = []
    node_counts = []
    for nodes in node_batch:
        tensor = nodes if torch.is_tensor(nodes) else torch.as_tensor(nodes, dtype=torch.long)
        if tensor.numel() == 0:
            tensor = tensor.reshape(0, 5)
        if tensor.dim() != 2 or tensor.shape[1] != 5:
            raise ValueError("Classifier nodes must have shape [N, 5]")
        tensor = tensor.to(device="cpu", dtype=torch.long)
        node_tensors.append(tensor)
        node_counts.append(tensor.shape[0])

    if not node_tensors or sum(node_counts) == 0:
        return None, None, []

    all_nodes = torch.cat(node_tensors, dim=0)
    batch_indices = torch.repeat_interleave(
        torch.arange(len(node_tensors), dtype=torch.long),
        torch.as_tensor(node_counts, dtype=torch.long),
    )

    shape_groups = []
    for grid_h, grid_w in CLASSIFIER_SUPPORTED_SIZE_ORDER:
        shape_indices = torch.nonzero(
            (all_nodes[:, 2] == grid_h) & (all_nodes[:, 3] == grid_w),
            as_tuple=False,
        ).flatten()
        shape_count = int(shape_indices.numel())
        if shape_count == 0:
            continue
        shape_groups.append((int(shape_indices[0]), grid_h, grid_w, shape_indices))

    # Match the legacy dictionary's first-occurrence shape order for exact A/B
    # comparisons while retaining every valid classifier node.
    selected_groups = []
    packed_indices = []
    packed_offset = 0
    for _, grid_h, grid_w, shape_indices in sorted(shape_groups):
        shape_count = int(shape_indices.numel())
        packed_indices.append(shape_indices)
        selected_groups.append((grid_h, grid_w, packed_offset, packed_offset + shape_count))
        packed_offset += shape_count

    if not packed_indices:
        return None, None, []

    selected_indices = torch.cat(packed_indices, dim=0)
    return (
        all_nodes.index_select(0, selected_indices),
        batch_indices.index_select(0, selected_indices),
        selected_groups,
    )


def extract_classifier_rois_vectorized(
    pred_gridmap,
    batch_indices,
    grid_y,
    grid_x,
    grid_h,
    grid_w,
):
    """Extract [N, C, H, W] ROIs with one differentiable indexed operation."""
    device = pred_gridmap.device
    batch_indices = batch_indices.to(device=device, dtype=torch.long)
    grid_y = grid_y.to(device=device, dtype=torch.long)
    grid_x = grid_x.to(device=device, dtype=torch.long)
    channels = torch.arange(pred_gridmap.shape[1], device=device, dtype=torch.long)
    row_offsets = torch.arange(grid_h, device=device, dtype=torch.long)
    col_offsets = torch.arange(grid_w, device=device, dtype=torch.long)
    return pred_gridmap[
        batch_indices[:, None, None, None],
        channels[None, :, None, None],
        grid_y[:, None, None, None] + row_offsets[None, None, :, None],
        grid_x[:, None, None, None] + col_offsets[None, None, None, :],
    ]


def classifier_node_loss_vectorized(
    classifier,
    pred_gridmap,
    node_batch,
    ce_loss,
    roi_chunk_size=0,
):
    if roi_chunk_size < 0:
        raise ValueError("roi_chunk_size must be non-negative")

    packed_nodes, packed_batch_indices, selected_groups = _pack_classifier_nodes(node_batch)
    if not selected_groups:
        zero = pred_gridmap.sum() * 0.0
        return zero, zero.detach(), 0, {}

    device = pred_gridmap.device
    packed_nodes = packed_nodes.to(device=device, non_blocking=True)
    packed_batch_indices = packed_batch_indices.to(device=device, non_blocking=True)
    total_nodes = 0
    losses = []
    correct = pred_gridmap.new_zeros((), dtype=torch.long)
    shape_stats = empty_classifier_shape_stats()

    for grid_h, grid_w, start, end in selected_groups:
        shape_nodes = packed_nodes[start:end]
        shape_batch_indices = packed_batch_indices[start:end]
        shape_count = end - start
        chunk_size = roi_chunk_size if roi_chunk_size > 0 else shape_count
        logits_chunks = []

        for chunk_start in range(0, shape_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, shape_count)
            chunk_nodes = shape_nodes[chunk_start:chunk_end]
            chunk_batch_indices = shape_batch_indices[chunk_start:chunk_end]

            def classifier_from_gridmap(
                gridmap, batch_ids, nodes, roi_h=grid_h, roi_w=grid_w
            ):
                roi = extract_classifier_rois_vectorized(
                    pred_gridmap=gridmap,
                    batch_indices=batch_ids,
                    grid_y=nodes[:, 0],
                    grid_x=nodes[:, 1],
                    grid_h=roi_h,
                    grid_w=roi_w,
                )
                return classifier(roi)

            if (
                torch.is_grad_enabled()
                and pred_gridmap.requires_grad
                and pred_gridmap.device.type == "cuda"
                and roi_chunk_size > 0
                and shape_count > roi_chunk_size
            ):
                logits = checkpoint(
                    classifier_from_gridmap,
                    pred_gridmap,
                    chunk_batch_indices,
                    chunk_nodes,
                )
            else:
                logits = classifier_from_gridmap(
                    pred_gridmap, chunk_batch_indices, chunk_nodes
                )
            logits_chunks.append(logits)

        logits = logits_chunks[0] if len(logits_chunks) == 1 else torch.cat(logits_chunks, dim=0)
        labels = shape_nodes[:, 4].long()
        shape_loss = ce_loss(logits, labels)
        losses.append(shape_loss)
        pred_labels = torch.argmax(logits, dim=1)
        shape_correct = torch.sum(pred_labels == labels)
        correct += shape_correct
        total_nodes += shape_count
        update_classifier_shape_stats(
            shape_stats,
            classifier_shape_name(grid_h, grid_w),
            shape_loss.detach(),
            shape_correct,
            shape_count,
        )

    return torch.stack(losses).mean(), correct / float(total_nodes), total_nodes, shape_stats


def classifier_node_loss(
    classifier,
    pred_gridmap,
    node_batch,
    ce_loss,
    roi_extraction="vectorized",
    roi_chunk_size=0,
):
    if roi_extraction == "legacy":
        return classifier_node_loss_legacy(
            classifier=classifier,
            pred_gridmap=pred_gridmap,
            node_batch=node_batch,
            ce_loss=ce_loss,
        )
    if roi_extraction != "vectorized":
        raise ValueError("roi_extraction must be 'legacy' or 'vectorized'")
    return classifier_node_loss_vectorized(
        classifier=classifier,
        pred_gridmap=pred_gridmap,
        node_batch=node_batch,
        ce_loss=ce_loss,
        roi_chunk_size=roi_chunk_size,
    )


def grid_positive_counts(pred_gridmap, label_gridmap, threshold=GRID_POSITIVE_THRESHOLD):
    pred_positive = pred_gridmap >= threshold
    label_positive = label_gridmap >= threshold
    true_positive = torch.logical_and(pred_positive, label_positive).sum()
    false_positive = torch.logical_and(pred_positive, ~label_positive).sum()
    false_negative = torch.logical_and(~pred_positive, label_positive).sum()
    return true_positive, false_positive, false_negative


def precision_recall(true_positive, false_positive, false_negative):
    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    if torch.is_tensor(true_positive):
        zero = true_positive.new_zeros((), dtype=torch.float32)
        precision = torch.where(
            precision_den > 0, true_positive.float() / precision_den.float(), zero
        )
        recall = torch.where(
            recall_den > 0, true_positive.float() / recall_den.float(), zero
        )
        return precision, recall
    precision = true_positive / float(precision_den) if precision_den > 0 else 0.0
    recall = true_positive / float(recall_den) if recall_den > 0 else 0.0
    return precision, recall


def train_one_epoch(
    swin_model,
    classifier,
    optimizer,
    data_loader,
    device,
    epoch,
    grid_loss_fn,
    cls_loss_fn,
    grid_weight,
    cls_weight,
    stage_name="train",
    roi_extraction="vectorized",
    classifier_roi_chunk_size=0,
    progress_update_interval=50,
    amp_enabled=False,
    grad_scaler=None,
):
    swin_model.train()
    classifier.train()
    optimizer.zero_grad()
    return _run_joint_epoch(
        swin_model=swin_model,
        classifier=classifier,
        optimizer=optimizer,
        data_loader=data_loader,
        device=device,
        epoch=epoch,
        grid_loss_fn=grid_loss_fn,
        cls_loss_fn=cls_loss_fn,
        grid_weight=grid_weight,
        cls_weight=cls_weight,
        stage_name=stage_name,
        training=True,
        roi_extraction=roi_extraction,
        classifier_roi_chunk_size=classifier_roi_chunk_size,
        progress_update_interval=progress_update_interval,
        amp_enabled=amp_enabled,
        grad_scaler=grad_scaler,
    )


@torch.no_grad()
def evaluate(
    swin_model,
    classifier,
    data_loader,
    device,
    epoch,
    grid_loss_fn,
    cls_loss_fn,
    grid_weight,
    cls_weight,
    stage_name="valid",
    roi_extraction="vectorized",
    classifier_roi_chunk_size=0,
    progress_update_interval=50,
    amp_enabled=False,
):
    swin_model.eval()
    classifier.eval()
    return _run_joint_epoch(
        swin_model=swin_model,
        classifier=classifier,
        optimizer=None,
        data_loader=data_loader,
        device=device,
        epoch=epoch,
        grid_loss_fn=grid_loss_fn,
        cls_loss_fn=cls_loss_fn,
        grid_weight=grid_weight,
        cls_weight=cls_weight,
        stage_name=stage_name,
        training=False,
        roi_extraction=roi_extraction,
        classifier_roi_chunk_size=classifier_roi_chunk_size,
        progress_update_interval=progress_update_interval,
        amp_enabled=amp_enabled,
        grad_scaler=None,
    )


def _run_joint_epoch(
    swin_model,
    classifier,
    optimizer,
    data_loader,
    device,
    epoch,
    grid_loss_fn,
    cls_loss_fn,
    grid_weight,
    cls_weight,
    stage_name,
    training,
    roi_extraction,
    classifier_roi_chunk_size,
    progress_update_interval,
    amp_enabled,
    grad_scaler,
):
    if progress_update_interval <= 0:
        raise ValueError("progress_update_interval must be positive")
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    if training and amp_enabled and grad_scaler is None:
        raise ValueError("grad_scaler is required when AMP training is enabled")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    accu_loss = torch.zeros(1).to(device)
    accu_grid_loss = torch.zeros(1).to(device)
    accu_cls_loss = torch.zeros(1).to(device)
    accu_cls_acc = torch.zeros(1).to(device)
    grid_tp = torch.zeros((), dtype=torch.long, device=device)
    grid_fp = torch.zeros((), dtype=torch.long, device=device)
    grid_fn = torch.zeros((), dtype=torch.long, device=device)
    total_cls_nodes = 0
    classifier_shape_stats = empty_classifier_shape_stats()

    progress = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress):
        input_batch, qp_batch, gridmap_batch, node_batch = data

        input_batch = input_batch.to(device, non_blocking=True)
        qp_batch = qp_batch.to(device, non_blocking=True)
        gridmap_batch = gridmap_batch.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            pred_gridmap = swin_model(input_batch, qp_batch)
        # The model returns sigmoid probabilities. BCELoss is unsafe under
        # autocast, and Classifier_I uses a large negative static mask that is
        # intentionally kept in FP32. Gradients still flow through this cast.
        with autocast(enabled=False):
            loss_gridmap = pred_gridmap.float() if amp_enabled else pred_gridmap
            grid_loss = grid_loss_fn(loss_gridmap, gridmap_batch.float())
            cls_loss, cls_acc, cls_nodes, batch_shape_stats = classifier_node_loss(
                classifier=classifier,
                pred_gridmap=loss_gridmap,
                node_batch=node_batch,
                ce_loss=cls_loss_fn,
                roi_extraction=roi_extraction,
                roi_chunk_size=classifier_roi_chunk_size,
            )
            loss = grid_weight * grid_loss + cls_weight * cls_loss

        tp, fp, fn = grid_positive_counts(pred_gridmap, gridmap_batch)
        grid_tp += tp
        grid_fp += fp
        grid_fn += fn
        if training:
            if amp_enabled:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            if not bool(torch.isfinite(loss).item()):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)
            if amp_enabled:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        accu_loss += loss.detach()
        accu_grid_loss += grid_loss.detach()
        accu_cls_loss += cls_loss.detach()
        accu_cls_acc += cls_acc
        total_cls_nodes += cls_nodes
        for shape_name, payload in batch_shape_stats.items():
            update_classifier_shape_stats(
                classifier_shape_stats,
                shape_name,
                payload["loss_sum"] / float(payload["count"]),
                payload["correct"],
                payload["count"],
            )

        if step == 0 or (step + 1) % progress_update_interval == 0:
            grid_precision, grid_recall = precision_recall(grid_tp, grid_fp, grid_fn)
            phase = "train" if training else "valid"
            progress.desc = (
                "[{} {} epoch {}] loss: {:.6f}, grid: {:.6f}, cls: {:.6f}, "
                "grid_precision: {:.6f}, grid_recall: {:.6f}, cls_acc: {:.6f}, cls_nodes: {:.2f}"
            ).format(
                phase,
                stage_name,
                epoch,
                accu_loss.item() / (step + 1),
                accu_grid_loss.item() / (step + 1),
                accu_cls_loss.item() / (step + 1),
                grid_precision.item(),
                grid_recall.item(),
                accu_cls_acc.item() / (step + 1),
                total_cls_nodes / float(step + 1),
            )

    step_count = step + 1
    grid_precision, grid_recall = precision_recall(grid_tp, grid_fp, grid_fn)
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        phase = "train" if training else "valid"
        print("{} peak CUDA memory allocated: {:.2f} MiB".format(phase, peak_memory_mb))
    return (
        accu_loss.item() / step_count,
        accu_grid_loss.item() / step_count,
        accu_cls_loss.item() / step_count,
        grid_precision.item(),
        grid_recall.item(),
        accu_cls_acc.item() / step_count,
        total_cls_nodes / float(step_count),
        finalize_classifier_shape_stats(classifier_shape_stats),
        peak_memory_mb,
    )


def train_one_epoch_classifier(model, optimizer, data_loaders, device, epoch, lossFunction="CE"):
    model.train()

    loss_function = get_loss_function(lossFunction)
    optimizer.zero_grad()

    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    total_steps = 0
    classifier_shape_stats = empty_classifier_shape_stats()

    for loader_name, data_loader in data_loaders:
        progress = tqdm(data_loader, file=sys.stdout)

        for data in progress:
            input_batch, label_batch = data

            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device).long()

            logits = model(input_batch)
            loss = loss_function(logits, label_batch)
            loss.backward()

            pred_labels = torch.argmax(logits, dim=1)
            batch_correct = torch.sum(pred_labels == label_batch).item()
            batch_acc = batch_correct / float(label_batch.numel())
            update_classifier_shape_stats(
                classifier_shape_stats,
                loader_name,
                loss.detach().item(),
                batch_correct,
                label_batch.numel(),
            )
            accu += batch_acc
            accu_loss += loss.detach()
            total_steps += 1

            progress.desc = "[train epoch {}][{}] loss: {:.6f}, acc: {:.6f}".format(
                epoch,
                loader_name,
                accu_loss.item() / total_steps,
                accu.item() / total_steps,
            )

            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)

            optimizer.step()
            optimizer.zero_grad()

    return (
        accu_loss.item() / total_steps,
        accu.item() / total_steps,
        finalize_classifier_shape_stats(classifier_shape_stats),
    )
