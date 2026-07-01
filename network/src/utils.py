import sys

import numpy as np
import pandas as pd
import paths
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


ID_COLUMNS = ["sequence_name", "qp", "frame_id", "ctu_id"]
GRID_ACC_DELTA = 1e-1
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

LOSS_FUNCTIONS = {
    "BCE": nn.BCELoss(),
    "HUBER": nn.SmoothL1Loss(),
    "L1": nn.L1Loss(),
    "MSE": nn.MSELoss(),
    "CE": nn.CrossEntropyLoss(),
}


def get_loss_function(loss_name):
    try:
        return LOSS_FUNCTIONS[loss_name.upper()]
    except KeyError:
        raise ValueError(
            "Unsupported lossFunction '{}'. Available options: {}".format(
                loss_name, ", ".join(sorted(LOSS_FUNCTIONS.keys()))
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

        common_ids = self.input_ids.index.intersection(self.gridmap_ids.index, sort=True)
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
                "Generate them first with createDataset.py/createDataset96.py --action cu-tree."
            )

        payload = pd.read_pickle(self.cu_tree_path)
        samples = payload["samples"]
        if list(samples.index.names) != ID_COLUMNS:
            raise RuntimeError(f"CU tree samples must be indexed by {ID_COLUMNS}")
        if payload.get("format") != "cu_tree_binary_v1":
            raise RuntimeError(
                f"Unsupported CU tree label format in {self.cu_tree_path}. "
                "Regenerate with the current createDataset.py/createDataset96.py --action cu-tree."
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
        nodes_path = self.cu_tree_path.with_name(payload["nodes_file"])
        if not nodes_path.exists():
            raise FileNotFoundError(f"CU tree node array not found: {nodes_path}")
        self.node_memmap = np.memmap(
            nodes_path,
            mode="r",
            dtype=np.dtype(payload["node_dtype"]),
            shape=tuple(payload["node_shape"]),
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


def classifier_node_loss(classifier, pred_gridmap, node_batch, ce_loss):
    node_batches = {}
    total_nodes = 0

    for batch_idx, nodes in enumerate(node_batch):
        nodes_iter = nodes.tolist() if torch.is_tensor(nodes) else nodes
        for node in nodes_iter:
            grid_y, grid_x, grid_h, grid_w, label = [int(v) for v in node]
            if not is_supported_classifier_size(grid_h, grid_w):
                continue
            key = (grid_h, grid_w)
            if key not in node_batches:
                node_batches[key] = {"roi": [], "label": []}
            node_batches[key]["roi"].append(
                pred_gridmap[
                    batch_idx:batch_idx + 1,
                    :,
                    grid_y:grid_y + grid_h,
                    grid_x:grid_x + grid_w,
                ]
            )
            node_batches[key]["label"].append(label)
            total_nodes += 1

    if total_nodes == 0:
        zero = pred_gridmap.sum() * 0.0
        return zero, 0.0, 0

    losses = []
    correct = 0
    for payload in node_batches.values():
        roi = torch.cat(payload["roi"], dim=0)
        labels = torch.tensor(payload["label"], dtype=torch.long, device=pred_gridmap.device)
        logits = classifier(roi)
        losses.append(ce_loss(logits, labels))
        correct += torch.sum(torch.argmax(logits, dim=1) == labels).item()

    return torch.stack(losses).mean(), correct / float(total_nodes), total_nodes


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
):
    accu_loss = torch.zeros(1).to(device)
    accu_grid_loss = torch.zeros(1).to(device)
    accu_cls_loss = torch.zeros(1).to(device)
    accu_grid_acc = torch.zeros(1).to(device)
    accu_cls_acc = torch.zeros(1).to(device)
    total_cls_nodes = 0

    progress = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress):
        input_batch, qp_batch, gridmap_batch, node_batch = data

        input_batch = input_batch.to(device)
        qp_batch = qp_batch.to(device)
        gridmap_batch = gridmap_batch.to(device)

        pred_gridmap = swin_model(input_batch, qp_batch)
        grid_loss = grid_loss_fn(pred_gridmap, gridmap_batch)
        cls_loss, cls_acc, cls_nodes = classifier_node_loss(
            classifier=classifier,
            pred_gridmap=pred_gridmap,
            node_batch=node_batch,
            ce_loss=cls_loss_fn,
        )
        loss = grid_weight * grid_loss + cls_weight * cls_loss

        grid_acc = torch.sum(abs(pred_gridmap - gridmap_batch) <= GRID_ACC_DELTA).item() / float(pred_gridmap.numel())

        if training:
            loss.backward()
            if not torch.isfinite(loss):
                print("WARNING: non-finite loss, ending training ", loss)
                sys.exit(1)
            optimizer.step()
            optimizer.zero_grad()

        accu_loss += loss.detach()
        accu_grid_loss += grid_loss.detach()
        accu_cls_loss += cls_loss.detach()
        accu_grid_acc += grid_acc
        accu_cls_acc += cls_acc
        total_cls_nodes += cls_nodes

        phase = "train" if training else "valid"
        progress.desc = (
            "[{} {} epoch {}] loss: {:.6f}, grid: {:.6f}, cls: {:.6f}, "
            "grid_acc: {:.6f}, cls_acc: {:.6f}, cls_nodes: {:.2f}"
        ).format(
            phase,
            stage_name,
            epoch,
            accu_loss.item() / (step + 1),
            accu_grid_loss.item() / (step + 1),
            accu_cls_loss.item() / (step + 1),
            accu_grid_acc.item() / (step + 1),
            accu_cls_acc.item() / (step + 1),
            total_cls_nodes / float(step + 1),
        )

    step_count = step + 1
    return (
        accu_loss.item() / step_count,
        accu_grid_loss.item() / step_count,
        accu_cls_loss.item() / step_count,
        accu_grid_acc.item() / step_count,
        accu_cls_acc.item() / step_count,
        total_cls_nodes / float(step_count),
    )


def train_one_epoch_classifier(model, optimizer, data_loaders, device, epoch, lossFunction="CE"):
    model.train()

    loss_function = get_loss_function(lossFunction)
    optimizer.zero_grad()

    accu_loss = torch.zeros(1).to(device)
    accu = torch.zeros(1).to(device)
    total_steps = 0

    for loader_name, data_loader in data_loaders:
        progress = tqdm(data_loader, file=sys.stdout)

        for data in progress:
            input_batch, label_batch = data

            input_batch = input_batch.to(device)
            label_batch = label_batch.to(device).long()

            logits = model(input_batch)
            loss = loss_function(logits, label_batch)
            loss.backward()

            batch_acc = torch.sum(torch.argmax(logits, dim=1) == label_batch).item() / float(label_batch.numel())
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

    return accu_loss.item() / total_steps, accu.item() / total_steps
