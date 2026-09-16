"""RD-aware Luma32 training: Swin gridmap + shape-specific classifiers."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import paths
from model import Classifier_I as classifier_i
# Luma32: 48x48 context input, 32x32 target block, 2x8x8 gridmap.
from model import SwinTransformer_Unet as model
from utils import (
    CLASSIFIER_SUPPORTED_SIZE_ORDER,
    IdAlignedGridmapCuTreeDataset,
    collate_gridmap_with_nodes,
    adjust_learning_rate,
    classifier_prediction_stats,
    classifier_shape_name,
    empty_classifier_prune_stats,
    empty_classifier_shape_stats,
    extract_classifier_rois_vectorized,
    finalize_classifier_prune_stats,
    finalize_classifier_shape_stats,
    get_loss_function,
    grid_positive_counts,
    precision_recall,
    update_classifier_prune_stats,
    update_classifier_shape_stats,
    _pack_classifier_nodes,
    classifier_node_loss_legacy,
    classifier_true_prob_safety_loss,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_log_file(args):
    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    os.makedirs(log_out_dir, exist_ok=True)
    log_path = os.path.join(log_out_dir, args.logFile)
    log_file = open(log_path, "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print("Log file:", log_path)
    return log_file


def setup_tensorboard(args, log_out_dir):
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is not installed in the FastPartitionVTM environment. "
            "Install it first, for example: "
            "python -m pip install tensorboard (inside FastPartitionVTM)"
        )
    tb_log_dir = args.tbLogDir
    if tb_log_dir is None:
        tb_log_dir = os.path.join(log_out_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print("TensorBoard log dir:", tb_log_dir)
    return writer


def add_classifier_shape_scalars(tb_writer, phase, shape_stats, epoch):
    for shape_name, stats in shape_stats.items():
        tb_writer.add_scalar(f"Classifier/{phase}/{shape_name}/loss", stats["loss"], epoch)
        tb_writer.add_scalar(f"Classifier/{phase}/{shape_name}/acc", stats["acc"], epoch)


def classifier_shape_sort_key(shape_name):
    try:
        h_str, w_str = shape_name.split("x", 1)
        h = int(h_str)
        w = int(w_str)
        return h * w, h, w
    except Exception:
        return 0, 0, 0


def format_classifier_shape_metric(shape_stats, metric_name):
    if not shape_stats:
        return "none"
    parts = []
    for shape_name, stats in sorted(shape_stats.items(), key=lambda item: classifier_shape_sort_key(item[0])):
        parts.append("{}={:.2f}%".format(shape_name, float(stats[metric_name]) * 100.0))
    return ", ".join(parts)


def format_prune_stats(prune_stats):
    if not prune_stats:
        return "none"
    parts = []
    for threshold, stats in sorted(prune_stats.items()):
        parts.append(
            "T={:.2f}:keep={:.2f}% false={:.2f}% reduction={:.2f}%".format(
                float(threshold),
                float(stats["keep_rate"]) * 100.0,
                float(stats["false_prune_rate"]) * 100.0,
                float(stats["candidate_reduction"]) * 100.0,
            )
        )
    return " | ".join(parts)


def write_epoch_summary(summary_path, epoch, stage_name, train_metrics, val_metrics, test_metrics=None):
    with open(summary_path, "a") as f:
        f.write("Epoch {} Stage {}\n".format(epoch, stage_name))
        f.write("Classifier shape acc train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "acc")))
        f.write("Classifier shape top2 train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "top2")))
        f.write("Classifier shape top3 train: {}\n".format(format_classifier_shape_metric(train_metrics[7], "top3")))
        f.write("Classifier prune train: {}\n".format(format_prune_stats(train_metrics[9])))
        f.write("Classifier shape acc val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "acc")))
        f.write("Classifier shape top2 val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "top2")))
        f.write("Classifier shape top3 val: {}\n".format(format_classifier_shape_metric(val_metrics[7], "top3")))
        f.write("Classifier prune val: {}\n".format(format_prune_stats(val_metrics[9])))
        if test_metrics is not None:
            f.write("Classifier shape acc test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "acc")))
            f.write("Classifier shape top2 test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "top2")))
            f.write("Classifier shape top3 test: {}\n".format(format_classifier_shape_metric(test_metrics[7], "top3")))
            f.write("Classifier prune test: {}\n".format(format_prune_stats(test_metrics[9])))
        f.write("\n")


def load_model_weights(model, checkpoint_path, device, model_name):
    if checkpoint_path is None:
        return
    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{model_name} checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    model.load_state_dict(checkpoint)
    print(f"Loaded {model_name} checkpoint:", checkpoint_path)


def classifier_safety_threshold_table(args, device):
    if args.classifierSafetyThresholdPreset == "none":
        return None
    if args.classifierSafetyThresholdPreset != "table3_lambda2000":
        raise ValueError(
            "Unsupported classifier safety threshold preset: {}".format(
                args.classifierSafetyThresholdPreset
            )
        )
    default_tau = float(args.classifierSafetyTauLarge)
    table_by_pixel_hw = {
        (32, 32): [0.006, 0.147, 0.114, 0.121, 0.073, 0.085],
        (16, 16): [0.011, 0.170, 0.025, 0.031, 0.028, 0.036],
        (16, 32): [0.039, None, 0.114, 0.108, 0.078, 0.091],
        (8, 32): [0.042, None, 0.062, 0.042, None, 0.052],
        (8, 16): [-0.010, None, 0.042, 0.049, None, 0.037],
        (32, 16): [0.031, None, 0.107, 0.115, 0.088, 0.080],
        (32, 8): [0.034, None, 0.045, 0.068, 0.052, None],
        (16, 8): [-0.010, None, 0.045, 0.051, 0.043, None],
    }
    thresholds = {}
    for (pixel_h, pixel_w), values in table_by_pixel_hw.items():
        grid_h = pixel_h // 4
        grid_w = pixel_w // 4
        filled = [default_tau if value is None else float(value) for value in values]
        thresholds[(grid_h, grid_w)] = torch.tensor(filled, dtype=torch.float32, device=device)
    return thresholds


def rd_cache_default_path(dataset_name, split_name, component):
    return paths.data_root() / "rdo_cost" / dataset_name / f"{split_name}_{component}_rd_cache.npz"


def build_rd_cache(dataset, rdo_root, sequence_list, cache_path, max_files=0):
    vtm_script_dir = paths.project_root() / "VVCSoftware_VTM" / "script"
    if str(vtm_script_dir) not in sys.path:
        sys.path.insert(0, str(vtm_script_dir))
    from ThSearch_RdoCost import discover_rdo_files, load_sequence_widths, parse_selected_file

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    total_nodes = int(dataset.node_memmap.shape[0])
    rd_delta = np.full((total_nodes, 6), np.inf, dtype=np.float32)
    rd_completed = np.zeros((total_nodes, 6), dtype=np.bool_)
    rd_valid = np.zeros((total_nodes,), dtype=np.bool_)

    sample_starts = {}
    sample_ends = {}
    available_sequence_qps = set()
    for aligned_idx, key in enumerate(dataset.common_ids):
        sequence, qp, frame_id, ctu_id = key
        available_sequence_qps.add((str(sequence), int(qp)))
        sample_index = int(dataset.tree_sample_indices[aligned_idx])
        start = int(dataset.node_offsets[sample_index])
        end = int(dataset.node_offsets[sample_index + 1])
        sample_key = (str(sequence), int(qp), int(frame_id), int(ctu_id))
        sample_starts[sample_key] = start
        sample_ends[sample_key] = end

    widths = load_sequence_widths(Path(sequence_list))
    files = discover_rdo_files(Path(rdo_root), int(max_files))
    matched = 0
    mismatched = 0
    missing_sample = 0

    for file_index, (path, sequence, qp) in enumerate(files, start=1):
        if (str(sequence), int(qp)) not in available_sequence_qps:
            print(f"[{file_index}/{len(files)}] skip {path.name}: not present in dataset split")
            continue
        if sequence not in widths:
            print(f"[{file_index}/{len(files)}] skip {path.name}: missing width for sequence {sequence}")
            continue
        print(f"[{file_index}/{len(files)}] build RD cache from {path.name}")
        parsed = parse_selected_file(path, widths[sequence])
        metadata = parsed["metadata"]
        if metadata.shape[0] == 0:
            continue

        rows_by_ctu = {}
        for row_idx, meta in enumerate(metadata):
            ctu_id = int(meta[0])
            coord = tuple(int(v) for v in meta[1:5])
            rows_by_ctu.setdefault(ctu_id, {})[coord] = row_idx

        file_matched = 0
        file_mismatched = 0
        for ctu_id, rows in rows_by_ctu.items():
            key = (str(sequence), int(qp), 0, int(ctu_id))
            if key not in sample_starts:
                missing_sample += len(rows)
                continue
            start = sample_starts[key]
            end = sample_ends[key]
            nodes = np.asarray(dataset.node_memmap[start:end])
            node_lookup = {
                tuple(int(v) for v in node[:4]): start + node_idx
                for node_idx, node in enumerate(nodes)
            }
            for coord, row_idx in rows.items():
                node_global_idx = node_lookup.get(coord)
                if node_global_idx is None:
                    file_mismatched += 1
                    mismatched += 1
                    continue
                best_cost = float(parsed["best_cost"][row_idx])
                denom = max(abs(best_cost), 1e-12)
                rd_delta[node_global_idx] = parsed["rd_delta"][row_idx] / denom
                rd_completed[node_global_idx] = parsed["completed"][row_idx]
                rd_valid[node_global_idx] = True
                file_matched += 1
                matched += 1
        print(f"  matched={file_matched:,}; mismatched={file_mismatched:,}")

    print(
        "RD cache matched nodes: {:,}; mismatched rows: {:,}; missing sample rows: {:,}".format(
            matched, mismatched, missing_sample
        )
    )
    if matched == 0:
        raise RuntimeError("No RD-cost rows matched the CU-tree dataset")
    dataset_dir = dataset.cu_tree_path.parent
    component = dataset.component
    array_path = dataset_dir / f"{component}_CU_RDCost.npy"
    metadata_path = dataset_dir / f"{component}_CU_RDCost.pkl"
    rd_dtype = np.dtype([
        ("rd_delta", np.float32, (6,)),
        ("completed", np.bool_, (6,)),
        ("valid", np.bool_),
    ])
    rd_array = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=rd_dtype,
        shape=(total_nodes,),
    )
    rd_array["rd_delta"] = rd_delta
    rd_array["completed"] = rd_completed
    rd_array["valid"] = rd_valid
    rd_array.flush()
    del rd_array

    payload = {
        "format": "cu_rdcost_numpy",
        "component": component,
        "id_columns": list(dataset.id_columns),
        "node_key_columns": ["sequence_name", "qp", "frame_id", "ctu_id", "cu_x", "cu_y", "cu_width", "cu_height"],
        "node_order": "identical to CU_Tree.npy global node order",
        "node_columns": ["rd_delta[6]", "completed[6]", "valid"],
        "array_file": array_path.name,
        "array_key": "cu_rdcost",
        "array_shape": (total_nodes,),
        "array_dtype": rd_dtype.descr,
        "offsets": np.asarray(dataset.node_offsets, dtype=np.int64),
        "sample_keys": list(dataset.common_ids),
        "class_order": ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"],
        "rd_delta_definition": "(candidate_cost - best_cost) / max(abs(best_cost), 1e-12)",
        "source_rdo_root": str(rdo_root),
        "source_sequence_list": str(sequence_list),
        "matched": int(matched),
        "mismatched": int(mismatched),
        "missing_sample": int(missing_sample),
    }
    with open(metadata_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Wrote CU RD-cost array:", array_path)
    print("Wrote CU RD-cost metadata:", metadata_path)


class RDAwareGridmapCuTreeDataset(IdAlignedGridmapCuTreeDataset):
    def __init__(self, *args, rd_cache_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        dataset_dir = self.cu_tree_path.parent
        self.rd_cache_path = dataset_dir / f"{self.component}_CU_RDCost.pkl"
        if not self.rd_cache_path.exists():
            raise FileNotFoundError(
                f"CU RD-cost cache not found: {self.rd_cache_path}. "
                "Run train.py with --prepareRdCacheOnly 1 first."
            )
        with open(self.rd_cache_path, "rb") as fp:
            cache = pickle.load(fp)
        if cache.get("format") != "cu_rdcost_numpy":
            raise RuntimeError(f"Unsupported CU RD-cost format in {self.rd_cache_path}")
        rd_array_path = self.rd_cache_path.with_name(cache["array_file"])
        self.rd_array = np.load(rd_array_path, mmap_mode="r")
        if self.rd_array.shape[0] != self.node_memmap.shape[0]:
            raise RuntimeError(
                f"RD cache node count {self.rd_array.shape[0]} does not match CU-tree "
                f"node count {self.node_memmap.shape[0]}"
            )
        if not np.array_equal(np.asarray(cache["offsets"]), np.asarray(self.node_offsets)):
            raise RuntimeError("CU RD-cost offsets do not match CU-tree offsets")
        print(
            f"{self.component} {self.type}: loaded CU RD-cost cache {self.rd_cache_path.name}, "
            f"matched nodes={int(cache['matched']):,}"
        )

    def __getitem__(self, idx):
        input_sample, qp_sample, gridmap_sample, node_sample = super().__getitem__(idx)
        sample_index = int(self.tree_sample_indices[idx])
        start = int(self.node_offsets[sample_index])
        end = int(self.node_offsets[sample_index + 1])
        rd_sample = np.asarray(self.rd_array[start:end])
        rd_delta_sample = torch.from_numpy(rd_sample["rd_delta"].copy()).float()
        rd_completed_sample = torch.from_numpy(rd_sample["completed"].copy()).bool()
        rd_valid_sample = torch.from_numpy(rd_sample["valid"].copy()).bool()
        return (
            input_sample,
            qp_sample,
            gridmap_sample,
            node_sample,
            rd_delta_sample,
            rd_completed_sample,
            rd_valid_sample,
        )


def collate_gridmap_with_nodes_and_rd(batch):
    input_batch = torch.stack([item[0] for item in batch], dim=0)
    qp_batch = torch.stack([item[1] for item in batch], dim=0)
    gridmap_batch = torch.stack([item[2] for item in batch], dim=0)
    node_batch = [item[3] for item in batch]
    rd_delta_batch = [item[4] for item in batch]
    rd_completed_batch = [item[5] for item in batch]
    rd_valid_batch = [item[6] for item in batch]
    return input_batch, qp_batch, gridmap_batch, node_batch, rd_delta_batch, rd_completed_batch, rd_valid_batch


def pack_nodes_with_rd(node_batch, rd_delta_batch, rd_completed_batch, rd_valid_batch):
    node_tensors = []
    delta_tensors = []
    completed_tensors = []
    valid_tensors = []
    node_counts = []
    for nodes, delta, completed, valid in zip(
        node_batch, rd_delta_batch, rd_completed_batch, rd_valid_batch
    ):
        nodes = nodes.to(device="cpu", dtype=torch.long)
        delta = delta.to(device="cpu", dtype=torch.float32)
        completed = completed.to(device="cpu", dtype=torch.bool)
        valid = valid.to(device="cpu", dtype=torch.bool)
        if nodes.numel() == 0:
            nodes = nodes.reshape(0, 5)
            delta = delta.reshape(0, 6)
            completed = completed.reshape(0, 6)
            valid = valid.reshape(0)
        node_tensors.append(nodes)
        delta_tensors.append(delta)
        completed_tensors.append(completed)
        valid_tensors.append(valid)
        node_counts.append(nodes.shape[0])

    if not node_tensors or sum(node_counts) == 0:
        return None, None, None, None, None, []

    all_nodes = torch.cat(node_tensors, dim=0)
    all_delta = torch.cat(delta_tensors, dim=0)
    all_completed = torch.cat(completed_tensors, dim=0)
    all_valid = torch.cat(valid_tensors, dim=0)
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
        if int(shape_indices.numel()) > 0:
            shape_groups.append((int(shape_indices[0]), grid_h, grid_w, shape_indices))

    selected_groups = []
    packed_indices = []
    packed_offset = 0
    for _, grid_h, grid_w, shape_indices in sorted(shape_groups):
        shape_count = int(shape_indices.numel())
        packed_indices.append(shape_indices)
        selected_groups.append((grid_h, grid_w, packed_offset, packed_offset + shape_count))
        packed_offset += shape_count
    if not packed_indices:
        return None, None, None, None, None, []

    selected_indices = torch.cat(packed_indices, dim=0)
    return (
        all_nodes.index_select(0, selected_indices),
        batch_indices.index_select(0, selected_indices),
        all_delta.index_select(0, selected_indices),
        all_completed.index_select(0, selected_indices),
        all_valid.index_select(0, selected_indices),
        selected_groups,
    )


def rd_soft_ce_loss(logits, rd_delta, completed, valid, tau, delta_clamp):
    if tau <= 0:
        raise ValueError("rdSoftTau must be positive")
    device = logits.device
    valid = valid.to(device=device, dtype=torch.bool)
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits.float()
    completed = completed.to(device=device, dtype=torch.bool)
    rd_delta = rd_delta.to(device=device, dtype=torch.float32)
    legal_mask = logits > -1e8
    available = completed & legal_mask & valid[:, None]
    valid_available = available.any(dim=1)
    if not torch.any(valid_available):
        return logits.sum() * 0.0
    safe_delta = torch.nan_to_num(rd_delta, nan=float(delta_clamp), posinf=float(delta_clamp))
    safe_delta = safe_delta.clamp(min=0.0, max=float(delta_clamp))
    target_scores = torch.where(
        available,
        -safe_delta / float(tau),
        torch.full_like(safe_delta, -1e9),
    )
    targets = torch.softmax(target_scores[valid_available], dim=1)
    log_probs = torch.log_softmax(logits[valid_available], dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def rd_delta_penalty_loss(logits, rd_delta, completed, valid, delta_clamp):
    device = logits.device
    valid = valid.to(device=device, dtype=torch.bool)
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits.float()
    completed = completed.to(device=device, dtype=torch.bool)
    rd_delta = rd_delta.to(device=device, dtype=torch.float32)
    legal_mask = logits > -1e8
    safe_delta = torch.nan_to_num(rd_delta, nan=float(delta_clamp), posinf=float(delta_clamp))
    safe_delta = safe_delta.clamp(min=0.0, max=float(delta_clamp))
    penalty_delta = torch.where(
        completed & legal_mask,
        safe_delta,
        torch.full_like(safe_delta, float(delta_clamp)),
    )
    probs = torch.softmax(logits, dim=1)
    per_node = (probs * penalty_delta).sum(dim=1)
    return per_node[valid].mean()


def rd_safety_loss(logits, rd_delta, completed, valid, good_delta, keep_tau):
    if keep_tau <= 0:
        return logits.sum() * 0.0
    device = logits.device
    valid = valid.to(device=device, dtype=torch.bool)
    if not torch.any(valid):
        return logits.sum() * 0.0
    logits = logits.float()
    completed = completed.to(device=device, dtype=torch.bool)
    rd_delta = rd_delta.to(device=device, dtype=torch.float32)
    legal_mask = logits > -1e8
    safe_delta = torch.nan_to_num(rd_delta, nan=float("inf"), posinf=float("inf"))
    good = completed & legal_mask & valid[:, None] & (safe_delta <= float(good_delta))
    if not torch.any(good):
        return logits.sum() * 0.0
    probs = torch.softmax(logits, dim=1)
    return torch.relu(float(keep_tau) - probs[good]).mean()


def rd_topk_rank_loss(
    logits,
    rd_delta,
    completed,
    valid,
    top_k,
    margin,
    temperature,
    regret_weight=4.0,
    regret_cap=0.20,
):
    """Differentiate the event that the best-RD legal mode falls outside Top-K.

    The existing delta loss optimizes expected cost, but it does not directly
    optimize the candidate budget used by VTM.  For each node this surrogate
    selects the lowest measured-RD legal mode and compares its logit against
    the K-th strongest competing logit.  Nodes whose current Top-K set drops
    the best-RD mode are weighted by the RD regret of the weakest retained
    mode.  This focuses capacity on the rare false-prune cases that can alter
    the recursive VTM tree, without using sequence-specific thresholds.
    """
    if top_k <= 0 or margin < 0 or temperature <= 0:
        return logits.sum() * 0.0
    logits = logits.float()
    device = logits.device
    valid = valid.to(device=device, dtype=torch.bool)
    completed = completed.to(device=device, dtype=torch.bool)
    rd_delta = rd_delta.to(device=device, dtype=torch.float32)
    legal = logits > -1e8
    finite_delta = torch.isfinite(rd_delta)
    usable = valid[:, None] & completed & legal & finite_delta
    row_usable = usable.any(dim=1)
    legal_count = legal.sum(dim=1)
    eligible = row_usable & (legal_count > int(top_k))
    if not torch.any(eligible):
        return logits.sum() * 0.0

    safe_delta = torch.where(usable, rd_delta, torch.full_like(rd_delta, float("inf")))
    target = torch.argmin(safe_delta, dim=1)
    competitor_k = min(int(top_k), logits.shape[1] - 1)
    ranked_logits = logits.masked_fill(~legal, -1e9)
    top_values, top_indices = torch.topk(ranked_logits, k=competitor_k, dim=1)
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    kth_competitor = top_values[:, -1]
    target_in_topk = (top_indices == target[:, None]).any(dim=1)

    # Estimate the cost of the weakest retained candidate.  The weight is
    # detached from the logits because it is a sample difficulty multiplier,
    # not another prediction target.
    kept_usable = torch.gather(usable, 1, top_indices)
    kept_delta = torch.gather(rd_delta, 1, top_indices)
    kept_delta = torch.where(kept_usable, kept_delta, torch.zeros_like(kept_delta))
    kept_delta = kept_delta.max(dim=1).values
    target_delta = torch.gather(rd_delta, 1, target[:, None]).squeeze(1)
    regret = (kept_delta - target_delta).clamp(min=0.0, max=float(regret_cap))
    difficulty = 1.0 + float(regret_weight) * regret / max(float(regret_cap), 1e-6)
    violation = (kth_competitor + float(margin) - target_logit) / float(temperature)
    false_prune = eligible & ~target_in_topk
    if not torch.any(false_prune):
        return logits.sum() * 0.0
    return (
        float(temperature)
        * torch.nn.functional.softplus(violation[false_prune])
        * difficulty[false_prune].detach()
    ).mean()


def classifier_node_loss_rd(
    classifier,
    pred_gridmap,
    node_batch,
    rd_delta_batch,
    rd_completed_batch,
    rd_valid_batch,
    ce_loss,
    rd_loss_type,
    hard_ce_weight,
    rd_soft_weight,
    rd_penalty_weight,
    rd_safety_weight,
    rd_soft_tau,
    rd_good_delta,
    rd_keep_tau,
    rd_delta_clamp,
    rd_topk_weight,
    rd_topk,
    rd_topk_margin,
    rd_topk_temperature,
    rd_topk_regret_weight,
    rd_topk_regret_cap,
    classifier_safety_tau,
    classifier_safety_loss_weight,
    classifier_safety_thresholds,
    classifier_safety_threshold_only_shapes,
):
    packed = pack_nodes_with_rd(node_batch, rd_delta_batch, rd_completed_batch, rd_valid_batch)
    packed_nodes, packed_batch_indices, packed_delta, packed_completed, packed_valid, groups = packed
    if not groups:
        zero = pred_gridmap.sum() * 0.0
        return zero, zero.detach(), 0, {}, empty_classifier_prune_stats()

    device = pred_gridmap.device
    packed_nodes = packed_nodes.to(device=device, non_blocking=True)
    packed_batch_indices = packed_batch_indices.to(device=device, non_blocking=True)
    packed_delta = packed_delta.to(device=device, non_blocking=True)
    packed_completed = packed_completed.to(device=device, non_blocking=True)
    packed_valid = packed_valid.to(device=device, non_blocking=True)

    total_nodes = 0
    correct = pred_gridmap.new_zeros((), dtype=torch.long)
    loss_sum = pred_gridmap.new_zeros((), dtype=torch.float32)
    shape_stats = empty_classifier_shape_stats()
    prune_stats = empty_classifier_prune_stats()

    for grid_h, grid_w, start, end in groups:
        shape_nodes = packed_nodes[start:end]
        shape_count = end - start
        roi = extract_classifier_rois_vectorized(
            pred_gridmap=pred_gridmap,
            batch_indices=packed_batch_indices[start:end],
            grid_y=shape_nodes[:, 0],
            grid_x=shape_nodes[:, 1],
            grid_h=grid_h,
            grid_w=grid_w,
        )
        logits = classifier(roi)
        labels = shape_nodes[:, 4].long()
        hard_ce = ce_loss(logits, labels)
        shape_delta = packed_delta[start:end]
        shape_completed = packed_completed[start:end]
        shape_valid = packed_valid[start:end]
        if classifier_safety_threshold_only_shapes and (
            classifier_safety_thresholds is None or (grid_h, grid_w) not in classifier_safety_thresholds
        ):
            continue
        safety = rd_safety_loss(
            logits, shape_delta, shape_completed, shape_valid, rd_good_delta, rd_keep_tau
        )
        threshold_by_label = None
        if classifier_safety_thresholds is not None:
            threshold_by_label = classifier_safety_thresholds.get((grid_h, grid_w))
        threshold_safety = classifier_true_prob_safety_loss(
            logits, labels, classifier_safety_tau, threshold_by_label
        )
        topk_rank = rd_topk_rank_loss(
            logits, shape_delta, shape_completed, shape_valid,
            rd_topk, rd_topk_margin, rd_topk_temperature,
            rd_topk_regret_weight, rd_topk_regret_cap,
        )
        shape_loss = (
            float(classifier_safety_loss_weight) * threshold_safety
            + float(rd_topk_weight) * topk_rank
        )
        if rd_loss_type == "soft":
            soft_ce = rd_soft_ce_loss(
                logits, shape_delta, shape_completed, shape_valid, rd_soft_tau, rd_delta_clamp
            )
            shape_loss = shape_loss + (
                float(hard_ce_weight) * hard_ce
                + float(rd_soft_weight) * soft_ce
                + float(rd_safety_weight) * safety
            )
        elif rd_loss_type == "delta":
            penalty = rd_delta_penalty_loss(
                logits, shape_delta, shape_completed, shape_valid, rd_delta_clamp
            )
            shape_loss = shape_loss + (
                float(hard_ce_weight) * hard_ce
                + float(rd_penalty_weight) * penalty
                + float(rd_safety_weight) * safety
            )
        else:
            raise ValueError("rdLossType must be soft or delta")

        loss_sum = loss_sum + shape_loss * float(shape_count)
        pred_labels = torch.argmax(logits, dim=1)
        shape_correct = torch.sum(pred_labels == labels)
        shape_top2, shape_top3, shape_prune = classifier_prediction_stats(logits, labels)
        correct += shape_correct
        total_nodes += shape_count
        update_classifier_shape_stats(
            shape_stats,
            classifier_shape_name(grid_h, grid_w),
            shape_loss.detach(),
            shape_correct,
            shape_count,
            shape_top2,
            shape_top3,
        )
        update_classifier_prune_stats(prune_stats, shape_prune)

    return loss_sum / float(total_nodes), correct / float(total_nodes), total_nodes, shape_stats, prune_stats


def run_rd_epoch(
    swin_model,
    classifier,
    optimizer,
    data_loader,
    device,
    epoch,
    grid_loss_fn,
    ce_loss,
    grid_weight,
    cls_weight,
    training,
    amp_enabled,
    grad_scaler,
    args,
    classifier_safety_thresholds,
):
    if training:
        swin_model.train()
        classifier.train()
        optimizer.zero_grad()
    else:
        swin_model.eval()
        classifier.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    accu_loss = torch.zeros(1, device=device)
    accu_grid_loss = torch.zeros(1, device=device)
    accu_cls_loss = torch.zeros(1, device=device)
    accu_cls_correct = torch.zeros(1, device=device)
    grid_tp = torch.zeros((), dtype=torch.long, device=device)
    grid_fp = torch.zeros((), dtype=torch.long, device=device)
    grid_fn = torch.zeros((), dtype=torch.long, device=device)
    total_cls_nodes = 0
    shape_stats = empty_classifier_shape_stats()
    prune_stats = empty_classifier_prune_stats()

    progress = tqdm(data_loader, file=sys.stdout)
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for step, data in enumerate(progress):
            (
                input_batch,
                qp_batch,
                gridmap_batch,
                node_batch,
                rd_delta_batch,
                rd_completed_batch,
                rd_valid_batch,
            ) = data
            input_batch = input_batch.to(device, non_blocking=True)
            qp_batch = qp_batch.to(device, non_blocking=True)
            gridmap_batch = gridmap_batch.to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                pred_gridmap = swin_model(input_batch, qp_batch)
            with autocast(enabled=False):
                loss_gridmap = pred_gridmap.float() if amp_enabled else pred_gridmap
                grid_loss = grid_loss_fn(loss_gridmap, gridmap_batch.float())
                cls_loss, cls_acc, cls_nodes, batch_shape_stats, batch_prune_stats = classifier_node_loss_rd(
                    classifier=classifier,
                    pred_gridmap=loss_gridmap,
                    node_batch=node_batch,
                    rd_delta_batch=rd_delta_batch,
                    rd_completed_batch=rd_completed_batch,
                    rd_valid_batch=rd_valid_batch,
                    ce_loss=ce_loss,
                    rd_loss_type=args.rdLossType,
                    hard_ce_weight=args.hardCeWeight,
                    rd_soft_weight=args.rdSoftWeight,
                    rd_penalty_weight=args.rdPenaltyWeight,
                    rd_safety_weight=args.rdSafetyWeight,
                    rd_soft_tau=args.rdSoftTau,
                    rd_good_delta=args.rdGoodDelta,
                    rd_keep_tau=args.rdKeepTau,
                    rd_delta_clamp=args.rdDeltaClamp,
                    rd_topk_weight=args.rdTopKWeight,
                    rd_topk=args.rdTopK,
                    rd_topk_margin=args.rdTopKMargin,
                    rd_topk_temperature=args.rdTopKTemperature,
                    rd_topk_regret_weight=args.rdTopKRegretWeight,
                    rd_topk_regret_cap=args.rdTopKRegretCap,
                    classifier_safety_tau=args.classifierSafetyTauLarge,
                    classifier_safety_loss_weight=args.classifierSafetyLossWeight,
                    classifier_safety_thresholds=classifier_safety_thresholds,
                    classifier_safety_threshold_only_shapes=bool(args.classifierSafetyThresholdOnlyShapes),
                )
                loss = float(grid_weight) * grid_loss + float(cls_weight) * cls_loss

            tp, fp, fn = grid_positive_counts(pred_gridmap, gridmap_batch)
            grid_tp += tp
            grid_fp += fp
            grid_fn += fn

            if training:
                if amp_enabled:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()
                if not bool(torch.isfinite(loss.detach()).item()):
                    raise RuntimeError(f"non-finite loss: {loss.detach().item()}")

            accu_loss += loss.detach()
            accu_grid_loss += grid_loss.detach()
            accu_cls_loss += cls_loss.detach()
            accu_cls_correct += cls_acc.detach() * float(cls_nodes)
            total_cls_nodes += cls_nodes
            for shape_name, payload in batch_shape_stats.items():
                update_classifier_shape_stats(
                    shape_stats,
                    shape_name,
                    payload["loss_sum"] / float(payload["count"]),
                    payload["correct"],
                    payload["count"],
                    payload.get("top2_correct"),
                    payload.get("top3_correct"),
                )
            update_classifier_prune_stats(prune_stats, batch_prune_stats)

            if step == 0 or (step + 1) % args.progressUpdateInterval == 0:
                grid_precision, grid_recall = precision_recall(grid_tp, grid_fp, grid_fn)
                phase = "train" if training else "valid"
                progress.desc = (
                    f"[{phase} rd-{args.rdLossType} epoch {epoch}] "
                    f"loss: {accu_loss.item() / (step + 1):.6f}, "
                    f"grid: {accu_grid_loss.item() / (step + 1):.6f}, "
                    f"cls: {accu_cls_loss.item() / (step + 1):.6f}, "
                    f"grid_precision: {grid_precision.item():.6f}, "
                    f"grid_recall: {grid_recall.item():.6f}, "
                    f"cls_acc: {(accu_cls_correct / max(total_cls_nodes, 1)).item():.6f}"
                )

    steps = max(len(data_loader), 1)
    grid_precision, grid_recall = precision_recall(grid_tp, grid_fp, grid_fn)
    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    cls_acc = accu_cls_correct / float(total_cls_nodes) if total_cls_nodes > 0 else torch.zeros(1, device=device)
    return (
        (accu_loss / steps).item(),
        (accu_grid_loss / steps).item(),
        (accu_cls_loss / steps).item(),
        grid_precision.item(),
        grid_recall.item(),
        cls_acc.item(),
        total_cls_nodes,
        finalize_classifier_shape_stats(shape_stats),
        peak_memory,
        finalize_classifier_prune_stats(prune_stats),
    )


@torch.no_grad()
def evaluate_prediction_only(model, classifier, data_loader, device, thresholds, progress_interval=50):
    """Evaluate a labelled dataset without using its RD-cost or computing loss."""
    model.eval()
    classifier.eval()
    threshold_values = tuple(float(value) for value in thresholds)
    prune_stats = empty_classifier_prune_stats(threshold_values)
    grid_tp = grid_fp = grid_fn = 0
    cls_top1 = cls_top2 = cls_top3 = total_nodes = 0
    shape_stats = empty_classifier_shape_stats()

    progress = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(progress):
        input_batch, qp_batch, gridmap_batch, node_batch = data
        input_batch = input_batch.to(device, non_blocking=True)
        qp_batch = qp_batch.to(device, non_blocking=True)
        gridmap_batch = gridmap_batch.to(device, non_blocking=True)
        pred_gridmap = model(input_batch, qp_batch)
        tp, fp, fn = grid_positive_counts(pred_gridmap, gridmap_batch)
        grid_tp += int(tp.item())
        grid_fp += int(fp.item())
        grid_fn += int(fn.item())

        packed_nodes, packed_batch_indices, groups = _pack_classifier_nodes(node_batch)
        if groups:
            packed_nodes = packed_nodes.to(device=device, non_blocking=True)
            packed_batch_indices = packed_batch_indices.to(device=device, non_blocking=True)
            for grid_h, grid_w, start, end in groups:
                nodes = packed_nodes[start:end]
                indices = packed_batch_indices[start:end]
                roi = extract_classifier_rois_vectorized(
                    pred_gridmap, indices, nodes[:, 0], nodes[:, 1], grid_h, grid_w
                )
                logits = classifier(roi)
                labels = nodes[:, 4].long()
                top1 = torch.argmax(logits, dim=1)
                top2, top3, batch_prune = classifier_prediction_stats(
                    logits, labels, thresholds=threshold_values
                )
                count = int(labels.numel())
                cls_top1 += int((top1 == labels).sum().item())
                cls_top2 += int(top2.item())
                cls_top3 += int(top3.item())
                total_nodes += count
                for threshold, payload in batch_prune.items():
                    update_classifier_prune_stats(prune_stats, {threshold: payload})
                shape_name = classifier_shape_name(grid_h, grid_w)
                update_classifier_shape_stats(
                    shape_stats, shape_name, torch.tensor(0.0),
                    (top1 == labels).sum(), count, top2, top3
                )
        if step == 0 or (step + 1) % progress_interval == 0:
            progress.desc = f"[eval-only] samples={step + 1}, classifier_nodes={total_nodes}"

    precision, recall = precision_recall(grid_tp, grid_fp, grid_fn)
    finalized_prune = finalize_classifier_prune_stats(prune_stats)
    return {
        "grid_precision": float(precision),
        "grid_recall": float(recall),
        "cls_top1": cls_top1 / float(total_nodes) if total_nodes else 0.0,
        "cls_top2": cls_top2 / float(total_nodes) if total_nodes else 0.0,
        "cls_top3": cls_top3 / float(total_nodes) if total_nodes else 0.0,
        "nodes": total_nodes,
        "prune": finalized_prune,
        "shape": finalize_classifier_shape_stats(shape_stats),
    }


@torch.no_grad()
def evaluate_test_metrics(
    model, classifier, data_loader, device, grid_loss_fn, grid_weight, cls_weight,
    thresholds, progress_interval, safety_tau, safety_loss_weight,
    safety_thresholds, safety_threshold_only_shapes,
):
    """Evaluate test data with the same grid/CE/safety reporting as train/val.

    Test data has no RD optimization step here; RD-delta supervision is only
    available for the training-style dataset. The reported test classifier
    loss is CE plus the configured threshold safety loss.
    """
    model.eval()
    classifier.eval()
    threshold_values = tuple(float(value) for value in thresholds)
    prune_stats = empty_classifier_prune_stats(threshold_values)
    shape_stats = empty_classifier_shape_stats()
    total_loss = total_grid_loss = total_cls_loss = 0.0
    grid_tp = grid_fp = grid_fn = 0
    total_nodes = 0
    cls_correct = 0
    progress = tqdm(data_loader, file=sys.stdout)
    ce_loss = nn.CrossEntropyLoss()
    for step, data in enumerate(progress):
        input_batch, qp_batch, gridmap_batch, node_batch = data
        input_batch = input_batch.to(device, non_blocking=True)
        qp_batch = qp_batch.to(device, non_blocking=True)
        gridmap_batch = gridmap_batch.to(device, non_blocking=True)
        pred_gridmap = model(input_batch, qp_batch)
        grid_loss = grid_loss_fn(pred_gridmap.float(), gridmap_batch.float())
        cls_loss, cls_acc, cls_nodes, batch_shapes, batch_prune = classifier_node_loss_legacy(
            classifier=classifier,
            pred_gridmap=pred_gridmap.float(),
            node_batch=node_batch,
            ce_loss=ce_loss,
            safety_tau_large=safety_tau,
            safety_loss_weight=safety_loss_weight,
            safety_thresholds=safety_thresholds,
            safety_threshold_only_shapes=safety_threshold_only_shapes,
        )
        loss = float(grid_weight) * grid_loss + float(cls_weight) * cls_loss
        total_loss += float(loss.item())
        total_grid_loss += float(grid_loss.item())
        total_cls_loss += float(cls_loss.item())
        cls_correct += int(round(float(cls_acc.item()) * cls_nodes))
        total_nodes += int(cls_nodes)
        tp, fp, fn = grid_positive_counts(pred_gridmap, gridmap_batch)
        grid_tp += int(tp.item()); grid_fp += int(fp.item()); grid_fn += int(fn.item())
        for shape_name, payload in batch_shapes.items():
            update_classifier_shape_stats(
                shape_stats, shape_name,
                payload["loss_sum"] / float(payload["count"]),
                payload["correct"], payload["count"],
                payload.get("top2_correct"), payload.get("top3_correct"),
            )
        update_classifier_prune_stats(prune_stats, batch_prune)
        if step == 0 or (step + 1) % progress_interval == 0:
            progress.desc = f"[test] samples={step + 1}, classifier_nodes={total_nodes}"
    steps = max(len(data_loader), 1)
    precision, recall = precision_recall(grid_tp, grid_fp, grid_fn)
    return (
        total_loss / steps, total_grid_loss / steps, total_cls_loss / steps,
        float(precision), float(recall),
        cls_correct / float(total_nodes) if total_nodes else 0.0,
        total_nodes, finalize_classifier_shape_stats(shape_stats), 0.0,
        finalize_classifier_prune_stats(prune_stats),
    )


def load_joint_optimizer_state(optimizer, state, classifier_state, swin):
    """Resume retained parameter moments after dropping the legacy CU64 head."""
    optimizer_state = state
    # Historical jobs optimized Swin, then all classifier heads in
    # ModuleDict order. The removed 16x16-grid head was the first.
    if "branches.16x16.0.weight" in classifier_state:
        import copy
        optimizer_state = copy.deepcopy(optimizer_state)
        groups = optimizer_state["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != sum(len(g["params"]) for g in optimizer.param_groups) + 4:
            raise ValueError("Unsupported legacy optimizer layout for removed 64x64 head")
        first = len(list(swin.parameters()))
        removed = groups[0]["params"][first:first + 4]
        del groups[0]["params"][first:first + 4]
        for parameter_id in removed:
            optimizer_state["state"].pop(parameter_id, None)
    optimizer.load_state_dict(optimizer_state)


def train_rd(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fasttrain_enabled = bool(args.fasttrain)
    amp_enabled = bool(fasttrain_enabled and device.type == "cuda")
    num_workers = args.numWorkers if args.numWorkers >= 0 else (4 if fasttrain_enabled else 2)
    classifier_safety_thresholds = classifier_safety_threshold_table(args, device)

    log_out_dir = os.path.join(str(paths.output_root()), args.outDir, args.jobID)
    ckpt_out_dir = os.path.join(str(paths.checkpoints_root()), args.outDir, args.jobID)
    os.makedirs(log_out_dir, exist_ok=True)
    os.makedirs(ckpt_out_dir, exist_ok=True)
    tb_writer = setup_tensorboard(args, log_out_dir)
    log_path = os.path.join(log_out_dir, "loss.txt")
    summary_path = os.path.join(log_out_dir, "summary.txt")

    print("Creating RD-aware data loader...")
    train_dataset = RDAwareGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.trainSplit,
        component=args.component,
        rd_cache_path=args.rdCache,
    )
    val_dataset = RDAwareGridmapCuTreeDataset(
        dataset_name=args.dataset,
        type=args.valSplit,
        component=args.component,
        rd_cache_path=args.rdCache,
    )
    loader_worker_args = {}
    if num_workers > 0:
        loader_worker_args.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gridmap_with_nodes_and_rd,
        **loader_worker_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_gridmap_with_nodes_and_rd,
        **loader_worker_args,
    )

    eval_loader = None
    if args.evalDataset:
        eval_dataset = IdAlignedGridmapCuTreeDataset(
            dataset_name=args.evalDataset,
            type=args.evalSplit,
            component=args.component,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_gridmap_with_nodes,
            **loader_worker_args,
        )

    swin = model(use_context_mask=args.useContextMask).to(device)
    classifier = classifier_i().to(device)
    # A full training checkpoint already contains the exact model architecture
    # state to resume.  Do not load the optional pretraining weights first:
    # those files may belong to an older/narrower model variant and can fail
    # before the resume state is reached.
    if not args.resume:
        load_model_weights(swin, args.swinCkpt, device, "SwinTransformer_Unet")
        load_model_weights(classifier, args.classifierCkpt, device, "Classifier_I")
    params = [p for p in list(swin.parameters()) + list(classifier.parameters()) if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=5e-2)
    grad_scaler = GradScaler(enabled=amp_enabled)
    start_epoch = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device)
        if not isinstance(resume_payload, dict) or "swin" not in resume_payload or "classifier" not in resume_payload:
            raise ValueError("--resume must point to a full training checkpoint")
        swin.load_state_dict(resume_payload["swin"])
        classifier.load_state_dict(resume_payload["classifier"])
        if "optimizer" in resume_payload:
            load_joint_optimizer_state(optimizer, resume_payload["optimizer"],
                                       resume_payload["classifier"], swin)
        start_epoch = int(resume_payload.get("epoch", 0))
        if start_epoch < 0 or start_epoch >= args.epoch:
            raise ValueError(f"resume epoch {start_epoch} is outside requested epoch count {args.epoch}")
        print(f"Resuming training from completed epoch {start_epoch}: {args.resume}")
    if args.freezeSwin:
        for parameter in swin.parameters():
            parameter.requires_grad_(False)
        print("SwinTransformer_Unet parameters frozen; optimizing Classifier_I only")
    grid_loss_fn = get_loss_function(
        args.gridLossType,
        positive_weight=args.gridPositiveWeight,
        negative_weight=args.gridNegativeWeight,
        l1_weight=args.gridL1Weight,
    )
    ce_loss = nn.CrossEntropyLoss()

    with open(log_path, "a") as f:
        f.write(
            "epoch_num, stage, grid_weight, cls_weight, lr, epoch_loss, grid_loss, cls_loss, "
            "grid_precision, grid_recall, cls_accu, val_loss, val_grid_loss, val_cls_loss, "
            "val_grid_precision, val_grid_recall, val_cls_accu, test_loss, test_grid_loss, "
            "test_cls_loss, test_grid_precision, test_grid_recall, test_cls_accu\n"
        )
        f.write(
            f"rdLossType={args.rdLossType},hardCeWeight={args.hardCeWeight},"
            f"rdSoftWeight={args.rdSoftWeight},rdPenaltyWeight={args.rdPenaltyWeight},"
            f"rdSafetyWeight={args.rdSafetyWeight},rdSoftTau={args.rdSoftTau},"
            f"rdGoodDelta={args.rdGoodDelta},rdKeepTau={args.rdKeepTau},"
                f"rdDeltaClamp={args.rdDeltaClamp},classifierSafetyPreset={args.classifierSafetyThresholdPreset},"
                f"classifierSafetyTau={args.classifierSafetyTauLarge},classifierSafetyWeight={args.classifierSafetyLossWeight},"
                f"fasttrain={args.fasttrain},amp={int(amp_enabled)}\n"
        )
        if args.evalDataset:
            f.write(f"evalDataset={args.evalDataset},evalSplit={args.evalSplit},"
                    f"evalThresholds={args.evalThresholds}\n")

    print(
        "RD-aware joint training: lossType={}, hardCE={}, softW={}, penaltyW={}, "
        "safetyW={}, softTau={}, goodDelta={}, keepTau={}, deltaClamp={}, AMP={}".format(
            args.rdLossType,
            args.hardCeWeight,
            args.rdSoftWeight,
            args.rdPenaltyWeight,
            args.rdSafetyWeight,
            args.rdSoftTau,
            args.rdGoodDelta,
            args.rdKeepTau,
            args.rdDeltaClamp,
            amp_enabled,
        )
    )

    def stage_config(epoch):
        if epoch < args.jointStage1Epoch:
            stage_lr = args.stage1Lr if args.stage1Lr is not None else args.lr
            return "rd_stage1_grid_first", args.stage1GridLossWeight, args.stage1ClsLossWeight, stage_lr, epoch
        stage_lr = args.stage2Lr if args.stage2Lr is not None else args.lr
        return "rd_stage2_joint", args.stage2GridLossWeight, args.stage2ClsLossWeight, stage_lr, epoch - args.jointStage1Epoch

    for epoch in range(start_epoch, args.epoch):
        stage_name, grid_weight, cls_weight, stage_lr, stage_epoch = stage_config(epoch)
        adjust_learning_rate(stage_lr, optimizer, stage_epoch, args.dr)
        train_metrics = run_rd_epoch(
            swin,
            classifier,
            optimizer,
            train_loader,
            device,
            epoch,
            grid_loss_fn,
            ce_loss,
            grid_weight,
            cls_weight,
            True,
            amp_enabled,
            grad_scaler,
            args,
            classifier_safety_thresholds,
        )
        val_metrics = run_rd_epoch(
            swin,
            classifier,
            None,
            val_loader,
            device,
            epoch,
            grid_loss_fn,
            ce_loss,
            grid_weight,
            cls_weight,
            False,
            amp_enabled,
            None,
            args,
            classifier_safety_thresholds,
        )
        eval_metrics = None
        if eval_loader is not None:
            eval_metrics = evaluate_test_metrics(
                swin, classifier, eval_loader, device,
                grid_loss_fn, grid_weight, cls_weight,
                args.evalThresholds, args.progressUpdateInterval,
                args.classifierSafetyTauLarge,
                args.classifierSafetyLossWeight,
                classifier_safety_thresholds,
                bool(args.classifierSafetyThresholdOnlyShapes),
            )
        if tb_writer is not None:
            tb_metrics = [("train", train_metrics), ("val", val_metrics)]
            if eval_metrics is not None:
                tb_metrics.append(("test", eval_metrics))
            for prefix, metrics in tb_metrics:
                tb_writer.add_scalar(f"Loss/{prefix}_total", metrics[0], epoch)
                tb_writer.add_scalar(f"Loss/{prefix}_grid", metrics[1], epoch)
                tb_writer.add_scalar(f"Loss/{prefix}_cls", metrics[2], epoch)
                tb_writer.add_scalar(f"Grid/{prefix}_precision", metrics[3], epoch)
                tb_writer.add_scalar(f"Grid/{prefix}_recall", metrics[4], epoch)
                tb_writer.add_scalar(f"Accu/{prefix}_cls", metrics[5], epoch)
                add_classifier_shape_scalars(tb_writer, prefix, metrics[7], epoch)
            tb_writer.add_scalar("Weight/grid", grid_weight, epoch)
            tb_writer.add_scalar("Weight/cls", cls_weight, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            tb_writer.flush()

        write_epoch_summary(
            summary_path, epoch, f"{stage_name}_{args.rdLossType}",
            train_metrics, val_metrics, eval_metrics,
        )
        print(
            "Epoch {} {} rd_{} GridW {:.6f} ClsW {:.6f} Loss {:.6f} Grid {:.6f} Cls {:.6f} Acc {:.6f} | "
            "Val Loss {:.6f} Grid {:.6f} Cls {:.6f} Acc {:.6f}".format(
                epoch,
                stage_name,
                args.rdLossType,
                grid_weight,
                cls_weight,
                train_metrics[0],
                train_metrics[1],
                train_metrics[2],
                train_metrics[5],
                val_metrics[0],
                val_metrics[1],
                val_metrics[2],
                val_metrics[5],
            )
        )
        print("Classifier shape acc train:", format_classifier_shape_metric(train_metrics[7], "acc"))
        print("Classifier shape acc val:", format_classifier_shape_metric(val_metrics[7], "acc"))
        print("Classifier prune val:", format_prune_stats(val_metrics[9]))
        if eval_metrics is not None:
            print(
                "{} test: loss={:.6f} grid={:.6f} cls={:.6f} "
                "grid_precision={:.6f} grid_recall={:.6f} cls_acc={:.6f}".format(
                    args.evalDataset, eval_metrics[0], eval_metrics[1], eval_metrics[2],
                    eval_metrics[3], eval_metrics[4], eval_metrics[5],
                )
            )
            print("{} shape acc: {}".format(
                args.evalDataset, format_classifier_shape_metric(eval_metrics[7], "acc")
            ))
            print("{} prune: {}".format(args.evalDataset, format_prune_stats(eval_metrics[9])))
        with open(log_path, "a") as f:
            values = [
                epoch,
                f"{stage_name}_{args.rdLossType}",
                grid_weight,
                cls_weight,
                optimizer.param_groups[0]["lr"],
                *train_metrics[:6],
                *val_metrics[:6],
                *(eval_metrics[:6] if eval_metrics is not None else [None] * 6),
            ]
            f.write(",".join(str(v) for v in values) + "\n")
        if args.checkpointInterval > 0 and (epoch + 1) % args.checkpointInterval == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "stage": stage_name,
                "swin": swin.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "train_metrics": train_metrics[:6],
                "val_metrics": val_metrics[:6],
                "eval_metrics": eval_metrics,
            }
            torch.save(checkpoint, os.path.join(ckpt_out_dir, f"checkpoint-epoch{epoch + 1:03d}.pth"))
            torch.save(swin.state_dict(), os.path.join(ckpt_out_dir, f"swin-epoch{epoch + 1:03d}.pth"))
            torch.save(classifier.state_dict(), os.path.join(ckpt_out_dir, f"classifier-epoch{epoch + 1:03d}.pth"))

    torch.save(swin.state_dict(), os.path.join(ckpt_out_dir, "swin-final.pth"))
    torch.save(classifier.state_dict(), os.path.join(ckpt_out_dir, "classifier-final.pth"))
    torch.save({
        "epoch": args.epoch,
        "swin": swin.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, os.path.join(ckpt_out_dir, "checkpoint-final.pth"))
    if tb_writer is not None:
        tb_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=45)
    parser.add_argument("--jobID", type=str, default="custom_rd_run")
    parser.add_argument("--batchSize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dr", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outDir", type=str, default="swin_luma32_custom_rd_delta_safety")
    parser.add_argument("--logFile", type=str, default="train.log")
    parser.add_argument("--dataset", type=str, default="CUSTOM_32")
    parser.add_argument("--trainSplit", type=str, default="training")
    parser.add_argument("--valSplit", type=str, default="validating")
    parser.add_argument("--component", type=str, choices=["Luma"], default="Luma")
    parser.add_argument("--tbLogDir", type=str, default=None)
    parser.add_argument("--useContextMask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swinCkpt", type=str, required=False)
    parser.add_argument("--classifierCkpt", type=str, required=False)
    parser.add_argument("--resume", type=str, default=None,
                        help="Full checkpoint produced by checkpointInterval; continue from its completed epoch")
    parser.add_argument("--gridLossType", type=str, default="BCE", choices=["BCE", "BCE_L1", "WBCE", "L1", "HUBER", "MSE"])
    parser.add_argument("--gridPositiveWeight", type=float, default=1.0)
    parser.add_argument("--gridNegativeWeight", type=float, default=1.0)
    parser.add_argument("--gridL1Weight", type=float, default=0.2)
    parser.add_argument("--jointStage1Epoch", type=int, default=15)
    parser.add_argument("--stage1Lr", type=float, default=None)
    parser.add_argument("--stage2Lr", type=float, default=None)
    parser.add_argument("--stage1GridLossWeight", type=float, default=1.0)
    parser.add_argument("--stage1ClsLossWeight", type=float, default=0.05)
    parser.add_argument("--stage2GridLossWeight", type=float, default=0.5)
    parser.add_argument("--stage2ClsLossWeight", type=float, default=1.0)
    parser.add_argument("--checkpointInterval", type=int, default=1)
    parser.add_argument("--fasttrain", type=int, choices=[0, 1], default=1)
    parser.add_argument("--numWorkers", type=int, default=0, help="DataLoader workers; use 0 in restricted multiprocessing environments")
    parser.add_argument("--progressUpdateInterval", type=int, default=50)
    parser.add_argument("--rdLossType", type=str, choices=["soft", "delta"], default="delta")
    parser.add_argument("--hardCeWeight", type=float, default=1.0)
    parser.add_argument("--rdSoftWeight", type=float, default=0.5)
    parser.add_argument("--rdPenaltyWeight", type=float, default=1.0)
    parser.add_argument("--rdSafetyWeight", type=float, default=0.0)
    parser.add_argument("--rdSoftTau", type=float, default=0.03)
    parser.add_argument("--rdGoodDelta", type=float, default=0.03)
    parser.add_argument("--rdKeepTau", type=float, default=0.10)
    parser.add_argument("--rdDeltaClamp", type=float, default=0.20)
    parser.add_argument("--rdTopKWeight", type=float, default=0.0,
                        help="Differentiable penalty for the best-RD mode falling outside Top-K")
    parser.add_argument("--rdTopK", type=int, default=3,
                        help="K used by the false-prune ranking surrogate")
    parser.add_argument("--rdTopKMargin", type=float, default=0.05,
                        help="Logit margin for the best-RD mode to enter Top-K")
    parser.add_argument("--rdTopKTemperature", type=float, default=0.10,
                        help="Temperature of the smooth Top-K ranking penalty")
    parser.add_argument("--rdTopKRegretWeight", type=float, default=4.0,
                        help="Extra weight for high-regret false-prune nodes")
    parser.add_argument("--rdTopKRegretCap", type=float, default=0.20,
                        help="Maximum RD regret used for the false-prune weight")
    parser.add_argument("--classifierSafetyTauLarge", type=float, default=0.10)
    parser.add_argument("--classifierSafetyLossWeight", type=float, default=0.5)
    parser.add_argument(
        "--classifierSafetyThresholdPreset", type=str, default="table3_lambda2000",
        choices=["none", "table3_lambda2000"],
    )
    parser.add_argument("--classifierSafetyThresholdOnlyShapes", type=int, choices=[0, 1], default=0)
    parser.add_argument("--freezeSwin", type=int, choices=[0, 1], default=0,
                        help="Freeze the Swin feature extractor and fine-tune only Classifier_I")
    parser.add_argument("--rdCache", type=str, default=None)
    parser.add_argument("--rdoRoot", type=str, default=str(paths.data_root() / "rdo_cost" / "DIV2K" / "validating"))
    parser.add_argument("--sequenceList", type=str, default=str(paths.ref_model_root() / "script" / "Validating_Sequences_DIV2K.txt"))
    parser.add_argument("--rdMaxFiles", type=int, default=0)
    parser.add_argument("--prepareRdCacheOnly", type=int, choices=[0, 1], default=0)
    parser.add_argument("--evalDataset", type=str, default=None)
    parser.add_argument("--evalSplit", type=str, default="testing")
    parser.add_argument(
        "--evalThresholds", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    args = parser.parse_args()

    if args.rdCache is None:
        args.rdCache = str(rd_cache_default_path(args.dataset, args.trainSplit, args.component))

    setup_log_file(args)
    if args.prepareRdCacheOnly:
        dataset = IdAlignedGridmapCuTreeDataset(
            dataset_name=args.dataset,
            type=args.trainSplit,
            component=args.component,
        )
        build_rd_cache(
            dataset=dataset,
            rdo_root=args.rdoRoot,
            sequence_list=args.sequenceList,
            cache_path=args.rdCache,
            max_files=args.rdMaxFiles,
        )
        return

    train_rd(args)


if __name__ == "__main__":
    main()
