#!/usr/bin/env python3
"""Search FastPartition thresholds with measured VTM RD costs.

The calibration samples are the luma nodes on the final selected coding tree.
For one node and a six-threshold vector T, the objective is

    min_measured_cost_of_kept_modes(T) - full_search_best_cost
      + lambda * sum_kept_legal_modes(area_in_4x4_blocks)

The VTM TSV files contain censored candidates (scheduled but not completed).
Their missing RD costs are never fabricated.  They still contribute to the
kept-compute term. If no retained mode has a measured cost, the RD outcome is
unknown; a final threshold vector with unknown outcomes cannot be exported.

Thresholds of the six modes interact through the minimum retained RD cost and
VTM's return-to-native fallback when all legal modes are rejected. The script uses exact discrete
coordinate descent.  With five thresholds fixed, crossing a sample probability
changes that sample exactly once, so sorting the probabilities and accumulating
the objective deltas finds the best sixth threshold without a grid or a
continuous golden-section approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
NETWORK_SRC = PROJECT_ROOT / "network" / "src"
if str(NETWORK_SRC) not in sys.path:
    sys.path.insert(0, str(NETWORK_SRC))

from model import Classifier_I
from utils import CLASSIFIER_SUPPORTED_SIZE_ORDER


CLASS_NAMES = ["NS", "QT", "BTH", "BTV", "TTH", "TTV"]
LUMA32_SUPPORTED_SIZE_ORDER = tuple(
    (grid_h, grid_w)
    for grid_h, grid_w in CLASSIFIER_SUPPORTED_SIZE_ORDER
    if grid_h <= 8 and grid_w <= 8
)
PAPER_ALPHA_BY_SIZE = {
    "32x32": 0.9266,
    "16x16": 0.1057,
    "32x16": 0.1136,
    "16x32": 0.1136,
    "32x8": 0.0362,
    "8x32": 0.0362,
    "16x8": 0.0806,
    "8x16": 0.0806,
}
PAPER_ALPHA_REFERENCE = PAPER_ALPHA_BY_SIZE["32x32"]
DEFAULT_RDO_ROOT = PROJECT_ROOT / "data" / "rdo_cost" / "DIV2K" / "validating"
DEFAULT_SEQUENCE_LIST = PROJECT_ROOT / "ref_model" / "script" / "Validating_Sequences_DIV2K.txt"
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "th_search_rdcost"
DEFAULT_LAMBDAS = "1000,2000,4000,5500,9000"
FILE_PATTERN = re.compile(r"^(?P<sequence>.+)_QP(?P<qp>[0-9]+)\.tsv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search shape x mode thresholds with measured RD cost and 4x4 compute."
    )
    parser.add_argument("--regret-weighting", choices=["uniform", "area4x4"], default="uniform",
                        help="ONNX normalized-regret cache only: weight RD proxy per node or by pixel area/16; neither restores absolute RD cost")
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument(
        "--max-threshold", type=float, default=1.0,
        help="Upper bound for every searched probability threshold (use 0.15 to stay near Table III)",
    )
    parser.add_argument("--dataset-dir", help="Directory containing createDataset NPY/PKL files")
    parser.add_argument("--probability-cache", help="Reuse probabilities generated from dataset NPY/PKL and ONNX")
    parser.add_argument("--batch-size", type=int, default=16, help="Swin inference batch; samples follow NPY order")
    parser.add_argument("--bundle", help="Deployment ONNX bundle directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--active-sizes",
        default="",
        help="comma-separated CU pixel sizes to emit in VTM cfg; empty keeps all sizes",
    )
    return parser.parse_args()


def parse_lambdas(text: str) -> List[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--lambdas must contain at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("All lambda values must be finite and non-negative")
    return values




def load_sequence_widths(path: Path) -> Dict[str, int]:
    widths = {}
    with path.open("r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#") or "end!!!!" in line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4:
                raise ValueError(f"Malformed sequence-list line: {raw_line.rstrip()}")
            widths[fields[0]] = int(fields[2])
    return widths


def discover_rdo_files(root: Path, max_files: int, allowed_sequences=None) -> List[Tuple[Path, str, int]]:
    files = []
    for path in sorted(root.glob("*_QP*.tsv")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        if allowed_sequences is not None and match.group("sequence") not in allowed_sequences:
            continue
        done_path = Path(str(path) + ".done")
        if not done_path.exists() or path.stat().st_size == 0:
            print(f"Skip incomplete RDO file: {path}")
            continue
        files.append((path, match.group("sequence"), int(match.group("qp"))))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No completed RDO TSV files found under {root}")
    return files










def selected_rows(path: Path, channel: str = "L") -> Iterable[List[str]]:
    """Yield selected-tree luma rows and luma roots in parent-before-child order.

    The classifier only supports the sizes in ``supported_size_expression()``,
    but the 128x128 luma roots are also needed as the non-overlapping
    normalization denominator.  A root best cost already contains its selected
    descendants, so summing every classifier node's best cost double-counts
    the same coding tree at several depths.
    """
    awk_program = (
        'NF==52 && $1!="node_id" {'
        'active=($2==-1 || (($2 in selected) && selected[$2]==$3));'
        'if(!active) next;'
        'selected[$1]=$51;'
        f'if($5=="{channel}") print;'
        '}'
    )
    tac_process = subprocess.Popen(
        ["tac", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if tac_process.stdout is None:
        raise RuntimeError("Failed to open tac stdout")
    awk_process = subprocess.Popen(
        ["awk", "-F", "\t", awk_program],
        stdin=tac_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tac_process.stdout.close()
    if awk_process.stdout is None:
        raise RuntimeError("Failed to open awk stdout")

    try:
        for line in awk_process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 52:
                yield fields
    finally:
        awk_process.stdout.close()

    awk_stderr = awk_process.stderr.read() if awk_process.stderr is not None else ""
    awk_code = awk_process.wait()
    tac_stderr = tac_process.stderr.read().decode("utf-8", errors="replace") if tac_process.stderr else ""
    tac_code = tac_process.wait()
    if tac_code != 0 or awk_code != 0:
        raise RuntimeError(
            f"Failed to extract selected nodes from {path}: "
            f"tac={tac_code} awk={awk_code} {tac_stderr} {awk_stderr}"
        )




def parse_selected_file(
    path: Path,
    picture_width: int,
    channel: str = "L",
    supported_size_order=None,
    model_block_size: int = 64,
) -> dict:
    if model_block_size not in (32, 64):
        raise ValueError("model_block_size must be 32 or 64 pixels")
    # Dataset ctu_id follows createDataset.block_origin_from_ctu_id:
    # raster-ordered 128x128 VTM CTU, then its four 64x64 feature blocks.
    outer_ctu_columns = (int(picture_width) + 127) // 128
    metadata = []
    frame_ids = []
    coordinates = []
    legal_rows = []
    completed_rows = []
    rd_delta_rows = []
    best_cost_rows = []
    tree_parent = []
    tree_depth = []
    tree_selected = []
    tree_legal = []
    tree_completed = []
    tree_rd_delta = []
    tree_best_cost = []
    tree_decision_indices = []
    node_to_tree_index = {}
    root_best_cost_sum = 0.0
    root_count = 0
    censored_legal = 0
    completed_legal = 0

    for fields in selected_rows(path, channel):
        parent_id = int(fields[1])
        best_cost = float(fields[51])
        if parent_id == -1 and math.isfinite(best_cost):
            root_best_cost_sum += best_cost
            root_count += 1

        x = int(fields[5])
        y = int(fields[6])
        width = int(fields[7])
        height = int(fields[8])
        if not math.isfinite(best_cost):
            continue
        legal = np.zeros(6, dtype=np.bool_)
        completed = np.zeros(6, dtype=np.bool_)
        rd_delta = np.full(6, np.inf, dtype=np.float32)
        for mode in range(6):
            base = 14 + 6 * mode
            legal[mode] = int(fields[base]) != 0
            completed_count = int(fields[base + 2])
            if completed_count > 0:
                mode_cost = float(fields[base + 3])
                if math.isfinite(mode_cost):
                    completed[mode] = True
                    rd_delta[mode] = np.float32(max(mode_cost - best_cost, 0.0))
        node_id = int(fields[0])
        if parent_id == -1:
            parent_index = -1
            depth = 0
        else:
            if parent_id not in node_to_tree_index:
                raise RuntimeError(
                    f"Selected-tree parent {parent_id} precedes no active luma node "
                    f"for node {node_id} in {path}"
                )
            parent_index = node_to_tree_index[parent_id]
            depth = tree_depth[parent_index] + 1
        selected_name = fields[50]
        selected_mode = CLASS_NAMES.index(selected_name) if selected_name in CLASS_NAMES else -1
        tree_index = len(tree_parent)
        node_to_tree_index[node_id] = tree_index
        tree_parent.append(parent_index)
        tree_depth.append(depth)
        tree_selected.append(selected_mode)
        tree_legal.append(legal)
        tree_completed.append(completed)
        tree_rd_delta.append(rd_delta)
        tree_best_cost.append(best_cost)

        grid_w = width // 4
        grid_h = height // 4
        if supported_size_order is None:
            supported_size_order = CLASSIFIER_SUPPORTED_SIZE_ORDER
        if (grid_h, grid_w) not in supported_size_order:
            continue

        if model_block_size == 32:
            # Luma32 inference is evaluated on independent 32x32 target
            # blocks, whose gridmap is exactly 8x8.  Do not use the 64x64
            # feature-CTU coordinates used by the legacy Luma96 path.
            feature_x = (x // 32) * 32
            feature_y = (y // 32) * 32
            ctu_id = (feature_y // 32) * max(1, (picture_width + 31) // 32) + feature_x // 32
        else:
            outer_ctu_x = x // 128
            outer_ctu_y = y // 128
            sub_x = (x % 128) // 64
            sub_y = (y % 128) // 64
            ctu_id = (
                (outer_ctu_y * outer_ctu_columns + outer_ctu_x) * 4
                + sub_y * 2
                + sub_x
            )
            feature_x = outer_ctu_x * 128 + sub_x * 64
            feature_y = outer_ctu_y * 128 + sub_y * 64
        grid_x = (x - feature_x) // 4
        grid_y = (y - feature_y) // 4
        max_grid_size = model_block_size // 4
        if grid_x < 0 or grid_y < 0 or grid_x + grid_w > max_grid_size or grid_y + grid_h > max_grid_size:
            raise RuntimeError(
                f"CU {x},{y},{width}x{height} does not fit its {model_block_size}x{model_block_size} feature block in {path}"
            )

        for mode in range(6):
            if legal[mode]:
                if completed[mode]:
                    completed_legal += 1
                else:
                    censored_legal += 1
        metadata.append((ctu_id, grid_y, grid_x, grid_h, grid_w))
        frame_ids.append(int(fields[3]))
        coordinates.append((x, y, width, height))
        tree_decision_indices.append(tree_index)
        legal_rows.append(legal)
        completed_rows.append(completed)
        rd_delta_rows.append(rd_delta)
        best_cost_rows.append(best_cost)

    if not metadata:
        return {
            "metadata": np.empty((0, 5), dtype=np.int32),
            "frame_ids": np.empty((0,), dtype=np.int32),
            "coordinates": np.empty((0, 4), dtype=np.int32),
            "legal": np.empty((0, 6), dtype=np.bool_),
            "completed": np.empty((0, 6), dtype=np.bool_),
            "rd_delta": np.empty((0, 6), dtype=np.float32),
            "best_cost": np.empty((0,), dtype=np.float64),
            "root_best_cost_sum": root_best_cost_sum,
            "root_count": root_count,
            "tree": None,
            "censored_legal": censored_legal,
            "completed_legal": completed_legal,
        }
    tree = {
        "parent": np.asarray(tree_parent, dtype=np.int32),
        "depth": np.asarray(tree_depth, dtype=np.uint8),
        "selected": np.asarray(tree_selected, dtype=np.int8),
        "legal": np.stack(tree_legal, axis=0),
        "completed": np.stack(tree_completed, axis=0),
        "rd_delta": np.stack(tree_rd_delta, axis=0),
        "best_cost": np.asarray(tree_best_cost, dtype=np.float64),
        "decision_indices": np.asarray(tree_decision_indices, dtype=np.int32),
        "decision_shapes": np.asarray(metadata, dtype=np.int32)[:, 3:5],
    }
    return {
        "metadata": np.asarray(metadata, dtype=np.int32),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "coordinates": np.asarray(coordinates, dtype=np.int32),
        "legal": np.stack(legal_rows, axis=0),
        "completed": np.stack(completed_rows, axis=0),
        "rd_delta": np.stack(rd_delta_rows, axis=0),
        "best_cost": np.asarray(best_cost_rows, dtype=np.float64),
        "root_best_cost_sum": root_best_cost_sum,
        "root_count": root_count,
        "tree": tree,
        "censored_legal": censored_legal,
        "completed_legal": completed_legal,
    }




    # File-level counts are distributed later from the actual masks.  Keeping
    # them here would double-count when a file contains several shapes.




def static_legal_modes(data: dict) -> np.ndarray:
    return np.any(data["legal"], axis=0)


def apply_vtm_fallback(keep: np.ndarray, legal: np.ndarray, probability: np.ndarray) -> np.ndarray:
    # Match deployment: an empty legal candidate set restores native search,
    # rather than silently switching a threshold policy to top-1.
    keep = keep & legal
    empty = ~np.any(keep, axis=1)
    keep[empty] = legal[empty]
    return keep


def measured_rd_delta(
    keep: np.ndarray,
    completed: np.ndarray,
    rd_delta: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    # A rejected candidate cannot supply the retained set's RD cost. If no
    # retained cost was observed, report unknown (+inf), never invented regret.
    measured_keep = keep & completed & np.isfinite(rd_delta)
    return np.min(np.where(measured_keep, rd_delta, np.inf), axis=1)


def evaluate_thresholds(data: dict, thresholds: np.ndarray, lambda_value: float) -> dict:
    probability = data["probability"]
    legal = data["legal"]
    keep = legal & (probability >= thresholds[None, :])
    keep = apply_vtm_fallback(keep, legal, probability)
    rd_per_node = measured_rd_delta(
        keep, data["completed"], data["rd_delta"], probability
    )
    kept_count = int(np.sum(keep))
    kept_compute = float(kept_count * data["area4x4"])
    rd_delta_raw_sum = float(np.sum(rd_per_node, dtype=np.float64))
    rd_delta_sum = float(data["rd_scale"]) * rd_delta_raw_sum
    return {
        "objective": rd_delta_sum + float(lambda_value) * kept_compute,
        "rd_delta_sum": rd_delta_sum,
        "rd_delta_raw_sum": rd_delta_raw_sum,
        "kept_count": kept_count,
        "kept_compute4x4": kept_compute,
        "unknown_rd_node_count": int(np.sum(~np.isfinite(rd_per_node))),
    }


def best_coordinate_threshold(
    data: dict,
    thresholds: np.ndarray,
    mode: int,
    lambda_value: float,
    max_threshold: float,
) -> float:
    probability = data["probability"]
    legal = data["legal"]
    completed = data["completed"]
    rd_delta = data["rd_delta"]
    mode_legal = legal[:, mode]
    if not np.any(mode_legal):
        return 1.0

    keep_other = legal & (probability >= thresholds[None, :])
    keep_other[:, mode] = False

    keep_before = keep_other.copy()
    keep_before[:, mode] = mode_legal
    keep_before = apply_vtm_fallback(keep_before, legal, probability)
    keep_after = apply_vtm_fallback(keep_other, legal, probability)

    rd_before = measured_rd_delta(keep_before, completed, rd_delta, probability)
    rd_after = measured_rd_delta(keep_after, completed, rd_delta, probability)
    compute_before = np.sum(keep_before, axis=1, dtype=np.int16)
    compute_after = np.sum(keep_after, axis=1, dtype=np.int16)

    indices = np.flatnonzero(mode_legal)
    mode_probability = probability[indices, mode]
    known_before = np.isfinite(rd_before)
    known_after = np.isfinite(rd_after)
    unknown_delta = (~known_after[indices]).astype(np.int64) - (~known_before[indices]).astype(np.int64)
    # Keep feasibility separate to avoid inf-inf producing NaNs. Unknown
    # nodes are never accepted as a zero-cost improvement. If the initial
    # coordinates are infeasible, first reduce their number toward zero.
    finite_before = np.where(known_before, rd_before, 0.0).astype(np.float64)
    finite_after = np.where(known_after, rd_after, 0.0).astype(np.float64)
    objective_delta = (
        float(data["rd_scale"]) * (finite_after[indices] - finite_before[indices])
        + float(lambda_value) * float(data["area4x4"])
        * (compute_after[indices] - compute_before[indices]).astype(np.float64)
    )

    order = np.argsort(mode_probability, kind="stable")
    ordered_probability = mode_probability[order]
    ordered_delta = objective_delta[order]
    unique_probability, first_indices = np.unique(
        ordered_probability, return_index=True
    )
    grouped_delta = np.add.reduceat(ordered_delta, first_indices)
    grouped_unknown = np.add.reduceat(unknown_delta[order], first_indices)
    candidate_unknown = int(np.sum(~known_before)) + np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(grouped_unknown, dtype=np.int64)))
    cumulative_delta = np.cumsum(grouped_delta, dtype=np.float64)
    candidate_delta = np.concatenate((np.asarray([0.0]), cumulative_delta))
    candidate_thresholds = np.concatenate((
        np.asarray([0.0], dtype=np.float64),
        np.nextafter(unique_probability.astype(np.float32), np.float32(np.inf)).astype(np.float64),
    ))
    # Optimize inside the constraint. Clamping an unconstrained minimizer can
    # skip a better feasible interval when the discrete objective is nonconvex.
    feasible = np.flatnonzero(candidate_thresholds <= max_threshold)
    ranking = np.lexsort((candidate_delta[feasible], candidate_unknown[feasible]))
    best_index = int(feasible[ranking[0]])
    return float(candidate_thresholds[best_index])


def coordinate_descent(
    data: dict,
    lambda_value: float,
    initial_thresholds: np.ndarray,
    max_passes: int,
    max_threshold: float,
) -> Tuple[np.ndarray, dict, int]:
    thresholds = np.minimum(initial_thresholds.astype(np.float64, copy=True), max_threshold)
    legal_modes = static_legal_modes(data)
    thresholds[~legal_modes] = 1.0
    passes = 0
    for pass_index in range(max_passes):
        changed = False
        passes = pass_index + 1
        for mode in range(6):
            if not legal_modes[mode]:
                continue
            new_threshold = best_coordinate_threshold(
                data, thresholds, mode, lambda_value, max_threshold
            )
            if new_threshold != thresholds[mode]:
                thresholds[mode] = new_threshold
                changed = True
        if not changed:
            break
    return thresholds, evaluate_thresholds(data, thresholds, lambda_value), passes


def search_shape(
    data: dict, lambda_value: float, max_passes: int, max_threshold: float
) -> dict:
    starts = [
        np.zeros(6, dtype=np.float64),
        np.ones(6, dtype=np.float64),
    ]
    candidates = [
        coordinate_descent(data, lambda_value, start, max_passes, max_threshold)
        for start in starts
    ]
    thresholds, metrics, passes = min(
        candidates, key=lambda candidate: candidate[1]["objective"]
    )
    if metrics["unknown_rd_node_count"] or not np.isfinite(metrics["objective"]):
        raise ValueError("No RD-observed feasible threshold solution; inspect completed candidate coverage")
    return {
        "thresholds": thresholds,
        "metrics": metrics,
        "passes": passes,
    }


def size_sort_key(size_name: str) -> Tuple[int, int, int]:
    width, height = [int(value) for value in size_name.split("x")]
    return -width * height, -height, -width


def cfg_line(
    thresholds_by_size: Dict[str, List[float]], active_sizes=None
) -> str:
    parts = []
    items = thresholds_by_size.items()
    if active_sizes:
        items = ((size, values) for size, values in items if size in active_sizes)
    for size_name, values in sorted(items, key=lambda item: size_sort_key(item[0])):
        value_text = ",".join(f"{value:.17g}" for value in values)
        parts.append(f"{size_name}:[{value_text}]")
    return "FastPartitionThBySize : " + ";".join(parts)


def lambda_tag(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")












def run_search(args, lambdas):
    from collect_threshold_probabilities import search_probabilities
    cache, nodes, probability, collected = search_probabilities(args, require_rd=True)
    dataset = Path(cache["dataset_dir"])
    rd_path = dataset / "Luma_CU_RDCost.npy"
    rd = np.load(rd_path, mmap_mode="r")
    if len(rd) != len(nodes) or not rd["valid"][collected].all():
        raise ValueError("RD cache does not cover collected nodes")
    classifier = Classifier_I()  # Static architecture masks only; no forward pass.
    shape_data = {}
    for h, w in CLASSIFIER_SUPPORTED_SIZE_ORDER:
        selected = collected & (nodes[:, 2] == h) & (nodes[:, 3] == w)
        if not np.any(selected):
            continue
        completed = rd["completed"][selected].copy()
        delta = rd["rd_delta"][selected].copy()
        if np.any(completed & (~np.isfinite(delta) | (delta < 0))):
            raise ValueError("Completed RD regrets must be finite and nonnegative")
        labels = nodes[selected, 4].astype(np.int64)
        row = np.arange(len(labels))
        if not completed[row, labels].all() or np.any(delta[row, labels] != 0):
            raise ValueError("Selected label must have observed zero regret")
        static = classifier._static_mask(h, w).cpu().numpy().astype(bool)
        # Completed modes prove availability at the recorded search state.
        # Remaining legality is only a static proxy; the NPY has no canSplit mask.
        legal = np.broadcast_to(static, completed.shape).copy() | completed
        delta[~completed] = np.inf
        shape_data[f"{w*4}x{h*4}"] = dict(probability=probability[selected], legal=legal,
            completed=completed, rd_delta=delta, labels=labels, area4x4=h*w,
            rd_scale=float(h*w) if args.regret_weighting == "area4x4" else 1.0,
            completed_outside_model_mask=int(np.sum(completed & ~static)))
    active = {v.strip() for v in args.active_sizes.split(",") if v.strip()}
    if active - set(shape_data):
        raise ValueError("Active sizes have no collected observations: " + str(active-set(shape_data)))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for weight in lambdas:
        thresholds = {}; rows = []
        for size in sorted(shape_data, key=size_sort_key):
            if active and size not in active:
                continue
            data = shape_data[size]
            result = search_shape(data, weight, args.max_passes, args.max_threshold)
            threshold = result["thresholds"]
            thresholds[size] = threshold.tolist()
            keep = apply_vtm_fallback(data["legal"] & (data["probability"] >= threshold), data["legal"], data["probability"])
            labels = data["labels"]
            rows.append(dict(size=size, nodes=len(labels), regret_weight=data["rd_scale"], selected_label_rejected=int(np.sum(~keep[np.arange(len(labels)), labels])),
                unknown_rd_nodes=result["metrics"]["unknown_rd_node_count"], completed_outside_model_mask=data["completed_outside_model_mask"],
                metrics=result["metrics"], passes=result["passes"]))
        tag = lambda_tag(weight)
        (out / f"thresholds_cache_lambda_{tag}.cfg").write_text(
            "# Normalized RD regret; weighting=" + args.regret_weighting + "; no paper alpha; static legality proxy\n"
            + "# Calibration scope: " + cache["scope"] + "; dataset-order ONNX batches\n"
            + cfg_line(thresholds) + "\n")
        print("Cache search", weight, "shapes", len(rows), "nodes", sum(row["nodes"] for row in rows))


def main() -> None:
    args = parse_args()
    if args.max_passes <= 0 or not 0 <= args.max_threshold <= 1:
        raise ValueError("Require positive max-passes and max-threshold in [0,1]")
    run_search(args, parse_lambdas(args.lambdas))



if __name__ == "__main__":
    main()
