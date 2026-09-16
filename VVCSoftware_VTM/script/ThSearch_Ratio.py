#!/usr/bin/env python3
"""Search FastPartition thresholds with the simple C/D ratio method.

This script reads validated node-aligned ONNX probabilities and searches
one threshold per CU size and split mode by taking
the low quantile of the true-class softmax probabilities.

For a target ratio r, the threshold of a given (size, mode) is chosen so that
about r of validation samples whose true label is that mode would fall below
the threshold.  This is the engineering version of the probability-adaptive
false-negative-ratio threshold method.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
NETWORK_SRC = PROJECT_ROOT / "network" / "src"
if str(NETWORK_SRC) not in sys.path:
    sys.path.insert(0, str(NETWORK_SRC))

from model import Classifier_I
from utils import CLASSIFIER_SUPPORTED_SIZE_ORDER


CLASS_NAMES = ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"]
DEFAULT_OUT_DIR = SCRIPT_DIR / "output" / "th_search_ratio"
DEFAULT_MODE_RATIOS = [0.003, 0.003, 0.006, 0.006, 0.012, 0.012]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple C/D threshold search by shape x mode false-negative ratio."
    )
    parser.add_argument("--ratio", type=float, default=None, help="Single fallback ratio for all modes.")
    parser.add_argument(
        "--mode-ratios",
        default=",".join(str(v) for v in DEFAULT_MODE_RATIOS),
        help="Comma-separated ratios in class order NO_SPLIT,QT,BTH,BTV,TTH,TTV.",
    )
    parser.add_argument("--ratio-denominator", choices=["positive", "all-nodes"], default="positive",
                        help="FN budget denominator: true-class nodes (legacy) or all nodes of the same pixel shape.")
    parser.add_argument("--default-threshold", type=float, default=0.1)
    parser.add_argument("--illegal-threshold", type=float, default=1.0)
    parser.add_argument("--dataset-dir", help="Directory containing createDataset NPY/PKL files")
    parser.add_argument("--probability-cache", help="Reuse probabilities generated from dataset NPY/PKL and ONNX")
    parser.add_argument("--batch-size", type=int, default=16, help="Swin inference batch; samples follow NPY order")
    parser.add_argument("--bundle", help="Deployment ONNX bundle directory")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--active-sizes",
        default="",
        help="comma-separated CU pixel sizes to emit in the VTM cfg; empty keeps all sizes",
    )
    return parser.parse_args()


def parse_mode_ratios(args: argparse.Namespace) -> List[float]:
    if args.ratio is not None:
        ratios = [float(args.ratio) for _ in CLASS_NAMES]
    else:
        parts = [part.strip() for part in str(args.mode_ratios).split(",") if part.strip()]
        if len(parts) != len(CLASS_NAMES):
            raise ValueError(
                "--mode-ratios must contain 6 values in class order "
                + ",".join(CLASS_NAMES)
            )
        ratios = [float(part) for part in parts]
    for ratio in ratios:
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("All ratios must be in [0, 1]")
    return ratios




def static_legal_mask(classifier: Classifier_I, grid_h: int, grid_w: int) -> List[bool]:
    mask = classifier._static_mask(grid_h, grid_w)
    return [bool(v) for v in mask.detach().cpu().tolist()]


def pixel_size_name(grid_h: int, grid_w: int) -> str:
    return f"{grid_w * 4}x{grid_h * 4}"


def ensure_stats_entry(
    stats: Dict[str, dict],
    classifier: Classifier_I,
    grid_h: int,
    grid_w: int,
) -> dict:
    size_name = pixel_size_name(grid_h, grid_w)
    if size_name not in stats:
        stats[size_name] = {
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "width": int(grid_w * 4),
            "height": int(grid_h * 4),
            "total_nodes": 0,
            "legal": static_legal_mask(classifier, grid_h, grid_w),
            "positive_probs": [[] for _ in CLASS_NAMES],
            "legal_counts": [0 for _ in CLASS_NAMES],
        }
    return stats[size_name]




def quantile_threshold(values: List[float], ratio: float) -> Tuple[float, int, float]:
    """Return threshold, estimated FN count, and estimated FN ratio.

    The encoder keeps candidates with probability >= threshold, so samples
    exactly equal to the threshold are counted as kept.
    """
    if not values:
        return math.nan, 0, math.nan
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = int(ordered.size)
    kth = int(math.floor(float(ratio) * n))
    kth = max(0, min(kth, n - 1))
    threshold = float(ordered[kth])
    fn_count = int(np.sum(ordered < threshold))
    fn_ratio = fn_count / float(n)
    return threshold, fn_count, fn_ratio


def build_thresholds(stats: Dict[str, dict], ratios: List[float], default_threshold: float,
                     illegal_threshold: float, ratio_denominator: str = "positive") -> dict:
    if ratio_denominator not in ("positive", "all-nodes"):
        raise ValueError("Unknown ratio denominator: " + ratio_denominator)
    thresholds = {}
    rows = []

    for grid_h, grid_w in CLASSIFIER_SUPPORTED_SIZE_ORDER:
        size_name = pixel_size_name(grid_h, grid_w)
        entry = stats.get(size_name)
        if entry is None:
            entry = {
                "grid_h": int(grid_h),
                "grid_w": int(grid_w),
                "width": int(grid_w * 4),
                "height": int(grid_h * 4),
                "total_nodes": 0,
                "legal": [False for _ in CLASS_NAMES],
                "positive_probs": [[] for _ in CLASS_NAMES],
                "legal_counts": [0 for _ in CLASS_NAMES],
            }

        shape_thresholds = []
        for cls, class_name in enumerate(CLASS_NAMES):
            values = entry["positive_probs"][cls]
            legal = bool(entry["legal"][cls])
            if not legal:
                threshold = float(illegal_threshold)
                fn_count = 0
                fn_ratio = math.nan
            elif values:
                # Paper-style all-node FN = class prior * conditional FN.
                # Retain tied probabilities and never exceed the integer FN budget.
                conditional_ratio = ratios[cls]
                if ratio_denominator == "all-nodes":
                    conditional_ratio = min(1.0, ratios[cls] * entry["total_nodes"] / len(values))
                threshold, fn_count, fn_ratio = quantile_threshold(values, conditional_ratio)
            else:
                threshold = float(default_threshold)
                fn_count = 0
                fn_ratio = math.nan

            shape_thresholds.append(threshold)
            rows.append(
                {
                    "size": size_name,
                    "width": entry["width"],
                    "height": entry["height"],
                    "grid_w": entry["grid_w"],
                    "grid_h": entry["grid_h"],
                    "mode": class_name,
                    "legal": int(legal),
                    "total_nodes": entry["total_nodes"],
                    "legal_count": entry["legal_counts"][cls],
                    "positive_count": len(values),
                    "target_fn_ratio": ratios[cls],
                    "ratio_denominator": ratio_denominator,
                    "estimated_fn_ratio_all_nodes": fn_count / entry["total_nodes"] if entry["total_nodes"] else math.nan,
                    "threshold": threshold,
                    "estimated_fn_count": fn_count,
                    "estimated_fn_ratio": fn_ratio,
                    "positive_min": min(values) if values else math.nan,
                    "positive_mean": float(np.mean(values)) if values else math.nan,
                    "positive_max": max(values) if values else math.nan,
                }
            )
        thresholds[size_name] = shape_thresholds

    return {"thresholds": thresholds, "rows": rows}


def cfg_line(
    thresholds: Dict[str, List[float]], active_sizes: Optional[set[str]] = None
) -> str:
    parts = []
    def sort_key(item):
        size = item[0]
        width, height = [int(v) for v in size.split("x")]
        return (-width * height, -height, -width)

    selected = thresholds.items()
    if active_sizes:
        selected = ((size, values) for size, values in thresholds.items() if size in active_sizes)
    for size_name, values in sorted(selected, key=sort_key):
        # Preserve the cutoff exactly when C++ reads the values as doubles.
        value_text = ",".join(f"{v:.17g}" for v in values)
        parts.append(f"{size_name}:[{value_text}]")
    return "FastPartitionThBySize : " + ";".join(parts)


def stats_from_dataset(args):
    from collect_threshold_probabilities import search_probabilities
    manifest, nodes, probability, valid = search_probabilities(args)
    # Only use the architecture's static masks; never run random/unloaded weights.
    classifier = Classifier_I()
    stats = {}
    total = 0
    for grid_h, grid_w in CLASSIFIER_SUPPORTED_SIZE_ORDER:
        selected = valid & (nodes[:, 2] == grid_h) & (nodes[:, 3] == grid_w)
        count = int(selected.sum())
        if not count:
            continue
        entry = ensure_stats_entry(stats, classifier, grid_h, grid_w)
        probs = probability[selected]
        labels = nodes[selected, 4]
        mask = np.asarray(entry["legal"], dtype=bool)
        if np.any(probs[:, ~mask] != 0) or not mask[labels].all():
            raise ValueError("Cached outputs or labels disagree with classifier static mask")
        entry["total_nodes"] = count
        entry["legal_counts"] = [count if legal else 0 for legal in mask]
        entry["positive_probs"] = [probs[labels == cls, cls].astype(float).tolist() for cls in range(6)]
        total += count
    return stats, total, manifest


def write_outputs(out_dir: Path, args: argparse.Namespace, result: dict, total_nodes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = result["thresholds"]
    active_sizes = {s.strip() for s in str(args.active_sizes).split(",") if s.strip()}
    unknown_sizes = active_sizes - set(thresholds)
    if unknown_sizes:
        raise ValueError("Unknown --active-sizes: " + ", ".join(sorted(unknown_sizes)))
    cfg_path = out_dir / "thresholds_ratio.cfg"

    with cfg_path.open("w", encoding="utf-8") as f:
        f.write("# Generated by ThSearch_Ratio.py\n")
        if args.cache_manifest:
            f.write("# Calibration scope: " + args.cache_manifest["scope"] + "; dataset-order ONNX batches\n")
        f.write("# Class order: " + ",".join(CLASS_NAMES) + "\n")
        f.write(cfg_line(thresholds, active_sizes or None))
        f.write("\n")

    print("Wrote:", cfg_path)


def main() -> None:
    args = parse_args()
    ratios = parse_mode_ratios(args)
    stats, total_nodes, args.cache_manifest = stats_from_dataset(args)
    dataset = Path(args.cache_manifest["dataset_dir"])
    args.dataset, args.split, args.component = dataset.parent.name, dataset.name, "Luma"
    result = build_thresholds(stats, ratios, args.default_threshold, args.illegal_threshold, args.ratio_denominator)
    write_outputs(Path(args.out_dir), args, result, total_nodes)



if __name__ == "__main__":
    main()
