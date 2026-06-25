import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import paths


def export_array(pkl_path: Path, key: str, npy_path: Path, overwrite: bool = False):
    if npy_path.exists() and not overwrite:
        print(f"skip existing: {npy_path}")
        return

    print(f"loading {pkl_path}")
    payload = pd.read_pickle(pkl_path)
    if key not in payload:
        raise KeyError(f"{key} not found in {pkl_path}")

    print(f"saving {key} array {payload[key].shape} {payload[key].dtype} -> {npy_path}")
    np.save(npy_path, payload[key])

    del payload
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DIV2K")
    parser.add_argument("--split", default="training96")
    parser.add_argument("--component", default="Luma", choices=["Luma", "Chroma"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = paths.dataset_root() / args.dataset / args.split
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {dataset_dir}")

    export_array(
        dataset_dir / f"{args.component}_Input.pkl",
        "input",
        dataset_dir / f"{args.component}_Input.npy",
        overwrite=args.overwrite,
    )
    export_array(
        dataset_dir / f"{args.component}_Gridmap.pkl",
        "gridmap",
        dataset_dir / f"{args.component}_Gridmap.npy",
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
