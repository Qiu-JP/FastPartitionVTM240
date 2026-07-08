from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]


def project_root() -> Path:
    return _PROJECT_ROOT


def data_root() -> Path:
    return project_root() / "data"


def network_root() -> Path:
    return project_root() / "network"


def ref_model_root() -> Path:
    return project_root() / "ref_model"


def codec_cfg_root() -> Path:
    return data_root() / "CodecTrainCfg"


def video_root() -> Path:
    return data_root() / "video"


def partition_root() -> Path:
    return data_root() / "partition"


def dataset_root() -> Path:
    return data_root() / "dataset"


def metadata_root() -> Path:
    return data_root() / "metadata"


def sequence_list_root() -> Path:
    return network_root() / "script"


def checkpoints_root() -> Path:
    return network_root() / "checkpoints"


def output_root() -> Path:
    return network_root() / "output"


def templates_root() -> Path:
    return sequence_list_root()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    video_subdir: str
    partition_subdir: str
    default_sequence_list: str


_DATASET_SPECS = {
    "DIV2K": DatasetSpec(
        name="DIV2K",
        video_subdir="DIV2K",
        partition_subdir="DIV2K",
        default_sequence_list="Training_Sequences_DIV2K.txt",
    ),
    "HEVC_CTC": DatasetSpec(
        name="HEVC_CTC",
        video_subdir="HEVC_CTC",
        partition_subdir="HEVC_CTC",
        default_sequence_list="Testing_Sequences.txt",
    ),
}


def dataset_spec(dataset_name: str) -> DatasetSpec:
    key = dataset_name.upper()
    if key not in _DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return _DATASET_SPECS[key]


def sequence_list_path(filename: Optional[str] = None, dataset_name: Optional[str] = None) -> Path:
    if filename is not None:
        return sequence_list_root() / filename
    if dataset_name is None:
        raise ValueError("Either filename or dataset_name must be provided")
    return sequence_list_root() / dataset_spec(dataset_name).default_sequence_list


def video_dataset_root(dataset_name: str) -> Path:
    return video_root() / dataset_spec(dataset_name).video_subdir


def partition_dataset_root(dataset_name: str, qp: Optional[int] = None) -> Path:
    base = partition_root() / dataset_spec(dataset_name).partition_subdir
    if qp is None:
        return base
    return base / f"qp_{qp}"


def codec_cfg_dataset_root(dataset_name: str, qp: Optional[int] = None) -> Path:
    base = codec_cfg_root() / dataset_spec(dataset_name).name
    if qp is None:
        return base
    return base / f"qp_{qp}"
