import argparse
import os
import time
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

import paths


DATA_TYPE_TO_NAME = {
    1: "Training",
    2: "Testing",
    3: "Validating",
}

DATA_TYPE_TO_SPLIT_DIR = {
    1: "training",
    2: "testing",
    3: "validating",
}

DATA_TYPE_TO_DATASET = {
    1: "DIV2K",
    2: "HEVC_CTC",
    3: "DIV2K",
}

PARTITION_INFO_COLUMNS = [
    "sequence_name",
    "qp",
    "frame_id",
    "ctu_id",
    "cu_x",
    "cu_y",
    "cu_width",
    "cu_height",
    "depth",
    "qt_depth",
    "bt_depth",
    "mt_depth",
    "split_0",
    "split_1",
    "split_2",
    "split_3",
    "split_4",
    "split_5",
    "split_6",
    "split_7",
]

ID_COLUMNS = ["sequence_name", "qp", "frame_id", "ctu_id"]

DEFAULT_BLOCK_SIZE_MAP = {
    "Luma": 64,
    "Chroma": 32,
}

YUV420_COMPONENTS = {
    "Luma": ("Y",),
    "Chroma": ("U", "V"),
}

COMPONENT_ALIASES = {
    "both": ("Luma", "Chroma"),
    "luma": ("Luma",),
    "chroma": ("Chroma",),
}

PROGRESS_INTERVAL = 500000

plt = None
mlines = None


def log_progress(message):
    print(f"[createDataset] {message}", flush=True)


def resolve_split_name(data_type):
    if data_type not in DATA_TYPE_TO_NAME:
        raise ValueError(f"Unsupported data_type: {data_type}")
    return DATA_TYPE_TO_NAME[data_type]


def resolve_split_dir(data_type):
    if data_type not in DATA_TYPE_TO_SPLIT_DIR:
        raise ValueError(f"Unsupported data_type: {data_type}")
    return DATA_TYPE_TO_SPLIT_DIR[data_type]


def resolve_dataset_name(data_type, dataset_name=None):
    if dataset_name is not None:
        return dataset_name
    return DATA_TYPE_TO_DATASET[data_type]


def load_sequence_info(data_type, dataset_name=None, sequence_list=None):
    split_name = resolve_split_name(data_type)
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    if sequence_list is None:
        sequence_path = paths.sequence_list_path(dataset_name=dataset_name)
    else:
        sequence_path = Path(sequence_list)
        if not sequence_path.exists():
            sequence_path = paths.sequence_list_root() / sequence_list
    data = []
    with open(sequence_path, 'r') as seqs_info_fp:
        for line in seqs_info_fp:
            if "#" in line:
                continue
            if "end!!!!" in line:
                break
            striped = line.rstrip('\n')
            if not striped:
                continue
            fields = striped.split(',')
            if len(fields) == 1:
                file_name = fields[0]
                seq_name = Path(file_name).stem
                data.append([seq_name, file_name, None, None, None, None])
            else:
                data.append(fields)
    return split_name, dataset_name, np.array(data, dtype=object), sequence_path


def require_pandas():
    if pd is None:
        raise ImportError(
            "pandas is required for the new partition-info gridmap pipeline. "
            "Install pandas in the FastPartitionVTM environment before running this action."
        )


def require_matplotlib():
    global plt, mlines
    if plt is None or mlines is None:
        try:
            from matplotlib import pyplot as mpl_pyplot
            from matplotlib import lines as mpl_lines
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required when --show is enabled. "
                "Install matplotlib in the FastPartitionVTM environment or run without --show."
            ) from exc
        plt = mpl_pyplot
        mlines = mpl_lines
    if plt is None or mlines is None:
        raise ImportError(
            "matplotlib is required when --show is enabled. "
            "Install matplotlib in the FastPartitionVTM environment or run without --show."
        )


def build_ctu_gridmap_from_records(records, block_size):
    grid_size = block_size // 4
    gridmap = np.zeros((2, grid_size, grid_size), dtype=np.uint8)

    origin_x = min(record["cu_x"] for record in records) // block_size * block_size
    origin_y = min(record["cu_y"] for record in records) // block_size * block_size
    valid_right = max(record["cu_x"] + record["cu_width"] for record in records)
    valid_bottom = max(record["cu_y"] + record["cu_height"] for record in records)

    for record in records:
        x0 = (record["cu_x"] - origin_x) // 4
        y0 = (record["cu_y"] - origin_y) // 4
        x1 = x0 + record["cu_width"] // 4
        y1 = y0 + record["cu_height"] // 4

        x0_clip = max(0, min(grid_size, x0))
        y0_clip = max(0, min(grid_size, y0))
        x1_clip = max(0, min(grid_size, x1))
        y1_clip = max(0, min(grid_size, y1))

        v_col = x1 - 1
        h_row = y1 - 1
        if 0 <= v_col < grid_size and y0_clip < y1_clip:
            gridmap[0, y0_clip:y1_clip, v_col] = 1
        if 0 <= h_row < grid_size and x0_clip < x1_clip:
            gridmap[1, h_row, x0_clip:x1_clip] = 1

    valid_width_units = max(1, min(grid_size, (valid_right - origin_x + 3) // 4))
    valid_height_units = max(1, min(grid_size, (valid_bottom - origin_y + 3) // 4))
    gridmap[0, :, valid_width_units - 1] = 0
    gridmap[1, valid_height_units - 1, :] = 0
    gridmap[0, :, grid_size - 1] = 0
    gridmap[1, grid_size - 1, :] = 0
    return gridmap


def parse_partition_info_line(line):
    fields = line.split()
    if len(fields) != len(PARTITION_INFO_COLUMNS):
        raise RuntimeError(f"Expected {len(PARTITION_INFO_COLUMNS)} fields, got {len(fields)}: {line[:120]}")
    return {
        "sequence_name": fields[0],
        "qp": int(fields[1]),
        "frame_id": int(fields[2]),
        "ctu_id": int(fields[3]),
        "cu_x": int(fields[4]),
        "cu_y": int(fields[5]),
        "cu_width": int(fields[6]),
        "cu_height": int(fields[7]),
    }


def iter_partition_groups(partition_info_path, progress_interval=PROGRESS_INTERVAL):
    current_key = None
    current_records = []
    line_count = 0
    group_count = 0
    start_time = time.time()
    with partition_info_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line_count += 1
            record = parse_partition_info_line(line)
            key = tuple(record[column] for column in ID_COLUMNS)
            if current_key is not None and key != current_key:
                group_count += 1
                yield current_key, current_records
                current_records = []
            current_key = key
            current_records.append(record)
            if line_count % progress_interval == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                log_progress(
                    f"read {line_count:,} CU rows from {partition_info_path.name}, "
                    f"generated {group_count:,} gridmaps, {line_count / elapsed:,.0f} rows/s"
                )
    if current_key is not None:
        group_count += 1
        yield current_key, current_records
    elapsed = max(time.time() - start_time, 1e-6)
    log_progress(
        f"finished {partition_info_path.name}: {line_count:,} CU rows -> "
        f"{group_count:,} gridmaps in {elapsed:.1f}s"
    )


def save_gridmap_preview(gridmaps, output_path, block_size, title):
    require_matplotlib()
    if gridmaps.size == 0:
        return
    edge_counts = gridmaps[:, 0].sum(axis=(1, 2)) + gridmaps[:, 1].sum(axis=(1, 2))
    block_idx = int(np.argmax(edge_counts))
    arr = gridmaps[block_idx]
    grid_size = arr.shape[1]

    fig, ax = plt.subplots()
    for i in range(grid_size):
        for j in range(grid_size):
            ax.add_line(mlines.Line2D([(j + 1) * 4, (j + 1) * 4], [i * 4, (i + 1) * 4], color='black', linewidth=0.6, linestyle=':'))
            ax.add_line(mlines.Line2D([j * 4, (j + 1) * 4], [(i + 1) * 4, (i + 1) * 4], color='black', linewidth=0.6, linestyle=':'))
    for i in range(grid_size):
        for j in range(grid_size):
            ax.add_line(mlines.Line2D([(j + 1) * 4, (j + 1) * 4], [i * 4, (i + 1) * 4], color='orange', linewidth=2, alpha=arr[0, i, j]))
            ax.add_line(mlines.Line2D([j * 4, (j + 1) * 4], [(i + 1) * 4, (i + 1) * 4], color=(0 / 255, 47 / 255, 167 / 255), linewidth=2, alpha=arr[1, i, j]))
    ax.set_xlim(0, block_size)
    ax.set_ylim(0, block_size)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    tick_values = list(range(0, block_size + 1, 16))
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    plt.title(f'{title} block {block_idx}')
    plt.savefig(output_path)
    plt.close(fig)


def convert_component_partition_to_gridmap(component, partition_info_path, save_path, block_size, show=False):
    require_pandas()
    log_progress(f"start {component} gridmap from {partition_info_path}")
    ids = []
    gridmaps = []
    for key, records in iter_partition_groups(partition_info_path):
        ids.append(key)
        gridmaps.append(build_ctu_gridmap_from_records(records, block_size))

    if not gridmaps:
        raise RuntimeError(f"No CTU gridmaps generated from {partition_info_path}")

    id_df = pd.DataFrame(ids, columns=ID_COLUMNS)
    id_df["sample_index"] = np.arange(len(id_df), dtype=np.int64)
    id_df = id_df.set_index(ID_COLUMNS, drop=False)
    log_progress(f"stack {component} gridmaps: {len(gridmaps):,} samples")
    gridmap_arr = np.stack(gridmaps, axis=0)
    payload = {
        "component": component,
        "block_size": block_size,
        "grid_size": block_size // 4,
        "id_columns": ID_COLUMNS,
        "ids": id_df,
        "gridmap": gridmap_arr,
    }
    paths.ensure_dir(save_path.parent)
    pd.to_pickle(payload, save_path)
    pd.to_pickle(id_df, save_path.with_name(f"{component}_Ids.pkl"))
    log_progress(f"saved {component} gridmap to {save_path}")

    if show:
        preview_path = paths.ensure_dir(paths.output_root()) / f"{component}_gridmap_preview.png"
        save_gridmap_preview(gridmap_arr, preview_path, block_size, component)
    return save_path


def select_block_size_map(block_size_map, component):
    selected_components = COMPONENT_ALIASES[component]
    return {name: block_size_map[name] for name in selected_components}


def convert_partition_to_gridmap(data_type, block_size_map=None, show=False, dataset_name=None, component="both"):
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    split_dir = resolve_split_dir(data_type)
    partition_dir = paths.partition_dataset_root(dataset_name) / split_dir
    save_dir = paths.ensure_dir(paths.dataset_root() / dataset_name / split_dir)
    if block_size_map is None:
        block_size_map = DEFAULT_BLOCK_SIZE_MAP
    block_size_map = select_block_size_map(block_size_map, component)

    output_paths = []
    for component, block_size in block_size_map.items():
        partition_info_path = partition_dir / f"{component}_Partition_Info.txt"
        save_path = save_dir / f"{component}_Gridmap.pkl"
        output_paths.append(
            convert_component_partition_to_gridmap(
                component=component,
                partition_info_path=partition_info_path,
                save_path=save_path,
                block_size=block_size,
                show=show,
            )
        )
    return output_paths


def sequence_info_to_metadata(data_type, dataset_name=None, sequence_list=None):
    _, dataset_name, data, _ = load_sequence_info(data_type, dataset_name, sequence_list)
    metadata = {}
    for row in data:
        if len(row) < 5:
            raise ValueError(f"Sequence metadata requires at least 5 fields: {row}")
        sequence_name = str(row[0])
        metadata[sequence_name] = {
            "file_name": str(row[1]),
            "width": int(row[2]),
            "height": int(row[3]),
            "frame_count": int(row[4]),
            "subsample_ratio": int(row[5]) if len(row) > 5 and row[5] not in (None, "") else 1,
            "is10bit": bool(int(row[6])) if len(row) > 6 and row[6] not in (None, "") else False,
        }
    return metadata


def load_component_ids(component, split_dir, dataset_name):
    require_pandas()
    save_dir = paths.dataset_root() / dataset_name / split_dir
    ids_path = save_dir / f"{component}_Ids.pkl"
    if ids_path.exists():
        return pd.read_pickle(ids_path)
    gridmap_path = save_dir / f"{component}_Gridmap.pkl"
    if not gridmap_path.exists():
        raise FileNotFoundError(
            f"Neither {ids_path} nor {gridmap_path} exists. "
            "Run --action gridmap before generating input."
        )
    return pd.read_pickle(gridmap_path)["ids"]


def read_yuv420_frame(file_path, width, height, frame_id, is10bit=False):
    data_type = np.uint16 if is10bit else np.uint8
    bytes_per_sample = np.dtype(data_type).itemsize
    y_size = width * height
    uv_size = y_size // 4
    frame_size = y_size + 2 * uv_size
    with open(file_path, "rb") as fp:
        fp.seek(frame_id * frame_size * bytes_per_sample, os.SEEK_SET)
        y = np.fromfile(fp, dtype=data_type, count=y_size)
        u = np.fromfile(fp, dtype=data_type, count=uv_size)
        v = np.fromfile(fp, dtype=data_type, count=uv_size)
    if y.size != y_size or u.size != uv_size or v.size != uv_size:
        raise RuntimeError(f"Failed to read frame {frame_id} from {file_path}")
    y = y.reshape((height, width))
    u = u.reshape((height // 2, width // 2))
    v = v.reshape((height // 2, width // 2))
    if is10bit:
        y = np.round(y / 4).clip(0, 255).astype(np.uint8)
        u = np.round(u / 4).clip(0, 255).astype(np.uint8)
        v = np.round(v / 4).clip(0, 255).astype(np.uint8)
    return y, u, v


def block_origin_from_ctu_id(ctu_id, source_width, component):
    luma_ctu_size = 128
    luma_block_size = 64
    scale = 0 if component == "Luma" else 1
    ctu_size = luma_ctu_size >> scale
    block_size = luma_block_size >> scale
    sub_blocks_per_row = ctu_size // block_size
    ctu_per_row = (source_width + luma_ctu_size - 1) // luma_ctu_size
    outer_ctu_id = ctu_id // (sub_blocks_per_row * sub_blocks_per_row)
    sub_id = ctu_id % (sub_blocks_per_row * sub_blocks_per_row)
    outer_ctu_x = outer_ctu_id % ctu_per_row
    outer_ctu_y = outer_ctu_id // ctu_per_row
    sub_x = sub_id % sub_blocks_per_row
    sub_y = sub_id // sub_blocks_per_row
    x = outer_ctu_x * ctu_size + sub_x * block_size
    y = outer_ctu_y * ctu_size + sub_y * block_size
    return x, y


def crop_with_padding(frame, x, y, block_size):
    block = np.zeros((block_size, block_size), dtype=np.uint8)
    valid_h = max(0, min(block_size, frame.shape[0] - y))
    valid_w = max(0, min(block_size, frame.shape[1] - x))
    if valid_h > 0 and valid_w > 0:
        block[:valid_h, :valid_w] = frame[y:y + valid_h, x:x + valid_w]
    return block


def save_component_input(component, ids, metadata, dataset_name, split_dir, block_size):
    components = YUV420_COMPONENTS[component]
    input_shape = (len(ids), len(components), block_size, block_size)
    log_progress(f"start {component} input: allocate array shape={input_shape}")
    input_blocks = np.zeros(input_shape, dtype=np.uint8)
    video_root = paths.video_dataset_root(dataset_name)
    total_groups = ids[["sequence_name", "frame_id"]].drop_duplicates().shape[0]
    processed_groups = 0
    processed_samples = 0
    start_time = time.time()

    for (sequence_name, frame_id), group in ids.groupby(level=["sequence_name", "frame_id"], sort=False):
        processed_groups += 1
        sequence_name = str(sequence_name)
        if sequence_name not in metadata:
            raise KeyError(f"Missing sequence metadata for {sequence_name}")
        seq_meta = metadata[sequence_name]
        file_path = video_root / seq_meta["file_name"]
        if not file_path.exists():
            raise FileNotFoundError(f"YUV file not found: {file_path}")
        y_frame, u_frame, v_frame = read_yuv420_frame(
            file_path=file_path,
            width=seq_meta["width"],
            height=seq_meta["height"],
            frame_id=int(frame_id),
            is10bit=seq_meta["is10bit"],
        )
        source_frames = (y_frame,) if component == "Luma" else (u_frame, v_frame)
        for row in group.itertuples(index=False):
            sample_index = int(row.sample_index)
            x, y = block_origin_from_ctu_id(int(row.ctu_id), seq_meta["width"], component)
            for channel_idx, frame in enumerate(source_frames):
                input_blocks[sample_index, channel_idx] = crop_with_padding(frame, x, y, block_size)
            processed_samples += 1
        if processed_groups == 1 or processed_groups % 50 == 0 or processed_groups == total_groups:
            elapsed = max(time.time() - start_time, 1e-6)
            log_progress(
                f"{component} input {processed_groups:,}/{total_groups:,} sequence-frame groups, "
                f"{processed_samples:,}/{len(ids):,} samples, {processed_samples / elapsed:,.0f} samples/s"
            )

    save_dir = paths.ensure_dir(paths.dataset_root() / dataset_name / split_dir)
    save_path = save_dir / f"{component}_Input.pkl"
    payload = {
        "component": component,
        "block_size": block_size,
        "id_columns": ID_COLUMNS,
        "ids": ids.copy(),
        "input": input_blocks,
    }
    pd.to_pickle(payload, save_path)
    log_progress(f"saved {component} input to {save_path}")
    return save_path


def convert_yuv_to_input(data_type, block_size_map=None, dataset_name=None, sequence_list=None, component="both"):
    require_pandas()
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    split_dir = resolve_split_dir(data_type)
    if block_size_map is None:
        block_size_map = DEFAULT_BLOCK_SIZE_MAP
    block_size_map = select_block_size_map(block_size_map, component)
    metadata = sequence_info_to_metadata(data_type, dataset_name, sequence_list)

    output_paths = []
    for component, block_size in block_size_map.items():
        ids = load_component_ids(component, split_dir, dataset_name)
        output_paths.append(
            save_component_input(
                component=component,
                ids=ids,
                metadata=metadata,
                dataset_name=dataset_name,
                split_dir=split_dir,
                block_size=block_size,
            )
        )
    return output_paths


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--sequence-list', type=str, default=None)
    parser.add_argument('--luma-block-size', type=int, default=64)
    parser.add_argument('--chroma-block-size', type=int, default=32)
    parser.add_argument('--component', choices=['both', 'luma', 'chroma'], default='both')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--action', choices=['gridmap', 'input', 'gridmap-input'], default='gridmap-input')
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    block_size_map = {
        "Luma": args.luma_block_size,
        "Chroma": args.chroma_block_size,
    }
    if args.action in ('gridmap', 'gridmap-input'):
        convert_partition_to_gridmap(
            args.data_type,
            block_size_map=block_size_map,
            show=args.show,
            dataset_name=args.dataset,
            component=args.component,
        )
    if args.action in ('input', 'gridmap-input'):
        convert_yuv_to_input(
            args.data_type,
            block_size_map=block_size_map,
            dataset_name=args.dataset,
            sequence_list=args.sequence_list,
            component=args.component,
        )
