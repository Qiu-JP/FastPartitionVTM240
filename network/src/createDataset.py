import argparse
import math
import os
import pickle
import re
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

import paths

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


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

DATA_TYPE_TO_OUTPUT_SPLIT_DIR = {
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
    "Luma": 32,
    "Chroma": 16,
}

DEFAULT_INPUT_SIZE_MAP = {
    "Luma": 48,
    "Chroma": 32,
}

RDO_FILE_PATTERN = re.compile(r"^(?P<sequence>.+)_QP(?P<qp>\d+)\.tsv$")
RDO_SUPPORTED_SIZES = {
    (16, 16), (8, 8), (8, 4), (4, 8), (8, 2), (2, 8),
    (8, 1), (1, 8), (4, 2), (2, 4), (4, 1), (1, 4),
    (4, 4), (2, 2), (2, 1), (1, 2),
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

CLASSIFIER_I_CLASS_NAMES = {
    0: "NO_SPLIT",
    1: "QT",
    2: "BTH",
    3: "BTV",
    4: "TTH",
    5: "TTV",
}

CLASSIFIER_I_GRID_SPECS = {
    (16, 16): [0, 1],
    (8, 8): [0, 1, 2, 3, 4, 5],
    (8, 4): [0, 2, 3, 4, 5],
    (4, 8): [0, 2, 3, 4, 5],
    (8, 2): [0, 2, 3, 4],
    (2, 8): [0, 2, 3, 5],
    (8, 1): [0, 2, 4],
    (1, 8): [0, 3, 5],
    (4, 2): [0, 2, 3, 4],
    (2, 4): [0, 2, 3, 5],
    (4, 1): [0, 2, 4],
    (1, 4): [0, 3, 5],
    (4, 4): [0, 1, 2, 3, 4, 5],
    (2, 2): [0, 2, 3],
    (2, 1): [0, 2],
    (1, 2): [0, 3],
}

CLASSIFIER_I_PRETRAIN_SAMPLES_PER_CLASS = 256

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


def resolve_output_split_dir(data_type):
    if data_type not in DATA_TYPE_TO_OUTPUT_SPLIT_DIR:
        raise ValueError(f"Unsupported data_type: {data_type}")
    return DATA_TYPE_TO_OUTPUT_SPLIT_DIR[data_type]


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
    record = {"sequence_name": fields[0]}
    for column, value in zip(PARTITION_INFO_COLUMNS[1:], fields[1:]):
        record[column] = int(value)
    return record


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


def center_crop(image, crop_size):
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


def read_payload_array(dataset_dir, component, kind):
    pkl_path = dataset_dir / f"{component}_{kind}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found")
    payload = pd.read_pickle(pkl_path)
    if not isinstance(payload, dict) or "array_file" not in payload:
        raise RuntimeError(f"{pkl_path} is not in the current metadata format")
    array_path = dataset_dir / payload["array_file"]
    if not array_path.exists():
        raise FileNotFoundError(f"{array_path} not found")
    return payload, np.load(array_path, mmap_mode="r")


def sample_position(ids, sample_id=None, sample_index=None):
    if sample_id is not None:
        if sample_id not in ids.index:
            raise KeyError(f"Sample id not found: {sample_id}")
        return int(ids.loc[sample_id, "sample_index"])
    if sample_index is None:
        raise ValueError("Either sample_id or sample_index must be provided")
    if sample_index < 0 or sample_index >= len(ids):
        raise IndexError(f"sample_index {sample_index} out of range [0, {len(ids)})")
    return int(sample_index)


def select_preview_sample(gridmap_payload, gridmap_array, sample_id=None, sample_index=None):
    ids = gridmap_payload["ids"]
    if sample_id is not None or sample_index is not None:
        pos = sample_position(ids, sample_id=sample_id, sample_index=sample_index)
    else:
        edge_counts = gridmap_array[:, 0].sum(axis=(1, 2)) + gridmap_array[:, 1].sum(axis=(1, 2))
        pos = int(np.argmax(edge_counts))
    row = ids.iloc[pos]
    resolved_id = tuple(row[column] for column in ID_COLUMNS)
    return resolved_id, pos


def build_preview_background(dataset_dir, sample_id):
    luma_payload, luma_array = read_payload_array(dataset_dir, "Luma", "Input")
    chroma_payload, chroma_array = read_payload_array(dataset_dir, "Chroma", "Input")
    luma_ids = luma_payload["ids"]
    chroma_ids = chroma_payload["ids"]
    if sample_id not in luma_ids.index:
        raise KeyError(f"Sample id not found in Luma input: {sample_id}")
    if sample_id not in chroma_ids.index:
        raise KeyError(f"Sample id not found in Chroma input: {sample_id}")

    luma = luma_array[sample_position(luma_ids, sample_id=sample_id)][0]
    luma_lcu = center_crop(luma, DEFAULT_BLOCK_SIZE_MAP["Luma"])
    chroma = chroma_array[sample_position(chroma_ids, sample_id=sample_id)]
    chroma_lcu = np.stack(
        (
            center_crop(chroma[0], DEFAULT_BLOCK_SIZE_MAP["Chroma"]),
            center_crop(chroma[1], DEFAULT_BLOCK_SIZE_MAP["Chroma"]),
        ),
        axis=0,
    )
    u = upsample_nearest(chroma_lcu[0], luma_lcu.shape)
    v = upsample_nearest(chroma_lcu[1], luma_lcu.shape)
    return yuv_to_rgb(np.stack((luma_lcu, u, v), axis=-1))


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


def draw_gridmap_panel(ax, value_ax, image, gridmap, title=None):
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

    value_ax.axis("off")
    value_ax.text(
        0.0,
        1.0,
        format_gridmap_values(gridmap),
        transform=value_ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=5.6,
    )


def visualize_dataset_gridmap(data_type, dataset_name=None, component="luma", sample_id=None, sample_index=None, output_name=None):
    require_pandas()
    require_matplotlib()
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    split_dir = resolve_output_split_dir(data_type)
    component_name = COMPONENT_ALIASES[component][0]
    dataset_dir = paths.dataset_root() / dataset_name / split_dir
    gridmap_payload, gridmap_array = read_payload_array(dataset_dir, component_name, "Gridmap")
    resolved_id, gridmap_pos = select_preview_sample(
        gridmap_payload=gridmap_payload,
        gridmap_array=gridmap_array,
        sample_id=sample_id,
        sample_index=sample_index,
    )
    image = build_preview_background(dataset_dir, resolved_id)
    gridmap = gridmap_array[gridmap_pos]

    if output_name is None:
        seq, qp, frame_id, ctu_id = resolved_id
        output_name = f"{dataset_name}_{split_dir}_{component_name}_{seq}_qp{qp}_f{frame_id}_ctu{ctu_id}_gridmap.png"
    output_path = paths.ensure_dir(paths.network_root() / "figures") / Path(output_name).name

    fig, (ax, value_ax) = plt.subplots(
        1,
        2,
        figsize=(8.0, 4.0),
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    draw_gridmap_panel(
        ax,
        value_ax,
        image=image,
        gridmap=gridmap,
        title=f"{dataset_name} {split_dir} {component_name} {resolved_id}",
    )
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log_progress(f"saved {component_name} gridmap preview to {output_path}")
    return output_path


def convert_component_partition_to_gridmap(component, partition_info_path, save_path, block_size):
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
    paths.ensure_dir(save_path.parent)
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, gridmap_arr)
    payload = {
        "component": component,
        "block_size": block_size,
        "grid_size": block_size // 4,
        "id_columns": ID_COLUMNS,
        "ids": id_df,
        "array_file": npy_path.name,
        "array_key": "gridmap",
        "array_shape": tuple(gridmap_arr.shape),
        "array_dtype": str(gridmap_arr.dtype),
    }
    pd.to_pickle(payload, save_path)
    log_progress(f"saved {component} gridmap array to {npy_path}")
    log_progress(f"saved {component} gridmap metadata to {save_path}")

    return save_path


def classifier_label_from_part_split(part_split):
    if part_split == 2000:
        return 0
    if part_split in (1, 2, 3, 4, 5):
        return int(part_split)
    if part_split == 0:
        return None
    raise ValueError(f"Unsupported CU PartSplit value for classifier label: {part_split}")


def split_node_children_abs(x, y, w, h, label):
    if label == 1:
        w2 = w // 2
        h2 = h // 2
        return [
            (x, y, w2, h2),
            (x + w2, y, w2, h2),
            (x, y + h2, w2, h2),
            (x + w2, y + h2, w2, h2),
        ]
    if label == 2:
        h2 = h // 2
        return [(x, y, w, h2), (x, y + h2, w, h2)]
    if label == 3:
        w2 = w // 2
        return [(x, y, w2, h), (x + w2, y, w2, h)]
    if label == 4:
        h1 = h // 4
        h2 = h // 2
        return [(x, y, w, h1), (x, y + h1, w, h2), (x, y + h1 + h2, w, h1)]
    if label == 5:
        w1 = w // 4
        w2 = w // 2
        return [(x, y, w1, h), (x + w1, y, w2, h), (x + w1 + w2, y, w1, h)]
    return []


def child_contains_leaf(child, leaf):
    cx, cy, cw, ch = child
    lx, ly, lw, lh = leaf
    return lx >= cx and ly >= cy and lx + lw <= cx + cw and ly + lh <= cy + ch


def node_contains_node(outer, inner):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh


def reconstruct_classifier_nodes_from_records(records, block_size):
    origin_x = min(record["cu_x"] for record in records) // block_size * block_size
    origin_y = min(record["cu_y"] for record in records) // block_size * block_size
    export_node = (origin_x, origin_y, block_size, block_size)
    partition_root_size = block_size * 2
    partition_root_x = origin_x // partition_root_size * partition_root_size
    partition_root_y = origin_y // partition_root_size * partition_root_size
    node_labels = {}
    conflicts = []

    for record in records:
        node = (partition_root_x, partition_root_y, partition_root_size, partition_root_size)
        leaf = (record["cu_x"], record["cu_y"], record["cu_width"], record["cu_height"])
        split_values = [record[f"split_{idx}"] for idx in range(8)]

        for depth, part_split in enumerate(split_values):
            label = classifier_label_from_part_split(part_split)
            if label is None:
                continue

            should_record = node_contains_node(export_node, node)
            if label == 0 and not should_record and child_contains_leaf(node, export_node):
                should_record = True
                node = export_node

            if should_record:
                if node in node_labels and node_labels[node] != label:
                    conflicts.append((node, node_labels[node], label, record))
                    break
                node_labels[node] = label

            if label == 0:
                break

            children = split_node_children_abs(*node, label)
            next_node = None
            for child in children:
                if child_contains_leaf(child, leaf):
                    next_node = child
                    break
            if next_node is None:
                raise RuntimeError(
                    "Cannot follow split path for leaf "
                    f"{leaf} at node {node} with label {label} in "
                    f"{record['sequence_name']} qp={record['qp']} "
                    f"frame={record['frame_id']} ctu={record['ctu_id']}"
                )
            node = next_node

    if conflicts:
        node, old_label, new_label, record = conflicts[0]
        raise RuntimeError(
            "Conflicting split labels while reconstructing tree: "
            f"node={node}, old={old_label}, new={new_label}, "
            f"sample={(record['sequence_name'], record['qp'], record['frame_id'], record['ctu_id'])}"
        )

    ordered_nodes = []
    stack = [(origin_x, origin_y, block_size, block_size, 0)]
    seen = set()
    while stack:
        x, y, w, h, depth = stack.pop()
        # VVC split syntax can describe a child narrower than the 4x4
        # classifier grid. Such a child is terminal/unsupported for this
        # dataset and must not become a zero-sized grid node.
        if w < 4 or h < 4:
            continue
        node = (x, y, w, h)
        if node in seen or node not in node_labels:
            continue
        seen.add(node)
        # A 4x4 luma CU is already at the VTM minimum and has no
        # partition decision to learn.
        if w == 4 and h == 4:
            continue
        label = node_labels[node]
        if x % 4 != 0 or y % 4 != 0 or w % 4 != 0 or h % 4 != 0:
            continue
        ordered_nodes.append((x, y, w, h, depth, label))
        if label != 0:
            children = split_node_children_abs(x, y, w, h, label)
            for child in reversed(children):
                stack.append((*child, depth + 1))

    return origin_x, origin_y, ordered_nodes


def convert_component_partition_to_cu_tree(component, partition_info_path, save_path, block_size):
    require_pandas()
    log_progress(f"start {component} CU tree labels from {partition_info_path}")
    paths.ensure_dir(save_path.parent)
    sample_rows = []
    offsets = [0]
    sample_index = 0
    total_nodes = 0
    start_time = time.time()
    nodes_path = save_path.with_suffix(".npy")
    raw_nodes_path = save_path.with_name(f"{component}_CU_Tree_Nodes.tmp.i2.bin")

    with open(raw_nodes_path, "wb") as nodes_fp:
        for key, records in iter_partition_groups(partition_info_path):
            origin_x, origin_y, nodes = reconstruct_classifier_nodes_from_records(records, block_size)
            sequence_name, qp, frame_id, ctu_id = key
            sample_rows.append(
                {
                    "sequence_name": sequence_name,
                    "qp": qp,
                    "frame_id": frame_id,
                    "ctu_id": ctu_id,
                    "sample_index": sample_index,
                    "node_start": total_nodes,
                    "node_end": total_nodes + len(nodes),
                    "node_count": len(nodes),
                }
            )
            node_values = []
            for x, y, w, h, depth, label in nodes:
                grid_x = (x - origin_x) // 4
                grid_y = (y - origin_y) // 4
                grid_w = w // 4
                grid_h = h // 4
                node_values.append((grid_y, grid_x, grid_h, grid_w, label))
            np.asarray(node_values, dtype=np.int16).tofile(nodes_fp)
            total_nodes += len(nodes)
            offsets.append(total_nodes)
            sample_index += 1
            if sample_index == 1 or sample_index % 50000 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                log_progress(
                    f"{component} CU tree {sample_index:,} samples, "
                    f"{total_nodes:,} nodes, {total_nodes / elapsed:,.0f} nodes/s"
                )

    if total_nodes == 0:
        raise RuntimeError(f"No CU tree nodes generated from {partition_info_path}")

    raw_nodes = np.memmap(raw_nodes_path, mode="r", dtype=np.int16, shape=(int(total_nodes), 5))
    node_array = np.lib.format.open_memmap(
        nodes_path,
        mode="w+",
        dtype=np.int16,
        shape=(int(total_nodes), 5),
    )
    chunk_size = 1_000_000
    for start in range(0, int(total_nodes), chunk_size):
        end = min(start + chunk_size, int(total_nodes))
        node_array[start:end] = raw_nodes[start:end]
    node_array.flush()
    del node_array
    del raw_nodes
    raw_nodes_path.unlink()

    samples_df = pd.DataFrame(sample_rows)
    samples_df = samples_df.set_index(ID_COLUMNS, drop=False)
    payload = {
        "format": "cu_tree_numpy",
        "component": component,
        "block_size": block_size,
        "grid_size": block_size // 4,
        "id_columns": ID_COLUMNS,
        "samples": samples_df,
        "node_columns": ["grid_y", "grid_x", "grid_h", "grid_w", "label"],
        "array_file": nodes_path.name,
        "array_key": "cu_tree_nodes",
        "array_shape": (int(total_nodes), 5),
        "array_dtype": "int16",
        "offsets": np.asarray(offsets, dtype=np.int64),
        "class_order": ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"],
    }
    pd.to_pickle(payload, save_path)
    log_progress(f"saved {component} CU tree node array to {nodes_path}")
    log_progress(f"saved {component} CU tree labels to {save_path}")
    return save_path


def convert_partition_to_cu_tree(data_type, block_size_map=None, dataset_name=None, component="both", output_split=None, source_dataset=None):
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    partition_split_dir = resolve_split_dir(data_type)
    output_split_dir = output_split or resolve_output_split_dir(data_type)
    source_dataset = source_dataset or dataset_name
    partition_dir = paths.partition_dataset_root(source_dataset) / partition_split_dir
    save_dir = paths.ensure_dir(paths.dataset_root() / dataset_name / output_split_dir)
    if block_size_map is None:
        block_size_map = DEFAULT_BLOCK_SIZE_MAP
    block_size_map = select_block_size_map(block_size_map, component)

    output_paths = []
    for component, block_size in block_size_map.items():
        partition_info_path = partition_dir / f"{component}_Partition_Info.txt"
        save_path = save_dir / f"{component}_CU_Tree.pkl"
        output_paths.append(
            convert_component_partition_to_cu_tree(
                component=component,
                partition_info_path=partition_info_path,
                save_path=save_path,
                block_size=block_size,
            )
        )
    return output_paths


def load_rdo_sequence_widths(path):
    widths = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for raw_line in source:
            line = raw_line.strip()
            if not line or line.startswith("#") or "end!!!!" in line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4:
                raise ValueError(f"Malformed sequence-list line: {raw_line.rstrip()}")
            widths[fields[0]] = int(fields[2])
    return widths


def discover_rdo_files_for_dataset(root):
    files = []
    for path in sorted(Path(root).glob("*_QP*.tsv")):
        match = RDO_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        done_path = Path(str(path) + ".done")
        if not done_path.exists() or path.stat().st_size == 0:
            continue
        files.append((path, match.group("sequence"), int(match.group("qp"))))
    if not files:
        raise FileNotFoundError(f"No completed RDO TSV files found under {root}")
    return files


def _parse_dataset_rdo_fields(fields, outer_ctu_columns, coordinate_scale=0):
    best_cost = float(fields[51])
    if not math.isfinite(best_cost):
        return None
    completed = np.zeros(6, dtype=np.bool_)
    rd_delta = np.full(6, np.inf, dtype=np.float32)
    for mode in range(6):
        base = 14 + 6 * mode
        if int(fields[base + 2]) > 0:
            mode_cost = float(fields[base + 3])
            if math.isfinite(mode_cost):
                completed[mode] = True
                rd_delta[mode] = np.float32(max(mode_cost - best_cost, 0.0))
    x, y = int(fields[5]), int(fields[6])
    width, height = int(fields[7]), int(fields[8])
    scaled_width = width >> coordinate_scale
    scaled_height = height >> coordinate_scale
    if (scaled_height // 4, scaled_width // 4) not in RDO_SUPPORTED_SIZES:
        return None
    if coordinate_scale and (scaled_width > 32 or scaled_height > 32):
        return None
    if coordinate_scale and ((x >> coordinate_scale) % 4 != 0 or (y >> coordinate_scale) % 4 != 0):
        return None
    outer_ctu_x, outer_ctu_y = x // 128, y // 128
    sub_x, sub_y = (x % 128) // 64, (y % 128) // 64
    ctu_id = (outer_ctu_y * outer_ctu_columns + outer_ctu_x) * 4 + sub_y * 2 + sub_x
    return {
        "ctu_id": ctu_id,
        "frame_id": int(fields[3]),
        "coordinates": (x, y, width, height),
        "rd_delta": rd_delta,
        "completed": completed,
        "best_cost": best_cost,
        "selected": {
            "NS": 0, "QT": 1, "BTH": 2, "BTV": 3, "TTH": 4, "TTV": 5,
        }.get(fields[50], -99),
        "node_id": int(fields[0]),
    }


def iter_dataset_rdo_records(path, picture_width, channel):
    """Yield all supported RDO rows for dataset conversion."""
    outer_ctu_columns = (int(picture_width) + 127) // 128
    awk_program = f'NF==52 && $1!="node_id" && $5=="{channel}" {{print;}}'
    process = subprocess.Popen(
        ["awk", "-F", "\t", awk_program, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 52:
                continue
            record = _parse_dataset_rdo_fields(fields, outer_ctu_columns, 1 if channel == "C" else 0)
            if record is not None:
                yield record
    finally:
        process.stdout.close()
    stderr = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(f"Failed to parse RDO file {path}: {stderr}")


def iter_final_dataset_rdo_records(path, picture_width, channel):
    """Yield nodes on the final selected partition tree only.

    A TSV can contain several searches that visit the same geometry.  The
    final tree is recovered by walking parent/selected-mode links backwards:
    a row is active when its parent is a root or its ``via_mode`` equals the
    parent's selected mode.  Its six candidate costs are then retained.
    """
    outer_ctu_columns = (int(picture_width) + 127) // 128
    mode_names = {"ROOT": -1, "NS": 0, "QT": 1, "BTH": 2, "BTV": 3, "TTH": 4, "TTV": 5}
    parents, via_modes, selected_modes = [], [], []
    scan = subprocess.Popen(
        ["awk", "-F", "\t", 'NF==52 && $1!="node_id" {print $1 "\t" $2 "\t" $3 "\t" $51;}', str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if scan.stdout is None:
        raise RuntimeError(f"Failed to scan RDO node links for {path}")
    for line in scan.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 4:
            continue
        node_id, parent_id = int(fields[0]), int(fields[1])
        while len(parents) <= node_id:
            parents.append(-2)
            via_modes.append(-2)
            selected_modes.append(-2)
        parents[node_id] = parent_id
        via_modes[node_id] = mode_names.get(fields[2], -99)
        selected_modes[node_id] = mode_names.get(fields[3], -99)
    scan_stderr = scan.stderr.read() if scan.stderr is not None else ""
    if scan.wait() != 0:
        raise RuntimeError(f"Failed to scan RDO node links for {path}: {scan_stderr}")
    active = [False] * len(parents)
    for node_id, parent_id in enumerate(parents):
        if parent_id == -2:
            continue
        if parent_id < 0:
            active[node_id] = True
        elif parent_id < len(active):
            active[node_id] = active[parent_id] and via_modes[node_id] == selected_modes[parent_id]

    awk_program = f'NF==52 && $1!="node_id" && $5=="{channel}" {{print;}}'
    awk_process = subprocess.Popen(
        ["awk", "-F", "\t", awk_program, str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if awk_process.stdout is None:
        raise RuntimeError(f"Failed to open final-tree RDO stream for {path}")
    try:
        for line in awk_process.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 52:
                continue
            node_id = int(fields[0])
            if node_id >= len(active) or not active[node_id]:
                continue
            record = _parse_dataset_rdo_fields(fields, outer_ctu_columns, 1 if channel == "C" else 0)
            if record is not None:
                yield record
    finally:
        awk_process.stdout.close()
    awk_stderr = awk_process.stderr.read()
    if awk_process.wait() != 0:
        raise RuntimeError(f"Failed to parse final RDO tree {path}: {awk_stderr}")


def convert_component_rdocost_to_cache(
    data_type,
    dataset_name,
    component,
    rdo_root,
    sequence_list,
    output_split=None,
):
    """Convert completed VTM RDO TSVs to a CU-tree-aligned pkl/npy cache."""
    if component not in ("Luma", "Chroma"):
        raise ValueError(f"unsupported component: {component}")
    if output_split is None:
        output_split = resolve_output_split_dir(data_type)
    dataset_dir = paths.dataset_root() / dataset_name / output_split
    tree_path = dataset_dir / f"{component}_CU_Tree.pkl"
    if not tree_path.exists():
        raise FileNotFoundError(f"CU tree metadata not found: {tree_path}")
    with open(tree_path, "rb") as fp:
        tree = pickle.load(fp)
    nodes_path = tree_path.with_name(tree["array_file"])
    nodes = np.load(nodes_path, mmap_mode="r")
    samples = tree["samples"]
    offsets = np.asarray(tree["offsets"], dtype=np.int64)
    sample_rows = list(samples.itertuples(index=False))
    widths = load_rdo_sequence_widths(sequence_list)
    node_key_to_global = {}
    for row in sample_rows:
        sample_index = int(row.sample_index)
        start = int(offsets[sample_index])
        end = int(offsets[sample_index + 1])
        if hasattr(row, "block_x"):
            block_x, block_y = int(row.block_x), int(row.block_y)
        else:
            block_x, block_y = block_origin_from_ctu_id(int(row.ctu_id), widths.get(str(row.sequence_name), 0), component)
        for i, node in enumerate(nodes[start:end]):
            grid_y, grid_x, grid_h, grid_w = (int(v) for v in node[:4])
            node_key = (
                str(row.sequence_name), int(row.qp), int(row.frame_id), int(row.ctu_id),
                block_x + grid_x * 4, block_y + grid_y * 4, grid_w * 4, grid_h * 4,
            )
            if node_key in node_key_to_global:
                raise RuntimeError(f"Duplicate CU-tree node key: {node_key}")
            node_key_to_global[node_key] = start + i

    channel = "L" if component == "Luma" else "C"
    coordinate_scale = 1 if component == "Chroma" else 0
    total_nodes = int(nodes.shape[0])
    rd_delta = np.full((total_nodes, 6), np.inf, dtype=np.float32)
    rd_completed = np.zeros((total_nodes, 6), dtype=np.bool_)
    rd_valid = np.zeros((total_nodes,), dtype=np.bool_)
    matched = mismatched = missing = duplicate = 0
    discarded_outer_nodes = 0
    tree_block_size = int(tree.get("block_size", 0))
    rd_dtype = np.dtype([("rd_delta", np.float32, (6,)), ("completed", np.bool_, (6,)), ("valid", np.bool_)])
    array_path = dataset_dir / f"{component}_CU_RDCost.npy"
    metadata_path = dataset_dir / f"{component}_CU_RDCost.pkl"
    previous_matched = 0
    if array_path.exists() and metadata_path.exists():
        previous = np.load(array_path, mmap_mode="r")
        if previous.shape != (total_nodes,):
            raise RuntimeError(f"Existing {array_path} has incompatible shape {previous.shape}")
        rd_delta = np.asarray(previous["rd_delta"], dtype=np.float32).copy()
        rd_completed = np.asarray(previous["completed"], dtype=np.bool_).copy()
        rd_valid = np.asarray(previous["valid"], dtype=np.bool_).copy()
        with open(metadata_path, "rb") as fp:
            previous_payload = pickle.load(fp)
        previous_matched = int(previous_payload.get("matched", 0))

    for path, sequence, qp in discover_rdo_files_for_dataset(rdo_root):
        if sequence not in widths:
            continue
        log_progress(f"RDCost {component}: parsing {path.name}")
        matched_file = 0
        for record in iter_final_dataset_rdo_records(path, widths[sequence], channel=channel):
            ctu_id = int(record["ctu_id"])
            x, y, width, height = record["coordinates"]
            # The minimum luma CU is not a classifier decision sample and is
            # intentionally absent from the compact CU tree.
            if component == "Luma" and (width, height) == (4, 4):
                continue
            if coordinate_scale:
                x, y, width, height = x // 2, y // 2, width // 2, height // 2
            frame_id = int(record["frame_id"])
            node_key = (str(sequence), int(qp), frame_id, ctu_id, x, y, width, height)
            global_index = node_key_to_global.get(node_key)
            if global_index is None:
                if component == "Luma" and tree_block_size == 32 and (width, height) == (64, 64):
                    discarded_outer_nodes += 1
                    continue
                mismatched += 1
                continue
            if rd_valid[global_index]:
                raise RuntimeError(
                    "Duplicate final-tree RDCost key; expected one selected node: "
                    f"{node_key} in {path}"
                )
            best_cost = float(record["best_cost"])
            denom = max(abs(best_cost), 1e-12)
            rd_delta[global_index] = record["rd_delta"] / denom
            rd_completed[global_index] = record["completed"]
            rd_valid[global_index] = True
            matched += 1
            matched_file += 1
        log_progress(f"RDCost {component}: matched {matched_file:,} rows from {path.name}")

    if matched == 0:
        raise RuntimeError("No RDO-cost rows matched the target CU tree")
    # The compact tree contains only nodes that still require a partition
    # decision; therefore every remaining node must have an RDCost row.
    missing_trainable = int(np.count_nonzero(~rd_valid))
    # The TSV also contains search nodes that are not part of the compact
    # final tree.  They are intentionally discarded: the training contract
    # is that every trainable tree node has one RDCost record.  Only missing
    # tree nodes or duplicate records for a tree node invalidate the cache.
    if duplicate != 0 or missing_trainable != 0:
        raise RuntimeError(
            f"RDCost validation failed: mismatched={mismatched}, duplicate={duplicate}, "
            f"missing_trainable_nodes={missing_trainable}"
        )
    array = np.lib.format.open_memmap(array_path, mode="w+", dtype=rd_dtype, shape=(total_nodes,))
    array["rd_delta"] = rd_delta
    array["completed"] = rd_completed
    array["valid"] = rd_valid
    array.flush()
    del array
    payload = {
        "format": "cu_rdcost_numpy",
        "component": component,
        "id_columns": list(tree["id_columns"]),
        "node_key_columns": ["sequence_name", "qp", "frame_id", "ctu_id", "cu_x", "cu_y", "cu_width", "cu_height"],
        "node_order": f"identical to {component}_CU_Tree.npy global node order",
        "node_columns": ["rd_delta[6]", "completed[6]", "valid"],
        "array_file": array_path.name,
        "array_shape": (total_nodes,),
        "array_dtype": rd_dtype.descr,
        "offsets": offsets,
        "class_order": ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"],
        "rd_delta_definition": "(candidate_cost - best_cost) / max(abs(best_cost), 1e-12)",
        "source_rdo_root": str(rdo_root),
        "source_sequence_list": str(sequence_list),
        "matched": matched + previous_matched,
        "duplicate_rows": duplicate,
        "mismatched": mismatched,
        "missing_sample": missing,
        "discarded_outer_nodes": discarded_outer_nodes,
        "ignored_non_tree_rows": mismatched,
        "missing_trainable_nodes": missing_trainable,
    }
    with open(metadata_path, "wb") as fp:
        pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
    log_progress(f"saved {component} CU RD-cost array to {array_path} (matched={matched:,})")
    log_progress(f"saved {component} CU RD-cost metadata to {metadata_path}")
    return metadata_path


def classifier_i_logical_gridmap(grid_h, grid_w, class_id):
    gridmap = np.zeros((2, grid_h, grid_w), dtype=np.float32)

    if class_id == 0:
        return gridmap
    if class_id == 1:
        gridmap[0, :, grid_w // 2 - 1] = 1
        gridmap[1, grid_h // 2 - 1, :] = 1
        return gridmap
    if class_id == 2:
        gridmap[1, grid_h // 2 - 1, :] = 1
        return gridmap
    if class_id == 3:
        gridmap[0, :, grid_w // 2 - 1] = 1
        return gridmap
    if class_id == 4:
        gridmap[1, grid_h // 4 - 1, :] = 1
        gridmap[1, (3 * grid_h) // 4 - 1, :] = 1
        return gridmap
    if class_id == 5:
        gridmap[0, :, grid_w // 4 - 1] = 1
        gridmap[0, :, (3 * grid_w) // 4 - 1] = 1
        return gridmap

    raise ValueError(f"Unsupported Classifier_I class id: {class_id}")


def create_classifier_i_pretrain_logical_dataset():
    save_root = paths.ensure_dir(paths.dataset_root() / "pretrain" / "Classifier_I" / "logical")
    output_paths = []

    for (grid_h, grid_w), allowed_classes in CLASSIFIER_I_GRID_SPECS.items():
        gridmaps = []
        labels = []

        for class_id in allowed_classes:
            minimal = classifier_i_logical_gridmap(grid_h, grid_w, class_id)
            for _ in range(CLASSIFIER_I_PRETRAIN_SAMPLES_PER_CLASS):
                gridmaps.append(minimal.copy())
                labels.append(class_id)

        gridmaps = np.stack(gridmaps, axis=0).astype(np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        save_dir = paths.ensure_dir(save_root / f"{grid_h}x{grid_w}")
        gridmap_path = save_dir / "gridmap.npy"
        label_path = save_dir / "label.npy"
        np.save(gridmap_path, gridmaps)
        np.save(label_path, labels)
        output_paths.append((gridmap_path, label_path))
        class_names = ", ".join(CLASSIFIER_I_CLASS_NAMES[class_id] for class_id in allowed_classes)
        log_progress(
            f"saved Classifier_I logical pretrain {grid_h}x{grid_w}: "
            f"{len(labels)} samples, classes [{class_names}]"
        )

    return output_paths


def select_block_size_map(block_size_map, component):
    selected_components = COMPONENT_ALIASES[component]
    return {name: block_size_map[name] for name in selected_components}


def convert_partition_to_gridmap(data_type, block_size_map=None, dataset_name=None, component="both", output_split=None, source_dataset=None):
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    partition_split_dir = resolve_split_dir(data_type)
    output_split_dir = output_split or resolve_output_split_dir(data_type)
    source_dataset = source_dataset or dataset_name
    partition_dir = paths.partition_dataset_root(source_dataset) / partition_split_dir
    save_dir = paths.ensure_dir(paths.dataset_root() / dataset_name / output_split_dir)
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
        width = int(row[2])
        height = int(row[3])
        explicit_is10bit = bool(int(row[6])) if len(row) > 6 and row[6] not in (None, "") else None
        # CUSTOM lists may use the sixth field for FPS (for example 59.94),
        # while older lists use it for an integer temporal subsample ratio.
        # FPS is descriptive metadata and must not affect frame indexing.
        subsample_ratio = 1
        if len(row) > 5 and row[5] not in (None, ""):
            try:
                subsample_ratio = int(row[5])
            except (TypeError, ValueError):
                subsample_ratio = 1
        metadata[sequence_name] = {
            "file_name": str(row[1]),
            "width": width,
            "height": height,
            "frame_count": int(row[4]),
            "subsample_ratio": subsample_ratio,
            # VVC CTC A1/A2 are stored as 10-bit 16-bit samples.  The
            # historical six-column lists do not carry a bit-depth field.
            "is10bit": (
                explicit_is10bit
                if explicit_is10bit is not None
                else dataset_name.upper() == "VVC_CTC" and width == 3840 and height == 2160
            ),
        }
    return metadata


def load_component_ids(component, split_dir, dataset_name):
    require_pandas()
    save_dir = paths.dataset_root() / dataset_name / split_dir
    gridmap_path = save_dir / f"{component}_Gridmap.pkl"
    if not gridmap_path.exists():
        raise FileNotFoundError(
            f"Gridmap metadata not found: {gridmap_path}. "
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


def crop_with_edge_padding(frame, x, y, crop_size):
    frame_h, frame_w = frame.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + crop_size)
    y1 = min(frame_h, y + crop_size)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"Crop window ({x}, {y}, {crop_size}, {crop_size}) does not overlap frame {frame_w}x{frame_h}"
        )

    block = frame[y0:y1, x0:x1]
    pad_left = x0 - x
    pad_top = y0 - y
    pad_right = x + crop_size - x1
    pad_bottom = y + crop_size - y1
    if pad_left or pad_top or pad_right or pad_bottom:
        block = np.pad(
            block,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
    return block.astype(np.uint8, copy=False)


def save_component_input(component, ids, metadata, output_dataset, source_dataset, split_dir, block_size, input_size):
    components = YUV420_COMPONENTS[component]
    context_margin = input_size - block_size
    if context_margin < 0:
        raise ValueError(f"input_size {input_size} must be >= block_size {block_size}")

    input_shape = (len(ids), len(components), input_size, input_size)
    log_progress(f"start {component} input: allocate array shape={input_shape}")
    input_blocks = np.zeros(input_shape, dtype=np.uint8)
    video_root = paths.video_dataset_root(source_dataset)
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
            crop_x = x - context_margin
            crop_y = y - context_margin
            for channel_idx, frame in enumerate(source_frames):
                input_blocks[sample_index, channel_idx] = crop_with_edge_padding(
                    frame,
                    crop_x,
                    crop_y,
                    input_size,
                )
            processed_samples += 1
        if processed_groups == 1 or processed_groups % 50 == 0 or processed_groups == total_groups:
            elapsed = max(time.time() - start_time, 1e-6)
            log_progress(
                f"{component} input {processed_groups:,}/{total_groups:,} sequence-frame groups, "
                f"{processed_samples:,}/{len(ids):,} samples, {processed_samples / elapsed:,.0f} samples/s"
            )

    save_dir = paths.ensure_dir(paths.dataset_root() / output_dataset / split_dir)
    save_path = save_dir / f"{component}_Input.pkl"
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, input_blocks)
    payload = {
        "component": component,
        "block_size": block_size,
        "input_size": input_size,
        "context_margin": context_margin,
        "padding": "edge",
        "id_columns": ID_COLUMNS,
        "ids": ids.copy(),
        "array_file": npy_path.name,
        "array_key": "input",
        "array_shape": tuple(input_blocks.shape),
        "array_dtype": str(input_blocks.dtype),
    }
    pd.to_pickle(payload, save_path)
    log_progress(f"saved {component} input array to {npy_path}")
    log_progress(f"saved {component} input metadata to {save_path}")
    return save_path


def convert_yuv_to_input(
    data_type,
    block_size_map=None,
    input_size_map=None,
    dataset_name=None,
    sequence_list=None,
    component="both",
    output_split=None,
    source_dataset=None,
):
    require_pandas()
    dataset_name = resolve_dataset_name(data_type, dataset_name)
    split_dir = output_split or resolve_output_split_dir(data_type)
    if block_size_map is None:
        block_size_map = DEFAULT_BLOCK_SIZE_MAP
    if input_size_map is None:
        input_size_map = DEFAULT_INPUT_SIZE_MAP
    block_size_map = select_block_size_map(block_size_map, component)
    input_size_map = select_block_size_map(input_size_map, component)
    source_dataset = source_dataset or dataset_name
    metadata = sequence_info_to_metadata(data_type, source_dataset, sequence_list)

    output_paths = []
    for component, block_size in block_size_map.items():
        ids = load_component_ids(component, split_dir, dataset_name)
        output_paths.append(
            save_component_input(
                component=component,
                ids=ids,
                metadata=metadata,
                output_dataset=dataset_name,
                source_dataset=source_dataset,
                split_dir=split_dir,
                block_size=block_size,
                input_size=input_size_map[component],
            )
        )
    return output_paths


LUMA32_ID_COLUMNS = ID_COLUMNS + ["sub_block_id"]
SUB_BLOCK_OFFSETS = (
    (0, 0),
    (32, 0),
    (0, 32),
    (32, 32),
)

def resolve_dataset_name(data_type, dataset_name=None):
    if dataset_name is not None:
        return dataset_name
    return DATA_TYPE_TO_DATASET[data_type]


def output_dataset_name(source_dataset_name, output_dataset=None):
    if output_dataset is not None:
        return output_dataset
    return source_dataset_name


def luma64_origin_from_ctu_id(ctu_id, source_width):
    luma_ctu_size = 128
    luma_block_size = 64
    sub_blocks_per_row = luma_ctu_size // luma_block_size
    ctu_per_row = (source_width + luma_ctu_size - 1) // luma_ctu_size
    outer_ctu_id = ctu_id // (sub_blocks_per_row * sub_blocks_per_row)
    sub_id = ctu_id % (sub_blocks_per_row * sub_blocks_per_row)
    outer_ctu_x = outer_ctu_id % ctu_per_row
    outer_ctu_y = outer_ctu_id // ctu_per_row
    sub_x = sub_id % sub_blocks_per_row
    sub_y = sub_id // sub_blocks_per_row
    return outer_ctu_x * luma_ctu_size + sub_x * luma_block_size, outer_ctu_y * luma_ctu_size + sub_y * luma_block_size


def crop_with_clamped_edge(frame, x, y, crop_size):
    frame_h, frame_w = frame.shape
    ys = np.clip(np.arange(y, y + crop_size), 0, frame_h - 1)
    xs = np.clip(np.arange(x, x + crop_size), 0, frame_w - 1)
    return frame[np.ix_(ys, xs)].astype(np.uint8, copy=False)


def record_intersects_block(record, x, y, size):
    return (
        record["cu_x"] < x + size
        and record["cu_x"] + record["cu_width"] > x
        and record["cu_y"] < y + size
        and record["cu_y"] + record["cu_height"] > y
    )


def record_leaf_node(record):
    return (record["cu_x"], record["cu_y"], record["cu_width"], record["cu_height"])


def luma64_origin_from_records(records):
    return (
        min(record["cu_x"] for record in records) // 64 * 64,
        min(record["cu_y"] for record in records) // 64 * 64,
    )


def label_at_node_for_record(record, target_node):
    root_size = 128
    target_x, target_y, _, _ = target_node
    node = (target_x // root_size * root_size, target_y // root_size * root_size, root_size, root_size)
    leaf = record_leaf_node(record)

    for split_idx in range(8):
        label = classifier_label_from_part_split(record[f"split_{split_idx}"])
        if label is None:
            continue
        if node == target_node:
            return label
        if label == 0:
            return None
        next_node = None
        for child in split_node_children_abs(*node, label):
            if child_contains_leaf(child, leaf):
                next_node = child
                break
        if next_node is None:
            return None
        node = next_node
    return None


def luma64_split_label(records, block_x, block_y):
    target_node = (block_x, block_y, 64, 64)
    counts = {}
    for record in records:
        label = label_at_node_for_record(record, target_node)
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], item[0] != 0, item[0]))[0]


def collect_luma32_samples(key, records):
    origin_x, origin_y = luma64_origin_from_records(records)
    if luma64_split_label(records, origin_x, origin_y) != 1:
        return []

    samples = []
    for sub_block_id, (dx, dy) in enumerate(SUB_BLOCK_OFFSETS):
        block_x = origin_x + dx
        block_y = origin_y + dy
        block_node = (block_x, block_y, 32, 32)
        local_records = [
            record for record in records
            if node_contains_node(block_node, record_leaf_node(record))
        ]
        if local_records:
            samples.append((key, sub_block_id, block_x, block_y, local_records))
    return samples


def build_luma32_gridmap(local_records):
    return build_ctu_gridmap_from_records(local_records, 32)


def reconstruct_luma32_nodes(local_records, block_x, block_y):
    export_node = (block_x, block_y, 32, 32)
    root_size = 128
    node_labels = {}
    conflicts = []

    for record in local_records:
        node = (block_x // root_size * root_size, block_y // root_size * root_size, root_size, root_size)
        leaf = record_leaf_node(record)

        for split_idx in range(8):
            label = classifier_label_from_part_split(record[f"split_{split_idx}"])
            if label is None:
                continue

            if node_contains_node(export_node, node):
                if node in node_labels and node_labels[node] != label:
                    conflicts.append((node, node_labels[node], label, record))
                    break
                node_labels[node] = label

            if label == 0:
                break

            next_node = None
            for child in split_node_children_abs(*node, label):
                if child_contains_leaf(child, leaf):
                    next_node = child
                    break
            if next_node is None:
                raise RuntimeError(
                    "Cannot follow Luma32 split path for leaf "
                    f"{leaf} at node {node} with label {label} in "
                    f"{record['sequence_name']} qp={record['qp']} "
                    f"frame={record['frame_id']} ctu={record['ctu_id']}"
                )
            node = next_node

    if conflicts:
        node, old_label, new_label, record = conflicts[0]
        raise RuntimeError(
            "Conflicting Luma32 split labels while reconstructing tree: "
            f"node={node}, old={old_label}, new={new_label}, "
            f"sample={(record['sequence_name'], record['qp'], record['frame_id'], record['ctu_id'])}"
        )

    node_values = []
    stack = [(block_x, block_y, 32, 32)]
    seen = set()
    while stack:
        x, y, w, h = stack.pop()
        node = (x, y, w, h)
        if node in seen or node not in node_labels:
            continue
        seen.add(node)
        label = node_labels[node]
        # A 4x4 luma CU has no remaining partition decision to learn.
        if w == 4 and h == 4:
            continue
        node_values.append(((y - block_y) // 4, (x - block_x) // 4, h // 4, w // 4, label))
        if label != 0:
            for child in reversed(split_node_children_abs(x, y, w, h, label)):
                if node_contains_node(export_node, child):
                    stack.append(child)
    return node_values


def convert_luma32_gridmap(data_type, source_dataset, output_dataset, output_split):
    partition_split = DATA_TYPE_TO_SPLIT_DIR[data_type]
    save_split = output_split or DATA_TYPE_TO_OUTPUT_SPLIT_DIR[data_type]
    partition_info_path = paths.partition_dataset_root(source_dataset) / partition_split / "Luma_Partition_Info.txt"
    save_dir = paths.ensure_dir(paths.dataset_root() / output_dataset / save_split)
    save_path = save_dir / "Luma_Gridmap.pkl"

    ids = []
    gridmaps = []
    for key, records in iter_partition_groups(partition_info_path):
        for _, sub_block_id, block_x, block_y, local_records in collect_luma32_samples(key, records):
            ids.append((*key, sub_block_id, block_x, block_y))
            gridmaps.append(build_luma32_gridmap(local_records))

    if not gridmaps:
        raise RuntimeError(f"No Luma32 QT child samples generated from {partition_info_path}")

    id_df = pd.DataFrame(ids, columns=LUMA32_ID_COLUMNS + ["block_x", "block_y"])
    id_df["sample_index"] = np.arange(len(id_df), dtype=np.int64)
    id_df = id_df.set_index(LUMA32_ID_COLUMNS, drop=False)
    arr = np.stack(gridmaps, axis=0)
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, arr)
    pd.to_pickle({
        "component": "Luma",
        "block_size": 32,
        "grid_size": 8,
        "id_columns": LUMA32_ID_COLUMNS,
        "ids": id_df,
        "array_file": npy_path.name,
        "array_key": "gridmap",
        "array_shape": tuple(arr.shape),
        "array_dtype": str(arr.dtype),
    }, save_path)
    log_progress(f"saved Luma32 gridmap array to {npy_path}")
    return save_path


def convert_luma32_cu_tree(data_type, source_dataset, output_dataset, output_split):
    partition_split = DATA_TYPE_TO_SPLIT_DIR[data_type]
    save_split = output_split or DATA_TYPE_TO_OUTPUT_SPLIT_DIR[data_type]
    partition_info_path = paths.partition_dataset_root(source_dataset) / partition_split / "Luma_Partition_Info.txt"
    save_dir = paths.ensure_dir(paths.dataset_root() / output_dataset / save_split)
    save_path = save_dir / "Luma_CU_Tree.pkl"
    nodes_path = save_path.with_suffix(".npy")
    raw_nodes_path = save_path.with_name("Luma_CU_Tree_Nodes.tmp.i2.bin")

    rows = []
    offsets = [0]
    total_nodes = 0
    sample_index = 0
    start_time = time.time()
    with open(raw_nodes_path, "wb") as fp:
        for key, records in iter_partition_groups(partition_info_path):
            for _, sub_block_id, block_x, block_y, local_records in collect_luma32_samples(key, records):
                nodes = reconstruct_luma32_nodes(local_records, block_x, block_y)
                rows.append({
                    "sequence_name": key[0],
                    "qp": key[1],
                    "frame_id": key[2],
                    "ctu_id": key[3],
                    "sub_block_id": sub_block_id,
                    "block_x": block_x,
                    "block_y": block_y,
                    "sample_index": sample_index,
                    "node_start": total_nodes,
                    "node_end": total_nodes + len(nodes),
                    "node_count": len(nodes),
                })
                np.asarray(nodes, dtype=np.int16).tofile(fp)
                total_nodes += len(nodes)
                offsets.append(total_nodes)
                sample_index += 1
            if sample_index % 200000 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                log_progress(f"Luma32 CU tree {sample_index:,} samples, {total_nodes:,} nodes, {total_nodes / elapsed:,.0f} nodes/s")

    if total_nodes == 0:
        raise RuntimeError(f"No Luma32 CU tree nodes generated from {partition_info_path}")

    raw_nodes = np.memmap(raw_nodes_path, mode="r", dtype=np.int16, shape=(int(total_nodes), 5))
    node_array = np.lib.format.open_memmap(nodes_path, mode="w+", dtype=np.int16, shape=(int(total_nodes), 5))
    node_array[:] = raw_nodes[:]
    node_array.flush()
    del node_array
    del raw_nodes
    raw_nodes_path.unlink()

    samples = pd.DataFrame(rows).set_index(LUMA32_ID_COLUMNS, drop=False)
    pd.to_pickle({
        "format": "cu_tree_numpy",
        "component": "Luma",
        "block_size": 32,
        "grid_size": 8,
        "id_columns": LUMA32_ID_COLUMNS,
        "samples": samples,
        "node_columns": ["grid_y", "grid_x", "grid_h", "grid_w", "label"],
        "array_file": nodes_path.name,
        "array_key": "cu_tree_nodes",
        "array_shape": (int(total_nodes), 5),
        "array_dtype": "int16",
        "offsets": np.asarray(offsets, dtype=np.int64),
        "class_order": ["NO_SPLIT", "QT", "BTH", "BTV", "TTH", "TTV"],
    }, save_path)
    log_progress(f"saved Luma32 CU tree to {save_path}")
    return save_path


def convert_luma32_input(data_type, source_dataset, output_dataset, output_split, sequence_list=None):
    save_split = output_split or DATA_TYPE_TO_OUTPUT_SPLIT_DIR[data_type]
    save_dir = paths.ensure_dir(paths.dataset_root() / output_dataset / save_split)
    gridmap_payload = pd.read_pickle(save_dir / "Luma_Gridmap.pkl")
    ids = gridmap_payload["ids"]
    metadata = sequence_info_to_metadata(data_type, source_dataset, sequence_list)
    save_path = save_dir / "Luma_Input.pkl"
    npy_path = save_path.with_suffix(".npy")
    input_blocks = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint8, shape=(len(ids), 1, 48, 48))
    video_root = paths.video_dataset_root(source_dataset)
    total_groups = ids[["sequence_name", "frame_id"]].drop_duplicates().shape[0]
    processed_groups = 0
    processed_samples = 0
    start_time = time.time()

    for (sequence_name, frame_id), group in ids.groupby(level=["sequence_name", "frame_id"], sort=False):
        processed_groups += 1
        seq_meta = metadata[str(sequence_name)]
        y_frame, _, _ = read_yuv420_frame(
            video_root / seq_meta["file_name"],
            seq_meta["width"],
            seq_meta["height"],
            int(frame_id),
            seq_meta["is10bit"],
        )
        for row in group.itertuples(index=False):
            block_x = int(row.block_x)
            block_y = int(row.block_y)
            input_blocks[int(row.sample_index), 0] = crop_with_clamped_edge(y_frame, block_x - 16, block_y - 16, 48)
            processed_samples += 1
        if processed_groups == 1 or processed_groups % 50 == 0 or processed_groups == total_groups:
            elapsed = max(time.time() - start_time, 1e-6)
            log_progress(
                f"Luma32 input {processed_groups:,}/{total_groups:,} sequence-frame groups, "
                f"{processed_samples:,}/{len(ids):,} samples, {processed_samples / elapsed:,.0f} samples/s"
            )

    input_shape = tuple(input_blocks.shape)
    input_dtype = str(input_blocks.dtype)
    input_blocks.flush()
    del input_blocks
    pd.to_pickle({
        "component": "Luma",
        "block_size": 32,
        "input_size": 48,
        "context_margin": 16,
        "padding": "edge",
        "id_columns": LUMA32_ID_COLUMNS,
        "ids": ids.copy(),
        "array_file": npy_path.name,
        "array_key": "input",
        "array_shape": input_shape,
        "array_dtype": input_dtype,
    }, save_path)
    log_progress(f"saved Luma32 input to {save_path}")
    return save_path

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-type', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--source-dataset', type=str, default=None)
    parser.add_argument('--sequence-list', type=str, default=None)
    parser.add_argument(
        '--output-split',
        type=str,
        default=None,
        help='Override output split directory under data/dataset/<dataset>/ without changing partition input split.',
    )
    parser.add_argument('--luma-block-size', type=int, default=32)
    parser.add_argument('--chroma-block-size', type=int, default=16)
    parser.add_argument('--luma-input-size', type=int, default=48)
    parser.add_argument('--chroma-input-size', type=int, default=32)
    parser.add_argument('--component', choices=['both', 'luma', 'chroma'], default='luma')
    parser.add_argument('--rdo-root', type=str, default=None)
    parser.add_argument('--rdo-sequence-list', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--show-output', type=str, default=None)
    parser.add_argument('--show-sample-index', type=int, default=None)
    parser.add_argument('--show-sequence', type=str, default=None)
    parser.add_argument('--show-qp', type=int, default=None)
    parser.add_argument('--show-frame-id', type=int, default=None)
    parser.add_argument('--show-ctu-id', type=int, default=None)
    parser.add_argument(
        '--action',
        choices=[
            'gridmap',
            'input',
            'gridmap-input',
            'cu-tree',
            'gridmap-input-cu-tree',
            'rdocost',
            'preview',
            'classifier-pretrain',
        ],
        default='gridmap-input',
    )
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    # Luma samples are 32x32 blocks extracted from QT children of the
    # original 128x128 CTU.
    source_dataset = args.source_dataset or args.dataset
    output_dataset = args.dataset or source_dataset
    if source_dataset is None:
        source_dataset = resolve_dataset_name(args.data_type)
    if output_dataset is None:
        output_dataset = source_dataset
    luma_actions = {'gridmap', 'input', 'gridmap-input', 'cu-tree', 'gridmap-input-cu-tree'}
    if args.component in ('luma', 'both') and args.action in luma_actions:
        if args.action in ('gridmap', 'gridmap-input', 'gridmap-input-cu-tree'):
            convert_luma32_gridmap(args.data_type, source_dataset, output_dataset, args.output_split)
        if args.action in ('input', 'gridmap-input', 'gridmap-input-cu-tree'):
            convert_luma32_input(
                args.data_type, source_dataset, output_dataset, args.output_split, args.sequence_list
            )
        if args.action in ('cu-tree', 'gridmap-input-cu-tree'):
            convert_luma32_cu_tree(args.data_type, source_dataset, output_dataset, args.output_split)
        if args.component == 'luma':
            raise SystemExit(0)
        args.component = 'chroma'
    block_size_map = {
        "Luma": args.luma_block_size,
        "Chroma": args.chroma_block_size,
    }
    input_size_map = {
        "Luma": args.luma_input_size,
        "Chroma": args.chroma_input_size,
    }
    if args.action in ('gridmap', 'gridmap-input', 'gridmap-input-cu-tree'):
        convert_partition_to_gridmap(
            args.data_type,
            block_size_map=block_size_map,
            dataset_name=args.dataset,
            component=args.component,
            output_split=args.output_split,
            source_dataset=args.source_dataset,
        )
    if args.action in ('input', 'gridmap-input', 'gridmap-input-cu-tree'):
        convert_yuv_to_input(
            args.data_type,
            block_size_map=block_size_map,
            input_size_map=input_size_map,
            dataset_name=args.dataset,
            sequence_list=args.sequence_list,
            component=args.component,
            output_split=args.output_split,
            source_dataset=args.source_dataset,
        )
    if args.action in ('cu-tree', 'gridmap-input-cu-tree'):
        convert_partition_to_cu_tree(
            args.data_type,
            block_size_map=block_size_map,
            dataset_name=args.dataset,
            component=args.component,
            output_split=args.output_split,
            source_dataset=args.source_dataset,
        )
    if args.action == 'rdocost':
        if args.dataset is None:
            raise ValueError('--dataset is required for --action rdocost')
        if args.rdo_root is None or args.rdo_sequence_list is None:
            raise ValueError('--rdo-root and --rdo-sequence-list are required for --action rdocost')
        for rdo_component in COMPONENT_ALIASES[args.component]:
            convert_component_rdocost_to_cache(
                data_type=args.data_type,
                dataset_name=args.dataset,
                component=rdo_component,
                rdo_root=args.rdo_root,
                sequence_list=args.rdo_sequence_list,
                output_split=args.output_split,
            )
    if args.action == 'classifier-pretrain':
        create_classifier_i_pretrain_logical_dataset()
    if args.show or args.action == 'preview':
        show_id_fields = (args.show_sequence, args.show_qp, args.show_frame_id, args.show_ctu_id)
        show_id_values = [value is not None for value in show_id_fields]
        if any(show_id_values) and not all(show_id_values):
            raise ValueError("--show-sequence, --show-qp, --show-frame-id, and --show-ctu-id must be provided together")
        sample_id = None
        if all(show_id_values):
            sample_id = (args.show_sequence, args.show_qp, args.show_frame_id, args.show_ctu_id)
        preview_component = args.component
        if preview_component == "both":
            preview_component = "luma"
        visualize_dataset_gridmap(
            data_type=args.data_type,
            dataset_name=args.dataset,
            component=preview_component,
            sample_id=sample_id,
            sample_index=args.show_sample_index,
            output_name=args.show_output,
        )
