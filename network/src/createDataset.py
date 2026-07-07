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

    if show:
        preview_path = paths.ensure_dir(paths.output_root()) / f"{component}_gridmap_preview.png"
        save_gridmap_preview(gridmap_arr, preview_path, block_size, component)
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
        node = (x, y, w, h)
        if node in seen or node not in node_labels:
            continue
        seen.add(node)
        label = node_labels[node]
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


def convert_partition_to_cu_tree(data_type, block_size_map=None, dataset_name=None, component="both"):
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
    frame_h, frame_w = frame.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + block_size)
    y1 = min(frame_h, y + block_size)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"Crop window ({x}, {y}, {block_size}, {block_size}) does not overlap frame {frame_w}x{frame_h}"
        )

    block = frame[y0:y1, x0:x1]
    pad_left = x0 - x
    pad_top = y0 - y
    pad_right = x + block_size - x1
    pad_bottom = y + block_size - y1
    if pad_left or pad_top or pad_right or pad_bottom:
        block = np.pad(
            block,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )
    return block.astype(np.uint8, copy=False)


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
    npy_path = save_path.with_suffix(".npy")
    np.save(npy_path, input_blocks)
    payload = {
        "component": component,
        "block_size": block_size,
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
    parser.add_argument(
        '--action',
        choices=['gridmap', 'input', 'gridmap-input', 'cu-tree', 'gridmap-input-cu-tree'],
        default='gridmap-input',
    )
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    block_size_map = {
        "Luma": args.luma_block_size,
        "Chroma": args.chroma_block_size,
    }
    if args.action in ('gridmap', 'gridmap-input', 'gridmap-input-cu-tree'):
        convert_partition_to_gridmap(
            args.data_type,
            block_size_map=block_size_map,
            show=args.show,
            dataset_name=args.dataset,
            component=args.component,
        )
    if args.action in ('input', 'gridmap-input', 'gridmap-input-cu-tree'):
        convert_yuv_to_input(
            args.data_type,
            block_size_map=block_size_map,
            dataset_name=args.dataset,
            sequence_list=args.sequence_list,
            component=args.component,
        )
    if args.action in ('cu-tree', 'gridmap-input-cu-tree'):
        convert_partition_to_cu_tree(
            args.data_type,
            block_size_map=block_size_map,
            dataset_name=args.dataset,
            component=args.component,
        )
