#!/usr/bin/env python3
"""
Self-check partition text files for DIV2K QP data.

Checks, using the current DecLib export definition:
  1. Whether the CU size fields behave like width,height or height,width.
     A valid block should have no overlapping cells, no CU extending outside
     its 64x64 luma or 32x32 chroma export block, and full coverage.
  2. Whether ctu_id matches DecLib's id derived from x/y and source width.

The script checks both current Partition_Info files:
    <seq>/Luma_Partition_Info.txt
    <seq>/Chroma_Partition_Info.txt

Legacy FastOff files are checked when Partition_Info files are absent:
    <seq>/<seq>_QPxx_Luma_Partition_FastOff_LFNST0.txt
    <seq>/<seq>_QPxx_Chroma_Partition_FastOff_LFNST0.txt

Legacy files do not contain ctu_id, so only geometry/order can be checked.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


COMPONENTS = ("Luma", "Chroma")
DEFAULT_QPS = (22, 27)
LUMA_GRID_UNIT = 4
LUMA_CTU = 128
LUMA_EXPORT_BLOCK = 64
EXAMPLE_LIMIT = 5


@dataclass
class SequenceDims:
    width: Optional[int]
    height: Optional[int]


@dataclass
class Record:
    line_no: int
    sequence: str
    qp: int
    frame_id: int
    ctu_id: Optional[int]
    x: int
    y: int
    size_a: int
    size_b: int
    raw: str


@dataclass
class PendingLine:
    line: str
    fields: Optional[List[str]]
    record: Optional[Record]


@dataclass
class GeometryStats:
    groups: int = 0
    ok_groups: int = 0
    bad_groups: int = 0
    overlap_groups: int = 0
    out_of_block_groups: int = 0
    coverage_groups: int = 0
    total_overlaps: int = 0
    total_out_of_block: int = 0
    total_missing_cells: int = 0
    examples: List[str] = field(default_factory=list)


@dataclass
class FileStats:
    path: Path
    component: str
    qp: int
    sequence: str
    file_format: str
    records: int = 0
    parse_errors: int = 0
    cfg_missing: bool = False
    id_checked: int = 0
    id_errors: int = 0
    id_examples: List[str] = field(default_factory=list)
    wh: GeometryStats = field(default_factory=GeometryStats)
    hw: GeometryStats = field(default_factory=GeometryStats)
    fixed: bool = False
    fixed_ids: bool = False


def merge_geometry_stats(dst: GeometryStats, src: GeometryStats) -> None:
    dst.groups += src.groups
    dst.ok_groups += src.ok_groups
    dst.bad_groups += src.bad_groups
    dst.overlap_groups += src.overlap_groups
    dst.out_of_block_groups += src.out_of_block_groups
    dst.coverage_groups += src.coverage_groups
    dst.total_overlaps += src.total_overlaps
    dst.total_out_of_block += src.total_out_of_block
    dst.total_missing_cells += src.total_missing_cells
    for example in src.examples:
        if len(dst.examples) < EXAMPLE_LIMIT:
            dst.examples.append(example)


def read_cfg_value(cfg_path: Path, key: str) -> Optional[int]:
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            body = line.split("#", 1)[0].strip()
            if not body or ":" not in body:
                continue
            name, value = body.split(":", 1)
            if name.strip() == key:
                return int(value.strip())
    return None


def cfg_path_for(cfg_root: Path, dataset: str, qp: int, sequence: str) -> Path:
    return cfg_root / dataset / f"qp_{qp}" / f"{sequence}_intra_vtm.cfg"


def dims_for(cfg_root: Path, dataset: str, qp: int, sequence: str, cache: Dict[Tuple[str, int], SequenceDims]) -> SequenceDims:
    key = (sequence, qp)
    if key not in cache:
        cfg_path = cfg_path_for(cfg_root, dataset, qp, sequence)
        cache[key] = SequenceDims(
            width=read_cfg_value(cfg_path, "SourceWidth"),
            height=read_cfg_value(cfg_path, "SourceHeight"),
        )
    return cache[key]


def block_size_for(component: str, file_format: str = "info") -> int:
    if component == "Luma":
        return LUMA_EXPORT_BLOCK
    return LUMA_EXPORT_BLOCK // 2


def grid_unit_for(component: str) -> int:
    return LUMA_GRID_UNIT if component == "Luma" else LUMA_GRID_UNIT // 2


def ctu_id_from_xy(x: int, y: int, source_width: int, component: str) -> int:
    ctu = LUMA_CTU if component == "Luma" else LUMA_CTU // 2
    block = LUMA_EXPORT_BLOCK if component == "Luma" else LUMA_EXPORT_BLOCK // 2
    ctu_per_row = (source_width + LUMA_CTU - 1) // LUMA_CTU
    sub_blocks_per_row = ctu // block
    ctu_x = x // ctu
    ctu_y = y // ctu
    sub_x = (x % ctu) // block
    sub_y = (y % ctu) // block
    return (ctu_y * ctu_per_row + ctu_x) * sub_blocks_per_row * sub_blocks_per_row + sub_y * sub_blocks_per_row + sub_x


def discover_component_file(
    seq_dir: Path,
    sequence: str,
    qp: int,
    component: str,
    include_legacy: bool,
) -> Optional[Tuple[Path, str]]:
    info = seq_dir / f"{component}_Partition_Info.txt"
    if info.exists():
        return info, "info"
    if not include_legacy:
        return None
    legacy = seq_dir / f"{sequence}_QP{qp}_{component}_Partition_FastOff_LFNST0.txt"
    if legacy.exists():
        return legacy, "legacy_fastoff"
    return None


def parse_info(path: Path, component: str, expected_qp: int) -> Tuple[List[Record], int]:
    records: List[Record] = []
    parse_errors = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                parse_errors += 1
                continue
            try:
                sequence = fields[0]
                qp = int(fields[1])
                frame_id = int(fields[2])
                ctu_id = int(fields[3])
                x = int(fields[4])
                y = int(fields[5])
                size_a = int(fields[6])
                size_b = int(fields[7])
            except ValueError:
                parse_errors += 1
                continue
            records.append(Record(line_no, sequence, qp, frame_id, ctu_id, x, y, size_a, size_b, line))
    return records, parse_errors


def parse_legacy(path: Path, sequence: str, qp: int, component: str) -> Tuple[List[Record], int]:
    records: List[Record] = []
    parse_errors = 0
    frame_id = -1
    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            if line == "frame++":
                frame_id += 1
                continue
            if frame_id < 0:
                frame_id = 0
            fields = line.split()
            if len(fields) < 4:
                parse_errors += 1
                continue
            try:
                x = int(fields[0])
                y = int(fields[1])
                size_a = int(fields[2])
                size_b = int(fields[3])
            except ValueError:
                parse_errors += 1
                continue
            records.append(Record(line_no, sequence, qp, frame_id, None, x, y, size_a, size_b, line))
    return records, parse_errors


def group_records_by_export_block(
    records: Iterable[Record],
    component: str,
    file_format: str,
) -> Dict[Tuple[str, int, int, int, int], List[Record]]:
    block = block_size_for(component, file_format)
    groups: Dict[Tuple[str, int, int, int, int], List[Record]] = defaultdict(list)
    for record in records:
        key = (
            record.sequence,
            record.qp,
            record.frame_id,
            record.x // block,
            record.y // block,
        )
        groups[key].append(record)
    return groups


def geometry_for_groups(
    groups: Dict[Tuple[str, int, int, int, int], List[Record]],
    component: str,
    file_format: str,
    swap_size_fields: bool,
    source_width: Optional[int],
    source_height: Optional[int],
) -> GeometryStats:
    block = block_size_for(component, file_format)
    grid_unit = grid_unit_for(component)
    scale = 1 if component == "Luma" else 2
    component_width = None if source_width is None else (source_width + scale - 1) // scale
    component_height = None if source_height is None else (source_height + scale - 1) // scale
    stats = GeometryStats(groups=len(groups))

    for key, group in groups.items():
        origin_x = key[3] * block
        origin_y = key[4] * block
        valid_width = block
        valid_height = block
        if component_width is not None:
            valid_width = max(0, min(block, component_width - origin_x))
        if component_height is not None:
            valid_height = max(0, min(block, component_height - origin_y))
        if valid_width <= 0 or valid_height <= 0:
            valid_width = block
            valid_height = block
        expected_cols = (valid_width + grid_unit - 1) // grid_unit
        expected_rows = (valid_height + grid_unit - 1) // grid_unit
        expected_cells = expected_cols * expected_rows
        occupied: Dict[Tuple[int, int], Record] = {}
        overlaps = 0
        out_of_block = 0

        for record in group:
            width, height = (record.size_b, record.size_a) if swap_size_fields else (record.size_a, record.size_b)
            if (
                record.x < origin_x
                or record.y < origin_y
                or record.x + width > origin_x + block
                or record.y + height > origin_y + block
                or width <= 0
                or height <= 0
                or width % grid_unit
                or height % grid_unit
            ):
                out_of_block += 1

            x0 = max(record.x, origin_x)
            y0 = max(record.y, origin_y)
            x1 = min(record.x + width, origin_x + block)
            y1 = min(record.y + height, origin_y + block)
            for yy in range(y0, y1, grid_unit):
                for xx in range(x0, x1, grid_unit):
                    cell = ((xx - origin_x) // grid_unit, (yy - origin_y) // grid_unit)
                    if cell[0] >= expected_cols or cell[1] >= expected_rows:
                        continue
                    if cell in occupied:
                        overlaps += 1
                    occupied[cell] = record

        missing = max(0, expected_cells - len(occupied))
        is_ok = overlaps == 0 and out_of_block == 0 and missing == 0
        if is_ok:
            stats.ok_groups += 1
        else:
            stats.bad_groups += 1
            stats.total_overlaps += overlaps
            stats.total_out_of_block += out_of_block
            stats.total_missing_cells += missing
            if overlaps:
                stats.overlap_groups += 1
            if out_of_block:
                stats.out_of_block_groups += 1
            if missing:
                stats.coverage_groups += 1
            if len(stats.examples) < EXAMPLE_LIMIT:
                first = group[0]
                stats.examples.append(
                    f"key={key}, occ={len(occupied)}/{expected_cells}, overlaps={overlaps}, "
                    f"out={out_of_block}, first_line={first.line_no}: {first.raw}"
                )

    return stats


def check_ids(stats: FileStats, records: Iterable[Record], source_width: Optional[int]) -> None:
    if source_width is None:
        stats.cfg_missing = True
        return
    for record in records:
        if record.ctu_id is None:
            continue
        expected = ctu_id_from_xy(record.x, record.y, source_width, stats.component)
        stats.id_checked += 1
        if record.ctu_id != expected:
            stats.id_errors += 1
            if len(stats.id_examples) < EXAMPLE_LIMIT:
                stats.id_examples.append(
                    f"line={record.line_no}, got={record.ctu_id}, expected={expected}, "
                    f"x={record.x}, y={record.y}, raw={record.raw}"
                )


def check_file(
    path: Path,
    file_format: str,
    component: str,
    qp: int,
    sequence: str,
    source_width: Optional[int],
    source_height: Optional[int],
) -> FileStats:
    stats = FileStats(path=path, component=component, qp=qp, sequence=sequence, file_format=file_format)
    if file_format == "info":
        records, stats.parse_errors = parse_info(path, component, qp)
    else:
        records, stats.parse_errors = parse_legacy(path, sequence, qp, component)

    stats.records = len(records)
    check_ids(stats, records, source_width)
    groups = group_records_by_export_block(records, component, file_format)
    stats.wh = geometry_for_groups(
        groups,
        component,
        file_format,
        swap_size_fields=False,
        source_width=source_width,
        source_height=source_height,
    )
    stats.hw = geometry_for_groups(
        groups,
        component,
        file_format,
        swap_size_fields=True,
        source_width=source_width,
        source_height=source_height,
    )
    return stats


def check_training_file(
    path: Path,
    component: str,
    cfg_root: Path,
    dataset: str,
    dim_cache: Dict[Tuple[str, int], SequenceDims],
) -> FileStats:
    stats = FileStats(path=path, component=component, qp=-1, sequence="training", file_format="training_info")
    records, stats.parse_errors = parse_info(path, component, expected_qp=-1)
    stats.records = len(records)

    records_by_seq_qp: Dict[Tuple[str, int], List[Record]] = defaultdict(list)
    for record in records:
        records_by_seq_qp[(record.sequence, record.qp)].append(record)

    for (sequence, qp), seq_records in records_by_seq_qp.items():
        dims = dims_for(cfg_root, dataset, qp, sequence, dim_cache)
        if dims.width is None or dims.height is None:
            stats.cfg_missing = True
        check_ids(stats, seq_records, dims.width)
        groups = group_records_by_export_block(seq_records, component, "info")
        merge_geometry_stats(
            stats.wh,
            geometry_for_groups(
                groups,
                component,
                "info",
                swap_size_fields=False,
                source_width=dims.width,
                source_height=dims.height,
            ),
        )
        merge_geometry_stats(
            stats.hw,
            geometry_for_groups(
                groups,
                component,
                "info",
                swap_size_fields=True,
                source_width=dims.width,
                source_height=dims.height,
            ),
        )

    return stats


def record_from_info_fields(line_no: int, fields: List[str], raw: str) -> Optional[Record]:
    if len(fields) < 8:
        return None
    try:
        return Record(
            line_no=line_no,
            sequence=fields[0],
            qp=int(fields[1]),
            frame_id=int(fields[2]),
            ctu_id=int(fields[3]),
            x=int(fields[4]),
            y=int(fields[5]),
            size_a=int(fields[6]),
            size_b=int(fields[7]),
            raw=raw.strip(),
        )
    except ValueError:
        return None


def training_group_key(record: Record, component: str) -> Tuple[str, int, int, int, int]:
    block = block_size_for(component, "info")
    return (
        record.sequence,
        record.qp,
        record.frame_id,
        record.x // block,
        record.y // block,
    )


def add_id_check(stats: FileStats, record: Record, source_width: Optional[int]) -> None:
    if source_width is None or record.ctu_id is None:
        stats.cfg_missing = True
        return
    expected = ctu_id_from_xy(record.x, record.y, source_width, stats.component)
    stats.id_checked += 1
    if record.ctu_id != expected:
        stats.id_errors += 1
        if len(stats.id_examples) < EXAMPLE_LIMIT:
            stats.id_examples.append(
                f"line={record.line_no}, got={record.ctu_id}, expected={expected}, "
                f"x={record.x}, y={record.y}, raw={record.raw}"
            )


def flush_training_group(
    out_fp,
    pending: List[PendingLine],
    component: str,
    cfg_root: Path,
    dataset: str,
    dim_cache: Dict[Tuple[str, int], SequenceDims],
    stats: FileStats,
    apply_fix: bool,
    fix_ids: bool,
) -> Tuple[int, int]:
    records = [item.record for item in pending if item.record is not None]
    if not records:
        for item in pending:
            out_fp.write(item.line)
        return 0, 0

    first = records[0]
    dims = dims_for(cfg_root, dataset, first.qp, first.sequence, dim_cache)
    groups = {training_group_key(first, component): records}
    wh = geometry_for_groups(
        groups,
        component,
        "info",
        swap_size_fields=False,
        source_width=dims.width,
        source_height=dims.height,
    )
    hw = geometry_for_groups(
        groups,
        component,
        "info",
        swap_size_fields=True,
        source_width=dims.width,
        source_height=dims.height,
    )
    merge_geometry_stats(stats.wh, wh)
    merge_geometry_stats(stats.hw, hw)

    should_swap = apply_fix and wh.bad_groups > 0 and hw.bad_groups == 0
    swapped_lines = 0
    fixed_ids = 0

    for item in pending:
        if item.fields is None or item.record is None:
            out_fp.write(item.line)
            continue

        fields = item.fields[:]
        if should_swap:
            fields[6], fields[7] = fields[7], fields[6]
            swapped_lines += 1

        if dims.width is not None:
            add_id_check(stats, item.record, dims.width)
            if fix_ids:
                expected = ctu_id_from_xy(item.record.x, item.record.y, dims.width, component)
                if fields[3] != str(expected):
                    fields[3] = str(expected)
                    fixed_ids += 1
        else:
            stats.cfg_missing = True

        out_fp.write(" ".join(fields) + "\n")

    return swapped_lines, fixed_ids


def stream_process_training_file(
    path: Path,
    component: str,
    cfg_root: Path,
    dataset: str,
    apply_fix: bool,
    fix_ids: bool,
    make_backup: bool = True,
) -> FileStats:
    stats = FileStats(path=path, component=component, qp=-1, sequence="training", file_format="training_info")
    dim_cache: Dict[Tuple[str, int], SequenceDims] = {}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pending: List[PendingLine] = []
    current_key: Optional[Tuple[str, int, int, int, int]] = None
    swapped_lines = 0
    fixed_id_lines = 0

    with path.open("r", encoding="utf-8", errors="ignore") as in_fp, tmp_path.open("w", encoding="utf-8") as out_fp:
        for line_no, line in enumerate(in_fp, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if pending:
                    a, b = flush_training_group(
                        out_fp, pending, component, cfg_root, dataset, dim_cache, stats, apply_fix, fix_ids
                    )
                    swapped_lines += a
                    fixed_id_lines += b
                    pending = []
                    current_key = None
                out_fp.write(line)
                continue

            fields = stripped.split()
            record = record_from_info_fields(line_no, fields, line)
            if record is None:
                stats.parse_errors += 1
                if pending:
                    a, b = flush_training_group(
                        out_fp, pending, component, cfg_root, dataset, dim_cache, stats, apply_fix, fix_ids
                    )
                    swapped_lines += a
                    fixed_id_lines += b
                    pending = []
                    current_key = None
                out_fp.write(line)
                continue

            stats.records += 1
            key = training_group_key(record, component)
            if current_key is not None and key != current_key:
                a, b = flush_training_group(
                    out_fp, pending, component, cfg_root, dataset, dim_cache, stats, apply_fix, fix_ids
                )
                swapped_lines += a
                fixed_id_lines += b
                pending = []
            current_key = key
            pending.append(PendingLine(line=line, fields=fields, record=record))

        if pending:
            a, b = flush_training_group(
                out_fp, pending, component, cfg_root, dataset, dim_cache, stats, apply_fix, fix_ids
            )
            swapped_lines += a
            fixed_id_lines += b

    if swapped_lines or fixed_id_lines:
        if make_backup:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(path, backup)
        tmp_path.replace(path)
        stats.fixed = bool(swapped_lines)
        stats.fixed_ids = bool(fixed_id_lines)
    else:
        tmp_path.unlink(missing_ok=True)

    return stats


def geometry_label(stats: FileStats) -> str:
    wh_ok = stats.wh.bad_groups == 0
    hw_ok = stats.hw.bad_groups == 0
    if wh_ok and not hw_ok:
        return "OK_width_height"
    if hw_ok and not wh_ok:
        return "SUSPECT_height_width"
    if wh_ok and hw_ok:
        return "AMBIGUOUS_both_ok"
    if stats.wh.bad_groups <= stats.hw.bad_groups:
        return "BAD_prefers_width_height"
    return "BAD_prefers_height_width"


def summarize_file(stats: FileStats) -> str:
    return (
        f"{stats.sequence} qp={stats.qp} {stats.component} {stats.file_format} "
        f"{geometry_label(stats)} records={stats.records} parse_errors={stats.parse_errors} "
        f"id_errors={stats.id_errors}/{stats.id_checked} "
        f"wh_bad={stats.wh.bad_groups}/{stats.wh.groups} hw_bad={stats.hw.bad_groups}/{stats.hw.groups} "
        f"fixed_size={int(stats.fixed)} fixed_id={int(stats.fixed_ids)} "
        f"path={stats.path}"
    )


def rewrite_swapped_size_fields(path: Path, file_format: str, make_backup: bool = True) -> int:
    """Swap height,width into width,height in-place.

    For current Partition_Info, size fields are columns 6 and 7:
      seq qp frame ctu x y width height ...

    For legacy FastOff, size fields are columns 2 and 3:
      x y width height ...
    """

    size_a_idx, size_b_idx = (6, 7) if file_format == "info" else (2, 3)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    out_lines: List[str] = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "frame++" or stripped.startswith("#"):
            out_lines.append(line)
            continue

        newline = ""
        body = line
        if body.endswith("\n"):
            newline = "\n"
            body = body[:-1]

        fields = body.split()
        if len(fields) <= size_b_idx:
            out_lines.append(line)
            continue
        fields[size_a_idx], fields[size_b_idx] = fields[size_b_idx], fields[size_a_idx]
        out_lines.append(" ".join(fields) + newline)
        changed += 1

    if make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    path.write_text("".join(out_lines), encoding="utf-8")
    return changed


def rewrite_declib_ctu_ids(path: Path, component: str, source_width: int, make_backup: bool = True) -> int:
    """Rewrite ctu_id in current Partition_Info files using DecLib's formula."""

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    out_lines: List[str] = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        newline = ""
        body = line
        if body.endswith("\n"):
            newline = "\n"
            body = body[:-1]

        fields = body.split()
        if len(fields) < 8:
            out_lines.append(line)
            continue
        try:
            x = int(fields[4])
            y = int(fields[5])
            expected = ctu_id_from_xy(x, y, source_width, component)
        except ValueError:
            out_lines.append(line)
            continue

        if fields[3] != str(expected):
            fields[3] = str(expected)
            changed += 1
        out_lines.append(" ".join(fields) + newline)

    if changed and make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    if changed:
        path.write_text("".join(out_lines), encoding="utf-8")
    return changed


def rewrite_training_declib_ctu_ids(
    path: Path,
    component: str,
    cfg_root: Path,
    dataset: str,
    make_backup: bool = True,
) -> int:
    """Rewrite ctu_id in a mixed training Partition_Info file."""

    dim_cache: Dict[Tuple[str, int], SequenceDims] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    out_lines: List[str] = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue

        newline = ""
        body = line
        if body.endswith("\n"):
            newline = "\n"
            body = body[:-1]

        fields = body.split()
        if len(fields) < 8:
            out_lines.append(line)
            continue
        try:
            sequence = fields[0]
            qp = int(fields[1])
            x = int(fields[4])
            y = int(fields[5])
        except ValueError:
            out_lines.append(line)
            continue

        dims = dims_for(cfg_root, dataset, qp, sequence, dim_cache)
        if dims.width is None:
            out_lines.append(line)
            continue
        expected = ctu_id_from_xy(x, y, dims.width, component)
        if fields[3] != str(expected):
            fields[3] = str(expected)
            changed += 1
        out_lines.append(" ".join(fields) + newline)

    if changed and make_backup:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    if changed:
        path.write_text("".join(out_lines), encoding="utf-8")
    return changed


def write_report(report_path: Path, all_stats: List[FileStats]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    labels = defaultdict(int)
    id_errors = 0
    id_checked = 0
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write("# Partition Info Self Check Report\n\n")
        for stats in all_stats:
            labels[geometry_label(stats)] += 1
            id_errors += stats.id_errors
            id_checked += stats.id_checked

        fp.write("## Summary\n")
        fp.write(f"files_checked: {len(all_stats)}\n")
        fp.write(f"id_errors: {id_errors}/{id_checked}\n")
        for label in sorted(labels):
            fp.write(f"{label}: {labels[label]}\n")

        fp.write("\n## Problem Files\n")
        for stats in all_stats:
            label = geometry_label(stats)
            if label.startswith("OK_") and stats.id_errors == 0 and stats.parse_errors == 0:
                continue
            fp.write(summarize_file(stats) + "\n")
            if stats.id_examples:
                fp.write("  id examples:\n")
                for example in stats.id_examples:
                    fp.write(f"    {example}\n")
            if stats.wh.examples:
                fp.write("  width,height geometry examples:\n")
                for example in stats.wh.examples:
                    fp.write(f"    {example}\n")
            if stats.hw.examples:
                fp.write("  height,width geometry examples:\n")
                for example in stats.hw.examples:
                    fp.write(f"    {example}\n")

        fp.write("\n## All Files\n")
        for stats in all_stats:
            fp.write(summarize_file(stats) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check partition info geometry and ctu_id consistency.")
    parser.add_argument("--dataset", default="DIV2K")
    parser.add_argument("--partition-root", type=Path, default=Path("data/partition"))
    parser.add_argument("--cfg-root", type=Path, default=Path("data/CodecTrainCfg"))
    parser.add_argument("--qps", nargs="+", type=int, default=list(DEFAULT_QPS))
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional sequence names to check, e.g. 0055 0332.")
    parser.add_argument("--limit-sequences", type=int, default=None)
    parser.add_argument("--report", type=Path, default=Path("network/output/partition_self_check_report.txt"))
    parser.add_argument("--print-problems", action="store_true")
    parser.add_argument(
        "--training",
        action="store_true",
        help="Check data/partition/<dataset>/training/{Luma,Chroma}_Partition_Info.txt.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream training files group-by-group instead of loading them into memory.",
    )
    parser.add_argument(
        "--apply-fix",
        action="store_true",
        help="Rewrite files classified as SUSPECT_height_width by swapping their size fields to width,height.",
    )
    parser.add_argument(
        "--fix-ids",
        action="store_true",
        help="Rewrite wrong ctu_id values in Partition_Info files using DecLib's current formula.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak files when --apply-fix is used.",
    )
    parser.add_argument(
        "--no-legacy",
        action="store_true",
        help="Only check Luma/Chroma_Partition_Info.txt and ignore legacy FastOff files.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    all_stats: List[FileStats] = []
    missing_files = 0
    checked_sequence_qp = 0
    dim_cache: Dict[Tuple[str, int], SequenceDims] = {}

    if args.training:
        training_dir = args.partition_root / args.dataset / "training"
        for component in args.components:
            path = training_dir / f"{component}_Partition_Info.txt"
            if not path.exists():
                missing_files += 1
                continue
            if args.stream or args.apply_fix or args.fix_ids:
                stats = stream_process_training_file(
                    path,
                    component,
                    args.cfg_root,
                    args.dataset,
                    apply_fix=args.apply_fix,
                    fix_ids=args.fix_ids,
                    make_backup=not args.no_backup,
                )
            else:
                stats = check_training_file(path, component, args.cfg_root, args.dataset, dim_cache)
            all_stats.append(stats)
            if args.print_problems and (not geometry_label(stats).startswith("OK_") or stats.id_errors):
                print(summarize_file(stats))

    for qp in ([] if args.training else args.qps):
        qp_dir = args.partition_root / args.dataset / f"qp_{qp}"
        if not qp_dir.exists():
            print(f"missing qp dir: {qp_dir}")
            continue
        seq_dirs = sorted(path for path in qp_dir.iterdir() if path.is_dir())
        if args.sequences is not None:
            wanted = set(args.sequences)
            seq_dirs = [path for path in seq_dirs if path.name in wanted]
        if args.limit_sequences is not None:
            seq_dirs = seq_dirs[: args.limit_sequences]

        for seq_dir in seq_dirs:
            sequence = seq_dir.name
            checked_sequence_qp += 1
            cfg_path = cfg_path_for(args.cfg_root, args.dataset, qp, sequence)
            source_width = read_cfg_value(cfg_path, "SourceWidth")
            source_height = read_cfg_value(cfg_path, "SourceHeight")
            for component in args.components:
                discovered = discover_component_file(
                    seq_dir,
                    sequence,
                    qp,
                    component,
                    include_legacy=not args.no_legacy,
                )
                if discovered is None:
                    missing_files += 1
                    continue
                path, file_format = discovered
                stats = check_file(path, file_format, component, qp, sequence, source_width, source_height)
                if args.apply_fix and geometry_label(stats) == "SUSPECT_height_width":
                    rewrite_swapped_size_fields(path, file_format, make_backup=not args.no_backup)
                    stats = check_file(path, file_format, component, qp, sequence, source_width, source_height)
                    stats.fixed = True
                if args.fix_ids and stats.file_format == "info" and stats.id_errors and source_width is not None:
                    rewrite_declib_ctu_ids(path, component, source_width, make_backup=not args.no_backup)
                    stats = check_file(path, file_format, component, qp, sequence, source_width, source_height)
                    stats.fixed_ids = True
                all_stats.append(stats)
                if args.print_problems and (not geometry_label(stats).startswith("OK_") or stats.id_errors):
                    print(summarize_file(stats))

    write_report(args.report, all_stats)

    label_counts = defaultdict(int)
    total_id_errors = 0
    total_id_checked = 0
    for stats in all_stats:
        label_counts[geometry_label(stats)] += 1
        total_id_errors += stats.id_errors
        total_id_checked += stats.id_checked

    print(f"sequence_qp_dirs_checked: {checked_sequence_qp}")
    print(f"files_checked: {len(all_stats)}")
    print(f"missing_component_files: {missing_files}")
    print(f"id_errors: {total_id_errors}/{total_id_checked}")
    for label in sorted(label_counts):
        print(f"{label}: {label_counts[label]}")
    print(f"report: {args.report}")
    return 1 if total_id_errors or any(not label.startswith("OK_") for label in label_counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
