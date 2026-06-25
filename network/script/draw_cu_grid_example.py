#!/usr/bin/env python3
"""
Draw a 64x64 CU example from real training partition rows.

Each training partition row is interpreted with the current DecLib format:
    sequence qp frame_id ctu_id x y width height ...

The CU rectangle is first converted to 4x4 grid units:
    grid_x = (x - lcu_x) / 4
    grid_y = (y - lcu_y) / 4
    grid_w = width / 4
    grid_h = height / 4

Then its rectangle boundary is drawn on a 16x16 grid.  This makes cases such
as an 8x16 CU visibly occupy 2 grid columns and 4 grid rows.
"""

from pathlib import Path
import subprocess


SEQ = "0055"
QP = 27
FRAME_IDX = 0
LCU_X = 1344
LCU_Y = 448
LCU_SIZE = 64
GRID_UNIT = 4
GRID = LCU_SIZE // GRID_UNIT
SCALE = 28
PAD = 10

ROOT = Path(__file__).resolve().parents[2]
YUV_PATH = ROOT / "data/video/DIV2K/0055_2040x1152.yuv"
PARTITION_PATH = ROOT / "data/partition/DIV2K/training/Luma_Partition_Info.txt"
OUT_DIR = ROOT / "network/figures/div2k_0055_qp27_block_example"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 2040
HEIGHT = 1152
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2
Y_SIZE = WIDTH * HEIGHT

V_CAND = (183, 91, 18)
V_REAL = (255, 224, 200)
H_CAND = (38, 86, 180)
H_REAL = (190, 210, 245)
OUTLINE = (255, 245, 235)
LINE_WIDTH = 3
DASH = 9
GAP = 7
TARGET_CTU_ID = 235


def parse_partition_rows():
    rows = []
    with PARTITION_PATH.open("r", encoding="utf-8", errors="ignore") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                sequence = fields[0]
                qp = int(fields[1])
                frame_id = int(fields[2])
                ctu_id = int(fields[3])
                x = int(fields[4])
                y = int(fields[5])
                w = int(fields[6])
                h = int(fields[7])
            except ValueError:
                continue
            if sequence != SEQ or qp != QP or frame_id != FRAME_IDX:
                continue
            if ctu_id != TARGET_CTU_ID:
                continue
            if LCU_X <= x < LCU_X + LCU_SIZE and LCU_Y <= y < LCU_Y + LCU_SIZE:
                rows.append((lineno, x, y, w, h))
    return rows


def row_to_grid(row):
    lineno, x, y, w, h = row
    if (x - LCU_X) % GRID_UNIT or (y - LCU_Y) % GRID_UNIT or w % GRID_UNIT or h % GRID_UNIT:
        raise ValueError("CU row is not aligned to 4x4 grid units: {}".format(row))
    gx = (x - LCU_X) // GRID_UNIT
    gy = (y - LCU_Y) // GRID_UNIT
    gw = w // GRID_UNIT
    gh = h // GRID_UNIT
    return lineno, gx, gy, gw, gh


def read_luma_block():
    with YUV_PATH.open("rb") as fp:
        fp.seek(FRAME_IDX * FRAME_SIZE)
        y_plane = fp.read(Y_SIZE)
    if len(y_plane) != Y_SIZE:
        raise RuntimeError("short YUV read")
    block = [
        [y_plane[(LCU_Y + y) * WIDTH + (LCU_X + x)] for x in range(LCU_SIZE)]
        for y in range(LCU_SIZE)
    ]
    return block


def make_canvas(block):
    side = LCU_SIZE * SCALE + 2 * PAD
    canvas = bytearray(side * side * 3)
    for yy in range(side):
        for xx in range(side):
            sx = min(LCU_SIZE - 1, max(0, (xx - PAD) // SCALE))
            sy = min(LCU_SIZE - 1, max(0, (yy - PAD) // SCALE))
            if PAD <= xx < PAD + LCU_SIZE * SCALE and PAD <= yy < PAD + LCU_SIZE * SCALE:
                v = block[sy][sx]
            else:
                v = 255
            idx = (yy * side + xx) * 3
            canvas[idx : idx + 3] = bytes((v, v, v))
    return canvas, side, side


def set_px(canvas, width, height, x, y, color):
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        canvas[idx : idx + 3] = bytes(color)


def draw_line(canvas, width, height, x1, y1, x2, y2, color, line_width=LINE_WIDTH, dashed=False):
    if x1 == x2:
        if y2 < y1:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            if dashed and ((y - y1) % (DASH + GAP)) >= DASH:
                continue
            for dx in range(-(line_width // 2), line_width - line_width // 2):
                set_px(canvas, width, height, x1 + dx, y, color)
    elif y1 == y2:
        if x2 < x1:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            if dashed and ((x - x1) % (DASH + GAP)) >= DASH:
                continue
            for dy in range(-(line_width // 2), line_width - line_width // 2):
                set_px(canvas, width, height, x, y1 + dy, color)


def grid_x(gx):
    return PAD + gx * GRID_UNIT * SCALE


def grid_y(gy):
    return PAD + gy * GRID_UNIT * SCALE


def draw_candidate_grid(canvas, width, height, mode):
    if mode in ("v", "both"):
        for gx in range(1, GRID):
            x = grid_x(gx)
            draw_line(canvas, width, height, x, PAD, x, PAD + LCU_SIZE * SCALE, V_CAND, dashed=True)
    if mode in ("h", "both"):
        for gy in range(1, GRID):
            y = grid_y(gy)
            draw_line(canvas, width, height, PAD, y, PAD + LCU_SIZE * SCALE, y, H_CAND, dashed=True)


def draw_patch_grid(canvas, width, height):
    for gx in range(1, GRID):
        x = grid_x(gx)
        draw_line(canvas, width, height, x, PAD, x, PAD + LCU_SIZE * SCALE, (0, 0, 0), line_width=2, dashed=True)
    for gy in range(1, GRID):
        y = grid_y(gy)
        draw_line(canvas, width, height, PAD, y, PAD + LCU_SIZE * SCALE, y, (0, 0, 0), line_width=2, dashed=True)


def draw_cu_boundaries(canvas, width, height, grid_rows, mode):
    for lineno, gx, gy, gw, gh in grid_rows:
        x0 = grid_x(gx)
        y0 = grid_y(gy)
        x1 = grid_x(gx + gw)
        y1 = grid_y(gy + gh)
        if mode in ("v", "both"):
            draw_line(canvas, width, height, x0, y0, x0, y1, V_REAL)
            draw_line(canvas, width, height, x1, y0, x1, y1, V_REAL)
        if mode in ("h", "both"):
            draw_line(canvas, width, height, x0, y0, x1, y0, H_REAL)
            draw_line(canvas, width, height, x0, y1, x1, y1, H_REAL)


def draw_outer_boundary(canvas, width, height, mode):
    if mode in ("v", "both"):
        draw_line(canvas, width, height, PAD, PAD, PAD, PAD + LCU_SIZE * SCALE, OUTLINE)
        draw_line(canvas, width, height, PAD + LCU_SIZE * SCALE, PAD, PAD + LCU_SIZE * SCALE, PAD + LCU_SIZE * SCALE, OUTLINE)
    if mode in ("h", "both"):
        draw_line(canvas, width, height, PAD, PAD, PAD + LCU_SIZE * SCALE, PAD, OUTLINE)
        draw_line(canvas, width, height, PAD, PAD + LCU_SIZE * SCALE, PAD + LCU_SIZE * SCALE, PAD + LCU_SIZE * SCALE, OUTLINE)


def save_png(stem, canvas, width, height):
    ppm = OUT_DIR / "{}.ppm".format(stem)
    png = OUT_DIR / "{}.png".format(stem)
    with ppm.open("wb") as fp:
        fp.write("P6\n{} {}\n255\n".format(width, height).encode("ascii"))
        fp.write(canvas)
    subprocess.run(
        ["/opt/tiger/ss_bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(ppm), str(png)],
        check=True,
    )
    ppm.unlink()
    return png


def draw(stem, mode, block, grid_rows, candidate=True):
    canvas, width, height = make_canvas(block)
    if candidate:
        draw_candidate_grid(canvas, width, height, mode)
    draw_cu_boundaries(canvas, width, height, grid_rows, mode)
    draw_outer_boundary(canvas, width, height, mode)
    return save_png(stem, canvas, width, height)


def draw_raw(stem, block):
    canvas, width, height = make_canvas(block)
    return save_png(stem, canvas, width, height)


def draw_grid(stem, block):
    canvas, width, height = make_canvas(block)
    draw_patch_grid(canvas, width, height)
    return save_png(stem, canvas, width, height)


def main():
    rows = parse_partition_rows()
    grid_rows = [row_to_grid(row) for row in rows]
    block = read_luma_block()

    outputs = [
        draw_raw("01_luma_raw", block),
        draw_grid("02_luma_4x4_patch_grid", block),
        draw("03_luma_vertical_boundaries_orange_real_qp27", "v", block, grid_rows),
        draw("04_luma_horizontal_boundaries_blue_real_qp27", "h", block, grid_rows),
        draw("05_luma_combined_boundaries_real_qp27", "both", block, grid_rows, candidate=False),
        draw("03_luma_vertical_boundaries_reference_palette", "v", block, grid_rows),
        draw("04_luma_horizontal_boundaries_reference_palette", "h", block, grid_rows),
    ]

    debug_path = OUT_DIR / "cu_rows_grid_debug.txt"
    with debug_path.open("w", encoding="utf-8") as fp:
        fp.write("source: {}\n".format(PARTITION_PATH))
        fp.write("target: seq={} qp={} frame={} ctu_id={} lcu=({}, {})\n".format(SEQ, QP, FRAME_IDX, TARGET_CTU_ID, LCU_X, LCU_Y))
        fp.write("lineno x y w h -> grid_x grid_y grid_w grid_h\n")
        for row, grid_row in zip(rows, grid_rows):
            lineno, x, y, w, h = row
            _, gx, gy, gw, gh = grid_row
            fp.write("{} {} {} {} {} -> {} {} {} {}\n".format(lineno, x, y, w, h, gx, gy, gw, gh))

    with (OUT_DIR / "metadata.txt").open("w", encoding="utf-8") as fp:
        fp.write("Sequence: {}\n".format(SEQ))
        fp.write("QP: {}\n".format(QP))
        fp.write("Frame: {}\n".format(FRAME_IDX))
        fp.write("CTU id: {}\n".format(TARGET_CTU_ID))
        fp.write("LCU origin: {}, {}\n".format(LCU_X, LCU_Y))
        fp.write("Partition: {}\n".format(PARTITION_PATH))
        fp.write("YUV: {}\n".format(YUV_PATH))
        fp.write(
            "Redrawn with network/script/draw_cu_grid_example.py from real training rows. "
            "Each row follows DecLib order x,y,width,height and is converted to grid_x,grid_y,grid_w,grid_h.\n"
        )
        fp.write("Debug grid rows: {}\n".format(debug_path))

    print("rows:", len(rows))
    print("first:", rows[0], "grid:", grid_rows[0])
    print("second:", rows[1], "grid:", grid_rows[1])
    for output in outputs:
        print(output)
    print(debug_path)


if __name__ == "__main__":
    main()
