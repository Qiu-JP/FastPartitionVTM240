#!/usr/bin/env python3
"""
Draw a hypothetical soft network prediction for the DIV2K 0055 QP27 example.

The script reuses the debug grid rows generated from the real training label
and renders a plausible prediction: most true CU boundaries are present with
probabilities around 0.75-0.98, and a few short boundary fragments are extended
or added with lower confidence.
"""

from pathlib import Path
import hashlib
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
OUT_DIR = ROOT / "network/figures/div2k_0055_qp27_block_example"
DEBUG_PATH = OUT_DIR / "cu_rows_grid_debug.txt"
OUT_PATH = OUT_DIR / "06_luma_predicted_boundaries_hypothetical_qp27.png"
DEBUG_OUT = OUT_DIR / "predicted_soft_boundary_debug.txt"

WIDTH = 2040
HEIGHT = 1152
FRAME_SIZE = WIDTH * HEIGHT * 3 // 2
Y_SIZE = WIDTH * HEIGHT

V_COLOR = (255, 185, 120)
H_COLOR = (150, 185, 255)
LINE_WIDTH = 3


def stable01(*items):
    key = "|".join(map(str, items)).encode("utf-8")
    digest = hashlib.sha1(key).digest()
    value = int.from_bytes(digest[:4], "big")
    return value / 0xFFFFFFFF


def read_luma_block():
    with YUV_PATH.open("rb") as fp:
        fp.seek(FRAME_IDX * FRAME_SIZE)
        y_plane = fp.read(Y_SIZE)
    if len(y_plane) != Y_SIZE:
        raise RuntimeError("short YUV read")
    return [
        [y_plane[(LCU_Y + y) * WIDTH + (LCU_X + x)] for x in range(LCU_SIZE)]
        for y in range(LCU_SIZE)
    ]


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


def blend_px(canvas, width, height, x, y, color, alpha):
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        for c in range(3):
            base = canvas[idx + c]
            canvas[idx + c] = int(round(base * (1.0 - alpha) + color[c] * alpha))


def draw_line(canvas, width, height, x1, y1, x2, y2, color, alpha):
    if x1 == x2:
        if y2 < y1:
            y1, y2 = y2, y1
        for y in range(y1, y2 + 1):
            for dx in range(-(LINE_WIDTH // 2), LINE_WIDTH - LINE_WIDTH // 2):
                blend_px(canvas, width, height, x1 + dx, y, color, alpha)
    elif y1 == y2:
        if x2 < x1:
            x1, x2 = x2, x1
        for x in range(x1, x2 + 1):
            for dy in range(-(LINE_WIDTH // 2), LINE_WIDTH - LINE_WIDTH // 2):
                blend_px(canvas, width, height, x, y1 + dy, color, alpha)


def grid_x(gx):
    return PAD + gx * GRID_UNIT * SCALE


def grid_y(gy):
    return PAD + gy * GRID_UNIT * SCALE


def parse_grid_rows():
    rows = []
    with DEBUG_PATH.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if "->" not in line or line.startswith("lineno"):
                continue
            left, right = line.split("->", 1)
            fields = left.split()
            grid = right.split()
            if len(fields) < 5 or len(grid) < 4:
                continue
            lineno = int(fields[0])
            gx, gy, gw, gh = map(int, grid[:4])
            rows.append((lineno, gx, gy, gw, gh))
    return rows


def collect_segments(grid_rows):
    v_segments = set()
    h_segments = set()
    for lineno, gx, gy, gw, gh in grid_rows:
        for yy in range(gy, gy + gh):
            v_segments.add((gx, yy))
            v_segments.add((gx + gw, yy))
        for xx in range(gx, gx + gw):
            h_segments.add((xx, gy))
            h_segments.add((xx, gy + gh))
    return v_segments, h_segments


def probability(kind, segment):
    base = 0.78 + 0.20 * stable01(kind, *segment)
    if stable01("weak", kind, *segment) < 0.07:
        base -= 0.25
    return max(0.35, min(0.98, base))


def add_soft_prediction_artifacts(v_segments, h_segments):
    v_pred = dict((seg, probability("v", seg)) for seg in v_segments)
    h_pred = dict((seg, probability("h", seg)) for seg in h_segments)

    # A few deterministic low-confidence false positives / one-cell extensions.
    extra_v = [(4, 8), (8, 4), (11, 10), (14, 13)]
    extra_h = [(2, 8), (7, 5), (10, 11), (13, 7)]
    for seg in extra_v:
        if 0 <= seg[0] <= GRID and 0 <= seg[1] < GRID:
            v_pred.setdefault(seg, 0.32 + 0.18 * stable01("extra_v", *seg))
    for seg in extra_h:
        if 0 <= seg[0] < GRID and 0 <= seg[1] <= GRID:
            h_pred.setdefault(seg, 0.32 + 0.18 * stable01("extra_h", *seg))

    return v_pred, h_pred


def save_png(canvas, width, height):
    ppm = OUT_PATH.with_suffix(".ppm")
    with ppm.open("wb") as fp:
        fp.write("P6\n{} {}\n255\n".format(width, height).encode("ascii"))
        fp.write(canvas)
    subprocess.run(
        ["/opt/tiger/ss_bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(ppm), str(OUT_PATH)],
        check=True,
    )
    ppm.unlink()


def main():
    rows = parse_grid_rows()
    v_label, h_label = collect_segments(rows)
    v_pred, h_pred = add_soft_prediction_artifacts(v_label, h_label)

    block = read_luma_block()
    canvas, width, height = make_canvas(block)

    for (gx, gy), score in sorted(v_pred.items()):
        draw_line(canvas, width, height, grid_x(gx), grid_y(gy), grid_x(gx), grid_y(gy + 1), V_COLOR, score)
    for (gx, gy), score in sorted(h_pred.items()):
        draw_line(canvas, width, height, grid_x(gx), grid_y(gy), grid_x(gx + 1), grid_y(gy), H_COLOR, score)

    save_png(canvas, width, height)

    with DEBUG_OUT.open("w", encoding="utf-8") as fp:
        fp.write("Hypothetical soft prediction for seq={} qp={} frame={} LCU=({}, {})\n".format(SEQ, QP, FRAME_IDX, LCU_X, LCU_Y))
        fp.write("Vertical segments: {}\n".format(len(v_pred)))
        fp.write("Horizontal segments: {}\n".format(len(h_pred)))
        fp.write("Each score is rendered as line opacity.\n\n")
        for seg, score in sorted(v_pred.items()):
            fp.write("V {} {:.3f}{}\n".format(seg, score, " extra" if seg not in v_label else ""))
        for seg, score in sorted(h_pred.items()):
            fp.write("H {} {:.3f}{}\n".format(seg, score, " extra" if seg not in h_label else ""))

    print(OUT_PATH)
    print(DEBUG_OUT)


if __name__ == "__main__":
    main()
