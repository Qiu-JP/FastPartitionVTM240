#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run in FastPartitionVTM, or explicitly select its interpreter.
"${FASTPARTITION_PYTHON:-python}" - "${SCRIPT_DIR}" "$@" <<'PY'
import argparse
import os
from pathlib import Path
import sys
import tempfile

from PIL import Image

script_dir = Path(sys.argv.pop(1))
parser = argparse.ArgumentParser(description="Generate a sequence list from PNG dimensions; does not convert images to YUV.")
parser.add_argument('--input-dir', type=Path, required=True)
parser.add_argument('--output-file', type=Path, default=script_dir / 'Training_Sequences_DIV2K.txt')
parser.add_argument('--frames', type=int, default=1)
parser.add_argument('--fps', type=int, default=1)
args = parser.parse_args()
if not args.input_dir.is_dir():
    parser.error('input directory does not exist')
if args.frames <= 0 or args.fps <= 0:
    parser.error('frames and fps must be positive')
images = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png')
if not images:
    parser.error('input directory contains no PNG images; existing output is unchanged')
rows = []
names = set()
for image in images:
    name = image.stem
    if any(c in name for c in ',\r\n#') or name in names:
        parser.error(f'unsupported or duplicate sequence name: {name!r}')
    names.add(name)
    with Image.open(image) as img:
        width, height = img.size
        img.verify()
    if width % 2 or height % 2:
        parser.error(f'{image.name}: YUV420 requires even width and height')
    rows.append(f'{name},{name}_{width}x{height}.yuv,{width},{height},{args.frames},{args.fps}\n')
args.output_file.parent.mkdir(parents=True, exist_ok=True)
# Replace only after every image has been validated.
tmp_name = None
try:
    with tempfile.NamedTemporaryFile(mode='w', dir=args.output_file.parent, delete=False) as tmp:
        tmp_name = tmp.name
        tmp.writelines(rows)
    os.replace(tmp_name, args.output_file)
finally:
    if tmp_name and os.path.exists(tmp_name):
        os.unlink(tmp_name)
print(f'Generated {len(rows)} sequences: {args.output_file}')
PY
