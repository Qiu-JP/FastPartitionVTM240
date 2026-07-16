#!/usr/bin/env python3

import math
import re
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit("usage: parseLog.py ENCODER_LOG")

lines = Path(sys.argv[1]).read_text(errors="replace").splitlines()
summary = {}
elapsed = None
number = r"(?:[-+]?nan|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
metric_re = re.compile(
    r"^\s*(\d+)\s+([aipb])\s+(" + number + r")\s+(" + number
    + r")\s+(" + number + r")\s+(" + number + r")"
)
time_re = re.compile(
    r"Total Time:\s*[0-9.]+\s*sec\.\s*\[user\]\s*"
    r"([0-9.]+)\s*sec\.\s*\[elapsed\]"
)

for line in lines:
    match = metric_re.match(line)
    if match:
        summary[match.group(2)] = [
            int(match.group(1)),
            *[float(match.group(index)) for index in range(3, 7)],
        ]
        continue
    match = time_re.search(line)
    if match:
        elapsed = float(match.group(1))

if "a" not in summary:
    raise SystemExit("missing average summary row")
if elapsed is None:
    raise SystemExit("missing elapsed time")

average = summary["a"]
intra = summary.get("i", average)
if any(not math.isfinite(value) for value in average[1:] + intra[1:] + [elapsed]):
    raise SystemExit("summary contains non-finite values")

print("\t".join(str(value) for value in average + intra + [elapsed]))
