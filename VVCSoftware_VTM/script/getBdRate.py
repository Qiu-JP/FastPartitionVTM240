#!/usr/bin/env python3
"""Aggregate paired intra evaluation curves into per-sequence/average metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from getBdRateCore import getBdRateCore


def bd_rate(anchor, test, component):
    if len(anchor) < 4:
        return None
    a, t = np.array(anchor), np.array(test)
    if (len(np.unique(a[:, component])) != len(a)
            or len(np.unique(t[:, component])) != len(t)
            or max(a[:, component].min(), t[:, component].min()) >= min(a[:, component].max(), t[:, component].max())):
        return None  # No usable common PSNR interval; never report a false zero.
    return float(getBdRateCore(a[:, 0], a[:, component], t[:, 0], t[:, component]))


def summarize(records):
    rows = []
    for name, runs in records.items():
        anchor = [r['anchor']['rd'] for r in runs]
        test = [r['test']['rd'] for r in runs]
        row = {'sequence': name, **{f'BD_{c}_pct': bd_rate(anchor, test, i) for i, c in enumerate('YUV', 1)}}
        for key, field, source in [('time_saving_pct', 'elapsed', 'anchor'),
                                   ('luma_compute_saving_pct', 'compute_luma', 'anchor'),
                                   ('total_compute_saving_pct', 'compute_total', 'anchor')]:
            values = []
            for r in runs:
                baseline = r[source][field]
                if baseline is None or baseline <= 0 or r['test'][field] is None:
                    raise ValueError(f'Missing/invalid {field} for {name}')
                values.append(100 * (1 - r['test'][field] / baseline))
            row[key] = float(np.mean(values))
        rows.append(row)
    if not rows:
        raise ValueError('No completed sequences')
    # Do not quietly average only a subset of sequences with valid BD intervals.
    rows.append({'sequence': 'Average', **{
        key: float(np.mean([r[key] for r in rows])) if all(r[key] is not None for r in rows) else None
        for key in rows[0] if key != 'sequence'}})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('curves', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = summarize(json.loads(args.curves.read_text()))
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows[-1], ensure_ascii=False))


if __name__ == '__main__':
    main()
