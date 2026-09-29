#!/usr/bin/env python3
"""Run the detector over frames exported by bag_to_npz.py and print a log.

    python3 tools/eval_frames.py frames.npz [--csv out.csv] [--pkg path]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('--csv')
ap.add_argument('--pkg')
ap.add_argument('--quiet', action='store_true')
ap.add_argument('--set', nargs='*', default=[], help='param=value overrides')
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, args.pkg or os.path.join(HERE, '..', 'src', 'metro_obstacle_core'))
from metro_obstacle_core.detector import GeometricDetector  # noqa: E402

overrides = {}
for item in args.set:
    k, v = item.split('=', 1)
    overrides[k] = eval(v)
data = np.load(args.npz)
sizes, stamps = data['sizes'], data['stamps']
xyz = data['xyz_cm'].astype(np.float32) / 100.0
offsets = np.r_[0, np.cumsum(sizes)]
det = GeometricDetector(**overrides)
rows = []
for i, stamp in enumerate(stamps):
    cloud = xyz[offsets[i]:offsets[i + 1]]
    t0 = time.perf_counter()
    r = det.process(cloud, 0.1, stamp=float(stamp))
    ms = (time.perf_counter() - t0) * 1e3
    acc = [c for c in r.get('candidates', []) if c.get('reason') == 'volume']
    row = dict(frame=i, t=round(float(stamp - stamps[0]), 2), detected=int(r['detected']),
               distance=round(float(r.get('distance', np.nan)), 2),
               lateral=round(float(r.get('y', np.nan)), 2),
               category=r.get('category', ''), conf=round(float(r.get('confidence', 0)), 2),
               rail_end=round(float(r.get('rail_range', [0, np.nan])[1]), 1),
               det_end=round(float(r.get('detection_range', [0, np.nan])[1]), 1),
               meas_end=round(float(r.get('measured_range', [0, np.nan])[1]), 1),
               src=r.get('corridor_source', r.get('reason')),
               speed=round(float(r.get('ego_speed', np.nan)), 2), ms=round(ms, 1),
               accepted=';'.join('%.1f/%.2f/%.2f/%d' % (c['distance'], c['y'], c['maximum'][2],
                                                       c['points']) for c in acc[:4]))
    rows.append(row)
    if not args.quiet:
        print(row, flush=True)
if args.csv:
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
det_frames = [r for r in rows if r['detected']]
print('frames', len(rows), 'detected', len(det_frames),
      'first', det_frames[0]['frame'] if det_frames else None,
      'max_dist', max([r['distance'] for r in det_frames], default=None),
      'ms_median', float(np.median([r['ms'] for r in rows])),
      'ms_p95', float(np.percentile([r['ms'] for r in rows], 95)))
