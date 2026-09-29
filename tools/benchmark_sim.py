#!/usr/bin/env python3
"""Reproducible experiments on the ray-cast tunnel simulator.

Usage:
    python3 tools/benchmark_sim.py            # full suite, prints a table
    python3 tools/benchmark_sim.py --quick    # shorter sequences
    python3 tools/benchmark_sim.py --json out.json

Metrics:
    * static detection rate / distance error for an obstacle at D metres,
    * first confirmed detection distance while the train approaches,
    * false alarm frames in empty tunnels (straight, curves, walkway, dust),
    * processing time per frame.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_pkg = None
if '--pkg' in sys.argv:
    _pkg = sys.argv[sys.argv.index('--pkg') + 1]
sys.path.insert(0, _pkg or os.path.join(HERE, '..', 'src', 'metro_obstacle_core'))

from metro_obstacle_core.simulation import (LidarModel, Obstacle,  # noqa: E402
                                            TunnelScene, TunnelSimulator)


def make_detector(pkg_path=None, **kwargs):
    from metro_obstacle_core.detector import GeometricDetector
    return GeometricDetector(lidar_xyz=(0., 0., 0.), lidar_rpy=(0., 0., 0.), **kwargs)


def run(scene, frames, speed, start=0.0, seed=0, lidar=None, factory=None):
    sim = TunnelSimulator(scene, lidar or LidarModel(), seed)
    det = factory()
    log = []
    for i in range(frames):
        t = 100.0 + i * 0.1
        s = start + speed * i * 0.1
        cloud, truth = sim.scan(s, i * 0.1)
        t0 = time.perf_counter()
        res = det.process(cloud, 0.1, stamp=t)
        ms = (time.perf_counter() - t0) * 1000.0
        log.append(dict(detected=bool(res.get('detected')), distance=res.get('distance'),
                        ms=ms, truth=truth, lateral=res.get('y')))
    return log


def static_case(kind, distance, factory, frames, v=0.0, scene_kw=None, seed=1):
    scene = TunnelScene(obstacles=[Obstacle(kind, distance, v)], **(scene_kw or {}))
    log = run(scene, frames, 0.0, seed=seed, factory=factory)
    tail = log[3:]
    hits = [r for r in tail if r['detected'] and abs(r['distance'] - distance) < 3.0]
    wrong = [r for r in tail if r['detected'] and abs(r['distance'] - distance) >= 3.0]
    err = float(np.median([abs(r['distance'] - distance) for r in hits])) if hits else None
    return dict(rate=len(hits) / len(tail), wrong=len(wrong), err=err,
                ms=float(np.median([r['ms'] for r in log])))


def approach_case(kind, start_distance, speed, factory, frames, scene_kw=None, seed=2, v=0.0):
    scene = TunnelScene(obstacles=[Obstacle(kind, start_distance, v)], **(scene_kw or {}))
    log = run(scene, frames, speed, seed=seed, factory=factory)
    first = None
    streak = 0
    for r in log:
        true_d = r['truth'][0]['distance']
        ok = r['detected'] and abs(r['distance'] - true_d) < max(3.0, .03 * true_d)
        streak = streak + 1 if ok else 0
        if streak >= 3 and first is None:
            first = true_d + 2 * speed * 0.1   # distance at start of stable streak
    return dict(first_stable_detection_m=first)


def empty_case(scene_kw, factory, frames, speed=12.0, seed=3, lidar=None):
    log = run(TunnelScene(**scene_kw), frames, speed, seed=seed, factory=factory, lidar=lidar)
    fa = [r for r in log if r['detected']]
    return dict(false_alarm_frames=len(fa), frames=len(log),
                fa_distances=[round(float(r['distance']), 1) for r in fa][:10],
                ms_median=float(np.median([r['ms'] for r in log])),
                ms_p95=float(np.percentile([r['ms'] for r in log], 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', help='path to an alternative metro_obstacle_core package root')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--json')
    ap.add_argument('--only', default='static,approach,empty')
    args = ap.parse_args()
    factory = lambda: make_detector(args.pkg)  # noqa: E731
    make_detector(args.pkg)   # import once with the right path
    frames = 12 if args.quick else 20
    out = {}
    if 'static' in args.only:
        out['static'] = {}
        for kind in ('person', 'box', 'low_box'):
            for d in (30, 60, 100, 150, 200, 250, 300):
                r = static_case(kind, float(d), factory, frames)
                out['static'][f'{kind}@{d}'] = r
                print(f'static {kind:8s} {d:4d} m  rate={r["rate"]:.2f} err={r["err"]} '
                      f'wrong={r["wrong"]} ms={r["ms"]:.0f}', flush=True)
        for v in (0.6, 1.0):
            r = static_case('person', 80.0, factory, frames, v=v)
            out['static'][f'person@80 v={v}'] = r
            print(f'static person lateral {v} m @80  rate={r["rate"]:.2f}', flush=True)
        r = static_case('person', 70.0, factory, frames,
                        scene_kw=dict(curve_start=10.0, curve_radius=600.0))
        out['static']['person@70 curve R600'] = r
        print(f'static person @70 in R600 curve rate={r["rate"]:.2f} err={r["err"]}', flush=True)
    if 'approach' in args.only:
        out['approach'] = {}
        n = 60 if args.quick else 160
        for kind, start in (('person', 320.0), ('box', 260.0)):
            r = approach_case(kind, start, 15.0, factory, n)
            out['approach'][kind] = r
            print(f'approach {kind} from {start} m @15 m/s: {r}', flush=True)
    if 'empty' in args.only:
        out['empty'] = {}
        n = 60 if args.quick else 150
        for name, kw in (('straight', {}),
                         ('curve_left_R300', dict(curve_start=30.0, curve_radius=300.0)),
                         ('curve_right_R600', dict(curve_start=0.0, curve_radius=-600.0)),
                         ('walkway', dict(walkway=True)),
                         ('no_third_rail_boxes30', dict(third_rail=False, wall_boxes_every=30.0))):
            r = empty_case(kw, factory, n)
            out['empty'][name] = r
            print(f'empty {name:22s} FA={r["false_alarm_frames"]}/{r["frames"]} '
                  f'{r["fa_distances"]} ms={r["ms_median"]:.0f}/{r["ms_p95"]:.0f}', flush=True)
        r = empty_case({}, factory, n, lidar=LidarModel(spurious_points=300, range_noise=.04))
        out['empty']['heavy_dust'] = r
        print(f'empty heavy_dust FA={r["false_alarm_frames"]}/{r["frames"]} {r["fa_distances"]}',
              flush=True)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=1, default=float)


if __name__ == '__main__':
    main()
