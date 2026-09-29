"""Multi-object tracking in track coordinates with range-dependent confirmation.

A candidate becomes an obstacle only after it is re-observed at a consistent
position in several frames.  Positions are predicted with the track's own
velocity, initialised to ``-ego_speed`` (a static object approaches the
train at the train speed), so a fast train does not break association.
Dust, multipath and one-frame artefacts do not survive this test.
"""
from collections import deque

import numpy as np


class ObstacleTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    def required(self, distance, category=None):
        c = self.cfg
        if category == 'low_object':
            return c.confirm_mid
        if distance < 40.0:
            return c.confirm_near
        if distance < 120.0:
            return c.confirm_mid
        return c.confirm_far

    def update(self, candidates, dt, ego_speed, observed=None):
        """``observed`` = (start, end) of the corridor measured in this frame.

        A confirmed track whose predicted position lies outside the measured
        corridor is coasted (not penalised) for a few frames: the object did
        not disappear, the sensor coverage shrank.
        """
        self.frame += 1
        for tr in self.tracks:
            tr['predicted'] = tr['center'] + tr['velocity'] * dt
            outside = observed is not None and not (
                observed[0] + 0.5 < tr['predicted'][0] < observed[1] - 0.5)
            if outside and tr.get('was_confirmed') and tr.get('coast', 0) < 5:
                tr['coast'] = tr.get('coast', 0) + 1
                tr['center'] = tr['predicted']
                tr['last'] = self.frame
                tr['coasting'] = True
            else:
                tr['hits'].append(0)
                tr['coasting'] = False
        used = set()
        for c in sorted(candidates, key=lambda item: -item['points']):
            d = float(c['center'][0])
            gate = np.array([max(1.0, 0.03 * d, 0.6 * ego_speed * dt + 0.5), 0.6, 0.8])
            best, cost = None, np.inf
            for tr in self.tracks:
                if tr['id'] in used:
                    continue
                delta = np.abs(c['center'] - tr['predicted'])
                if (delta > gate).any():
                    continue
                score = float((delta / gate).sum())
                if score < cost:
                    best, cost = tr, score
            if best is None:
                window = self.required(d)[1]
                best = dict(id=self.next_id, hits=deque([1], maxlen=max(window, 6)),
                            velocity=np.array([-ego_speed, 0.0, 0.0]), age=0,
                            center=c['center'].copy(), last=self.frame, jitter=0.5)
                self.next_id += 1
                self.tracks.append(best)
            else:
                elapsed = max(dt * (self.frame - best['last']), 1e-3)
                measured = (c['center'] - best['center']) / elapsed
                measured[0] = np.clip(measured[0], -self.cfg.max_speed - 5, 5.0)
                best['velocity'] = 0.7 * best['velocity'] + 0.3 * measured
                best['hits'][-1] = 1
                # association residual: real objects re-appear where predicted,
                # clutter/dust matches only somewhere inside the gate
                best['jitter'] = 0.6 * best['jitter'] + 0.4 * float(
                    (np.abs(c['center'] - best['predicted']) / gate).max())
            best.update({k: v for k, v in c.items() if k != 'velocity'})
            best['last'] = self.frame
            best['coast'] = 0
            best['coasting'] = False
            best['age'] += 1
            used.add(best['id'])
        self.tracks = [t for t in self.tracks
                       if self.frame - t['last'] <= self.cfg.max_track_misses]
        confirmed = []
        for tr in self.tracks:
            need, window = self.required(float(tr['center'][0]), tr.get('category'))
            hits = int(sum(list(tr['hits'])[-window:]))
            tr['confidence'] = float(min(1.0, hits / max(need, 1)) *
                                     (0.6 + 0.4 * min(1.0, tr['points'] / 12.0)))
            steady = tr['jitter'] < (0.6 if tr['center'][0] < 60.0 else 0.45)
            if tr['last'] == self.frame and hits >= need and steady:
                tr['was_confirmed'] = True
                confirmed.append(tr)
        return confirmed
