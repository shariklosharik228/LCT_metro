"""Train speed from the motion of wall protrusions between consecutive frames.

No odometry is needed: brackets, boxes and joints fixed to the tunnel move
towards the train at the train speed; the smooth wall and the scan pattern do
not move and are excluded.  The shift is found by 1-D cross-correlation with
sub-bin refinement and a continuity prior against periodic structure.
"""
import numpy as np


class EgoSpeedEstimator:
    BIN = 0.1
    X0, X1 = 6.0, 46.0

    def __init__(self, max_speed=30.0):
        self.max_speed = max_speed
        self.prev = None
        self.speed = 0.0
        self.quality = 0.0
        self.valid = False

    def reset(self):
        self.prev = None
        self.speed = 0.0
        self.quality = 0.0
        self.valid = False

    def _profile(self, track_pts):
        """Histogram (along track) of wall *protrusions*.

        A smooth tunnel wall is translation invariant, and its returns sit at
        fixed ranges given by the scan pattern, so it carries no motion
        information.  Brackets, boxes, joints and niches stick out of the
        smooth wall; they are fixed to the tunnel and move with it.
        """
        d, lat, h = track_pts[:, 0], np.abs(track_pts[:, 1]), track_pts[:, 2]
        side = track_pts[:, 1] > 0
        m = (d > self.X0) & (d < self.X1) & (lat > 1.3) & (lat < 4.0) & (h > 0.3) & (h < 3.2)
        if m.sum() < 200:
            return None
        d, lat, h, side = d[m], lat[m], h[m], side[m]
        hb = np.clip(((h - 0.3) / 0.1).astype(int), 0, 28) + 29 * side
        med = np.zeros(58)
        for k in np.unique(hb):
            sel = hb == k
            med[k] = np.median(lat[sel]) if sel.sum() >= 5 else 0.0
        prot = (med[hb] > 0) & (lat < med[hb] - 0.05)
        n = int((self.X1 - self.X0) / self.BIN)
        hist = np.bincount(((d[prot] - self.X0) / self.BIN).astype(int).clip(0, n - 1),
                           minlength=n).astype(float)
        if hist.sum() < 10:
            return None
        hist = np.convolve(hist, [0.25, 0.5, 0.25], mode='same')
        hist -= hist.mean()
        return hist[None, :]

    def update(self, track_pts, dt):
        prof = self._profile(track_pts) if len(track_pts) else None
        if prof is None or dt <= 0:
            self.prev = prof
            return self.speed
        if self.prev is not None:
            max_shift = min(int(self.max_speed * dt / self.BIN) + 2, prof.shape[1] // 3)
            scores = []
            for s in range(0, max_shift + 1):
                a = self.prev[:, s:]
                b = prof[:, :prof.shape[1] - s]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                scores.append(float((a * b).sum() / (na * nb)) if na > 1e-6 and nb > 1e-6 else -1.0)
            scores = np.asarray(scores)
            k = int(np.argmax(scores))
            if self.valid:
                # periodic structure (brackets every few metres) -> prefer
                # the peak consistent with the current speed if nearly as good
                expected = self.speed * dt / self.BIN
                near = [i for i in range(1, len(scores) - 1)
                        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]
                        and scores[i] > 0.85 * scores[k]]
                if near:
                    k = min(near, key=lambda i: abs(i - expected))
            best = scores[k]
            if 0 < k < len(scores) - 1:
                y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
                den = y0 - 2 * y1 + y2
                frac = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
            else:
                frac = 0.0
            if best > 0.5:
                measured = float(np.clip((k + frac) * self.BIN / dt, 0.0, self.max_speed))
                alpha = 0.5 if not self.valid else 0.3
                self.speed = (1 - alpha) * self.speed + alpha * measured if self.valid else measured
                self.valid = True
                self.quality = best
        self.prev = prof
        return self.speed
