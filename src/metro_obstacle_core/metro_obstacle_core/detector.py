"""Obstacle detector for a metro tunnel: "describe the normal tunnel, flag the rest".

Pipeline (one LiDAR frame):

1.  Preprocess: drop invalid returns, LiDAR -> train frame, crop forward.
2.  Track geometry near: trackbed plane + raised rail heads -> centreline
    (``rail_geometry.fit_rails`` + ``rail_heads.fit_heads``), stabilised over
    time by registering the previous path to the fresh measurements.
3.  Track geometry far: the corridor is continued along the tunnel walls,
    calibrated against the rails (``wall_corridor``).  This is what extends the
    detection range from ~55 m (end of visible rails) to 200-300 m.
4.  Every point is expressed in track coordinates (distance along the path,
    lateral offset, height above trackbed).  Points inside the clearance
    profile, above the trackbed and not on the rail heads are candidates.
5.  Up to ``accumulate_frames`` frames are accumulated with ego-motion
    compensation along the path (important for sparse far targets).
6.  Range-adaptive clustering, shape checks (flat/long/wall-like surfaces
    rejected), range-aware point support.
7.  Temporal confirmation in a multi-object tracker (hits-of-window grows
    with range); the nearest confirmed object is the reported obstacle.

Nothing is learned from labelled obstacles, so the method does not depend on
the appearance of specific objects and generalises to unseen recordings.
"""
import math
import time
from collections import deque

import numpy as np
from sklearn.cluster import DBSCAN

from . import rail_geometry as _curve
from . import rail_heads as _heads
from . import wall_corridor as _walls
from .config import DetectorConfig
from .ego_motion import EgoSpeedEstimator
from .tracking import ObstacleTracker


def euler_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def stabilize(model, observations, previous):
    """Blend fresh rail centreline with the previous one (after registration)."""
    if model is None or previous is None:
        return model
    values = np.asarray(observations, float)
    overlap = ((values[:, 0] >= previous.x_nodes[0]) &
               (values[:, 0] <= previous.x_nodes[-1]))
    if np.count_nonzero(overlap) < 4:
        return model
    x = values[overlap, 0]
    old = previous.center_y(x)
    weights = np.sqrt(np.clip(values[overlap, 4], 2., 80.))
    target = values[overlap, 1] - old
    robust = np.ones(len(x))
    for _ in range(4):
        correction = np.polyfit(x, target, 1, w=weights * np.sqrt(robust))
        residual = target - np.polyval(correction, x)
        robust = np.minimum(1., .1 / np.maximum(abs(residual), 1e-6))
    rms = np.sqrt(np.average(residual ** 2, weights=weights ** 2 * robust))
    if rms >= .18:
        return model
    current = np.clip(values[overlap, 4] / (values[overlap, 4] + 20.), .1, .8)
    centers = model.path_center.copy()
    centers[overlap] = (current * model.center_y(x) +
                        (1 - current) * (old + np.polyval(correction, x)))
    result = _curve.PathModel(values, center_nodes=centers)
    result.selected_left = model.selected_left
    result.selected_right = model.selected_right
    return result


def downsample_adaptive(points):
    """Voxel grid whose cell grows with distance (beam spacing grows too)."""
    if len(points) == 0:
        return points
    parts = []
    for low, high, cell in ((0., 30., .08), (30., 60., .11), (60., 400., .16)):
        band = points[(points[:, 0] >= low) & (points[:, 0] < high)]
        if len(band) == 0:
            continue
        _, index = np.unique(np.floor(band / cell).astype(np.int32),
                             axis=0, return_index=True)
        parts.append(band[np.sort(index)])
    return np.vstack(parts) if parts else points


def dedup_clusters(clusters, gap=.65):
    kept = []
    for item in sorted(clusters, key=lambda c: -c['points']):
        if any(np.linalg.norm(item['center'] - other['center']) < gap for other in kept):
            continue
        kept.append(item)
    return kept


def beam_spacing(distance):
    """Vertical spacing of adjacent LiDAR rings at a distance (0.125 deg)."""
    return max(0.02, float(distance) * 0.00218)


class GeometricDetector:
    def __init__(self, config=None, **kwargs):
        self.cfg = DetectorConfig() if config is None else config
        # legacy keyword aliases accepted from older launch files
        kwargs.pop('semantic_model_path', None)
        self.cfg.update(**kwargs)
        c = self.cfg
        self.T_l2b = np.eye(4)
        self.T_l2b[:3, :3] = euler_to_matrix(*c.lidar_rpy)
        self.T_l2b[:3, 3] = np.asarray(c.lidar_xyz, float)
        self.tracker = ObstacleTracker(c)
        self.ego = EgoSpeedEstimator(c.max_speed)
        self.calib = _walls.WallCalibration()
        self.reset()

    # -- attributes used by the ROS visualisation ---------------------------
    @property
    def W(self):
        return self.cfg.clearance_width

    @property
    def H(self):
        return self.cfg.clearance_height

    @property
    def min_d(self):
        return self.cfg.min_distance

    @property
    def max_d(self):
        return self.cfg.max_distance

    @property
    def floor_clearance(self):
        return self.cfg.floor_margin

    def reset(self):
        self.ground_seed = None
        self.confirmed_track = None
        self.last_track = None
        self.last_fresh_track = None
        self.last_track_observations = []
        self.track_hold = 0
        self.previous_stamp = None
        self.smoothed_rail_end = None
        self.bed_profile = None
        self.free_space = np.full((2, 3), np.inf)
        self.rail_top = 0.17
        self.rail_gap = self.cfg.rail_gauge + 0.04
        self.buffer = deque(maxlen=max(1, self.cfg.accumulate_frames))
        self._end_history = deque(maxlen=6)
        self.tracker.reset()
        self.ego.reset()
        self.calib.reset()
        self.frame = 0
        self._dt = 0.1
        self._measured_end = np.inf
        self._end_history = deque(maxlen=6)
        self.timing = {}

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _fit_near(self, points):
        previous = self.confirmed_track
        seed, _ = _curve.fit_rails(points, self.cfg.rail_gauge, self.cfg.rail_fit_distance,
                                   previous=self.ground_seed, presorted=True)
        self.ground_seed = seed
        model, observations = None, []
        if seed is not None:
            model, observations = _heads.fit_heads(
                points, seed, _curve.PathModel, self.cfg.rail_gauge, self.cfg.rail_fit_distance,
                presorted=True)
        self.last_fresh_track = model
        if model is None:
            if previous is not None and self.track_hold < 2:
                self.track_hold += 1
                return previous, list(self.last_track_observations)
            self.confirmed_track = None
            return None, observations
        self.track_hold = 0
        model = stabilize(model, observations, previous)
        self.confirmed_track = model
        # rail head height / spacing from the actual selected returns
        heads = [p for p in (model.selected_left, model.selected_right) if len(p)]
        if heads:
            hp = np.vstack(heads)
            h = hp[:, 2] - model.z(np.clip(hp[:, 0], model.x_nodes[0], model.x_nodes[-1]))
            if len(h) >= 10:
                self.rail_top = 0.8 * self.rail_top + 0.2 * float(np.percentile(h, 90))
        gaps = np.asarray(observations, float)[:, 3] if observations else []
        if len(gaps) >= 3:
            self.rail_gap = 0.8 * self.rail_gap + 0.2 * float(np.median(gaps))
        return model, observations

    def _bootstrap_start(self, points):
        """No rails: start the wall walk from the walls around the train."""
        near = points[(points[:, 0] > 4.) & (points[:, 0] < 20.)]
        if len(near) < 200:
            return None
        floor_z = float(np.percentile(near[np.abs(near[:, 1]) < 1.2, 2], 10)) \
            if np.count_nonzero(np.abs(near[:, 1]) < 1.2) > 30 else None
        if floor_z is None:
            return None
        h = near[:, 2] - floor_z
        lw, _, rw, _ = _walls._nearest_wall(near[:, 1], h, self.cfg.wall_band,
                                            self.cfg.wall_min_lateral, self.cfg.wall_max_lateral)
        if lw is None or rw is None:
            return None
        if self.calib.left is None or self.calib.right is None:
            half = 0.5 * (lw + rw)
            self.calib.update(half, half, alpha=1.0)
        center = 0.5 * ((lw - self.calib.left) + (self.calib.right - rw))
        return (12.0, center, 0.0, 0.0, floor_z, 0.0)

    def _find_track(self, points):
        near, observations = self._fit_near(points)
        self.last_track_observations = observations
        far = []
        prior = None
        prev = self.last_track
        if prev is not None and getattr(prev, 'source', '') != 'rails':
            shift = self._speed_hint() * self._dt
            prev_end = float(np.interp(getattr(prev, 'measured_end', prev.observed_end),
                                       prev.grid_s, prev.grid_x))

            def prior(x, prev=prev, shift=shift, prev_end=prev_end):
                xp = x + shift
                if xp > prev_end or xp < prev.x_nodes[0]:
                    return None
                return float(prev.center_y(xp))
        if self.cfg.use_wall_corridor:
            if near is not None:
                _walls.calibrate(points, near, self.cfg, self.calib, presorted=True)
                far = _walls.extend(points, near, self.cfg, self.calib, self.cfg.max_distance,
                                    prior=prior, presorted=True)
            else:
                start = self._bootstrap_start(points)
                if start is not None:
                    far = _walls.extend(points, None, self.cfg, self.calib,
                                        self.cfg.max_distance, start=start, prior=prior,
                                        presorted=True)
        measured = ([] if near is None else
                    [(x, y, z, 1.52, 30.0) for x, y, z in zip(near.x_nodes, near.path_center,
                                                               near.z_nodes)]) + list(far)
        extra = []
        if (self.cfg.use_wall_corridor and self.cfg.corridor_extrapolation > 0
                and len(measured) >= 4):
            extra = _walls.extrapolate(points, measured, self.cfg, self.calib,
                                       self.cfg.max_distance,
                                       straight_len=self.cfg.corridor_extrapolation,
                                       curve_len=0.4 * self.cfg.corridor_extrapolation)
        far = list(far) + extra
        if near is None and len(far) < 4:
            self.last_track = None
            return None
        if near is None:
            model = _curve.PathModel(np.asarray(far), center_nodes=np.asarray(far)[:, 1],
                                     z_nodes=np.asarray(far)[:, 2])
            model.rail_end = float(model.x_nodes[0])
            model.selected_left = model.selected_right = np.empty((0, 3))
            model.source = 'walls_only'
        elif far:
            far_arr = np.asarray(far)
            obs = np.vstack((np.column_stack((near.x_nodes, near.path_center, near.z_nodes,
                                              np.full(len(near.x_nodes), 1.52),
                                              np.full(len(near.x_nodes), 30.))), far_arr))
            model = _curve.PathModel(obs, center_nodes=obs[:, 1], z_nodes=obs[:, 2])
            model.rail_end = float(near.observed_end)
            model.selected_left, model.selected_right = near.selected_left, near.selected_right
            model.source = 'rails+walls'
        else:
            model = near
            model.rail_end = float(near.observed_end)
            model.source = 'rails'
        model.predicted_end = float(model.observed_end)
        if extra:
            last_measured = measured[-1][0]
            model.measured_end = float(np.interp(last_measured, model.grid_x, model.grid_s))
        else:
            model.measured_end = float(model.observed_end)
        self.last_track = model
        return model

    def _parity(self, n):
        cached = getattr(self, '_parity_cache', None)
        if cached is None or len(cached) != n:
            cached = (np.arange(n) & 1) == 0
            self._parity_cache = cached
        return cached

    def _speed_hint(self):
        return self.cfg.ego_speed if self.cfg.ego_speed >= 0 else self.ego.speed

    def track_to_base(self, points):
        if self.last_track is None:
            return np.zeros((0, 3))
        return self.last_track.to_base(np.asarray(points, float))

    def _smooth_rail_end(self, raw_end):
        raw_end = float(raw_end)
        if self.smoothed_rail_end is None:
            self.smoothed_rail_end = raw_end
            return raw_end
        previous = self.smoothed_rail_end
        if raw_end >= previous:
            blended = .45 * previous + .55 * raw_end
        else:
            blended = .85 * max(raw_end, previous - 1.5) + .15 * raw_end
        self.smoothed_rail_end = min(blended, raw_end + 3.)
        return self.smoothed_rail_end

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------
    PROFILE_HALF = 1.45
    PROFILE_BIN = 0.1

    def _update_bed_profile(self, q, rail_end):
        """Cross-section of the normal trackbed: surface height per lateral bin.

        Real track is not a flat plane: cant (one rail higher), drainage
        channel, concrete shelves, rail heads.  For every 0.1 m lateral bin we
        take the surface height (80th percentile) in 1 m longitudinal slices and
        the median over slices, so an object occupying a few slices cannot
        enter the model; the result is smoothed over frames.
        """
        end = min(rail_end, 45.0)
        d, lat, h = q[:, 0], q[:, 1], q[:, 2]
        m = (d > 4.0) & (d < end) & (np.abs(lat) < self.PROFILE_HALF) & (h > -0.6) & (h < 0.7)
        nb = int(round(2 * self.PROFILE_HALF / self.PROFILE_BIN))
        ns = int(max(1, np.ceil(end - 4.0)))
        if m.sum() < 200:
            return
        b = np.clip(((lat[m] + self.PROFILE_HALF) / self.PROFILE_BIN).astype(int), 0, nb - 1)
        sl = np.clip((d[m] - 4.0).astype(int), 0, ns - 1)
        key = b * ns + sl
        order = np.lexsort((h[m], key))
        key_s, h_s = key[order], h[m][order]
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        counts = np.diff(np.r_[starts, len(key_s)])
        top = h_s[starts + (0.8 * (counts - 1)).astype(int)]
        grid = np.full(nb * ns, np.nan)
        grid[key_s[starts]] = top
        grid = grid.reshape(nb, ns)
        valid = np.isfinite(grid).sum(axis=1) >= max(3, ns // 5)
        with np.errstate(all='ignore'):
            prof = np.where(valid, np.nanmedian(np.where(np.isfinite(grid), grid, np.nan), axis=1),
                            np.nan)
        if self.bed_profile is None:
            self.bed_profile = prof
        else:
            old = self.bed_profile
            both = np.isfinite(old) & np.isfinite(prof)
            new = np.where(np.isfinite(old), old, prof)
            new[both] = 0.8 * old[both] + 0.2 * prof[both]
            self.bed_profile = new

    FREE_BANDS = (0.35, 1.0, 2.0, 3.6)

    def _update_free_space(self, q, rail_end):
        """Nearest *continuous* structure per side and height band.

        The train passes this tunnel every day, so permanent structures
        (walkways, cable trays, platform edges) bound the real envelope.  A
        structure counts if it is present in most 1 m slices; a person or a box
        occupies only a few slices and cannot shrink the envelope.
        """
        end = min(rail_end, 40.0)
        d, lat, h = q[:, 0], q[:, 1], q[:, 2]
        base = (d > 5.0) & (d < end) & (np.abs(lat) > 0.8) & (np.abs(lat) < 2.6)
        ns = int(max(1, end - 5.0))
        if base.sum() < 100 or ns < 10:
            return
        for bi in range(len(self.FREE_BANDS) - 1):
            hb = base & (h >= self.FREE_BANDS[bi]) & (h < self.FREE_BANDS[bi + 1])
            for side, sign in enumerate((1.0, -1.0)):
                m = hb & (sign * lat > 0)
                if not m.any():
                    value = np.inf
                else:
                    sl = np.clip((d[m] - 5.0).astype(int), 0, ns - 1)
                    mins = np.full(ns, np.inf)
                    np.minimum.at(mins, sl, np.abs(lat[m]))
                    value = float(np.median(mins))   # inf if most slices are empty
                old = self.free_space[side, bi]
                if np.isfinite(old) and np.isfinite(value):
                    value = 0.8 * old + 0.2 * value
                elif np.isfinite(old) and not np.isfinite(value):
                    value = old + 0.02            # relax slowly when it disappears
                self.free_space[side, bi] = value

    def _local_structure_limits(self, q, corridor_end):
        """Per 5 m slab: where the known permanent side structure is seen now.

        The corridor centreline has errors of a few decimetres towards the
        end of the rail range.  If the continuous structure learned near the
        train (walkway, platform edge) is visible in a slab, the envelope in
        that slab is bounded by *its* observed position.  A compact object
        (person, box) does not cover >=40 % of the slab length, so it cannot
        be mistaken for the structure.
        """
        self.slab_limits = {}
        d, lat, h = q[:, 0], q[:, 1], q[:, 2]
        for bi in range(len(self.FREE_BANDS) - 1):
            for side, sign in enumerate((1.0, -1.0)):
                ref = self.free_space[side, bi]
                if not np.isfinite(ref) or ref > 2.2:
                    continue
                m = ((h >= self.FREE_BANDS[bi]) & (h < self.FREE_BANDS[bi + 1]) &
                     (sign * lat > ref - 0.6) & (sign * lat < ref + 0.3) &
                     (d > 5.0) & (d < corridor_end))
                if m.sum() < 3:
                    continue
                dd, ll = d[m], np.abs(lat[m])
                slab = (dd // 5.0).astype(int)
                sub = ((dd % 5.0) // 0.5).astype(int)
                for k in np.unique(slab):
                    sel = slab == k
                    if len(np.unique(sub[sel])) >= 4:
                        self.slab_limits[(side, bi, int(k))] = float(np.percentile(ll[sel], 10))

    def effective_half(self, lat, h, far, d=None):
        """Half width of the search envelope for points (side/height aware)."""
        c = self.cfg
        half = c.half_width(h) - np.where(far, c.far_width_margin, 0.0)
        bi = np.clip(np.searchsorted(self.FREE_BANDS, h, side='right') - 1, 0, 2)
        side = (lat < 0).astype(int)
        free = self.free_space[side, bi] - c.structure_margin
        free = np.where(h < self.FREE_BANDS[0], np.inf, free)
        lim = getattr(self, 'slab_limits', None)
        if d is not None and lim:
            k = (np.asarray(d) // 5.0).astype(int)
            local = np.array([lim.get((int(s_), int(b_), int(k_)), np.inf)
                              for s_, b_, k_ in zip(np.broadcast_to(side, k.shape).ravel(),
                                                    np.broadcast_to(bi, k.shape).ravel(),
                                                    k.ravel())]).reshape(k.shape) \
                if k.size < 5000 else None
            if local is not None:
                free = np.minimum(free, local - self.cfg.structure_margin)
        return np.minimum(half, free)

    def _correct_far_floor(self, q, rail_end, corridor_end):
        """Beyond the rails the vertical datum is extrapolated; re-level it
        per 5 m slab from the floor returns themselves (stations, grade
        changes).  Obstacles occupy a small part of a slab's floor area."""
        d, lat, h = q[:, 0], q[:, 1], q[:, 2]
        cand = (d > rail_end) & (d < corridor_end + 5) & (np.abs(lat) < 1.2) & (h > -0.8) & (h < 0.9)
        if not cand.any():
            return
        edges = np.arange(rail_end, corridor_end + 5.0, 5.0)
        idx = np.digitize(d[cand], edges)
        offsets = np.zeros(len(edges) + 1)
        ok = np.zeros(len(edges) + 1, bool)
        hc = h[cand]
        for k in np.unique(idx):
            v = hc[idx == k]
            if len(v) >= 10:
                offsets[k] = float(np.percentile(v, 10))
                ok[k] = True
        if not ok.any():
            return
        ks = np.flatnonzero(ok)
        allk = np.arange(len(offsets))
        offsets = np.interp(allk, ks, offsets[ks])
        far = d > rail_end
        q[far, 2] -= np.clip(offsets[np.digitize(d[far], edges)], -0.8, 0.8)

    def _attached_to_structure(self, cl, far, reach=0.45):
        """True if the cluster continues outside the envelope (platform edge,
        wall, equipment) instead of being a compact object inside it."""
        q = getattr(self, '_frame_q', None)
        if q is None:
            return False
        lo, hi = cl['minimum'], cl['maximum']
        side = 1.0 if (lo[1] + hi[1]) > 0 else -1.0
        outer = max(abs(lo[1]), abs(hi[1]))
        mid_h = np.array([0.5 * (lo[2] + hi[2])])
        edge = float(self.effective_half(np.array([side]), mid_h, np.array([far]))[0])
        if edge - outer > reach:
            return False
        m = ((q[:, 0] > lo[0] - 1.0) & (q[:, 0] < hi[0] + 1.0) &
             (side * q[:, 1] > edge) & (side * q[:, 1] < edge + 1.5) &
             (q[:, 2] > lo[2] - 0.2) & (q[:, 2] < hi[2] + 0.3))
        return int(m.sum()) >= max(3, 0.2 * cl['points'])

    def bed_height(self, lat):
        """Trackbed surface height at a lateral offset (0 where unknown)."""
        if self.bed_profile is None:
            return np.zeros_like(lat)
        centers = -self.PROFILE_HALF + self.PROFILE_BIN * (np.arange(len(self.bed_profile)) + .5)
        ok = np.isfinite(self.bed_profile)
        if ok.sum() < 5:
            return np.zeros_like(lat)
        prof = np.interp(lat, centers[ok], np.maximum(self.bed_profile[ok], 0.0))
        return np.where(np.abs(lat) < self.PROFILE_HALF, prof, 0.0)

    def roi_mask(self, q, corridor_end, rail_end):
        c = self.cfg
        d, lat, h = q[:, 0], q[:, 1], q[:, 2]
        floor = np.where(d <= rail_end, c.floor_margin, c.far_floor_margin)
        half = self.effective_half(lat, h, d > min(rail_end, c.precise_range))
        m = ((d >= c.min_distance) & (d <= corridor_end) & (h > floor) &
             (h < c.clearance_height) & (np.abs(lat) <= half))
        # rail heads (and their fastenings) are part of the normal tunnel
        on_rail = (np.abs(np.abs(lat) - 0.5 * self.rail_gap) < 0.09) & (h < 0.12)
        return m & ~on_rail

    def evaluate(self, cl, rail_end):
        """Return (verdict, reason, category) for a cluster in track coords."""
        c = self.cfg
        size, lo, hi, n = cl['size'], cl['minimum'], cl['maximum'], cl['points']
        d = float(lo[0])
        far = d > rail_end + 0.5
        cl['beyond_rails'] = bool(far)
        spacing = beam_spacing(d)
        nearest = float(min(abs(lo[1]), abs(hi[1]))) if lo[1] * hi[1] > 0 else 0.0
        side_lat = lo[1] if abs(lo[1]) < abs(hi[1]) else hi[1]
        mid_h = np.array([0.5 * (lo[2] + hi[2])])
        narrow = far or d > c.precise_range
        edge = float(self.effective_half(np.array([side_lat if nearest > 0 else 1.0]), mid_h,
                                         np.array([narrow]), d=np.array([0.5 * (lo[0] + hi[0])]))[0])
        intrusion = edge - nearest
        cl['intrusion'] = float(intrusion)
        if size[0] > c.max_cluster_length:
            return 'rejected', 'long_structure', None
        if size[2] < 0.06 and hi[2] < 0.35:
            return 'rejected', 'flat_surface', None
        n_min = 5 if d < 60 else (3 if d < 150 else 2)
        if n < n_min:
            return 'rejected', 'sparse', None
        # confirmation must be driven by new measurements, not by re-counting
        # accumulated points of earlier frames
        if cl.get('fresh', n) < (2 if d < 150 else 1):
            return 'rejected', 'no_fresh_returns', None
        if far:
            if hi[2] < c.far_min_top or size[2] < max(0.3, min(0.6, 1.5 * spacing)):
                return 'uncertain', 'far_low_object', None
            if size[1] > 2.0 or size[0] > 2.0:
                return 'uncertain', 'far_wide_structure', None
            if lo[2] > 1.2:
                return 'uncertain', 'far_overhead_structure', None
            if intrusion < 0.30:
                return 'uncertain', 'far_boundary_object', None
            if self._attached_to_structure(cl, True):
                return 'uncertain', 'attached_to_structure', None
            # where the wall walk stops the line of sight usually meets a wall
            # (curve, junction); the last metres of the corridor are not trusted
            if d > getattr(self, '_corridor_end', np.inf) - 10.0:
                return 'uncertain', 'corridor_end', None
            # far coverage must be stable: where the measured corridor end keeps
            # jumping (junction, curve, dust) the wall model is not reliable
            if d > c.far_stable_range and d > min(self._end_history, default=np.inf) - 5.0:
                return 'uncertain', 'corridor_unstable', None
        else:
            if hi[2] < c.min_obstacle_top or size[2] < min(c.min_obstacle_height, spacing):
                return 'uncertain', 'low_object', None
            if hi[2] < 0.45 and (n < 8 or size[1] < 0.15):
                return 'uncertain', 'low_object_sparse', None
            if hi[2] < 0.5 and d > c.low_object_range:
                return 'uncertain', 'low_object_far', None
        if intrusion < 0.12:
            return 'uncertain', 'boundary_graze', None
        if not far and self._attached_to_structure(cl, False, reach=0.25):
            return 'uncertain', 'attached_to_structure', None
        if d > getattr(self, '_measured_end', np.inf):
            cl['extrapolated'] = True
            if intrusion < 0.35 or size[0] > 1.5:
                return 'uncertain', 'extrapolated_corridor_edge', None
        top = float(hi[2])
        if top < 0.45:
            category = 'low_object'
        elif size[1] > 0.9 or top > 2.3:
            category = 'large_object'
        elif top > 1.1 and size[1] < 0.9:
            category = 'person_like'
        else:
            category = 'object'
        return 'accepted', 'volume', category

    def _cluster(self, roi):
        found = []
        for low, high, eps in self.cfg.cluster_bands:
            band4 = roi[(roi[:, 0] >= low) & (roi[:, 0] < high)]
            if len(band4) < 2:
                continue
            band = band4[:, :3]
            min_samples = 2 if low >= 60. else 3
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(band)
            for label in np.unique(labels):
                if label < 0:
                    continue
                cp = band[labels == label]
                fresh = int(np.count_nonzero(band4[labels == label, 3] < 1e-6))
                lo, hi = np.percentile(cp, [2, 98], axis=0)
                center = (lo + hi) / 2
                nearest = float(np.percentile(cp[:, 0], 5)) if len(cp) >= 20 else float(cp[:, 0].min())
                found.append(dict(center=center, size=hi - lo, minimum=lo, maximum=hi,
                                  distance=nearest, y=float(center[1]),
                                  z=float(center[2]), points=int(len(cp)), fresh=fresh))
        return dedup_clusters(found)

    # ------------------------------------------------------------------
    def process(self, cloud_lidar, dt=0.1, stamp=None):
        t_start = time.perf_counter()
        c = self.cfg
        if stamp is None:
            stamp = (self.previous_stamp or 0.0) + dt
        if self.previous_stamp is not None:
            delta = stamp - self.previous_stamp
            if delta <= 0 or delta > .7:
                self.reset()
            else:
                dt = delta
        self.previous_stamp = stamp
        self._dt = dt
        self.frame += 1
        result = dict(detected=False, state='insufficient_data', reason='no_track',
                      candidates=[], rejected=[], unresolved=[], confirmed=[],
                      accumulation=False, frame=self.frame)
        pts = np.asarray(cloud_lidar, dtype=np.float32).reshape(-1, 3)
        rot = self.T_l2b[:3, :3].T.astype(np.float32)
        trn = self.T_l2b[:3, 3].astype(np.float32)
        base = pts @ rot + trn
        # NaN/inf rows fail every comparison below, so no separate isfinite pass
        bx = base[:, 0]
        keep = (bx >= 1.0) & (bx <= c.max_distance + 5.0) & (np.abs(base[:, 1]) <= 6.0 + 0.25 * bx)
        if float(trn[0]) >= 0.9:  # sensor far from the base origin: drop returns on the sensor itself
            keep &= np.einsum('ij,ij->i', pts, pts) > .01
        if c.near_thinning:
            # near field is oversampled (density ~1/r^2): drop every second near point
            # (by scan order); far points are all kept
            keep &= (bx >= 15.0) | (self._parity(len(base)))
        base = base[keep].astype(float)
        # the rail fits slice the cloud along x: sort once and share it
        base = base[np.argsort(base[:, 0])]
        t_pre = time.perf_counter()
        tm = self._find_track(base)
        t_geo = time.perf_counter()
        if tm is None:
            self.buffer.clear()
            self.tracker.update([], dt, self.ego.speed)
            self.timing = dict(pre=(t_pre - t_start) * 1e3, geometry=(t_geo - t_pre) * 1e3)
            result['timing_ms'] = self.timing
            return result
        corridor_end = min(c.max_distance, float(tm.observed_end))
        rail_end = self._smooth_rail_end(min(corridor_end, float(tm.rail_end)))
        self._measured_end = float(getattr(tm, 'measured_end', corridor_end))
        self._end_history.append(self._measured_end)
        self._corridor_end = corridor_end
        q = tm.to_track(base)
        # ego speed
        if c.ego_speed >= 0:
            speed = c.ego_speed
        else:
            speed = self.ego.update(q, dt)
        self._update_bed_profile(q, rail_end)
        self._update_free_space(q, rail_end)
        self._local_structure_limits(q, corridor_end)
        near = q[:, 0] <= rail_end + 2.0
        q[near, 2] -= self.bed_height(q[near, 1])
        self._frame_q = q
        roi = q[self.roi_mask(q, corridor_end, rail_end)]
        roi = downsample_adaptive(roi)
        self.buffer.append((stamp, roi))
        parts = []
        for s, pts_q in self.buffer:
            shifted = np.column_stack((pts_q, np.full(len(pts_q), stamp - s)))
            shifted[:, 0] -= speed * (stamp - s)
            parts.append(shifted)
        merged = np.vstack(parts) if parts else np.column_stack((roi, np.zeros(len(roi))))
        result['accumulation'] = len(self.buffer) > 1
        clusters = self._cluster(merged)
        t_clu = time.perf_counter()
        accepted, uncertain = [], []
        for cl in clusters:
            verdict, reason, category = self.evaluate(cl, rail_end)
            cl['reason'] = reason
            cl['category'] = category or 'unknown'
            result['candidates'].append(cl)
            if verdict == 'rejected':
                result['rejected'].append(cl)
            elif verdict == 'uncertain':
                uncertain.append(cl)
            else:
                accepted.append(cl)
        confirmed = self.tracker.update(accepted, dt, speed, (c.min_distance, corridor_end))
        result.update(confirmed=confirmed, unresolved=uncertain,
                      observed_range=[c.min_distance, rail_end],
                      rail_range=[c.min_distance, rail_end],
                      detection_range=[c.min_distance, corridor_end],
                      measured_range=[c.min_distance, self._measured_end],
                      display_range=[c.min_distance, corridor_end],
                      corridor_source=getattr(tm, 'source', 'rails'),
                      ego_speed=float(speed), ego_speed_valid=bool(self.ego.valid or c.ego_speed >= 0),
                      state='candidate' if accepted or uncertain else 'observed_clear',
                      reason='pending_confirmation' if accepted else
                      ('low_or_boundary_object' if uncertain else 'observed_clear'))
        if confirmed:
            best = min(confirmed, key=lambda t: t['distance'])
            for key in ('distance', 'y', 'z', 'size', 'minimum', 'maximum', 'category'):
                result[key] = best[key]
            result['confidence'] = float(best['confidence'])
            result['track_id'] = int(best['id'])
            closing = max(speed, 0.0)
            result['ttc'] = float(best['distance'] / closing) if closing > 0.5 else None
            stop = closing * c.reaction_time + closing * closing / (2 * c.decel)
            result['stopping_distance'] = float(stop)
            result['brake_required'] = bool(best['distance'] <= stop + 10.0)
            result.update(detected=True, state='obstacle',
                          reason='confirmed_%d_frames' % int(sum(best['hits'])))
        result['obstacles'] = [dict(id=int(t['id']), distance=float(t['distance']),
                                    lateral=float(t['y']), height=float(t['maximum'][2]),
                                    size=[float(v) for v in t['size']],
                                    category=t.get('category', 'unknown'),
                                    confidence=float(t['confidence']),
                                    beyond_rails=bool(t.get('beyond_rails', False)))
                               for t in sorted(confirmed, key=lambda t: t['distance'])]
        t_end = time.perf_counter()
        self.timing = dict(pre=(t_pre - t_start) * 1e3, geometry=(t_geo - t_pre) * 1e3,
                           cluster=(t_clu - t_geo) * 1e3, total=(t_end - t_start) * 1e3)
        result['timing_ms'] = self.timing
        return result
