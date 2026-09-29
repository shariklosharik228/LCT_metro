"""Ray-cast LiDAR simulator of a metro running tunnel.

Used for repeatable experiments and end-to-end tests when real rosbags are not
available.  Unlike simply sampling surfaces, every point is produced by a
beam of a Hesai-128-like scan pattern, so density falls with range, the
trackbed/rail heads fade at grazing incidence (as in the real recordings
where rails disappear at ~55 m) and obstacles occlude what is behind them.

World frame: x/y horizontal, z up, z = 0 on the trackbed.  The track
centreline is a polyline built from a curvature profile.  The train (and the
LiDAR) move along the centreline; obstacles are fixed in track coordinates
(s = arc length, v = lateral offset, z = height).

Output clouds are expressed in the LiDAR frame defined by ``lidar_rpy`` so that
they can be fed to the detector exactly like PointCloud2 data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Scan pattern
# --------------------------------------------------------------------------

def hesai128_elevations() -> np.ndarray:
    """Approximate Pandar128 channel layout: 0.125 deg in the dense band."""
    dense = np.arange(-6.0, 2.0, 0.125)                 # 64 channels
    low = np.linspace(-25.0, -6.0, 41)[:-1]             # 40 channels
    high = np.linspace(2.0, 15.0, 25)[1:]               # 24 channels
    return np.deg2rad(np.sort(np.r_[low, dense, high]))


@dataclass
class ScanPattern:
    elevations: np.ndarray = field(default_factory=hesai128_elevations)
    azimuth_step_deg: float = 0.1
    azimuth_half_fov_deg: float = 35.0

    def azimuths(self) -> np.ndarray:
        h = self.azimuth_half_fov_deg
        return np.deg2rad(np.arange(-h, h + 1e-9, self.azimuth_step_deg))


# --------------------------------------------------------------------------
# Geometry helpers (vectorised ray / primitive intersections)
# --------------------------------------------------------------------------

def _ray_box(o, d, lo, hi):
    """Slab test. Returns (t, normal_axis_sign) with t = inf on miss."""
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tmin = np.where(np.isnan(tmin), -np.inf, tmin)
    tmax = np.where(np.isnan(tmax), np.inf, tmax)
    t_enter = tmin.max(axis=1)
    t_exit = tmax.min(axis=1)
    axis = tmin.argmax(axis=1)
    hit = (t_exit >= t_enter) & (t_exit > 1e-6)
    t = np.where(hit, np.where(t_enter > 1e-6, t_enter, np.inf), np.inf)
    return t, axis


def _ray_cylinder_x(o, d, yc, zc, r, inside):
    """Infinite cylinder along local x.  inside=True -> exit hit."""
    oy, oz = o[:, 1] - yc, o[:, 2] - zc
    dy, dz = d[:, 1], d[:, 2]
    a = dy * dy + dz * dz
    b = 2.0 * (oy * dy + oz * dz)
    c = oy * oy + oz * oz - r * r
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_near = (-b - sq) / (2 * a)
        t_far = (-b + sq) / (2 * a)
    if inside:
        t = np.where(ok & (t_far > 1e-6), t_far, np.inf)
    else:
        t = np.where(ok & (t_near > 1e-6), t_near, np.inf)
    return t


# --------------------------------------------------------------------------
# Scene description
# --------------------------------------------------------------------------

@dataclass
class Obstacle:
    kind: str
    s: float                 # arc length along the track (world, fixed)
    v: float = 0.0           # lateral offset from centreline
    z: float = 0.0           # base height above trackbed
    size: Optional[Tuple[float, float, float]] = None
    velocity_v: float = 0.0  # lateral speed (m/s), e.g. person crossing

    PRESETS = {
        'person': (0.35, 0.50, 1.75),
        'child': (0.25, 0.35, 1.10),
        'box': (0.50, 0.50, 0.50),
        'low_box': (0.40, 0.40, 0.25),
        'trolley': (1.00, 1.20, 1.00),
        'pipe': (0.15, 1.80, 0.15),    # lying across the rails
        'barrel': (0.55, 0.55, 0.90),
    }

    def box(self, time_s: float = 0.0):
        sx, sy, sz = self.size or self.PRESETS[self.kind]
        v = self.v + self.velocity_v * time_s
        z0 = self.z + (0.17 if self.kind == 'pipe' else 0.0)
        return (np.array([-sx / 2, v - sy / 2, z0]),
                np.array([sx / 2, v + sy / 2, z0 + sz]))


@dataclass
class TunnelScene:
    radius: float = 2.75
    axis_height: float = 1.80
    gauge: float = 1.52
    rail_width: float = 0.07
    rail_height: float = 0.17
    curve_start: float = 1e9          # arc length where a curve begins
    curve_radius: float = 400.0       # signed: + left, - right
    length: float = 700.0
    third_rail: bool = True
    walkway: bool = False             # service walkway on the -v side
    wall_boxes_every: float = 70.0    # signal/cable boxes on the wall
    bracket_every: float = 3.0        # cable brackets on both walls
    segment: float = 4.0
    obstacles: List[Obstacle] = field(default_factory=list)

    def __post_init__(self):
        ds = 0.5
        s = np.arange(-50.0, self.length + ds, ds)
        k = np.where(s >= self.curve_start, 1.0 / self.curve_radius, 0.0)
        heading = np.concatenate(([0.0], np.cumsum(k[:-1] * ds)))
        x = np.concatenate(([0.0], np.cumsum(np.cos(heading[:-1]) * ds)))
        y = np.concatenate(([0.0], np.cumsum(np.sin(heading[:-1]) * ds)))
        x -= np.interp(0.0, s, x)
        y -= np.interp(0.0, s, y)
        self._s, self._x, self._y, self._h = s, x, y, heading
        self.seg_s = np.arange(-40.0, self.length - self.segment, self.segment)

    def pose(self, s):
        return (np.interp(s, self._s, self._x), np.interp(s, self._s, self._y),
                np.interp(s, self._s, self._h))

    def local_primitives(self, seg_s0: float):
        """Boxes and cylinders in the local frame of a straight segment."""
        L = self.segment + 0.02
        half = self.gauge / 2
        w = self.rail_width
        boxes = [
            (np.array([0, -half - w / 2, 0.0]), np.array([L, -half + w / 2, self.rail_height]), 'rail'),
            (np.array([0, half - w / 2, 0.0]), np.array([L, half + w / 2, self.rail_height]), 'rail'),
        ]
        if self.third_rail:
            boxes.append((np.array([0, 1.55, 0.10]), np.array([L, 1.68, 0.32]), 'third_rail'))
        if self.walkway:
            boxes.append((np.array([0, -2.35, 0.55]), np.array([L, -1.55, 0.65]), 'walkway'))
        if self.wall_boxes_every > 0:
            k = math.floor((seg_s0 + L) / self.wall_boxes_every)
            s_box = k * self.wall_boxes_every
            if seg_s0 <= s_box < seg_s0 + L and k > 0:
                side = 1.0 if k % 2 else -1.0
                u = s_box - seg_s0
                v0 = side * 1.95
                boxes.append((np.array([u, min(v0, v0 + side * .45), 1.2]),
                              np.array([u + .5, max(v0, v0 + side * .45), 1.8]), 'wall_box'))
        if self.bracket_every > 0:
            first = math.ceil(seg_s0 / self.bracket_every) * self.bracket_every
            for s_b in np.arange(first, seg_s0 + L, self.bracket_every):
                u = s_b - seg_s0
                for side in (-1.0, 1.0):
                    boxes.append((np.array([u, min(side * 2.12, side * 2.40), 1.0]),
                                  np.array([u + .06, max(side * 2.12, side * 2.40), 1.7]), 'bracket'))
        cylinders = [(2.30, 1.10, .045), (-2.30, 1.10, .045),
                     (2.18, 1.55, .05), (-2.18, 1.55, .05)]
        return boxes, cylinders


@dataclass
class LidarModel:
    height: float = 1.6              # above trackbed
    lateral: float = 0.0
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    range_noise: float = 0.02
    dropout: float = 0.03
    full_range: float = 130.0        # 100 % return probability up to here
    max_range: float = 320.0
    grazing_ref: float = 0.03        # cos(incidence) giving 100 % return
    spurious_points: int = 25        # dust / multipath returns per frame
    pattern: ScanPattern = field(default_factory=ScanPattern)


def _rot(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class TunnelSimulator:
    def __init__(self, scene: TunnelScene, lidar: Optional[LidarModel] = None, seed: int = 0):
        self.scene = scene
        self.lidar = lidar or LidarModel()
        self.rng = np.random.default_rng(seed)
        pat = self.lidar.pattern
        self.el = pat.elevations
        self.az = pat.azimuths()
        E, A = np.meshgrid(self.el, self.az, indexing='ij')
        # Directions in the "base" frame of the train: x forward, y left.
        self.dirs_base = np.stack((np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                                   np.sin(E)), axis=-1)       # (nel, naz, 3)

    # ------------------------------------------------------------------
    def scan(self, s_train: float, time_s: float = 0.0):
        """Return (points_in_lidar_frame, truth) for a train at arc length s."""
        sc, li = self.scene, self.lidar
        x0, y0, h0 = sc.pose(s_train)
        ch, sh = math.cos(h0), math.sin(h0)
        # lidar origin in world
        ox = x0 - sh * li.lateral
        oy = y0 + ch * li.lateral
        origin = np.array([ox, oy, li.height])
        Rw = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])
        dirs_w = self.dirs_base @ Rw.T                          # (nel, naz, 3)
        nel, naz = dirs_w.shape[:2]
        best_t = np.full((nel, naz), np.inf)
        best_cos = np.zeros((nel, naz))

        def update(sel_e, sel_a, t, cosn):
            sub = best_t[sel_e, sel_a]
            better = t < sub
            if not better.any():
                return
            sub = np.where(better, t, sub)
            best_t[sel_e, sel_a] = sub
            cs = best_cos[sel_e, sel_a]
            best_cos[sel_e, sel_a] = np.where(better, cosn, cs)

        # Floor plane z = 0 (global): analytic, all rays.
        dz = dirs_w[..., 2]
        with np.errstate(divide='ignore'):
            tf = np.where(dz < -1e-6, -li.height / dz, np.inf)
        best_t = np.minimum(best_t, tf)
        best_cos = np.where(np.isfinite(tf), np.abs(dz), 0.0)

        el_sorted = self.el
        az_sorted = self.az

        def cull(corners_w):
            rel = (corners_w - origin) @ Rw           # back to train base frame
            dist = np.linalg.norm(rel[:, :2], axis=1)
            if dist.min() < 6.0 or (rel[:, 0] <= 0.5).any():
                if (rel[:, 0] <= 0.5).all():
                    return None
                return slice(0, nel), slice(0, naz)
            az = np.arctan2(rel[:, 1], rel[:, 0])
            el = np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1))
            a0, a1 = np.searchsorted(az_sorted, [az.min() - 1e-3, az.max() + 1e-3])
            e0, e1 = np.searchsorted(el_sorted, [el.min() - 1e-3, el.max() + 1e-3])
            if a1 <= a0 or e1 <= e0:
                return None
            return slice(e0, e1), slice(a0, a1)

        rel_s = sc.seg_s[(sc.seg_s > s_train - 12.0) & (sc.seg_s < s_train + li.max_range + 10)]
        for s0 in rel_s:
            px, py, ph = sc.pose(s0)
            c, s_ = math.cos(ph), math.sin(ph)
            Rl = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])     # local->world
            L = sc.segment
            corners_local = np.array([[u, v, z] for u in (0, L) for v in (-sc.radius, sc.radius)
                                      for z in (0.0, sc.axis_height + sc.radius)])
            corners_w = corners_local @ Rl.T + np.array([px, py, 0])
            sl = cull(corners_w)
            if sl is None:
                continue
            se, sa = sl
            d = dirs_w[se, sa].reshape(-1, 3) @ Rl        # world->local (Rl^T applied)
            o = (origin - np.array([px, py, 0])) @ Rl
            o = np.broadcast_to(o, d.shape)
            shape = best_t[se, sa].shape

            def in_segment(t):
                u = o[:, 0] + t * d[:, 0]
                return np.where((u >= -0.01) & (u <= L + 0.01), t, np.inf)

            # Tunnel wall (exit through cylinder), only above the floor.
            t = in_segment(_ray_cylinder_x(o, d, 0.0, sc.axis_height, sc.radius, inside=True))
            with np.errstate(invalid="ignore"):
                hz = o[:, 2] + t * d[:, 2]
            t = np.where(hz > 0.0, t, np.inf)
            hy = o[:, 1] + t * d[:, 1]
            nrm = np.stack((np.zeros_like(hy), hy, hz - sc.axis_height), 1) / sc.radius
            cosn = np.abs(np.einsum('ij,ij->i', nrm, d))
            update(se, sa, t.reshape(shape), cosn.reshape(shape))

            boxes, cylinders = sc.local_primitives(s0)
            for lo, hi, _ in boxes:
                t, axis = _ray_box(o, d, lo, hi)
                t = in_segment(t)
                cosn = np.abs(d[np.arange(len(d)), axis])
                update(se, sa, t.reshape(shape), cosn.reshape(shape))
            for yc, zc, r in cylinders:
                t = in_segment(_ray_cylinder_x(o, d, yc, zc, r, inside=False))
                update(se, sa, t.reshape(shape), np.full(shape, 0.8))

        truth = []
        for ob in sc.obstacles:
            px, py, ph = sc.pose(ob.s)
            c, s_ = math.cos(ph), math.sin(ph)
            Rl = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            lo, hi = ob.box(time_s)
            corners_local = np.array([[u, v, z] for u in (lo[0], hi[0]) for v in (lo[1], hi[1])
                                      for z in (lo[2], hi[2])])
            corners_w = corners_local @ Rl.T + np.array([px, py, 0])
            rel = (corners_w - origin) @ Rw
            truth.append(dict(kind=ob.kind, distance=float(ob.s - s_train),
                              along=float(rel[:, 0].min()),
                              lateral=float(rel[:, 1].mean())))
            sl = cull(corners_w)
            if sl is None:
                continue
            se, sa = sl
            d = dirs_w[se, sa].reshape(-1, 3) @ Rl
            o = np.broadcast_to((origin - np.array([px, py, 0])) @ Rl, d.shape)
            shape = best_t[se, sa].shape
            t, axis = _ray_box(o, d, lo, hi)
            cosn = np.abs(d[np.arange(len(d)), axis])
            update(se, sa, t.reshape(shape), np.maximum(cosn, .3).reshape(shape))

        # Return probability: range, incidence, dropout.
        t = best_t
        valid = np.isfinite(t) & (t > 1.0) & (t < li.max_range)
        p_range = np.clip(1.0 - (t - li.full_range) / (li.max_range - li.full_range) * 0.85, 0.15, 1.0)
        p_inc = np.clip(best_cos / li.grazing_ref, 0.0, 1.0)
        p = p_range * p_inc * (1.0 - li.dropout)
        keep = valid & (self.rng.random(t.shape) < p)
        r = t[keep] + self.rng.normal(scale=li.range_noise, size=keep.sum())
        pts_base = self.dirs_base[keep] * r[:, None]
        pts_base[:, 2] += 0.0
        # spurious returns (dust) inside the tunnel volume
        n_sp = self.rng.poisson(li.spurious_points)
        if n_sp:
            sp = np.column_stack((self.rng.uniform(3, 120, n_sp),
                                  self.rng.uniform(-2.0, 2.0, n_sp),
                                  self.rng.uniform(-li.height + .2, 2.5, n_sp)))
            pts_base = np.vstack((pts_base, sp))
        R = _rot(li.rpy)
        # base = R @ lidar  ->  lidar = R^T base
        pts_lidar = pts_base @ R
        return pts_lidar.astype(np.float32), truth


def run_sequence(scene: TunnelScene, frames: int, speed: float, start_s: float = 0.0,
                 lidar: Optional[LidarModel] = None, seed: int = 0, dt: float = 0.1):
    sim = TunnelSimulator(scene, lidar, seed)
    for i in range(frames):
        t = i * dt
        s = start_s + speed * t
        cloud, truth = sim.scan(s, t)
        yield t, cloud, truth
