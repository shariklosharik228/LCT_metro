"""Continue the track corridor beyond the visible rails using the tunnel walls.

Why: at grazing incidence the trackbed and rail heads stop returning points
at roughly 50-70 m, while the walls stay visible for hundreds of metres.  The
track is laid at a (locally) constant offset from the walls, so:

1. In the section where rails are measured, calibrate the distance from the
   track centreline to the nearest wall structure on each side
   (``left_offset``, ``right_offset``).
2. Beyond the rails, march forward slab by slab.  In every slab measure the
   nearest wall on each side, convert each into a centreline estimate and
   accept the one(s) consistent with a smooth continuation of the path
   (constant-curvature prediction + innovation gate).  One visible wall is
   enough (platforms, niches, one-sided occlusion in curves).
3. The vertical datum is extrapolated with a bounded grade and refreshed from
   sparse trackbed returns whenever they exist.

The walk stops as soon as walls are no longer measured, so the corridor only
exists where there is evidence for it; it never extends into unseen space.
"""
import numpy as np

MAX_SLAB = 20.0      # longer slabs = too little evidence to steer the corridor


class WallCalibration:
    """Temporally smoothed wall offsets relative to the track centreline."""

    def __init__(self):
        self.left = None
        self.right = None

    def reset(self):
        self.left = self.right = None

    def update(self, left, right, alpha=0.25):
        if left is not None:
            self.left = left if self.left is None else (1 - alpha) * self.left + alpha * left
        if right is not None:
            self.right = right if self.right is None else (1 - alpha) * self.right + alpha * right


def _nearest_wall(lat, h, band, lo, hi, q=50.0):
    wall = (h > band[0]) & (h < band[1])
    left = lat[wall & (lat > lo) & (lat < hi)]
    right = -lat[wall & (lat < -lo) & (lat > -hi)]
    lw = float(np.percentile(left, q)) if len(left) >= 3 else None
    rw = float(np.percentile(right, q)) if len(right) >= 3 else None
    return lw, len(left), rw, len(right)


def _expected_walls(lat, h, band, calib, tol):
    """Nearest wall structure searched around where the wall is expected."""
    wall = (h > band[0]) & (h < band[1])
    out = []
    for sign, ref in ((1.0, calib.left), (-1.0, calib.right)):
        if ref is None:
            out += [None, 0]
            continue
        v = sign * lat[wall]
        v = v[(v > ref - tol - 0.4) & (v < ref + 1.5)]
        out += ([float(np.percentile(v, 50)), len(v)] if len(v) >= 3 else [None, len(v)])
    return out[0], out[1], out[2], out[3]


def calibrate(points, near, cfg, calib, presorted=False):
    """Measure wall offsets where the rail centreline is known."""
    if near is None:
        return
    if not presorted:
        points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    end = float(near.x_nodes[-1])
    lefts, rights = [], []
    for low in np.arange(max(6.0, float(near.x_nodes[0])), end - 2.0, 5.0):
        i, j = np.searchsorted(x, [low, low + 5.0])
        if j - i < 20:
            continue
        p = points[i:j]
        c = near.center_y(p[:, 0])
        slope = near.dy(np.clip(p[:, 0], near.x_nodes[0], near.x_nodes[-1]))
        lat = (p[:, 1] - c) / np.sqrt(1 + slope * slope)
        h = p[:, 2] - near.z(np.clip(p[:, 0], near.x_nodes[0], near.x_nodes[-1]))
        lw, _, rw, _ = _nearest_wall(lat, h, cfg.wall_band, cfg.wall_min_lateral,
                                     cfg.wall_max_lateral)
        if lw is not None:
            lefts.append(lw)
        if rw is not None:
            rights.append(rw)
    def robust(values):
        # a real wall is continuous: many sections, small spread
        if len(values) < 4:
            return None
        v = np.asarray(values)
        med = float(np.median(v))
        return med if float(np.median(np.abs(v - med))) < 0.15 else None
    left, right = robust(lefts), robust(rights)
    calib.update(left, right)
    if left is None and calib.left is not None and len(lefts) >= 4:
        calib.left = None if abs(np.median(lefts) - calib.left) > 0.3 else calib.left
    if right is None and calib.right is not None and len(rights) >= 4:
        calib.right = None if abs(np.median(rights) - calib.right) > 0.3 else calib.right


def extend(points, near, cfg, calib, max_distance, start=None, prior=None, presorted=False):
    """Return far observations (x, centre, z, gauge, support) past ``near``."""
    if calib.left is None and calib.right is None:
        return []
    pts = points if presorted else points[np.argsort(points[:, 0])]
    xs = np.ascontiguousarray(pts[:, 0])
    if near is not None:
        x_nodes, y_nodes = near.x_nodes, near.path_center
        last_x = float(x_nodes[-1])
        c = float(y_nodes[-1])
        n = min(6, len(x_nodes))
        t = x_nodes[-n:] - last_x
        deg = 2 if n >= 4 else 1
        coef = np.polyfit(t, y_nodes[-n:], deg)
        heading = float(np.clip(np.polyval(np.polyder(coef), 0.0), -.35, .35))
        curv = float(np.clip(coef[0], -.004, .004)) if deg == 2 else 0.0
        z = float(near.z_nodes[-1])
        grade = float(np.clip(near.rail_coeff[0], -.04, .04))
    else:
        if start is None:
            return []
        last_x, c, heading, curv, z, grade = start
    x0_start, z0_start = last_x, z
    history = []
    anchor = []
    if near is not None:
        anchor = [(float(xn), float(yn), 30.0) for xn, yn in
                  zip(near.x_nodes[-4:], near.path_center[-4:])]
    observations = []
    misses = 0
    x = last_x
    while True:
        # Wall returns per metre fall ~1/r^2, so far slabs grow until they hold
        # enough evidence (or reach 40 m).
        step = 5.0
        x_lo = x
        chosen = None
        while step <= MAX_SLAB:
            x_hi = x_lo + step
            if x_hi > max_distance + 0.5 * step:
                break
            x_mid = 0.5 * (x_lo + x_hi)
            dx = x_mid - last_x
            c_p = c + heading * dx + curv * dx * dx
            if prior is not None:
                c_prior = prior(x_mid)
                if c_prior is not None and abs(c_prior - c_p) < 0.5:
                    c_p = 0.5 * (c_p + c_prior)
            head_p = heading + 2 * curv * dx
            z_p = z + grade * dx
            i, j = np.searchsorted(xs, [x_lo, x_hi])
            local = pts[i:j]
            norm = np.sqrt(1 + head_p * head_p)
            lat = (local[:, 1] - c_p - head_p * (local[:, 0] - x_mid)) / norm
            h = local[:, 2] - z_p
            lw, nl, rw, nr = _expected_walls(lat, h, cfg.wall_band, calib, 0.3 + 0.006 * dx)
            chosen = (x_mid, dx, c_p, head_p, z_p, norm, lat, h, lw, nl, rw, nr)
            if nl + nr >= 8 or (min(nl, nr) >= 3):
                break
            step *= 2.0
        if chosen is None:
            break
        x_mid, dx, c_p, head_p, z_p, norm, lat, h, lw, nl, rw, nr = chosen
        x = x_lo + min(step, MAX_SLAB)
        cands = []
        if lw is not None and calib.left is not None:
            cands.append((lw - calib.left, nl))
        if rw is not None and calib.right is not None:
            cands.append((calib.right - rw, nr))
        gate = 0.30 + 0.006 * dx
        good = [cd for cd in cands if abs(cd[0]) < gate]
        if len(good) == 2 and abs(good[0][0] - good[1][0]) > 0.30:
            good = [min(good, key=lambda cd: abs(cd[0]))]
        if not good:
            misses += 1
            if misses >= 2 or x > max_distance:
                break
            continue
        misses = 0
        w = np.array([np.sqrt(n) for _, n in good])
        offset = float(np.dot([o for o, _ in good], w) / w.sum())
        support = float(sum(n for _, n in good))
        gain = support / (support + 6.0)
        if len(good) == 1 and support < 6:
            gain = min(gain, 0.3)
        c_new = c_p + gain * offset * norm
        # vertical datum from sparse trackbed returns, if any
        floor = (np.abs(lat) < 0.6) & (h > -0.3) & (h < 0.2)
        z_new = z_p
        if floor.sum() >= 4:
            z_new = z_p + 0.25 * float(np.percentile(h[floor], 30))
        # platforms / shelves must not lift the datum: stay near the grade line
        z_line = z0_start + grade * (x_mid - x0_start)
        z_new = float(np.clip(z_new, z_line - 0.3, z_line + 0.3))
        observations.append((x_mid, c_new, z_new, 1.52, min(support, 60.0)))
        history.append((x_mid, c_new, support))
        if support < 6:
            heading = head_p
        if support >= 6:
            recent = np.asarray(anchor + history[-6:])[-7:]
            tt = recent[:, 0] - x_mid
            deg = 2 if len(recent) >= 5 else 1
            wts = np.sqrt(np.clip(recent[:, 2], 1.0, 40.0))
            coef = np.polyfit(tt, recent[:, 1], deg, w=wts)
            new_heading = float(np.clip(np.polyval(np.polyder(coef), 0.0), -.5, .5))
            new_curv = float(np.clip(coef[0], -.004, .004)) if deg == 2 else curv
            # alignment curvature changes gradually (transition curves)
            curv = curv + float(np.clip(new_curv - curv, -3e-4, 3e-4))
            heading = new_heading
        c, z, last_x = c_new, z_new, x_mid
        if x >= max_distance:
            break
    return observations


def extrapolate(points, nodes, cfg, calib, max_distance, straight_len=80.0, curve_len=30.0):
    """Short constant-curvature continuation past the last measured node.

    Tunnel alignments are smooth (straights, circular curves, transitions),
    so a short continuation is predictable.  Its length depends on how
    straight the measured end is, and every slab that still has wall
    returns must agree with it, otherwise the continuation stops.
    Returns extra nodes (x, centre, z, gauge, support=1).
    """
    nodes = np.asarray(nodes, float)
    if len(nodes) < 4:
        return []
    last_x = float(nodes[-1, 0])
    sel = nodes[nodes[:, 0] >= last_x - 70.0]
    if len(sel) < 3:
        sel = nodes[-4:]
    t = sel[:, 0] - last_x
    w = np.sqrt(np.clip(sel[:, 4], 1.0, 40.0))
    coef = np.polyfit(t, sel[:, 1], 1, w=w)
    rms1 = float(np.sqrt(np.average((np.polyval(coef, t) - sel[:, 1]) ** 2, weights=w * w)))
    curv = 0.0
    if len(sel) >= 5:
        coef2 = np.polyfit(t, sel[:, 1], 2, w=w)
        rms2 = float(np.sqrt(np.average((np.polyval(coef2, t) - sel[:, 1]) ** 2, weights=w * w)))
        if rms2 < 0.6 * rms1 and abs(coef2[0]) < .004:
            coef, curv, rms1 = coef2, float(coef2[0]), rms2
    # only a well-determined end is continued
    if rms1 > 0.10 or np.count_nonzero(sel[:, 4] >= 6) < 3:
        return []
    length = straight_len if abs(curv) < 5e-4 else curve_len
    zc = np.polyfit(t, sel[:, 2], 1)
    zc[0] = np.clip(zc[0], -.04, .04)
    out = []
    pts = points[np.argsort(points[:, 0])]
    xs = np.ascontiguousarray(pts[:, 0])
    x = last_x
    while x + 10.0 <= min(max_distance, last_x + length) + 1e-6:
        x += 10.0
        tt = x - last_x
        c = float(np.polyval(coef, tt))
        z = float(np.polyval(zc, tt))
        head = float(np.polyval(np.polyder(coef), tt))
        i, j = np.searchsorted(xs, [x - 5.0, x + 5.0])
        local = pts[i:j]
        if len(local):
            norm = np.sqrt(1 + head * head)
            lat = (local[:, 1] - c - head * (local[:, 0] - x)) / norm
            h = local[:, 2] - z
            lw, nl, rw, nr = _nearest_wall(lat, h, cfg.wall_band, 0.9, cfg.wall_max_lateral, q=0.0)
            bad = ((lw is not None and calib.left is not None and lw < calib.left - 0.5) or
                   (rw is not None and calib.right is not None and rw < calib.right - 0.5))
            if bad:
                break
        out.append((x, c, z, 1.52, 1.0))
    return out
