"""Detector configuration: one place for every tunable parameter.

The ROS node declares a parameter for every field below with the same name, so
``config/detector.yaml`` really controls the algorithm.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple


@dataclass
class DetectorConfig:
    # Sensor mounting: LiDAR frame -> train base frame (x forward, y left, z up).
    lidar_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lidar_rpy: Tuple[float, float, float] = (0.0, 0.0, 1.5707963267948966)

    # Search range along the track (metres).
    min_distance: float = 3.0
    max_distance: float = 300.0

    # Clearance (kinematic envelope) around the track centreline.  Two-level
    # profile: a narrow low zone around the rails (keeps the contact rail and
    # its cover outside) and the full car-body zone above it.
    clearance_width: float = 2.5          # full width of the body zone
    clearance_height: float = 3.5         # above the trackbed
    clearance_low_width: float = 2.0      # width below clearance_low_height
    clearance_low_height: float = 0.35
    structure_margin: float = 0.15        # keep this far from permanent tunnel structures
    far_width_margin: float = 0.20        # narrower where the centreline is less certain
    precise_range: float = 45.0           # ... i.e. beyond this distance or beyond the rails

    # Track geometry.
    rail_gauge: float = 1.52
    rail_fit_distance: float = 80.0       # where rails are searched for

    # Far corridor from tunnel walls (beyond the visible rails).
    use_wall_corridor: bool = True
    wall_band: Tuple[float, float] = (0.9, 2.4)   # height of wall points used
    wall_min_lateral: float = 1.45                 # ignore anything nearer than this
    wall_max_lateral: float = 4.5
    corridor_extrapolation: float = 0.0   # metres past the last wall measurement (0 = off)

    # Vertical filters.
    floor_margin: float = 0.10            # points below are trackbed
    far_floor_margin: float = 0.22        # beyond the rails (vertical datum less certain)
    far_min_top: float = 0.80             # beyond the rails only objects reaching this height
    min_obstacle_height: float = 0.10     # vertical extent of a low obstacle
    far_stable_range: float = 120.0       # beyond: corridor end must stay stable over 6 frames
    low_object_range: float = 35.0        # low objects (<0.5 m) only where the floor is measured
    min_obstacle_top: float = 0.30        # its top above the trackbed

    # Clustering: (start, end, eps) bands; eps grows with beam spacing.
    cluster_bands: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 35.0, 0.25), (30.0, 70.0, 0.40), (65.0, 130.0, 0.65),
        (120.0, 210.0, 0.95), (200.0, 400.0, 1.30))
    max_cluster_length: float = 3.0       # longer = wall / cable structure
    accumulate_frames: int = 3
    near_thinning: bool = True            # speed: thin the oversampled near field            # ego-motion compensated accumulation

    # Temporal confirmation.
    confirm_near: Tuple[int, int] = (2, 3)      # hits of window, d < 40 m
    confirm_mid: Tuple[int, int] = (3, 5)       # 40..120 m
    confirm_far: Tuple[int, int] = (4, 6)       # > 120 m
    max_track_misses: int = 3

    # Ego motion.
    ego_speed: float = -1.0               # >= 0 forces a known speed (m/s)
    max_speed: float = 30.0
    decel: float = 1.1                    # service braking for TTC / stop flag (m/s^2)
    reaction_time: float = 1.0

    def update(self, **kwargs):
        names = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key not in names:
                raise TypeError(f'Unknown detector parameter: {key}')
            current = getattr(self, key)
            if isinstance(current, tuple):
                if current and isinstance(current[0], tuple):
                    flat = list(value)
                    if flat and not isinstance(flat[0], (list, tuple)):
                        width = len(current[0])
                        flat = [tuple(flat[i:i + width]) for i in range(0, len(flat), width)]
                    value = tuple(tuple(float(v) for v in item) for item in flat)
                else:
                    value = tuple(type(current[0])(v) for v in value) if current else tuple(value)
            elif isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(self, key, value)
        return self

    def as_dict(self):
        return asdict(self)

    def half_width(self, height):
        """Half width of the clearance profile at a given height (vectorised)."""
        import numpy as np
        h = np.asarray(height)
        return np.where(h < self.clearance_low_height,
                        0.5 * self.clearance_low_width, 0.5 * self.clearance_width)


def ros_parameter_defaults():
    """Flat defaults suitable for ``Node.declare_parameter``."""
    out = {}
    for f in fields(DetectorConfig):
        value = getattr(DetectorConfig(), f.name)
        if isinstance(value, tuple):
            if value and isinstance(value[0], tuple):
                value = [float(v) for item in value for v in item]
            elif value and isinstance(value[0], int) and not isinstance(value[0], bool):
                value = [int(v) for v in value]
            else:
                value = [float(v) for v in value]
        out[f.name] = value
    return out
