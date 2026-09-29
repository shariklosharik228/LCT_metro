"""Unit and end-to-end tests (ray-cast tunnel simulator, no ROS needed)."""
import numpy as np
import pytest

from metro_obstacle_core.config import DetectorConfig, ros_parameter_defaults
from metro_obstacle_core.detector import GeometricDetector, dedup_clusters, downsample_adaptive
from metro_obstacle_core.ego_motion import EgoSpeedEstimator
from metro_obstacle_core.simulation import LidarModel, Obstacle, TunnelScene, TunnelSimulator


def detector():
    return GeometricDetector(lidar_xyz=(0., 0., 0.), lidar_rpy=(0., 0., 0.))


def run(scene, frames=8, speed=0.0, lidar=None, seed=1):
    sim, det = TunnelSimulator(scene, lidar, seed), detector()
    results = []
    for i in range(frames):
        cloud, truth = sim.scan(speed * i * 0.1, i * 0.1)
        results.append((det.process(cloud, 0.1, stamp=10.0 + 0.1 * i), truth))
    return det, results


def test_ros_parameters_round_trip():
    assert DetectorConfig().update(**ros_parameter_defaults()) == DetectorConfig()


def test_unknown_parameter_is_an_error():
    with pytest.raises(TypeError):
        DetectorConfig().update(no_such_parameter=1)


def test_downsample_keeps_far_points():
    rng = np.random.default_rng(0)
    near = rng.normal(scale=.02, size=(80, 3)) + [10., 0., .5]
    far = rng.normal(scale=.05, size=(20, 3)) + [160., 0., .8]
    assert np.any(downsample_adaptive(np.vstack((near, far)))[:, 0] > 150.)


def test_dedup_merges_overlapping_band_clusters():
    a = dict(center=np.array([40., 0., .8]), points=20)
    b = dict(center=np.array([40.2, .05, .82]), points=9)
    c = dict(center=np.array([50., 0., .8]), points=7)
    assert len(dedup_clusters([a, b, c])) == 2


def test_clearance_profile_is_narrow_near_rails():
    cfg = DetectorConfig()
    assert cfg.half_width(0.2) < cfg.half_width(1.0)


@pytest.mark.parametrize('distance', [40.0, 100.0])
def test_person_detected_with_correct_distance(distance):
    _, results = run(TunnelScene(obstacles=[Obstacle('person', distance)]))
    res = results[-1][0]
    assert res['detected']
    assert abs(res['distance'] - distance) < 1.0
    assert res['detection_range'][1] > distance


def test_corridor_reaches_far_beyond_rails():
    _, results = run(TunnelScene(), frames=4)
    res = results[-1][0]
    assert res['rail_range'][1] < 90.0          # rails fade at grazing incidence
    assert res['detection_range'][1] > 110.0    # walls carry the corridor further


@pytest.mark.parametrize('scene', [
    TunnelScene(), TunnelScene(walkway=True),
    TunnelScene(curve_start=30.0, curve_radius=300.0)])
def test_empty_tunnel_has_no_false_alarm(scene):
    _, results = run(scene, frames=12, speed=12.0)
    assert not any(r['detected'] for r, _ in results)


def test_obstacle_beside_clearance_is_ignored():
    _, results = run(TunnelScene(obstacles=[Obstacle('person', 50.0, v=2.0)]))
    assert not any(r['detected'] for r, _ in results)


def test_ego_speed_from_wall_protrusions():
    det, results = run(TunnelScene(), frames=6, speed=15.0)
    assert abs(results[-1][0]['ego_speed'] - 15.0) < 1.0


def test_brake_flag_when_close_at_speed():
    _, results = run(TunnelScene(obstacles=[Obstacle('person', 60.0)]), frames=10, speed=12.0)
    res = results[-1][0]
    assert res['detected'] and res['brake_required']


def test_permanent_side_structure_is_not_an_obstacle():
    # service walkway along the whole tunnel close to the envelope
    _, results = run(TunnelScene(walkway=True), frames=12, speed=12.0)
    assert not any(r['detected'] for r, _ in results)
