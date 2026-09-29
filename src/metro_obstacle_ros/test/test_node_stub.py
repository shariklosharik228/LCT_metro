"""Runs DeveloperDetectorNode._on_cloud with stubbed ROS modules.

Skipped when real rclpy is installed (test_interface covers that case). Purpose:
catch NameError/typing bugs in the node glue on machines without ROS.
"""
import importlib.util
import json
import sys
import types

import numpy as np
import pytest

if importlib.util.find_spec('rclpy') is not None:
    pytest.skip('real rclpy present', allow_module_level=True)


class Auto:
    """Permissive message: any attribute is a nested Auto, kwargs become attributes."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        value = Auto()
        self.__dict__[name] = value
        return value


def _msg(name, **consts):
    cls = type(name, (Auto,), {'markers': None, **consts})
    return cls


class ArrayMsg(Auto):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.markers = []


class MarkerMsg(Auto):
    (ADD, DELETEALL, TEXT_VIEW_FACING, LINE_LIST, POINTS, LINE_STRIP, SPHERE) = 0, 3, 9, 5, 8, 4, 2

    def __init__(self, **kw):
        super().__init__(**kw)
        self.points = []


class FakeNode:
    def __init__(self, name):
        self.params, self.published = {}, {}

    def declare_parameter(self, name, value=None):
        self.params[name] = value

    def get_parameter(self, name):
        return Auto(value=self.params[name])

    def create_subscription(self, *a, **k):
        return None

    def create_publisher(self, typ, topic, depth):
        out = self.published.setdefault(topic, [])
        return Auto(publish=out.append)

    def get_logger(self):
        return Auto(info=lambda *a: None, error=lambda *a: None, warn=lambda *a: None)


@pytest.fixture(scope='module')
def node_module():
    names = ['rclpy', 'rclpy.node', 'rclpy.qos', 'sensor_msgs', 'sensor_msgs.msg',
             'std_msgs', 'std_msgs.msg', 'geometry_msgs', 'geometry_msgs.msg',
             'visualization_msgs', 'visualization_msgs.msg', 'builtin_interfaces',
             'builtin_interfaces.msg']
    saved = {n: sys.modules.get(n) for n in names}
    mods = {n: types.ModuleType(n) for n in names}
    mods['rclpy.node'].Node = FakeNode
    class Enum:
        def __getattr__(self, name):
            return name
    for cls in ('DurabilityPolicy', 'HistoryPolicy', 'ReliabilityPolicy'):
        setattr(mods['rclpy.qos'], cls, Enum())
    mods['rclpy.qos'].QoSProfile = Auto
    mods['sensor_msgs.msg'].PointCloud2 = _msg('PointCloud2')
    for cls in ('Bool', 'Float32', 'String'):
        setattr(mods['std_msgs.msg'], cls, _msg(cls))
    mods['geometry_msgs.msg'].Point = _msg('Point')
    mods['visualization_msgs.msg'].Marker = MarkerMsg
    mods['visualization_msgs.msg'].MarkerArray = ArrayMsg
    mods['builtin_interfaces.msg'].Duration = _msg('Duration')
    sys.modules.update(mods)
    sys.modules.pop('metro_obstacle_ros.detector_node', None)
    try:
        import metro_obstacle_ros.detector_node as module
        yield module
    finally:
        sys.modules.pop('metro_obstacle_ros.detector_node', None)
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def make_cloud(points, sec):
    data = np.zeros(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
    data['x'], data['y'], data['z'] = points.T
    fields = [Auto(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate('xyz')]
    fields.append(Auto(name='intensity', offset=12, datatype=7, count=1))
    msg = Auto(fields=fields, point_step=16, height=1, width=len(points),
               row_step=16 * len(points), is_bigendian=False, data=data.tobytes())
    msg.header.stamp.sec, msg.header.stamp.nanosec = sec, 0
    msg.header.frame_id = 'lidar'
    return msg


def test_node_processes_simulated_frames(node_module):
    from metro_obstacle_core.simulation import Obstacle, TunnelScene, TunnelSimulator
    orig = FakeNode.declare_parameter

    def declare(self, name, value=None):
        if name == 'lidar_rpy':
            value = [0.0, 0.0, 0.0]   # simulator frame == base frame
        orig(self, name, value)
    FakeNode.declare_parameter = declare
    try:
        node = node_module.DeveloperDetectorNode()
    finally:
        FakeNode.declare_parameter = orig
    sim = TunnelSimulator(TunnelScene(), None, 1)
    for i in range(8):
        cloud, _ = sim.scan(0.0, i * 0.1)
        node._on_cloud(make_cloud(np.asarray(cloud, dtype=np.float32)[:, :3], 10 + i))
    pub = node.published
    assert len(pub['/developer_obstacle/detected']) == 8
    assert len(pub['/developer_obstacle/processing_ms']) == 8
    status = json.loads(pub['/developer_obstacle/status'][-1].data)
    assert 'state' in status and status['state'] != 'insufficient_data'
    assert pub['/developer_obstacle/markers'][-1].markers
