from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker, MarkerArray
from metro_obstacle_core import detector as obstacle_algorithm
from metro_obstacle_core.config import DetectorConfig, ros_parameter_defaults


_FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def corridor_stations(start: float, end: float) -> np.ndarray:
    """Anchor cross-sections to distance, not to the fluctuating visible range."""
    interior = np.arange(np.ceil(start / 5.) * 5., end, 5.)
    return np.unique(np.r_[start, interior, end])


class DisplayHorizon:
    """Grow recovered geometry gently; never draw beyond current evidence.

    Only affects the green wireframe, not the detector's measurement range.
    Use measurement time so playback speed does not change this behaviour.
    """
    def __init__(self):
        self.end = None
        self.stamp = None

    def update(self, start, end, stamp):
        if self.stamp is not None and 0 < stamp - self.stamp <= .7:
            end = min(end, self.end + 25. * (stamp - self.stamp))
        self.end, self.stamp = max(start, end), stamp
        return self.end


def latest_sensor_qos() -> QoSProfile:
    """Keep only the newest cloud when processing is slower than the sensor."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def pointcloud_xyz(message: PointCloud2) -> np.ndarray:
    """PointCloud2 -> (N, 3) float32 array, any field layout / endianness."""
    fields = {field.name: field for field in message.fields}
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PointCloud2 must contain x, y and z fields")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + _FIELD_TYPES[fields[k].datatype] for k in ("x", "y", "z")],
        "offsets": [fields[k].offset for k in ("x", "y", "z")],
        "itemsize": message.point_step,
    })
    cloud = np.ndarray((message.height, message.width), buffer=message.data, dtype=dtype,
                       strides=(message.row_step, message.point_step)).reshape(-1)
    return np.column_stack((cloud["x"], cloud["y"], cloud["z"])).astype(np.float32, copy=False)


class DeveloperDetectorNode(Node):
    """Runs the detector on every incoming cloud and publishes the results."""
    def __init__(self):
        super().__init__("developer_obstacle_detector")
        self.declare_parameter("input_topic", "/lidar_points")
        defaults = ros_parameter_defaults()
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        # Every DetectorConfig field is a ROS parameter with the same name,
        # so config/detector.yaml really controls the algorithm.
        config = DetectorConfig().update(**{
            name: self.get_parameter(name).value for name in defaults})
        self.detector = obstacle_algorithm.GeometricDetector(config)
        self.last_stamp = None
        self.display_horizon = DisplayHorizon()
        self.frame_count = 0
        topic = str(self.get_parameter("input_topic").value)
        self.subscription = self.create_subscription(PointCloud2, topic, self._on_cloud,
                                                     latest_sensor_qos())
        self.processed_cloud_pub = self.create_publisher(
            PointCloud2, "/developer_obstacle/processed_cloud", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/developer_obstacle/markers", 10)
        self.detected_pub = self.create_publisher(Bool, "/developer_obstacle/detected", 10)
        self.distance_pub = self.create_publisher(Float32, "/developer_obstacle/distance", 10)
        self.confidence_pub = self.create_publisher(Float32, "/developer_obstacle/confidence", 10)
        self.range_pub = self.create_publisher(Float32, "/developer_obstacle/detection_range", 10)
        self.brake_pub = self.create_publisher(Bool, "/developer_obstacle/brake_required", 10)
        self.processing_pub = self.create_publisher(Float32, "/developer_obstacle/processing_ms", 10)
        self.status_pub = self.create_publisher(String, "/developer_obstacle/status", 10)
        self.get_logger().info(
            f"Obstacle detector ready on {topic}; max_distance={config.max_distance} m, "
            f"clearance {config.clearance_width}x{config.clearance_height} m")

    def _detector_value(self, legacy_name: str, config_name: str) -> float:
        return float(getattr(self.detector, legacy_name))

    def _on_cloud(self, message: PointCloud2):
        started = time.perf_counter()
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and stamp < self.last_stamp - 0.5:
            if hasattr(self.detector, "reset"):
                self.detector.reset()
            self.get_logger().info("Bag timestamp restarted; track model reset")
        dt = 0.1 if self.last_stamp is None else min(1.0, max(0.001, stamp - self.last_stamp))
        self.last_stamp = stamp
        try:
            result = self.detector.process(pointcloud_xyz(message), dt, stamp=stamp)
        except Exception as error:
            self.get_logger().error(f"Developer algorithm failed: {error}")
            self.status_pub.publish(String(data=json.dumps({'state': 'insufficient_data', 'reason': str(error)})))
            self.detected_pub.publish(Bool(data=False))
            self.distance_pub.publish(Float32(data=float('nan')))
            self.detector.last_track = None
            self.processed_cloud_pub.publish(message)
            self._publish_markers(message, {})
            return

        detected = bool(result.get("detected", False))
        distance = float(result.get("distance", float("nan")))
        self.detected_pub.publish(Bool(data=detected))
        self.distance_pub.publish(Float32(data=distance))
        # Publish the exact cloud used by the detector immediately before its
        # markers. RViz then renders one coherent measurement instead of a
        # newer raw cloud underneath an older clearance corridor.
        self.confidence_pub.publish(Float32(data=float(result.get('confidence', 0.0))))
        self.brake_pub.publish(Bool(data=bool(result.get('brake_required', False))))
        if 'detection_range' in result:
            self.range_pub.publish(Float32(data=float(result['detection_range'][1])))
        self.processed_cloud_pub.publish(message)
        self._publish_markers(message, result)
        self.status_pub.publish(String(data=json.dumps({key: result[key] for key in
            ('state', 'reason', 'distance', 'confidence', 'category', 'ttc',
             'stopping_distance', 'brake_required', 'obstacles', 'observed_range',
             'rail_range', 'measured_range', 'detection_range', 'corridor_source',
             'accumulation', 'ego_speed', 'ego_speed_valid', 'timing_ms')
            if key in result}, default=float)))
        elapsed = (time.perf_counter() - started) * 1000.0
        self.processing_pub.publish(Float32(data=elapsed))
        self.frame_count += 1
        if self.frame_count % 10 == 0 or (detected and self.frame_count < 5):
            state = f"obstacle at {distance:.1f} m" if detected else result.get("reason", "clear")
            self.get_logger().info(f"{state}; processing={elapsed:.1f} ms")

    def _track_to_lidar(self, points: np.ndarray) -> np.ndarray:
        base = self.detector.track_to_base(points)
        rotation = self.detector.T_l2b[:3, :3]
        translation = self.detector.T_l2b[:3, 3]
        return (base - translation) @ rotation

    def _publish_markers(self, message: PointCloud2, result: dict):
        array = MarkerArray()
        clear = Marker()
        clear.header = message.header
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        if 'state' in result:
            status = Marker()
            status.header = message.header
            status.ns, status.id = 'detector_status', 0
            status.type, status.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            status.pose.orientation.w = 1.
            status.pose.position.z = 2.
            status.scale.z = .35
            status.color.r, status.color.g, status.color.a = 1., .8, 1.
            status.text = (f"{result['state']} | {result.get('reason', '')} | range "
                           f"{result.get('detection_range', [0, 0])[1]:.0f} m | "
                           f"v={result.get('ego_speed', 0.0):.1f} m/s")
            array.markers.append(status)

        tm = self.detector.last_track
        if tm is not None and tm.valid:
            x0 = self._detector_value("min_d", "min_distance")
            x1 = self._detector_value("max_d", "max_distance")
            predicted_display = 'display_range' in result
            if predicted_display:
                x0, x1 = result['display_range']
            elif 'observed_range' in result:
                x0 = result['observed_range'][0]
                x1 = result['observed_range'][1]
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            if predicted_display:
                # Prediction is explicitly separated from the detector's
                # observed range; the user accepts display jumps here.
                self.display_horizon.end, self.display_horizon.stamp = x1, stamp
            else:
                x1 = self.display_horizon.update(x0, x1, stamp)
            width = self._detector_value("W", "clearance_width")
            height = self._detector_value("H", "clearance_height")
            y0, y1 = -width * 0.5, width * 0.5
            z0, z1 = getattr(self.detector, "floor_clearance", 0.02), height
            stations = corridor_stations(x0, x1)
            sections = [np.array([[x, y0, z0], [x, y0, z1],
                                  [x, y1, z0], [x, y1, z1]], dtype=float)
                        for x in stations]
            line_pairs = []
            for section in sections:
                for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
                    line_pairs.extend((section[a], section[b]))
            for previous, current in zip(sections[:-1], sections[1:]):
                for corner in range(4):
                    line_pairs.extend((previous[corner], current[corner]))
            lidar_lines = self._track_to_lidar(np.asarray(line_pairs))
            corridor = Marker()
            corridor.header = message.header
            corridor.ns = "clearance"
            corridor.id = 1
            corridor.type = Marker.LINE_LIST
            corridor.action = Marker.ADD
            corridor.pose.orientation.w = 1.0
            corridor.scale.x = 0.05
            corridor.color.r, corridor.color.g, corridor.color.b, corridor.color.a = 0.1, 1.0, 0.2, 0.8
            corridor.points = [Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                               for point in lidar_lines]
            array.markers.append(corridor)

            # Actual input returns used by the rail search (not reconstructed
            # geometry). Their position shares the processed cloud timestamp.
            for name, color in (('selected_left', (0.,.6,1.)),
                                ('selected_right', (1.,.2,1.))):
                base_points=getattr(tm,name,np.empty((0,3)))
                if not len(base_points):
                    continue
                marker=Marker()
                marker.header=message.header
                marker.ns,marker.id='rail_returns_'+name,0
                marker.type,marker.action=Marker.POINTS,Marker.ADD
                marker.pose.orientation.w=1.
                marker.scale.x=marker.scale.y=.09
                marker.color.r,marker.color.g,marker.color.b=color
                marker.color.a=1.
                rotation=self.detector.T_l2b[:3,:3]
                translation=self.detector.T_l2b[:3,3]
                lidar_points=(base_points-translation) @ rotation
                marker.points=[Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
                               for p in lidar_points]
                array.markers.append(marker)

            # Show independent measurement stages, not just the final corridor.
            # The blue/magenta lines are SELECTED rail hypotheses, not labels.
            observations = getattr(self.detector, 'last_track_observations', [])
            fresh = getattr(self.detector, 'last_fresh_track', None)
            if fresh is not None and len(observations) >= 4:
                values = np.asarray(observations)
                x = values[:, 0]
                dense_x = np.linspace(x[0], x[-1], 150)
                y = fresh.center_y(dense_x)
                slope = fresh.dy(x)
                norm = np.sqrt(1. + slope*slope)
                half = values[:, 3] / 2.
                traces = [
                    ('rail_hypothesis_left', (0., .6, 1.), np.column_stack(
                        (x+half*slope/norm, values[:,1]-half/norm, values[:,2]))),
                    ('rail_hypothesis_right', (1., .2, 1.), np.column_stack(
                        (x-half*slope/norm, values[:,1]+half/norm, values[:,2]))),
                    ('fresh_rail_center', (1., .8, 0.), np.column_stack(
                        (dense_x, y, fresh.z(dense_x)+.08))),
                    ('stabilized_center', (0., 1., .2), np.column_stack(
                        (dense_x, tm.center_y(dense_x), tm.z(dense_x)+.08))),
                ]
                rotation = self.detector.T_l2b[:3,:3]
                translation = self.detector.T_l2b[:3,3]
                for namespace, color, base_points in traces:
                    marker = Marker()
                    marker.header = message.header
                    marker.ns, marker.id = namespace, 0
                    marker.type, marker.action = Marker.LINE_STRIP, Marker.ADD
                    marker.pose.orientation.w = 1.
                    marker.scale.x = .035
                    marker.color.r, marker.color.g, marker.color.b = color
                    marker.color.a = 1.
                    lidar_points = (base_points-translation) @ rotation
                    marker.points = [Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
                                     for p in lidar_points]
                    array.markers.append(marker)

        if tm is not None and tm.valid:
            for index, candidate in enumerate(result.get('candidates', [])):
                lo, hi = candidate['minimum'], candidate['maximum']
                corners = np.array([[x,y,z] for x in (lo[0],hi[0])
                                    for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
                corners = self._track_to_lidar(corners)
                marker = Marker()
                marker.header = message.header
                marker.ns, marker.id = 'candidate_bounds', index
                marker.type, marker.action = Marker.LINE_LIST, Marker.ADD
                marker.pose.orientation.w = 1.
                marker.scale.x = .025
                color = (1., .7, 0.)
                if any(candidate is c for c in result.get('rejected', [])):
                    color = (.2, .4, 1.)
                elif any(np.allclose(candidate['center'], c['center']) for c in result.get('confirmed', [])):
                    color = (1., 0., 0.)
                marker.color.r, marker.color.g, marker.color.b = color
                marker.color.a = .85
                for i in range(8):
                    for bit in (1,2,4):
                        if i < (i ^ bit):
                            for corner in (corners[i], corners[i ^ bit]):
                                marker.points.append(Point(x=float(corner[0]),y=float(corner[1]),z=float(corner[2])))
                array.markers.append(marker)

        if result.get("detected", False) and tm is not None and tm.valid:
            track_point = np.array([[result["distance"], result["y"], result["z"]]], dtype=float)
            point = self._track_to_lidar(track_point)[0]
            obstacle = Marker()
            obstacle.header = message.header
            obstacle.ns = "obstacle"
            obstacle.id = 10
            obstacle.type = Marker.SPHERE
            obstacle.action = Marker.ADD
            obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.position.z = map(float, point)
            obstacle.pose.orientation.w = 1.0
            size = np.asarray(result.get("size", [0.8, 0.8, 0.8]), dtype=float)
            obstacle.scale.x = max(0.25, float(size[0]) + 0.15)
            obstacle.scale.y = max(0.25, float(size[1]) + 0.15)
            obstacle.scale.z = max(0.25, float(size[2]) + 0.15)
            obstacle.color.r, obstacle.color.g, obstacle.color.b, obstacle.color.a = 1.0, 0.05, 0.02, 0.9
            obstacle.lifetime = Duration(sec=0, nanosec=400_000_000)
            array.markers.append(obstacle)

            label = Marker()
            label.header = message.header
            label.ns = "label"
            label.id = 11
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x, label.pose.position.y = float(point[0]), float(point[1])
            label.pose.position.z = float(point[2] + 0.8)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 0.1, 1.0
            label.text = (f"OBSTACLE {result['distance']:.1f} m "
                          f"({result.get('category', '')}, {result.get('confidence', 0):.2f})")
            label.lifetime = Duration(sec=0, nanosec=400_000_000)
            array.markers.append(label)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = DeveloperDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
