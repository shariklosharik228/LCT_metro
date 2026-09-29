import os
import signal
import subprocess

import rclpy
from rclpy.node import Node


class BagPlayerNode(Node):
    def __init__(self):
        super().__init__('metro_bag_player')
        self.declare_parameter('bag_path', '')
        self.declare_parameter('bag_topic', '/lidar_points')
        self.declare_parameter('output_topic', '/lidar_points')
        self.declare_parameter('loop', True)
        self.declare_parameter('rate', 1.0)
        self.declare_parameter('read_ahead_queue_size', 10)
        self.declare_parameter('start_paused', False)
        path = os.path.abspath(os.path.expanduser(str(self.get_parameter('bag_path').value)))
        if not self.get_parameter('bag_path').value or not os.path.isdir(path):
            raise ValueError(f'bag_path must be an existing rosbag directory: {path}')
        source = str(self.get_parameter('bag_topic').value)
        target = str(self.get_parameter('output_topic').value)
        command = ['ros2', 'bag', 'play', path, '--rate',
                   str(float(self.get_parameter('rate').value)),
                   '--read-ahead-queue-size',
                   str(int(self.get_parameter('read_ahead_queue_size').value)),
                   '--remap', f'{source}:={target}']
        if bool(self.get_parameter('loop').value):
            command.append('--loop')
        if bool(self.get_parameter('start_paused').value):
            command.append('--start-paused')
        self.process = subprocess.Popen(command, start_new_session=True)
        self.create_timer(.5, self._check_process)
        self.get_logger().info(f'Playing external bag {path}: {source} -> {target}')

    def _check_process(self):
        code = self.process.poll()
        if code is not None:
            self.get_logger().error(f'ros2 bag play exited with code {code}')
            rclpy.shutdown()

    def destroy_node(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BagPlayerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
