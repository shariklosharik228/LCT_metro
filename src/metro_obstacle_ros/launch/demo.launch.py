import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('metro_obstacle_ros')
    detector = os.path.join(share, 'config', 'detector.yaml')
    rviz = os.path.join(share, 'rviz', 'obstacle_detector.rviz')
    arguments = [
        DeclareLaunchArgument('bag_path'),
        DeclareLaunchArgument('bag_topic', default_value='/lidar_points'),
        DeclareLaunchArgument('input_topic', default_value='/lidar_points'),
        DeclareLaunchArgument('fixed_frame', default_value='hesai_lidar'),
        DeclareLaunchArgument('loop', default_value='true'),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]
    player = Node(package='metro_obstacle_ros', executable='bag_player_node',
                  name='metro_bag_player', output='screen', parameters=[{
                      'bag_path': LaunchConfiguration('bag_path'),
                      'bag_topic': LaunchConfiguration('bag_topic'),
                      'output_topic': LaunchConfiguration('input_topic'),
                      'loop': LaunchConfiguration('loop'),
                      'rate': LaunchConfiguration('rate')}])
    detection = Node(package='metro_obstacle_ros', executable='detector_node',
                     name='developer_obstacle_detector', output='screen',
                     parameters=[detector, {'input_topic': LaunchConfiguration('input_topic')}])
    viewer = Node(package='rviz2', executable='rviz2', name='developer_obstacle_rviz',
                  arguments=['-d', rviz, '-f', LaunchConfiguration('fixed_frame')],
                  condition=IfCondition(LaunchConfiguration('rviz')), output='screen')
    return LaunchDescription(arguments + [player, detection, viewer])
