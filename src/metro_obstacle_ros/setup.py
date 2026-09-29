from glob import glob
from setuptools import find_packages, setup

setup(name='metro_obstacle_ros', version='1.0.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages',['resource/metro_obstacle_ros']),
                  ('share/metro_obstacle_ros',['package.xml']),
                  ('share/metro_obstacle_ros/config',glob('config/*.yaml')),
                  ('share/metro_obstacle_ros/launch',glob('launch/*.launch.py')),
                  ('share/metro_obstacle_ros/rviz',glob('rviz/*.rviz'))],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='Metro team', maintainer_email='team@example.com',
      description='ROS 2 integration and external rosbag player.', license='Apache-2.0',
      entry_points={'console_scripts':['detector_node=metro_obstacle_ros.detector_node:main',
                                      'bag_player_node=metro_obstacle_ros.bag_player_node:main']})
