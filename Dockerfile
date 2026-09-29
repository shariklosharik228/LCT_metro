# ROS 2 Humble / Ubuntu 22.04 image with the obstacle detector.
FROM ros:humble-ros-base-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-scipy python3-sklearn python3-pytest \
        ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins \
        ros-humble-rosbag2-storage-mcap ros-humble-rviz2 \
        ros-humble-sensor-msgs ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY src ./src
COPY tools ./tools
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

RUN printf '#!/bin/bash\nset -e\nsource /opt/ros/humble/setup.bash\nsource /ws/install/setup.bash\nexec "$@"\n' \
        > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
# default: detector + bag player + RViz; bag mounted to /data/bag
CMD ["ros2", "launch", "metro_obstacle_ros", "demo.launch.py", "bag_path:=/data/bag"]
