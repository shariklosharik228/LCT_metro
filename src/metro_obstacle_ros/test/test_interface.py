from metro_obstacle_ros.bag_player_node import BagPlayerNode
from metro_obstacle_ros.detector_node import DeveloperDetectorNode


def test_nodes_are_packaged():
    assert BagPlayerNode.__name__ == 'BagPlayerNode'
    assert DeveloperDetectorNode.__name__ == 'DeveloperDetectorNode'
