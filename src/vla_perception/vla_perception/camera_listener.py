import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory


def load_camera_topic():
    """Read the canonical camera topic name from vla_actuation's topics.yaml,
    instead of hardcoding the string here."""
    config_path = os.path.join(
        get_package_share_directory('vla_actuation'), 'config', 'topics.yaml')
    with open(config_path, 'r') as f:
        topics = yaml.safe_load(f)
    return topics['camera_topic']


class CameraListener(Node):
    def __init__(self):
        super().__init__('camera_listener')
        topic = load_camera_topic()
        self.sub = self.create_subscription(Image, topic, self.callback, 10)
        self.get_logger().info(f'Subscribed to camera topic: {topic}')
        self.count = 0

    def callback(self, msg):
        self.count += 1
        if self.count % 30 == 0:
            self.get_logger().info(
                f'Frame {self.count}: {msg.width}x{msg.height}, encoding={msg.encoding}')


def main():
    rclpy.init()
    node = CameraListener()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
