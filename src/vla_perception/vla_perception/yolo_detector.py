import os
import json
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


def load_topics():
    config_path = os.path.join(
        get_package_share_directory('vla_actuation'), 'config', 'topics.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        topics = load_topics()
        camera_topic = topics['camera_topic']
        detections_topic = topics['detections_topic']

        self.bridge = CvBridge()
        self.get_logger().info('Loading YOLOv10n model...')
        self.model = YOLO(os.path.expanduser('~/yolo_test/yolov10n.pt'))

        self.sub = self.create_subscription(Image, camera_topic, self.callback, 10)
        self.pub = self.create_publisher(String, detections_topic, 10)
        self.get_logger().info(
            f'Subscribed to {camera_topic}, publishing detections on {detections_topic}')
        self.frame_count = 0

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, device=0, verbose=False)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xywh = box.xywh[0]
            detections.append({
                'class': self.model.names[cls_id],
                'confidence': round(conf, 4),
                'x': float(xywh[0]),
                'y': float(xywh[1]),
                'width': float(xywh[2]),
                'height': float(xywh[3]),
            })

        out_msg = String()
        out_msg.data = json.dumps({'detections': detections})
        self.pub.publish(out_msg)

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(f'Frame {self.frame_count}: {len(detections)} detections')


def main():
    rclpy.init()
    node = YoloDetector()
    rclpy.spin(node)


if __name__ == '__main__':
    main()