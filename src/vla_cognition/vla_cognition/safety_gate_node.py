import os
import json
import math
import time
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

CONTROL_LOOP_HZ = 20.0

# --- Reactive avoidance: driven ENTIRELY by YOLO bounding-box geometry. ---
# The VLM's numeric outputs (distance estimate, trajectory) are not used for
# any actuation decision -- only its reasoning text is used, for explainability.
IMAGE_WIDTH_PX = 800.0
IMAGE_HEIGHT_PX = 800.0
IMAGE_CENTER_X = IMAGE_WIDTH_PX / 2.0

REACT_HEIGHT_FRACTION = 0.08       # bbox height > 8% of frame  -> start steering away
EMERGENCY_HEIGHT_FRACTION = 0.30   # bbox height > 30% of frame -> too close to steer, stop

STEER_GAIN = 0.35
REACT_MAX_LINEAR_VEL = 1.0
REACT_MAX_ANGULAR_VEL = 0.3
CRUISE_LINEAR_VEL = 1.6


def load_topics():
    config_path = os.path.join(
        get_package_share_directory('vla_actuation'), 'config', 'topics.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SafetyGateNode(Node):
    def __init__(self):
        super().__init__('safety_gate_node')
        topics = load_topics()

        self.latest_trace = None
        self.latest_detections = None

        self.trace_sub = self.create_subscription(
            String, topics['reasoning_trace_topic'], self.trace_callback, 10)
        self.detections_sub = self.create_subscription(
            String, topics['detections_topic'], self.detections_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, topics['safety_gate_cmd_vel_topic'], 10)
        self.real_cmd_pub = self.create_publisher(Twist, topics['cmd_vel_topic'], 10)

        period = 1.0 / CONTROL_LOOP_HZ
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            'Safety gate ready. ALL actuation decisions (stop/react/cruise) are '
            'computed from YOLO bounding-box geometry. The VLM trace is logged '
            'for explainability only and never drives the vehicle.')

    def trace_callback(self, msg):
        try:
            self.latest_trace = json.loads(msg.data)
            self.get_logger().info(f'[EXPLAIN] {self.latest_trace.get("reasoning", "")[:100]}')
        except json.JSONDecodeError:
            self.get_logger().warn('Received malformed trace JSON — ignoring update')

    def detections_callback(self, msg):
        try:
            self.latest_detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Received malformed detections JSON — ignoring update')

    def publish_stop(self, reason):
        t_start = time.time()
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.real_cmd_pub.publish(cmd)
        t_safe = (time.time() - t_start) * 1000
        self.get_logger().warn(f'[OVERRIDE] Publishing STOP ({reason}) | T_safe={t_safe:.3f}ms')

    def closest_detection(self):
        dets = (self.latest_detections or {}).get('detections', [])
        if not dets:
            return None
        return max(dets, key=lambda d: d.get('width', 0) * d.get('height', 0))

    def control_loop(self):
        t_start = time.time()

        det = self.closest_detection()
        height_fraction = 0.0
        if det is not None:
            height_fraction = det.get('height', 0) / IMAGE_HEIGHT_PX

        if height_fraction > EMERGENCY_HEIGHT_FRACTION:
            self.publish_stop(f'obstacle too close to steer safely (height_frac={height_fraction:.2f})')
            return

        if height_fraction > REACT_HEIGHT_FRACTION:
            det_x = det.get('x', IMAGE_CENTER_X)
            offset = (det_x - IMAGE_CENTER_X) / IMAGE_CENTER_X
            angular_vel = max(-REACT_MAX_ANGULAR_VEL, min(REACT_MAX_ANGULAR_VEL, -STEER_GAIN * offset))
            linear_vel = REACT_MAX_LINEAR_VEL
            maneuver_note = f' [REACT] (height_frac={height_fraction:.2f}, offset={offset:+.2f})'
        else:
            linear_vel = CRUISE_LINEAR_VEL
            angular_vel = 0.0
            maneuver_note = ''

        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)
        self.real_cmd_pub.publish(cmd)

        t_safe = (time.time() - t_start) * 1000
        self.get_logger().info(
            f'[PASS]{maneuver_note} linear.x={linear_vel:.2f} angular.z={angular_vel:.2f} | T_safe={t_safe:.3f}ms')


def main():
    rclpy.init()
    node = SafetyGateNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()