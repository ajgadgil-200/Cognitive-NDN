import os
import json
import base64
import time
import yaml
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ament_index_python.packages import get_package_share_directory

MISSION_GOAL = "Navigate forward along the road safely, avoiding all obstacles."
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava-phi3"

SYSTEM_PROMPT = """You are the perception-reasoning module of an autonomous vehicle.

CRITICAL RULE: You are FORBIDDEN from naming any object class that does not appear
in the "Detected objects" list below. The detector below is ground truth and has
already correctly identified every object in this scene. Do not use your own visual
guess for what an object is — only use the class names given to you.

Mission goal: {goal}
Detected objects (ground truth, from YOLO): {detections}

If the detected objects list is empty, say so and report no obstacles by class name.

IMPORTANT: If an obstacle IS in the path, do not simply stop. Propose a trajectory
that steers AROUND it — shift the "y" (lateral) values of your waypoints by at least
2.5 meters away from the obstacle's side, while still moving forward in "x", so the
vehicle can safely pass the obstacle rather than halting indefinitely.

Respond with ONLY valid JSON in exactly this schema, no other text:
{{
  "reasoning": "one or two short sentences, referring to obstacles ONLY by the exact
class names given above (e.g. 'cow', not 'dog' or 'animal')",
  "obstacle_in_path": true or false,
  "obstacle_distance_estimate_m": number or null,
  "trajectory": [{{"x": number, "y": number}}, ...]
}}
The trajectory is 3 to 5 waypoints, in meters, relative to the vehicle's current
position (x = forward, y = lateral). If bypassing an obstacle, waypoints should
show increasing lateral offset (y) before returning toward y=0 once past it."""


def load_topics():
    config_path = os.path.join(
        get_package_share_directory('vla_actuation'), 'config', 'topics.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ReasoningNode(Node):
    def __init__(self):
        super().__init__('reasoning_node')
        topics = load_topics()
        self.bridge = CvBridge()
        self.latest_detections = None

        self.image_sub = self.create_subscription(
            Image, topics['camera_topic'], self.image_callback, 10)
        self.detections_sub = self.create_subscription(
            String, topics['detections_topic'], self.detections_callback, 10)

        self.trace_pub = self.create_publisher(String, topics['reasoning_trace_topic'], 10)
        self.trajectory_pub = self.create_publisher(String, topics['trajectory_topic'], 10)

        self.get_logger().info('Reasoning node ready. Waiting for camera frames...')
        self.last_call_time = 0.0
        self.min_interval_sec = 2.0  # throttle: don't call the LLM on every single frame

    def detections_callback(self, msg):
        self.latest_detections = msg.data

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_call_time < self.min_interval_sec:
            return
        if self.latest_detections is None:
            return  # haven't received a real detections message yet — skip this frame
        self.last_call_time = now

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        success, buffer = cv2.imencode('.jpg', cv_image)
        if not success:
            self.get_logger().warn('Failed to encode frame')
            return
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        prompt = SYSTEM_PROMPT.format(goal=MISSION_GOAL, detections=self.latest_detections)
        self.get_logger().info(f'Detections being sent to model: {self.latest_detections}')
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "images": [image_b64],
            "format": "json",
            "stream": False,
        }

        t_start = time.time()
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f'Ollama request failed: {e}')
            return
        t_llm = time.time() - t_start

        try:
            result_text = response.json()['response']
            parsed = json.loads(result_text)
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f'Could not parse model output as JSON: {e}')
            return

        self.get_logger().info(
            f'T_llm = {t_llm*1000:.1f}ms | obstacle_in_path={parsed.get("obstacle_in_path")} | '
            f'reasoning="{parsed.get("reasoning", "")[:80]}"')

        obstacle_flag = parsed.get('obstacle_in_path')
        if obstacle_flag is None:
            self.get_logger().warn('Model returned obstacle_in_path=None — treating as UNSAFE (obstacle assumed present)')
            obstacle_flag = True  # fail-safe: ambiguous output defaults to "assume danger"

        trace_msg = String()
        trace_msg.data = json.dumps({
            'reasoning': parsed.get('reasoning'),
            'obstacle_in_path': obstacle_flag,
            'obstacle_distance_estimate_m': parsed.get('obstacle_distance_estimate_m'),
            't_llm_ms': round(t_llm * 1000, 1),
        })
        self.trace_pub.publish(trace_msg)

        traj_msg = String()
        traj_msg.data = json.dumps({'trajectory': parsed.get('trajectory', [])})
        self.trajectory_pub.publish(traj_msg)


def main():
    rclpy.init()
    node = ReasoningNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()