"""Safety gate v3 (ROS 2 node). Commands come ONLY from YOLO detections + elapsed time (see gate_core.py).
The VLM trace is subscribed to for logging only; nothing from it reaches a decision."""
import json
import os
import signal
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from vla_cognition.gate_core import CRUISE, GateCore, GateParams
from vla_cognition.trace_check import TraceChecker

CONTROL_LOOP_HZ = 20.0


def load_topics():
    config_path = os.path.join(get_package_share_directory('vla_actuation'), 'config', 'topics.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def stats(vals_s):
    """n, mean, p95, max in milliseconds."""
    if not vals_s:
        return 0, 0.0, 0.0, 0.0
    v = sorted(x * 1e3 for x in vals_s)
    p95 = v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]
    return len(v), sum(v) / len(v), p95, v[-1]


class SafetyGateNode(Node):
    def __init__(self):
        super().__init__('safety_gate_node')
        topics = load_topics()
        self.core = GateCore(GateParams(stop_bottom_frac=float(os.environ.get('GATE_STOP_BOTTOM', '0')),
                                        avoid_enabled=os.environ.get('GATE_AVOID', '0') == '1'))
        self.checker = TraceChecker()            # read-only monitor of the VLM text; no path to control
        self.t_step, self.t_det, self.t_d2c = [], [], []
        self.pending_arrival = None
        self.paced = os.environ.get('GATE_PACED', '0') == '1'
        self.held = (self.core.p.cruise_v, 0.0)

        self.create_subscription(String, topics['reasoning_trace_topic'], self.trace_callback, 10)
        self.create_subscription(String, topics['detections_topic'], self.detections_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, topics['safety_gate_cmd_vel_topic'], 10)
        self.real_cmd_pub = self.create_publisher(Twist, topics['cmd_vel_topic'], 10)

        self.create_timer(1.0 / CONTROL_LOOP_HZ, self.control_loop)
        self.create_timer(10.0, self.report_timing)
        self.get_logger().info(
            'Safety gate v3.2 ready (stop_bottom=%.2f avoid=%s). Actuation comes ONLY from YOLO bbox geometry; VLM text is logged only. '
            'STOP is valid in every state and latched; watchdog %.1f s.' % (self.core.p.stop_bottom_frac, self.core.p.avoid_enabled, self.core.p.det_timeout))

    # ---- subscribers ---------------------------------------------------------
    def trace_callback(self, msg):
        if self.paced:
            self.held = self.core.step(time.monotonic())   # baseline: command refreshed only when a trace arrives
        try:
            trace = json.loads(msg.data)
            self.get_logger().info('[EXPLAIN] %s' % str(trace.get('reasoning', ''))[:100])
            verdict, flags = self.checker.check(time.monotonic(), trace)       # label only; never used for control
            self.get_logger().info('[TRACE-CHECK] %s %s' % (verdict, ' '.join(flags)))
        except (json.JSONDecodeError, AttributeError):
            self.get_logger().warn('Malformed trace JSON - ignoring')

    def detections_callback(self, msg):
        t0 = time.perf_counter()
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('Malformed detections JSON - ignoring')
            return
        dets = data.get('detections', []) if isinstance(data, dict) else data
        now = time.monotonic()
        self.core.on_detections(now, dets)
        self.checker.observe(now, dets, self.core.obs[2] if self.core.obs else 0.0)
        if self.pending_arrival is None:
            self.pending_arrival = t0
        self.t_det.append(time.perf_counter() - t0)
        self.log_events()

    # ---- control -------------------------------------------------------------
    def publish_cmd(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)
        self.real_cmd_pub.publish(cmd)

    def control_loop(self):
        t0 = time.perf_counter()
        if self.paced:
            lin, ang = self.held
        else:
            lin, ang = self.core.step(time.monotonic())
        self.publish_cmd(lin, ang)
        t1 = time.perf_counter()
        self.t_step.append(t1 - t0)
        if self.pending_arrival is not None:
            self.t_d2c.append(t1 - self.pending_arrival)       # detection message arrival -> command published
            self.pending_arrival = None
        self.log_events()
        if lin == 0.0 and ang == 0.0:
            self.get_logger().warn('[OVERRIDE] STOP (%s)' % self.core.stop_reason, throttle_duration_sec=0.5)
        elif self.core.state == CRUISE:
            self.get_logger().info('[PASS] linear.x=%.2f angular.z=0.00' % lin, throttle_duration_sec=1.0)

    def log_events(self):
        for e in self.core.pop_events():
            if e.startswith('[OVERRIDE]'):
                self.get_logger().warn(e)
            else:
                self.get_logger().info(e)

    def report_timing(self, final=False):
        self.get_logger().info('[TRACE-STATS]%s %s' % (' final' if final else '', self.checker.summary()))
        for name, vals in (('gate_step_ms', self.t_step), ('det_callback_ms', self.t_det), ('det_to_cmd_ms', self.t_d2c)):
            n, mean, p95, mx = stats(vals)
            if n:
                self.get_logger().info('[TIMING]%s %s n=%d mean=%.3f p95=%.3f max=%.3f' % (
                    ' final' if final else '', name, n, mean, p95, mx))


def _term(*_):
    raise KeyboardInterrupt


def main():
    rclpy.init()
    node = SafetyGateNode()
    signal.signal(signal.SIGTERM, _term)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(3):                      # leave the car stopped when the gate exits
                node.publish_cmd(0.0, 0.0)
                time.sleep(0.05)
            node.report_timing(final=True)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
