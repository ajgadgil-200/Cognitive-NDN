import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from transforms3d.euler import quat2euler # Ensure python3-transforms3d is installed via apt

class PriusPidController(Node):
    def __init__(self):
        super().__init__('prius_pid_controller')
        
        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, '/vla/odom', self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, '/vla/predicted_path', self.path_callback, 10)
        
        # Publisher to Layer 1 Bridge
        self.cmd_pub = self.create_publisher(Twist, '/model/prius_vla/cmd_vel', 10)
        
        # Controller State Variables
        self.current_pose = None
        self.target_path = None
        
        # Longitudinal PID Parameters (Speed Control)
        self.Kp_v = 0.8
        self.target_speed = 5.0 # m/s default testing cap
        
        # Lateral Pure-Pursuit / PID Parameters (Steering Control)
        self.Kp_w = 1.5
        self.lookahead_distance = 3.0 

        self.get_logger().info("System 1 Actuation PID Core online.")

    def odom_callback(self, msg):
        # Extract location coordinates
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        
        self.current_pose = {'x': pos.x, 'y': pos.y, 'yaw': yaw, 'v': msg.twist.twist.linear.x}
        self.control_loop()

    def path_callback(self, msg):
        self.target_path = msg.poses

    def control_loop(self):
        if self.current_pose is None or self.target_path is None or len(self.target_path) == 0:
            return
        
        # Find lookahead target point from System 2 trajectory planner array
        target_pt = self.target_path[min(len(self.target_path)-1, 3)].pose.position
        
        # Compute tracking errors
        dx = target_pt.x - self.current_pose['x']
        dy = target_pt.y - self.current_pose['y']
        
        # Cross track error transformed to vehicle heading coordinate space
        target_heading = math.atan2(dy, dx)
        heading_error = target_heading - self.current_pose['yaw']
        
        # Normalize angle error between [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Compute Actuation Reflex Outputs
        cmd = Twist()
        
        # Linear tracking velocity control (simplistic error loop scaling)
        speed_error = self.target_speed - self.current_pose['v']
        cmd.linear.x = self.current_pose['v'] + (speed_error * self.Kp_v)
        
        # Angular rate control (Steering Reflex Engine)
        cmd.angular.z = heading_error * self.Kp_w
        
        # Execute Actuation command transmission
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PriusPidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()