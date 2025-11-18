import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from math import atan2, sqrt, pi


class GoToPointSafe(Node):
    """
    Safe Go-To-Point Controller:
    -------------------------------------------
    This node moves the robot toward a target point (x, y) while:
      - Continuously checking the distance to the goal
      - Adjusting linear and angular motion accordingly
      - Avoiding obstacles using LaserScan data
      - Publishing ROS info messages for monitoring (rqt_console)
      - Publishing error for controller evaluation (rqt_plot)

    It demonstrates:
      ✔ Proportional control for goal seeking
      ✔ Reactive obstacle avoidance
      ✔ Safety speed limits
      ✔ ROS2 logging best practices
      ✔ Real-time feedback via custom topics
    """

    def __init__(self, goal_x, goal_y):
        super().__init__("go_to_point_safe")

        # ---------------------------
        # 1. STORE GOAL COORDINATES
        # ---------------------------
        # Robot will move until it reaches (goal_x, goal_y)
        self.goal_x = goal_x
        self.goal_y = goal_y

        # ---------------------------
        # 2. ROBOT POSE VARIABLES
        # ---------------------------
        # These are updated from /odom topic
        self.x = 0.0      # current X position
        self.y = 0.0      # current Y position
        self.yaw = 0.0    # current orientation (yaw angle)

        # ---------------------------
        # 3. OBSTACLE DISTANCE VARIABLES
        # ---------------------------
        # These are updated from /scan (LaserScan)
        self.front = 999
        self.left = 999
        self.right = 999
        self.min_safe_distance = 0.35   # Robot will avoid if obstacle is closer than this

        # ---------------------------
        # 4. CONTROL GAINS
        # ---------------------------
        # k_linear  : speed toward goal
        # k_angular : turning strength toward goal
        # k_avoidance : turning strength when avoiding obstacle
        self.k_linear = 0.25
        self.k_angular = 0.8
        self.k_avoidance = 0.6

        # ---------------------------
        # 5. SPEED LIMITS (SAFETY)
        # ---------------------------
        self.max_linear_speed = 0.18
        self.max_angular_speed = 1.0

        # ---------------------------
        # 6. ROS PUBLISHERS
        # ---------------------------
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Publish distance error for plotting in rqt_plot
        self.error_pub = self.create_publisher(Float32, "/distance_error", 10)

        # ---------------------------
        # 7. ROS SUBSCRIBERS
        # ---------------------------
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Log startup message
        self.get_logger().info(
            f"SAFE Go-To-Point controller started. GOAL → ({goal_x}, {goal_y})"
        )

    # =====================================================================
    # LASERSCAN CALLBACK
    # =====================================================================

    def scan_callback(self, msg):
        """
        Receives 360° laser scan.
        We divide the view into 3 zones:
        - front : directly in front
        - left  : approx +90 degrees
        - right : approx -90 degrees
        """

        ranges = msg.ranges

        # Front = ±10 degrees → indices 0–10 and 350–360
        self.front = min(min(ranges[0:10]), min(ranges[-10:]))

        # Left side ≈ +90°
        self.left = min(ranges[70:110])

        # Right side ≈ -90°
        self.right = min(ranges[250:290])

    # =====================================================================
    # ODOM CALLBACK
    # =====================================================================

    def odom_callback(self, msg):
        """
        Extract robot position and orientation from /odom.
        Then trigger the main control loop.
        """

        # Update robot position
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Extract yaw orientation from quaternion
        o = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(o)

        # Run control logic each time odom updates
        self.control_loop()

    # =====================================================================
    # ANGLE NORMALIZATION
    # =====================================================================

    def normalize_angle(self, a):
        """
        Ensure angle is always between -pi and +pi.
        Prevents spinning more than needed.
        """
        while a > pi:
            a -= 2*pi
        while a < -pi:
            a += 2*pi
        return a

    # =====================================================================
    # SPEED LIMITING FUNCTION
    # =====================================================================

    def limit_speed(self, cmd):
        """
        Apply speed limits to avoid sudden fast movements.
        Improves stability in simulation and real robot.
        """

        cmd.linear.x = max(min(cmd.linear.x, self.max_linear_speed), -self.max_linear_speed)
        cmd.angular.z = max(min(cmd.angular.z, self.max_angular_speed), -self.max_angular_speed)

        return cmd

    # =====================================================================
    # MAIN CONTROL LOOP
    # =====================================================================

    def control_loop(self):
        """
        MAIN BRAIN OF THE ROBOT
        ------------------------
        This function chooses robot motion based on:
           1. Distance to goal
           2. Angle to goal
           3. Obstacle distances (from laser)
        """

        cmd = Twist()

        # ---------------------------
        # A) COMPUTE DISTANCE + ANGLE TO GOAL
        # ---------------------------
        dist = sqrt((self.goal_x - self.x)**2 + (self.goal_y - self.y)**2)
        angle_to_goal = atan2(self.goal_y - self.y, self.goal_x - self.x)
        angle_error = self.normalize_angle(angle_to_goal - self.yaw)

        # Publish error (for rqt_plot)
        self.error_pub.publish(Float32(data=dist))

        # ---------------------------
        # B) CHECK IF ROBOT REACHED GOAL
        # ---------------------------
        if dist < 0.08:
            self.get_logger().info("Goal reached safely!")
            self.cmd_pub.publish(Twist())  # stop robot
            return

        # ---------------------------
        # C) OBSTACLE AVOIDANCE BEHAVIOUR
        # ---------------------------
        if self.front < self.min_safe_distance:

            self.get_logger().warn(
                f"Obstacle detected → front={self.front:.2f} left={self.left:.2f} right={self.right:.2f}"
            )

            # CASE 1: TURN LEFT (more free space)
            if self.left > self.right:
                cmd.angular.z = +self.k_avoidance
                self.get_logger().info("↩ Avoiding → Turning LEFT")

            # CASE 2: TURN RIGHT (more free space on right)
            elif self.right > self.left:
                cmd.angular.z = -self.k_avoidance
                self.get_logger().info("↪ Avoiding → Turning RIGHT")

            # CASE 3: BOTH BLOCKED → back up + rotate
            else:
                cmd.linear.x = -0.08
                cmd.angular.z = +self.k_avoidance
                self.get_logger().info("🔙 Avoiding → BACKING UP")

        # ---------------------------
        # D) NORMAL GO-TO-GOAL MOTION
        # ---------------------------
        else:
            cmd.linear.x = self.k_linear * dist       # move forward
            cmd.angular.z = self.k_angular * angle_error  # rotate toward target

            self.get_logger().info(
                f"[GOAL TRACKING] dist={dist:.2f} angle_err={angle_error:.2f}"
            )

        # Apply safety speed limits
        cmd = self.limit_speed(cmd)

        # Send movement command
        self.cmd_pub.publish(cmd)

    # =====================================================================
    # QUATERNION → YAW
    # =====================================================================

    def quaternion_to_yaw(self, o):
        """
        Convert quaternion (robot orientation format) to yaw angle in radians.
        """
        siny = 2.0 * (o.w * o.z + o.x * o.y)
        cosy = 1.0 - 2.0 * (o.y**2 + o.z**2)
        return atan2(siny, cosy)


# =====================================================================
# MAIN PROGRAM ENTRY
# =====================================================================

def main(args=None):
    rclpy.init(args=args)
    import sys

    if len(sys.argv) != 3:
        print("Usage: ros2 run turtlebot3_controller go_to_point_safe x y")
        return

    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])

    node = GoToPointSafe(goal_x, goal_y)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
