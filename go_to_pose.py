# turtlebot3_controller/go_to_pose.py
#
# Continuous controller for moving TurtleBot3 to a full pose:
#   [x, y, theta]
#
# This controller uses the unicycle model equations and performs
# smooth curved motions without discrete stages.
#
# Usage:
#   ros2 run turtlebot3_controller go_to_pose x y theta

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt, pi


class GoToPose(Node):
    """
    GoToPose Controller
    -------------------
    This node drives the TurtleBot3 to a specified pose (x, y, theta)
    using a continuous feedback control law based on the unicycle model.

    Unlike the FSM (finite-state machine) method, this controller:
        ✔ Rotates and moves at the same time
        ✔ Produces smooth curved paths
        ✔ Does not "stop → rotate → go → rotate"
        ✔ Adjusts velocity continuously according to errors

    Useful for higher-performance motion where smoothness matters.
    """

    def __init__(self, gx, gy, gtheta):
        super().__init__('go_to_pose')

        # ---------------------------
        # 1. Store Goal Pose (Target)
        # ---------------------------
        self.goal_x = gx
        self.goal_y = gy
        self.goal_theta = gtheta   # final desired orientation

        # ---------------------------
        # 2. Robot's Current State
        # ---------------------------
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0   # robot's orientation (in radians)

        # ---------------------------
        # 3. Controller Gains
        # ---------------------------
        # k_rho    : how fast to reduce distance to goal
        # k_alpha  : how strongly robot rotates toward goal direction
        # k_beta   : how strongly robot corrects final orientation
        #
        # NOTE: k_beta must be NEGATIVE for stability.
        self.k_rho = 0.5
        self.k_alpha = 1.2
        self.k_beta = -0.2

        # ---------------------------
        # 4. Speed Limits (Safety)
        # ---------------------------
        self.max_lin = 0.18
        self.max_ang = 1.0

        # ---------------------------
        # 5. ROS Publishers / Subscribers
        # ---------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Debug message for rqt_console
        self.get_logger().info(
            f"GoToPose started → Goal = ({gx:.2f}, {gy:.2f}, θ={gtheta:.2f})"
        )

    # ----------------------------------------------------------------------
    # Helper: Normalize angle to range [-pi, pi]
    # ----------------------------------------------------------------------
    def normalize(self, a):
        """Wrap angle 'a' so it always stays within [-pi, pi]."""
        while a > pi:
            a -= 2*pi
        while a < -pi:
            a += 2*pi
        return a

    # ----------------------------------------------------------------------
    # Odometry Callback
    # ----------------------------------------------------------------------
    def odom_callback(self, msg):
        """
        Extract robot's current position and orientation from Odometry.
        Trigger the pose control loop after updating the robot state.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Convert quaternion → yaw angle
        o = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(o)

        # Execute control step
        self.control_loop()

    # ----------------------------------------------------------------------
    # MAIN CONTROL LOOP
    # ----------------------------------------------------------------------
    def control_loop(self):
        """
        Continuous pose controller using unicycle model equations.
        Adjusts linear and angular velocity simultaneously.
        """
        cmd = Twist()

        # ---------------------------
        # A) Compute Errors
        # ---------------------------
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        rho = sqrt(dx*dx + dy*dy)      # distance-to-goal

        angle_to_goal = atan2(dy, dx)

        # alpha = angle difference between robot orientation & goal direction
        alpha = self.normalize(angle_to_goal - self.yaw)

        # beta = how much final heading differs from movement direction
        beta = self.normalize(self.goal_theta - angle_to_goal)

        # ---------------------------
        # B) When very close to goal → stop moving forward
        # ---------------------------
        if rho < 0.05:
            # Stop linear movement
            cmd.linear.x = 0.0

            # Only rotate to final orientation
            angle_err = self.normalize(self.goal_theta - self.yaw)

            if abs(angle_err) < 0.05:
                # Goal pose reached completely
                cmd.angular.z = 0.0
                self.get_logger().info("Final pose reached!")
            else:
                # Rotate toward final angle
                cmd.angular.z = self.k_alpha * angle_err

            self.cmd_pub.publish(cmd)
            return

        # ---------------------------
        # C) Continuous Control Law
        # ---------------------------
        # Unicycle-model controller:
        #   v = k_rho * rho
        #   w = k_alpha * alpha + k_beta * beta
        cmd.linear.x = self.k_rho * rho
        cmd.angular.z = self.k_alpha * alpha + self.k_beta * beta

        # ---------------------------
        # D) Speed limiting (safety)
        # ---------------------------
        cmd.linear.x = max(min(cmd.linear.x, self.max_lin), -self.max_lin)
        cmd.angular.z = max(min(cmd.angular.z, self.max_ang), -self.max_ang)

        # ---------------------------
        # E) Send velocity command
        # ---------------------------
        self.cmd_pub.publish(cmd)

    # ----------------------------------------------------------------------
    # Quaternion → Yaw Conversion
    # ----------------------------------------------------------------------
    def quaternion_to_yaw(self, o):
        """Convert quaternion orientation into yaw angle (robot heading)."""
        siny = 2.0 * (o.w * o.z + o.x * o.y)
        cosy = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        return atan2(siny, cosy)


# --------------------------------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    import sys

    if len(sys.argv) != 4:
        print("Usage: ros2 run turtlebot3_controller go_to_pose x y theta")
        return

    gx = float(sys.argv[1])
    gy = float(sys.argv[2])
    gtheta = float(sys.argv[3])

    node = GoToPose(gx, gy, gtheta)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
