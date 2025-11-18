# turtlebot3_controller/follow_path.py
#
# FOLLOW A PREDEFINED PATH USING WAYPOINTS
# This controller demonstrates a Finite State Machine (FSM) approach.
#
# The robot moves through a list of waypoints. For each waypoint, the robot:
#   1) Rotates to face the waypoint
#   2) Moves straight toward it
#   3) When reached, switches to the next waypoint
#
# This ensures stable and predictable movement, even with abrupt direction changes.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt, pi


class FollowPath(Node):
    """
    FollowPath Node
    ----------------
    This node enables TurtleBot3 to follow a predefined path of waypoints.
    It uses a simple and reliable FSM (finite-state machine):

       Stage 0: Rotate to face next waypoint
       Stage 1: Drive straight toward waypoint

    Once the waypoint is reached, robot moves to the next point.
    """

    def __init__(self):
        super().__init__('follow_path')

        # ============================================================
        # WAYPOINT LIST (modifiable)
        # ============================================================
        # Example: a square path
        self.path = [
            (0.0, -6.0),
            (-1.0, -6.0),
            (-1.0, -7.0),
            (0.0, -7.0),
            (0.0, -6.0),
        ]

        self.current_idx = 0   # index of current waypoint
        self.stage = 0         # 0=rotate, 1=drive

        # ============================================================
        # ROBOT STATE VARIABLES (updated from /odom)
        # ============================================================
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0  # orientation

        # ============================================================
        # CONTROL GAINS
        # ============================================================
        self.k_lin = 0.3      # linear velocity gain
        self.k_ang = 1.0      # angular velocity gain

        # ============================================================
        # SAFETY SPEED LIMITS (to avoid instability)
        # ============================================================
        self.max_lin = 0.18
        self.max_ang = 0.6

        # ============================================================
        # ROS PUB-SUB
        # ============================================================
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.get_logger().info("FollowPath node started. Following square path.")

    # ----------------------------------------------------------------------
    # ANGLE NORMALIZATION
    # ----------------------------------------------------------------------
    def normalize(self, a):
        """Keep angle within [-pi, pi] to avoid unnecessary spinning."""
        while a > pi:
            a -= 2*pi
        while a < -pi:
            a += 2*pi
        return a

    # ----------------------------------------------------------------------
    # ODOMETRY CALLBACK
    # ----------------------------------------------------------------------
    def odom_callback(self, msg):
        """
        Update robot pose (x, y, yaw) from Odometry.
        Then call control loop to compute motion.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        o = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(o)

        self.control_loop()

    # ----------------------------------------------------------------------
    # MAIN CONTROL LOGIC (FSM)
    # ----------------------------------------------------------------------
    def control_loop(self):
        cmd = Twist()

        # If all waypoints completed → stop robot
        if self.current_idx >= len(self.path):
            self.get_logger().info("Path completed successfully!")
            self.cmd_pub.publish(Twist())  # stop
            return

        # Get the current waypoint
        goal_x, goal_y = self.path[self.current_idx]

        # Compute distance and direction to waypoint
        dx = goal_x - self.x
        dy = goal_y - self.y
        dist = sqrt(dx*dx + dy*dy)               # distance to waypoint
        ang_to_goal = atan2(dy, dx)              # desired heading
        ang_err = self.normalize(ang_to_goal - self.yaw)

        # ============================================================
        # STAGE 0 → ROTATE TO FACE WAYPOINT
        # ============================================================
        if self.stage == 0:
            if abs(ang_err) > 0.1:
                cmd.angular.z = self.k_ang * ang_err
            else:
                # Finished rotating → ready to drive straight
                self.stage = 1
                self.get_logger().info(
                    f"Facing waypoint {self.current_idx}: starting to drive."
                )

        # ============================================================
        # STAGE 1 → DRIVE STRAIGHT TOWARD WAYPOINT
        # ============================================================
        elif self.stage == 1:
            if dist > 0.1:
                cmd.linear.x = self.k_lin * dist
            else:
                # Waypoint reached!
                self.get_logger().info(
                    f"Reached waypoint {self.current_idx} at ({goal_x:.2f}, {goal_y:.2f})"
                )
                self.current_idx += 1
                self.stage = 0  # Return to rotate stage for next waypoint

        # ============================================================
        # APPLY SPEED LIMITS
        # ============================================================
        cmd.linear.x = max(min(cmd.linear.x, self.max_lin), -self.max_lin)
        cmd.angular.z = max(min(cmd.angular.z, self.max_ang), -self.max_ang)

        # Publish velocity command
        self.cmd_pub.publish(cmd)

    # ----------------------------------------------------------------------
    # QUATERNION → YAW CONVERSION
    # ----------------------------------------------------------------------
    def quaternion_to_yaw(self, o):
        """Extract yaw angle (heading) from quaternion."""
        siny = 2.0 * (o.w * o.z + o.x * o.y)
        cosy = 1.0 - 2.0 * (o.y**2 + o.z**2)
        return atan2(siny, cosy)


# ----------------------------------------------------------------------
# PROGRAM ENTRY POINT
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = FollowPath()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
