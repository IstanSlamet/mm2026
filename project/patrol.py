#!/usr/bin/env python3
"""
Patrol script for Stretch 3.

Follows a set of hardcoded waypoints in a loop using Nav2 and monitors
/gripper_detector/object_found.  When the target object is detected the
navigation task is cancelled and the object's pose is published so the
next stage (pre-grasp approach) can take over.

Run alongside:
  python3 gripper_object_detector.py

Published topics:
  /task/object_pose  (geometry_msgs/PoseStamped)
      Pose of the found object — consumed by the pre-grasp approach stage.

Requires (start in this order before running this script):
  ros2 launch stretch_core stretch_driver.launch.py
  ros2 launch stretch_nav2 navigation.launch.py use_sim_time:=False map:=/path/to/your_map.yaml
  ros2 launch stretch_core d405_basic.launch.py
"""

import threading
from copy import deepcopy

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult


# ---------------------------------------------------------------------------
# Patrol waypoints  [x, y, qz, qw]  — map frame
# Replace with your actual waypoints (drive + read /amcl_pose, or use RViz)
# ---------------------------------------------------------------------------

PATROL_ROUTE = [
    [-0.59146, -2.0741, -0.5522, 0.83371],
    [ 0.71151, -3.8080, -0.2812, 0.95964],
    [ 2.65800, -4.6610,  0.2806, 0.95982],
    [ 3.49210, -6.3836, -0.4494, 0.89335],
    [ 4.83040, -8.6511,  0.0425, 0.99910],
    [ 6.04890, -9.8384, -0.4463, 0.89487],
]

# Navigation timeout for a full patrol loop
PATROL_TIMEOUT_SEC = 600.0

# Polling interval while waiting for navigation to complete
POLL_PERIOD_SEC = 0.5


# ---------------------------------------------------------------------------
# Patrol monitor — thin Node for detection subscriptions
# ---------------------------------------------------------------------------

class PatrolMonitor(Node):
    """
    Subscribes to the gripper detector and exposes the latest state as plain
    attributes so the main patrol loop can poll them without callbacks.
    """

    def __init__(self):
        super().__init__('patrol_monitor')
        self.object_found = False
        self.object_pose: PoseStamped | None = None

        self.create_subscription(
            Bool, '/gripper_detector/object_found',
            self._found_callback, 10)
        self.create_subscription(
            PoseStamped, '/gripper_detector/goal_pose',
            self._pose_callback, 10)

        # Publish the found pose so the task manager / pre-grasp node picks it up
        self.object_pose_pub = self.create_publisher(
            PoseStamped, '/task/object_pose', 10)

    def _found_callback(self, msg: Bool):
        self.object_found = msg.data

    def _pose_callback(self, msg: PoseStamped):
        self.object_pose = msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pose(navigator: BasicNavigator, x, y, qz, qw) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.orientation.z = float(qz)
    pose.pose.orientation.w = float(qw)
    return pose


# ---------------------------------------------------------------------------
# Main patrol loop
# ---------------------------------------------------------------------------

def main():
    rclpy.init()

    navigator = BasicNavigator()
    monitor   = PatrolMonitor()

    # Spin both nodes in a background thread so Nav2 feedback and detection
    # callbacks both process while the main thread drives the patrol logic.
    executor = MultiThreadedExecutor()
    executor.add_node(navigator)
    executor.add_node(monitor)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Switch robot to navigation mode so Nav2 can send cmd_vel.
    # Any prior HelloNode script (grasp, IK, etc.) may have left it in position mode.
    nav_mode_client = monitor.create_client(Trigger, '/switch_to_navigation_mode')
    if nav_mode_client.wait_for_service(timeout_sec=5.0):
        nav_mode_client.call_async(Trigger.Request())
        navigator.get_logger().info('Switched to navigation mode.')
    else:
        navigator.get_logger().warning('/switch_to_navigation_mode service not available — continuing anyway.')

    navigator.waitUntilNav2Active()
    navigator.get_logger().info('Nav2 active — starting patrol.')

    route = list(PATROL_ROUTE)   # mutable copy so we can reverse it
    object_found = False

    while rclpy.ok() and not object_found:

        # Build PoseStamped list — skip first waypoint on first pass
        # (assumed to be home; matches reference script behaviour)
        route_poses = [make_pose(navigator, *wp) for wp in route[1:]]

        nav_start = navigator.get_clock().now()
        navigator.followWaypoints(route_poses)
        navigator.get_logger().info(f'Patrol pass started — {len(route_poses)} waypoints.')

        i = 0
        while not navigator.isTaskComplete():

            # Object detected → stop
            if monitor.object_found and monitor.object_pose is not None:
                navigator.get_logger().info('Object detected — cancelling patrol.')
                navigator.cancelTask()
                object_found = True
                break

            # Progress log every 5 polls
            i += 1
            feedback = navigator.getFeedback()
            if feedback and i % 5 == 0:
                navigator.get_logger().info(
                    f'Waypoint {feedback.current_waypoint + 1}/{len(route_poses)}')

            # Hard timeout guard
            if navigator.get_clock().now() - nav_start > Duration(seconds=PATROL_TIMEOUT_SEC):
                navigator.get_logger().warning('Patrol timeout — restarting loop.')
                navigator.cancelTask()
                break

            navigator.get_clock().sleep_for(Duration(seconds=POLL_PERIOD_SEC))

        if object_found:
            break

        # End of route — reverse and go again (same as reference script)
        route.reverse()
        result = navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            navigator.get_logger().info('Route complete — reversing.')
        elif result == TaskResult.FAILED:
            navigator.get_logger().warning('Route failed — restarting from other end.')

    # -----------------------------------------------------------------------
    # Publish found pose for the pre-grasp stage
    # -----------------------------------------------------------------------
    if object_found and monitor.object_pose is not None:
        navigator.get_logger().info('Publishing object pose for pre-grasp stage.')
        for _ in range(5):
            monitor.object_pose_pub.publish(monitor.object_pose)
            navigator.get_clock().sleep_for(Duration(seconds=0.1))

    navigator.get_logger().info('Patrol complete.')
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
