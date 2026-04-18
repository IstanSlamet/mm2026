#!/usr/bin/env python3
"""
Return Home script for Stretch 3.

After grasping, navigates the robot back to the home waypoint (first entry
in PATROL_ROUTE).  grasp_objects.py already set the travel pose (lift 0.8 m,
arm retracted) before signalling grasp_complete, so this script just needs to
switch to navigation mode and drive home.

Published topics:
  /task/return_home_complete  (std_msgs/Bool)

Requires (already running):
  ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
  ros2 launch stretch_nav2 navigation.launch.py use_sim_time:=False map:=/path/to/map.yaml
"""

import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult


# ---------------------------------------------------------------------------
# Monitor node — subscribes to home pose + publishes completion
# ---------------------------------------------------------------------------

class ReturnHomeMonitor(Node):
    def __init__(self):
        super().__init__('return_home_monitor')
        self.home_pose: PoseStamped | None = None
        self.done_pub = self.create_publisher(Bool, '/task/return_home_complete', 10)
        # Home pose published continuously by mission_manager from the
        # 2D initial pose estimate the operator set in RViz.
        self.create_subscription(
            PoseStamped, '/task/home_pose',
            self._home_pose_callback, 10)

    def _home_pose_callback(self, msg: PoseStamped):
        self.home_pose = msg


# ---------------------------------------------------------------------------

def main():
    rclpy.init()

    navigator = BasicNavigator()
    monitor   = ReturnHomeMonitor()

    executor = MultiThreadedExecutor()
    executor.add_node(navigator)
    executor.add_node(monitor)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Switch to navigation mode — grasp_objects.py left the driver in position mode.
    nav_client = monitor.create_client(Trigger, '/switch_to_navigation_mode')
    if nav_client.wait_for_service(timeout_sec=5.0):
        future = nav_client.call_async(Trigger.Request())
        deadline = time.time() + 5.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.1)
        navigator.get_logger().info('Switched to navigation mode.')
    else:
        navigator.get_logger().warning('/switch_to_navigation_mode not available — continuing.')

    navigator.waitUntilNav2Active()

    # Wait for home pose from mission_manager (published from the RViz
    # 2D initial pose estimate set by the operator before the mission).
    navigator.get_logger().info('Waiting for home pose from /task/home_pose...')
    deadline = time.time() + 10.0
    while monitor.home_pose is None and time.time() < deadline:
        time.sleep(0.2)

    if monitor.home_pose is None:
        navigator.get_logger().error('No home pose received — cannot navigate home.')
        executor.shutdown()
        rclpy.shutdown()
        return

    home = monitor.home_pose
    home.header.stamp = navigator.get_clock().now().to_msg()
    navigator.get_logger().info(
        f'Navigating home: ({home.pose.position.x:.2f}, {home.pose.position.y:.2f})')
    navigator.goToPose(home)

    while not navigator.isTaskComplete():
        navigator.get_clock().sleep_for(Duration(seconds=0.5))

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        navigator.get_logger().info('Arrived home.')
    else:
        navigator.get_logger().warning(f'Navigation ended with result: {result}')

    # Publish completion regardless — mission manager advances to DONE
    for _ in range(5):
        try:
            monitor.done_pub.publish(Bool(data=True))
        except Exception:
            break
        time.sleep(0.1)

    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
