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
# Home pose — must match the first waypoint in patrol.py's PATROL_ROUTE
# ---------------------------------------------------------------------------
HOME_X  =  0.0
HOME_Y  =  0.0
HOME_QZ =  0.0
HOME_QW =  1.0


# ---------------------------------------------------------------------------
# Publisher node — thin companion to BasicNavigator
# ---------------------------------------------------------------------------

class ReturnHomeMonitor(Node):
    def __init__(self):
        super().__init__('return_home_monitor')
        self.done_pub = self.create_publisher(Bool, '/task/return_home_complete', 10)


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

    # Build home pose
    home = PoseStamped()
    home.header.frame_id = 'map'
    home.header.stamp = navigator.get_clock().now().to_msg()
    home.pose.position.x = HOME_X
    home.pose.position.y = HOME_Y
    home.pose.orientation.z = HOME_QZ
    home.pose.orientation.w = HOME_QW

    navigator.get_logger().info(f'Navigating home: ({HOME_X}, {HOME_Y})')
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
