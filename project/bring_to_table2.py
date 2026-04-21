#!/usr/bin/env python3
"""
Return to Table Node

Runs during the RETURNING mission state. Navigates the robot to the
pre-defined table position and releases the grasped object by moving
the arm to the release configuration and opening the gripper.

Sequence:
  1. Navigate base to table pose via Nav2
  2. Move arm to release configuration (lift + extension + wrist angles)
  3. Open gripper — object drops onto table
  4. Stow arm back to safe travel pose
  5. Publish /return/result True → mission_manager transitions to IDLE

How to run (terminals 1-2 must already be running):
  # Terminal 1
  ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
  # Terminal 2
  ros2 launch stretch_nav2 navigation.launch.py use_sim_time:=False map:=/path/to/map.yaml
  # Terminal 3
  python3 return_to_table.py

Published topics:
  /return/result  (std_msgs/Bool)
      True when object has been released and arm is stowed.
      False if navigation fails.

Subscribed topics:
  /mission/state  (std_msgs/String) — gate: only runs during RETURNING
"""

import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from hello_helpers.hello_misc import HelloNode
from std_msgs.msg import Bool, String
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult


# ---------------------------------------------------------------------------
# Table pose  (map frame)
# ---------------------------------------------------------------------------

TABLE_X  = -0.15255
TABLE_Y  = -1.38
TABLE_QZ =  1.0
TABLE_QW =  0.63644

# ---------------------------------------------------------------------------
# Release arm configuration
# Measured from robot with object in hand at table height
# ---------------------------------------------------------------------------

RELEASE_POSE = {
    'joint_lift':        0.8,
    'wrist_extension':   0.3,
    'joint_wrist_yaw':   0.03004045709609381,
    'joint_wrist_pitch': -0.06749515466696822,
    'joint_wrist_roll':  0.035281558121369745,
}

# Gripper open aperture — wide enough to cleanly release the object
GRIPPER_OPEN  =  0.2
# Gripper closed aperture — matches execute_grasp() in grasp_objects.py
GRIPPER_CLOSED = -0.15

# Navigation timeout
NAV_TIMEOUT_SEC = 120.0
POLL_PERIOD_SEC  = 0.2


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ReturnToTable(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)
        self.mission_state  = ''
        self.return_active  = False

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _mission_state_callback(self, msg: String):
        prev               = self.mission_state
        self.mission_state = msg.data

        if prev != 'RETURNING' and msg.data == 'RETURNING':
            if self.return_active:
                self.get_logger().warn('Return already running, ignoring re-trigger.')
                return
            self.get_logger().info('State → RETURNING. Starting return sequence.')
            threading.Thread(target=self._run_return, daemon=True).start()

    # ------------------------------------------------------------------
    # Return sequence  (runs in a dedicated thread — blocking is fine)
    # ------------------------------------------------------------------

    def _run_return(self):
        self.return_active = True
        try:
            self.switch_to_navigation_mode()
            # 1. Navigate to table
            success = self._navigate_to_table()
            if not success:
                self.get_logger().error('Navigation to table failed.')
                self.return_result_pub.publish(Bool(data=False))
                return

            # Small settle pause after navigation before moving the arm
            time.sleep(0.8)

            # Must be in position mode for individual joint commands to work
            self.switch_to_position_mode()

              # move mast up before releaing
            self.move_to_pose({'joint_lift': 1.0}, blocking=True)

            self.get_logger().info('Extending arm.')
            self.move_to_pose(
                {'wrist_extension': RELEASE_POSE['wrist_extension']},
                blocking=True,
            )

            # 2. Move arm to release configuration — order matters:
            #    (a) lift mast to table height first
            #    (b) extend arm out over the table
            #    (c) set wrist angles one at a time
            self.get_logger().info('Lifting mast.')
            self.move_to_pose(
                {'joint_lift': RELEASE_POSE['joint_lift']},
                blocking=True,
            )

            self.get_logger().info('Setting wrist yaw.')
            self.move_to_pose(
                {'joint_wrist_yaw': RELEASE_POSE['joint_wrist_yaw']},
                blocking=True,
            )

            self.get_logger().info('Setting wrist pitch.')
            self.move_to_pose(
                {'joint_wrist_pitch': RELEASE_POSE['joint_wrist_pitch']},
                blocking=True,
            )

            self.get_logger().info('Setting wrist roll.')
            self.move_to_pose(
                {'joint_wrist_roll': RELEASE_POSE['joint_wrist_roll']},
                blocking=True,
            )

            # 3. Open gripper — object drops onto table surface
            self.get_logger().info('Opening gripper — releasing object.')
            self.move_to_pose({'gripper_aperture': GRIPPER_OPEN}, blocking=True)

            
            # move mast up before stowing
            self.move_to_pose({'joint_lift': 1.0}, blocking=True)

            # Brief pause so object settles before we retract
            time.sleep(1.5)

            # 4. Stow arm back to safe travel pose
            self.get_logger().info('Stowing arm.')
            self.stow_the_robot()

            # 5. Notify mission_manager — triggers RETURNING → IDLE
            self.get_logger().info('Return complete.')
            self.return_result_pub.publish(Bool(data=True))

        finally:
            self.return_active = False

    # ------------------------------------------------------------------
    # Nav2 navigation  (blocking — called from return thread)
    # ------------------------------------------------------------------

    def _navigate_to_table(self) -> bool:
        goal = PoseStamped()
        goal.header.frame_id        = 'map'
        goal.header.stamp           = self.get_clock().now().to_msg()
        goal.pose.position.x        = TABLE_X
        goal.pose.position.y        = TABLE_Y
        goal.pose.position.z        = 0.0
        goal.pose.orientation.z     = TABLE_QZ
        goal.pose.orientation.w     = TABLE_QW

        self.get_logger().info(
            f'Navigating to table: x={TABLE_X}, y={TABLE_Y}')
        self.navigator.goToPose(goal)

        nav_start = self.get_clock().now()

        while not self.navigator.isTaskComplete():
            time.sleep(POLL_PERIOD_SEC)

            elapsed = (self.get_clock().now() - nav_start).nanoseconds / 1e9
            if elapsed > NAV_TIMEOUT_SEC:
                self.get_logger().error('Navigation timeout — cancelling.')
                self.navigator.cancelTask()
                return False

            # Abort if mission state changes unexpectedly
            if self.mission_state not in ('RETURNING',):
                self.get_logger().warn(
                    f'Mission state changed to {self.mission_state} '
                    f'mid-navigation — cancelling.')
                self.navigator.cancelTask()
                return False

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Reached table position.')
            return True
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('Navigation was cancelled.')
        elif result == TaskResult.FAILED:
            self.get_logger().error('Navigation failed.')
        return False

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def main(self):
        HelloNode.main(
            self,
            node_name='return_to_table',
            node_topic_namespace='return_to_table',
            wait_for_first_pointcloud=False,
        )

        # Nav2 navigator
        self.navigator = BasicNavigator()
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Nav2 active.')

        # Subscribers
        self.create_subscription(
            String, '/mission/state',
            self._mission_state_callback, 10)

        # Publisher — result back to mission_manager
        self.return_result_pub = self.create_publisher(
            Bool, '/return/result', 10)

        self.get_logger().info('Return to table node ready.')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    node = ReturnToTable()
    node.main()
    node.new_thread.join()