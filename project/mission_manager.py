#!/usr/bin/env python3
"""
Mission Manager for Stretch 3 autonomous object retrieval.

Orchestrates the patrol → pre-grasp pipeline as a state machine,
launching and terminating child processes for each stage.

State machine:
  IDLE       → start immediately
  PATROL     gripper_object_detector + patrol running
               transition trigger: /task/object_pose received
  PRE_GRASP  pre_grasp_approach running
               transition trigger: /task/pre_grasp_complete received
  DONE       all processes terminated, mission complete

Published topics:
  /mission/state  (std_msgs/String) — current state name, for monitoring

Subscribed topics:
  /task/object_pose        (geometry_msgs/PoseStamped) — published by patrol.py
  /task/pre_grasp_complete (std_msgs/Bool)             — published by pre_grasp_approach.py

Requires (running before this script):
  ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
  ros2 launch stretch_core d405_basic.launch.py
  ros2 launch stretch_nav2 navigation.launch.py use_sim_time:=False map:=/path/to/map.yaml

Usage:
  python3 mission_manager.py
"""

import os
import subprocess
import sys
import threading
import time
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable


class MissionState(Enum):
    IDLE        = auto()
    PATROL      = auto()
    PRE_GRASP   = auto()
    GRASP       = auto()
    RETURN_HOME = auto()
    DONE        = auto()


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        self.state        = MissionState.IDLE
        self._processes: dict[str, subprocess.Popen] = {}
        self._transitioning = False   # guard against concurrent transitions

        # Flags set by ROS callbacks, consumed by the state machine tick
        self._object_found      = False
        self._pre_grasp_done    = False
        self._grasp_done        = False
        self._return_home_done  = False

        # --- subscribers (stage-completion signals) ---
        self.create_subscription(
            PoseStamped, '/task/object_pose',
            self._object_pose_callback, 10)
        self.create_subscription(
            Bool, '/task/pre_grasp_complete',
            self._pre_grasp_done_callback, 10)
        self.create_subscription(
            Bool, '/task/grasp_complete',
            self._grasp_done_callback, 10)
        self.create_subscription(
            Bool, '/task/return_home_complete',
            self._return_home_done_callback, 10)

        # --- publisher: current state for external monitoring ---
        self.state_pub = self.create_publisher(String, '/mission/state', 10)

        # State machine runs every 0.5 s; transitions happen in background threads
        # so the timer callback never blocks the ROS executor.
        self.create_timer(0.5, self._tick)

        self.get_logger().info('Mission Manager initialised — starting in 1 s.')

    # ------------------------------------------------------------------
    # ROS callbacks — only set flags, no heavy work
    # ------------------------------------------------------------------

    def _object_pose_callback(self, _msg: PoseStamped):
        if self.state == MissionState.PATROL:
            self.get_logger().info('Object pose received.')
            self._object_found = True

    def _pre_grasp_done_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.PRE_GRASP:
            self.get_logger().info('Pre-grasp complete signal received.')
            self._pre_grasp_done = True

    def _grasp_done_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.GRASP:
            self.get_logger().info('Grasp complete signal received.')
            self._grasp_done = True

    def _return_home_done_callback(self, msg: Bool):
        if msg.data and self.state == MissionState.RETURN_HOME:
            self.get_logger().info('Return home complete signal received.')
            self._return_home_done = True

    # ------------------------------------------------------------------
    # State machine tick  (runs every 0.5 s in the ROS timer thread)
    # ------------------------------------------------------------------

    def _tick(self):
        self.state_pub.publish(String(data=self.state.name))

        if self._transitioning:
            return  # a transition is already in progress

        if self.state == MissionState.IDLE:
            self._transition(self._enter_patrol)

        elif self.state == MissionState.PATROL and self._object_found:
            self._object_found = False
            self._transition(self._enter_pre_grasp)

        elif self.state == MissionState.PRE_GRASP and self._pre_grasp_done:
            self._pre_grasp_done = False
            self._transition(self._enter_grasp)

        elif self.state == MissionState.GRASP and self._grasp_done:
            self._grasp_done = False
            self._transition(self._enter_return_home)

        elif self.state == MissionState.RETURN_HOME and self._return_home_done:
            self._return_home_done = False
            self._transition(self._enter_done)

    def _transition(self, fn):
        """Run a state-transition function in a daemon thread so the timer
        callback returns immediately and the ROS executor stays unblocked."""
        self._transitioning = True
        threading.Thread(target=self._run_transition, args=(fn,), daemon=True).start()

    def _run_transition(self, fn):
        try:
            fn()
        finally:
            self._transitioning = False

    # ------------------------------------------------------------------
    # State entry functions  (run in background threads — blocking is fine)
    # ------------------------------------------------------------------

    def _enter_patrol(self):
        self.get_logger().info('=== STATE: PATROL ===')
        self.state = MissionState.PATROL

        # Start detector first, give it 2 s to load the YOLO model before
        # patrol.py begins following waypoints and checking for detections.
        self._launch('gripper_detector', 'gripper_object_detector.py')
        time.sleep(2.0)
        self._launch('patrol', 'patrol.py')

    def _enter_pre_grasp(self):
        self.get_logger().info('=== STATE: PRE_GRASP ===')
        self.state = MissionState.PRE_GRASP

        # Only kill patrol — gripper_object_detector keeps running because
        # pre_grasp_approach.py subscribes to /gripper_detector/goal_pose.
        self._kill('patrol')
        time.sleep(1.0)   # brief pause so ports / topics settle

        self._launch('pre_grasp', 'pre_grasp_approach.py')

    def _enter_grasp(self):
        self.get_logger().info('=== STATE: GRASP ===')
        self.state = MissionState.GRASP

        self._kill('pre_grasp')
        time.sleep(1.0)

        self._launch('grasp', 'grasp_objects.py')

    def _enter_return_home(self):
        self.get_logger().info('=== STATE: RETURN_HOME ===')
        self.state = MissionState.RETURN_HOME

        # grasp_objects.py already set the travel pose (lift 0.8 m, arm retracted).
        # Kill the grasp node and detector, then navigate back.
        self._kill('grasp')
        self._kill('gripper_detector')
        time.sleep(1.0)

        self._launch('return_home', 'return_home.py')

    def _enter_done(self):
        self.get_logger().info('=== STATE: DONE — Mission complete ===')
        self.state = MissionState.DONE
        self._kill_all()

    # ------------------------------------------------------------------
    # Process helpers
    # ------------------------------------------------------------------

    def _launch(self, name: str, script: str):
        path = os.path.join(PROJECT_DIR, script)
        self.get_logger().info(f'Launching {script}')
        self._processes[name] = subprocess.Popen(
            [PYTHON, path],
            cwd=PROJECT_DIR,
        )

    def _kill(self, name: str):
        proc = self._processes.pop(name, None)
        if proc is None or proc.poll() is not None:
            return
        self.get_logger().info(f'Stopping {name}')
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _kill_all(self):
        for name in list(self._processes):
            self._kill(name)

    def destroy_node(self):
        self._kill_all()
        super().destroy_node()


# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
