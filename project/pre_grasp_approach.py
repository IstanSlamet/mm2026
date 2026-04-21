#!/usr/bin/env python3
"""
Pre-grasp approach node for Stretch 3.

Subscribes to /gripper_detector/goal_pose published by gripper_object_detector.py,
drives the robot base to 15 cm in front of the detected object, then rotates 90
degrees counter-clockwise to align the arm with the object for grasping.

How to run:
  # Terminal 1
  ros2 launch stretch_core stretch_driver.launch.py
  # Terminal 2  (D405 gripper camera)
  ros2 launch stretch_core d405_basic.launch.py
  # Terminal 3
  python3 gripper_object_detector.py
  # Terminal 4
  python3 pre_grasp_approach.py
"""

import math
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from hello_helpers.hello_misc import HelloNode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import tf2_ros
import tf2_geometry_msgs
import numpy as np

APPROACH_DISTANCE = 0.9   # meters — desired standoff from object
DISTANCE_THRESHOLD = 0.05  # meters — tolerance to consider "arrived"
ANGLE_THRESHOLD = 0.05     # radians — tolerance to consider heading aligned
MAX_TRANSLATE_STEP = 0.10  # meters — max base translation per step


class PreGraspApproach(HelloNode):

    def __init__(self):
        HelloNode.__init__(self)
        self.target_frame = 'base_link'
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
        self.joint_state = {}
        self.done = False
        self.last_command_time = None

    def joint_states_callback(self, msg):
        with self.joint_states_lock:
            joint_names = ['joint_lift', 'joint_arm_l0', 'joint_wrist_yaw',
                           'joint_wrist_pitch', 'joint_wrist_roll']
            for name in joint_names:
                if name in msg.name:
                    i = msg.name.index(name)
                    self.joint_state[name] = msg.position[i]

    def get_goal_in_base_frame(self, goal_msg):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                goal_msg.header.frame_id,
                goal_msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            pose_transformed = tf2_geometry_msgs.do_transform_pose(goal_msg.pose, transform)
            result = PoseStamped()
            result.header.frame_id = self.target_frame
            result.pose = pose_transformed
            return result
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF Error: {e}')
            return None

    def goal_callback(self, goal_msg):
        if self.done:
            return

        # Rate-limit to avoid flooding the robot with commands while it is still moving
        current_time = self.get_clock().now()
        if self.last_command_time is not None and \
                (current_time - self.last_command_time).nanoseconds < 1.0e9:
            return

        # Use camera-frame depth directly — goal_msg.pose is in the gripper
        # camera optical frame, so position.z is the physical camera-to-object
        # distance. Base_link Euclidean distance never converges because the
        # camera is already ~0.3-0.4 m ahead of the base.
        depth = goal_msg.pose.position.z

        # Still need base_link for the heading angle so the base drives toward the object.
        goal_in_base = self.get_goal_in_base_frame(goal_msg)
        if goal_in_base is None:
            self.get_logger().warn('Waiting for TF tree...', throttle_duration_sec=1.0)
            return

        obj_x = goal_in_base.pose.position.x
        obj_y = goal_in_base.pose.position.y
        angle = math.atan2(obj_y, obj_x)  # angle from robot's +x to the object

        self.get_logger().info(
            f'Camera depth: {depth:.3f} m  |  heading error: {math.degrees(angle):.1f} deg'
        )

        if depth <= APPROACH_DISTANCE + DISTANCE_THRESHOLD:
            # Set done immediately — prevents concurrent callbacks (ReentrantCallbackGroup)
            # from issuing conflicting commands while the blocking moves below run.
            self.done = True
            self.get_logger().info('At approach distance. Rotating 90° CCW to align arm...')
            self.move_to_pose({'rotate_mobile_base': math.pi / 2}, blocking=True)

            # self.get_logger().info('Raising mast and pointing wrist camera down...')
            # self.move_to_pose({
            #     'joint_lift': 1.0,          # raise arm up the mast
            #     'joint_wrist_pitch': -0.9,  # tilt gripper/camera to face down
            #     'wrist_extension': 0.1,     # small extension to clear the robot body
            # }, blocking=True)

            self.get_logger().info('Pre-grasp approach complete!')
            self.done_pub.publish(Bool(data=True))
            return

        # Still need to close the gap — first align heading, then drive forward
        if abs(angle) > ANGLE_THRESHOLD:
            self.get_logger().info(f'Aligning heading with object: {math.degrees(angle):.1f} deg')
            self.last_command_time = self.get_clock().now()
            self.move_to_pose({'rotate_mobile_base': angle}, blocking=True)
        else:
            drive = min(depth - APPROACH_DISTANCE, MAX_TRANSLATE_STEP)
            self.get_logger().info(f'Driving toward object: {drive:.3f} m')
            self.last_command_time = self.get_clock().now()
            self.move_to_pose({'translate_mobile_base': drive}, blocking=True)

    def main(self):
        HelloNode.main(self, 'pre_grasp_approach', 'pre_grasp_approach',
                       wait_for_first_pointcloud=False)
        self.callback_group = ReentrantCallbackGroup()
        self.switch_to_position_mode()

        # Low + forward-facing so camera looks at the floor ahead during approach.
        # Arm raises and tilts down only once APPROACH_DISTANCE is reached.
        READY_POSE = {
            'joint_lift': 0.15,
            'wrist_extension': 0.1,
            'joint_wrist_yaw': np.pi/2,
            'joint_wrist_pitch': -np.pi/12,
            'joint_wrist_roll': 0.0,
            'gripper_aperture': 0.5,
        }

        self.switch_to_position_mode()
        self.move_to_pose(READY_POSE, blocking=True)
        self.get_logger().info('At ready pose — waiting for detections on /gripper_detector/goal_pose')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            JointState, '/stretch/joint_states', self.joint_states_callback, 1)

        self.done_pub = self.create_publisher(Bool, '/task/pre_grasp_complete', 10)

        self.create_subscription(
            PoseStamped, '/gripper_detector/goal_pose', self.goal_callback, 10)


if __name__ == '__main__':
    node = PreGraspApproach()
    node.main()
    node.new_thread.join()
