#!/usr/bin/env python3
"""
Set patrol pose: lift low (20 cm), gripper facing forward.
Run once synchronously before launching patrol.py and gripper_object_detector.py
so the D405 camera looks at the floor ahead of the robot during patrol.
"""

import rclpy
from hello_helpers.hello_misc import HelloNode
import numpy as np


PATROL_POSE = {
    'joint_lift': 0.2,
    'wrist_extension': 0.1,
    'joint_wrist_yaw': np.pi/2,
    'joint_wrist_pitch': 0.0,
    'joint_wrist_roll': 0.0,
    'gripper_aperture': 0.5,
}

class PatrolPoseSetter(HelloNode):
    def main(self):
        HelloNode.main(self, 'patrol_pose_setter', 'patrol_pose_setter',
                       wait_for_first_pointcloud=False)
        self.switch_to_position_mode()
        self.move_to_pose(PATROL_POSE, blocking=True)
        self.get_logger().info('Patrol pose set.')
        rclpy.shutdown()


if __name__ == '__main__':
    node = PatrolPoseSetter()
    node.main()
    node.new_thread.join()
