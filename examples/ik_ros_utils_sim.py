import os
import urchin as urdfpy
import numpy as np
import ikpy.chain

class IK:
    def __init__(self, m, env, robot):
        self.m = m
        self.env = env
        self.robot = robot
        new_urdf_path = self.get_modified_urdf()
        self.chain = ikpy.chain.Chain.from_urdf_file(new_urdf_path)
        self.link_names = [l.name for l in self.chain.links]

        for link in self.chain.links:
            print(f"* Link Name: {link.name}, Type: {link.joint_type}")

    def get_modified_urdf(self):
        urdf_file_path = os.path.join(self.env.directory, 'stretch3', 'stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf')

        # Remove unnecessary links/joints
        original_urdf = urdfpy.URDF.load(urdf_file_path)
        modified_urdf = original_urdf.copy()

        names_of_links_to_remove = ['link_right_wheel', 'link_left_wheel', 'caster_link', 'link_head', 'link_head_pan', 'link_head_tilt', 'link_aruco_right_base', 'link_aruco_left_base', 'link_aruco_shoulder', 'link_aruco_top_wrist', 'link_aruco_inner_wrist', 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_depth_optical_frame', 'camera_infra1_frame', 'camera_infra1_optical_frame', 'camera_infra2_frame', 'camera_infra2_optical_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_accel_frame', 'camera_accel_optical_frame', 'camera_gyro_frame', 'camera_gyro_optical_frame', 'gripper_camera_bottom_screw_frame', 'gripper_camera_link', 'gripper_camera_depth_frame', 'gripper_camera_depth_optical_frame', 'gripper_camera_infra1_frame', 'gripper_camera_infra1_optical_frame', 'gripper_camera_infra2_frame', 'gripper_camera_infra2_optical_frame', 'gripper_camera_color_frame', 'gripper_camera_color_optical_frame', 'laser', 'base_imu', 'respeaker_base', 'link_wrist_quick_connect', 'link_gripper_finger_right', 'link_gripper_fingertip_right', 'link_aruco_fingertip_right', 'link_gripper_finger_left', 'link_gripper_fingertip_left', 'link_aruco_fingertip_left', 'link_aruco_d405', 'link_head_nav_cam']
        # links_kept = ['base_link', 'link_mast', 'link_lift', 'link_arm_l4', 'link_arm_l3', 'link_arm_l2', 'link_arm_l1', 'link_arm_l0', 'link_wrist_yaw', 'link_wrist_yaw_bottom', 'link_wrist_pitch', 'link_wrist_roll', 'link_gripper_s3_body', 'link_grasp_center']
        links_to_remove = [l for l in modified_urdf._links if l.name in names_of_links_to_remove]
        for lr in links_to_remove:
            modified_urdf._links.remove(lr)
        names_of_joints_to_remove = ['joint_right_wheel', 'joint_left_wheel', 'caster_joint', 'joint_head', 'joint_head_pan', 'joint_head_tilt', 'joint_aruco_right_base', 'joint_aruco_left_base', 'joint_aruco_shoulder', 'joint_aruco_top_wrist', 'joint_aruco_inner_wrist', 'camera_joint', 'camera_link_joint', 'camera_depth_joint', 'camera_depth_optical_joint', 'camera_infra1_joint', 'camera_infra1_optical_joint', 'camera_infra2_joint', 'camera_infra2_optical_joint', 'camera_color_joint', 'camera_color_optical_joint', 'camera_accel_joint', 'camera_accel_optical_joint', 'camera_gyro_joint', 'camera_gyro_optical_joint', 'gripper_camera_joint', 'gripper_camera_link_joint', 'gripper_camera_depth_joint', 'gripper_camera_depth_optical_joint', 'gripper_camera_infra1_joint', 'gripper_camera_infra1_optical_joint', 'gripper_camera_infra2_joint', 'gripper_camera_infra2_optical_joint', 'gripper_camera_color_joint', 'gripper_camera_color_optical_joint', 'joint_laser', 'joint_base_imu', 'joint_respeaker', 'joint_wrist_quick_connect', 'joint_gripper_finger_right', 'joint_gripper_fingertip_right', 'joint_aruco_fingertip_right', 'joint_gripper_finger_left', 'joint_gripper_fingertip_left', 'joint_aruco_fingertip_left', 'joint_aruco_d405', 'joint_head_nav_cam']
        # joints_kept = ['joint_mast', 'joint_lift', 'joint_arm_l4', 'joint_arm_l3', 'joint_arm_l2', 'joint_arm_l1', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_yaw_bottom', 'joint_wrist_pitch', 'joint_wrist_roll', 'joint_gripper_s3_body', 'joint_grasp_center']
        joints_to_remove = [l for l in modified_urdf._joints if l.name in names_of_joints_to_remove]
        for jr in joints_to_remove:
            modified_urdf._joints.remove(jr)

        joint_base_rotation = urdfpy.Joint(name='joint_base_rotation',
                                        parent='base_link',
                                        child='link_base_rotation',
                                        joint_type='revolute',
                                        axis=np.array([0.0, 0.0, 1.0]),
                                        origin=np.eye(4, dtype=np.float64),
                                        limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-np.pi, upper=np.pi))
        modified_urdf._joints.append(joint_base_rotation)
        link_base_rotation = urdfpy.Link(name='link_base_rotation',
                                        inertial=None,
                                        visuals=None,
                                        collisions=None)
        modified_urdf._links.append(link_base_rotation)

        joint_base_translation = urdfpy.Joint(name='joint_base_translation',
                                            parent='link_base_rotation',
                                            child='link_base_translation',
                                            joint_type='prismatic',
                                            axis=np.array([1.0, 0.0, 0.0]),
                                            origin=np.eye(4, dtype=np.float64),
                                            limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
        modified_urdf._joints.append(joint_base_translation)
        link_base_translation = urdfpy.Link(name='link_base_translation',
                                            inertial=None,
                                            visuals=None,
                                            collisions=None)
        modified_urdf._links.append(link_base_translation)
        # amend the chain
        for j in modified_urdf._joints:
            if j.name == 'joint_mast':
                j.parent = 'link_base_translation'

        new_urdf_path = "/tmp/iktutorial/stretch.urdf"
        modified_urdf.save(new_urdf_path)
        return new_urdf_path

    def bound_range(self, name, value):
        index = self.link_names.index(name)
        bounds = self.chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1])

    def get_current_configuration(self):
        j = self.robot.get_joint_angles(self.robot.controllable_joints)

        q_base_rotation = 0.0
        q_base_translation = 0.0
        q_lift = self.bound_range('joint_lift', j[2])
        q_arml = self.bound_range('joint_arm_l0', np.sum(j[3:7]) / 4.0)
        q_yaw = self.bound_range('joint_wrist_yaw', j[7])
        q_pitch = self.bound_range('joint_wrist_pitch', j[8])
        q_roll = self.bound_range('joint_wrist_roll', j[9])

        q = [0.0, q_base_rotation, q_base_translation, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]

        return q

    def get_current_grasp_pose(self):
        q = self.get_current_configuration()
        transform_matrix = self.chain.forward_kinematics(q)
        translation, rot_matrix = ikpy.utils.geometry.from_transformation_matrix(transform_matrix)
        return translation, rot_matrix

    def move_to_grasp_goal(self, target_point, target_orientation, max_rotation_steps=100, max_translation_steps=100):
        q = self.chain.inverse_kinematics(target_point, target_orientation, orientation_mode='X', initial_position=self.get_current_configuration())

        err = np.linalg.norm(self.chain.forward_kinematics(q)[:3, 3] - target_point)
        if not np.isclose(err, 0.0, atol=1e-2):
            print("IKPy did not find a valid solution")
            return

        q_base_rotation = q[1]
        q_base_translation = q[2]
        q_lift = q[4]
        q_arm = q[6] + q[7] + q[8] + q[9]
        q_yaw = q[10]
        q_pitch = q[12]
        q_roll = q[13]

        q = [q_lift, q_arm/4.0, q_arm/4.0, q_arm/4.0, q_arm/4.0, q_yaw, q_pitch, q_roll]

        radius = 0.0508
        wheel_base = 0.36

        rotation_wheel_right = (q_base_rotation * wheel_base / 2) / radius
        rotation_wheel_left = -(q_base_rotation * wheel_base / 2) / radius
        translation_wheel = q_base_translation / radius

        # Perform base rotation
        angles = self.robot.get_joint_angles(self.robot.controllable_joints)
        targets = np.array([angles[0]+rotation_wheel_right, angles[1]+rotation_wheel_left])
        self.control(targets, joints=[0, 1], max_steps=max_rotation_steps)

        # Perform base translation and move all other motors
        angles = self.robot.get_joint_angles(self.robot.controllable_joints)
        self.control([angles[0]+translation_wheel, angles[1]+translation_wheel] + q, max_steps=max_translation_steps)

    def control(self, targets, joints=None, max_steps=100):
        joints = list(range(len(targets))) if joints is None else joints
        self.robot.control(targets, joints=np.array(self.robot.controllable_joints)[joints])
        for i in range(max_steps):
            angles = self.robot.get_joint_angles(self.robot.controllable_joints)
            if np.allclose(targets - angles[joints], 0, atol=1e-3):
                self.robot.control(angles)
                break
            self.m.step_simulation(steps=1, realtime=True)

    def print_q(self, q):
        if q is None:
            print('INVALID Q')

        else:
            print("IK Config")
            print("     Base Rotation:", q[1])
            print("     Base Translation:", q[2])
            print("     Lift", q[4])
            print("     Arm", q[6] + q[7] + q[8] + q[9])
            print("     Gripper Yaw:", q[10])
            print("     Gripper Pitch:", q[12])
            print("     Gripper Roll:", q[13])

