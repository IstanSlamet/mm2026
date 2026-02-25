import ikpy.urdf.utils
import urchin as urdfpy
import numpy as np
import ikpy.chain
import importlib.resources as importlib_resources
import hello_helpers.hello_misc as hm
from std_srvs.srv import Trigger

# NOTE before running: `python3 -m pip install --upgrade ikpy graphviz urchin networkx`


class MyNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        self.main('my_ik_node', 'my_ik_node', wait_for_first_pointcloud=False)

    def move_to_config(self, q):
        q_base_rot = q[1]
        q_base = q[2]
        q_lift = q[4]
        q_arm = q[6] + q[7] + q[8] + q[9]
        q_yaw = q[10]
        q_pitch = q[12]
        q_roll = q[13]
        self.move_to_pose({'rotate_mobile_base': q_base_rot}, blocking=True)
        self.move_to_pose({'joint_lift':q_lift, 'joint_arm':q_arm}, blocking=True)
        self.move_to_pose({'joint_wrist_yaw':q_yaw, 'joint_wrist_roll':q_roll, 'joint_wrist_pitch':q_pitch}, blocking=True)
        self.move_to_pose({'translate_mobile_base':q_base}, blocking=True)
        
    # Get the current joint state and bound them
    def get_current_configuration_ros(self):
        if self.joint_state is None:
            return None
        
        # Helper function to grab the joint position by joint name
        def get_pos(name):
            try:
                idx = self.joint_state.name.index(name)
                return self.joint_state.position[idx]
            except ValueError:
                return 0.0
            
        # 1. Get raw values from the JointState message
        # bounding is handled by ROS2 internally
        q_lift = get_pos('joint_lift')
        q_arm = get_pos('joint_arm') # Total arm extension
        q_yaw = get_pos('joint_wrist_yaw')
        q_pitch = get_pos('joint_wrist_pitch')
        q_roll = get_pos('joint_wrist_roll')


        # To get the 4-segment arm, dividing total extension by 4.0
        q_arml = q_arm / 4.0


        # Return the array in the specific order
        q = [
            0.0,       # 0. base_link (Fixed)
            0.0,       # 1. joint_base_rotation (Revolute)
            0.0,       # 2. joint_base_translation (Prismatic)
            0.0,       # 3. joint_mast (Fixed)
            q_lift,    # 4. joint_lift (Prismatic)
            0.0,       # 5. joint_arm_l4 (Fixed)
            q_arml,    # 6. joint_arm_l3 (Prismatic)
            q_arml,    # 7. joint_arm_l2 (Prismatic)
            q_arml,    # 8. joint_arm_l1 (Prismatic)
            q_arml,    # 9. joint_arm_l0 (Prismatic)
            q_yaw,     # 10. joint_wrist_yaw (Revolute)
            0.0,       # 11. joint_wrist_yaw_bottom (Fixed)
            q_pitch,   # 12. joint_wrist_pitch (Revolute)
            q_roll,    # 13. joint_wrist_roll (Revolute)
            0.0,       # 14. joint_gripper_s3_body (Fixed)
            0.0        # 15. joint_grasp_center (Fixed)
        ]
        return q


target_point_1 = [1.5, 0.2, 0.40] #[-0.043, -0.441, 0.654]
target_orientation_1 = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, np.pi/4) # [roll, pitch, yaw]

target_point_2 = [0.5, 0.2, 0.3] #[-0.043, -0.441, 0.654]
target_orientation_2 = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/4) # [roll, pitch, yaw]

target_point_3 = [0.5, 0.2, 0.4] #[-0.043, -0.441, 0.654]
target_orientation_3 = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, np.pi/2) # [roll, pitch, yaw]

target_point_4 = [-0.5, 0.1, 0.2] #[-0.043, -0.441, 0.654]
target_orientation_4 = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, -np.pi/2) # [roll, pitch, yaw]

pkg_path = str(importlib_resources.files('stretch_urdf'))
urdf_file_path = pkg_path + '/SE3/stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf'

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

# Add virtual base rotation joint
joint_base_rotation = urdfpy.Joint(name='joint_base_rotation',
                                      parent='base_link',
                                      child='link_base_rotation',
                                      joint_type='revolute',
                                      axis=np.array([0.0, 0.0, 1.0]),
                                      origin=np.eye(4, dtype=np.float64),
                                      limit=urdfpy.JointLimit(effort=100.0, velocity=1.0, lower=-1.0, upper=1.0))
modified_urdf._joints.append(joint_base_rotation)
link_base_rotation = urdfpy.Link(name='link_base_rotation',
                                    inertial=None,
                                    visuals=None,
                                    collisions=None)
modified_urdf._links.append(link_base_rotation)

# Add virtual base joint
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

chain = ikpy.chain.Chain.from_urdf_file(new_urdf_path)

# Bounding the links
indices_to_freeze = [0, 3, 5, 11, 14, 15]
for i in indices_to_freeze:
    chain.links[i].bounds = (-1e-3, 1e-3)
chain.links[4].bounds = (0.0, 1.1)
for i in range(6, 10):
    chain.links[i].bounds = (0.0, 0.13)


for link in chain.links:
    print(f"* Link Name: {link.name}, Type: {link.joint_type}")

node = MyNode()
def move_to_grasp_goal(target_point, target_orientation):
    q_init = node.get_current_configuration_ros()
    for i in range(6, 10):
        q_init[i] = 0.025

    q_soln = chain.inverse_kinematics(
        target_point,
        target_orientation,
        orientation_mode='all',
        initial_position=q_init
        )
    print('Solution:', q_soln)

    final_pose = chain.forward_kinematics(q_soln)
    err = np.linalg.norm(final_pose[:3, 3] - target_point)
    print(f"Distance Error: {err:.4f} m")
    if not np.isclose(err, 0.0, atol=1e-2):
        print("IKPy did not find a valid solution")
        return
    node.move_to_config(q=q_soln)
    return q_soln

def get_current_grasp_pose():
    q = node.get_current_configuration_ros()
    return chain.forward_kinematics(q)


# robot.stow()
move_to_grasp_goal(target_point_1, target_orientation_1)
print(get_current_grasp_pose())
move_to_grasp_goal(target_point_2, target_orientation_2)
print(get_current_grasp_pose())
move_to_grasp_goal(target_point_3, target_orientation_3)
print(get_current_grasp_pose())
move_to_grasp_goal(target_point_4, target_orientation_4)