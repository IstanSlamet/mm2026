import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from hello_helpers.hello_misc import HelloNode
import threading
import tf2_ros
from tf2_geometry_msgs import TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import ik_ros_utils as ik
import ikpy

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.10 # m
        self.target_frame = 'base_link'
        self.gripper_frame = 'link_grasp_center'
        #------------------
        self.ik_chain = ik.chain
        #------------------
        self.tf_buffer = None
        self.tf_listener = None
        self.joint_states_lock = threading.Lock()
        
        # Temporary
        self.last_command_time = None

        # Grasping
        self.is_grasping = False
    
    def joint_states_callback(self, msg):
        # unpacks joint state messages for what works with/is expected by ikpy
        with self.joint_states_lock:
            joint_states = msg
        # extract information needed for ik_solver
        joint_names = [
            'joint_lift', 'joint_arm_l0', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll'
        ]
        self.joint_state = {}
        for joint_name in joint_names:
            i = joint_states.name.index(joint_name)
            self.joint_state[joint_name] = joint_states.position[i]

    def get_goal_pose_in_base_frame(self, goal_msg):
        # TODO: ------------- start --------------
        # fill with your response
        #   transform the goal pose to the base frame
        
        try:
        # 1. Get the transform from the base to the camera
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                goal_msg.header.frame_id,
                rclpy.time.Time(),                          # get the latest available transform
                timeout=rclpy.duration.Duration(seconds=0.1)    # Get the latest data
            )
            
            # 2. Apply the transform to the actual coordinates of the object
            import tf2_geometry_msgs
            pose_transformed = tf2_geometry_msgs.do_transform_pose(goal_msg.pose, transform)
            
            # 3. Package it back into a PoseStamped message for the IK helper
            goal_transformed = PoseStamped()
            goal_transformed.header.frame_id = self.target_frame
            goal_transformed.pose = pose_transformed
            return goal_transformed
            
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Goal Pose TF Error: {e}")
            return None

        # TODO: -------------- end ---------------

    def get_gripper_pose_in_base_frame(self):
        # TODO: ------------- start --------------
        # fill with your response
        #   transform the gripper pose to the base frame
        try:
            t = gripper_transformed = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.gripper_frame,
                rclpy.time.Time()
            )
            # Convert TransformStamped to PoseStamped
            p = PoseStamped()
            p.header = t.header
            p.pose.position.x = t.transform.translation.x
            p.pose.position.y = t.transform.translation.y
            p.pose.position.z = t.transform.translation.z
            p.pose.orientation = t.transform.rotation
            return p # PoseStamped          
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Gripper TF Error: {e}")
        # TODO: -------------- end ---------------

        return gripper_transformed

    def goal_callback(self, goal_msg):
        print(f"[goal_callback] Received goal. is_grasping={self.is_grasping}")
        if self.is_grasping is True:
              return

        try:
            goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
            if goal_transformed is None:
                print("[goal_callback] ERROR: get_goal_pose_in_base_frame returned None")
                return
            gripper_transformed = self.get_gripper_pose_in_base_frame()
            if gripper_transformed is None:
                print("[goal_callback] ERROR: get_gripper_pose_in_base_frame returned None")
                return

            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
            print(f"[goal_callback] goal_pos={np.round(goal_pos, 3)}  gripper_pos={np.round(gripper_pos, 3)}")

            # --------- DEBUG LINES ---------
            goal_transformed.header.stamp = self.get_clock().now().to_msg()
            self.debug_goal_pub.publish(goal_transformed)
            # -----------------------------------------
        except Exception as e:
            print(f"[goal_callback] Exception getting transforms: {e}")
            return

        # TODO: ------------- start --------------
        # fill with your response
        #   use the same functions you used for IK in Lab 2, now in `ik_ros_utils.py`,
        #   to move the robot to the transformed goal point.

        # Part 2 changes: if the object is close enough to grab, grab it
        distance = np.linalg.norm(goal_pos - gripper_pos)
        print(f"[goal_callback] distance to goal={distance:.4f}  (grasp threshold=0.25)")
        if distance < 0.18:
            print("[goal_callback] Close enough — executing grasp")
            self.is_grasping = True
            self.execute_grasp(goal_pos)
            return

###################

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)
        print(f"[goal_callback] waypoint_pos={np.round(waypoint_pos, 3)}")
        with self.joint_states_lock:
            q_init = ik.get_current_configuration(self.joint_state)

        if q_init is None:
            print("[goal_callback] ERROR: joint_state not yet received — q_init is None")
            return
        print(f"[goal_callback] q_init obtained. Calling IK...")

        q_soln = ik.chain.inverse_kinematics(
                waypoint_pos,
                initial_position=q_init)
        err = np.linalg.norm(ik.chain.forward_kinematics(q_soln)[:3, 3] - waypoint_pos)
        print(f"[goal_callback] IK error={err:.6f}  (tolerance=0.01)")

        if not np.isclose(err, 0.0, atol=1e-2):
            print(f"[goal_callback] IK FAILED — no valid solution (err={err:.4f})")
            q_soln = None
        else:
            print("[goal_callback] IK succeeded")
        ik.print_q(q_soln)


# # adding time gate 
#         if q_soln is not None:
#             current_time = self.get_clock().now()
#             if self.last_command_time is None or \
#                (current_time - self.last_command_time).nanoseconds > 0.4 * 1e9:
#                 print("[goal_callback] Calling move_to_configuration...")
#                 ik.move_to_configuration(self, q_soln)
#                 self.last_command_time = current_time
#                 print("[goal_callback] move_to_configuration returned")
#             else:
#                 print("[goal_callback] Skipping — too soon since last command")
#         return




        if q_soln is not None:
            print("[goal_callback] Calling move_to_configuration...")
            ik.move_to_configuration(self, q_soln)
            print("[goal_callback] move_to_configuration returned")
        return

#########################




        
        # waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)
        # with self.joint_states_lock:
        #     q_init = ik.get_current_configuration(self.joint_state)
        # for i, link in enumerate(self.ik_chain.links):
        #     if link.joint_type != "fixed":
        #         val = q_init[i]
        #         low, high = link.bounds
        #         if val < low or val > high:
        #             print(f"!!! JOINT OUT OF BOUNDS: {link.name} !!!")
        #             print(f"Value: {val}, Limits: [{low}, {high}]")
    
        # q_soln = ik.get_grasp_goal(waypoint_pos, waypoint_orient, q_init)






        # TODO: -------------- end ---------------

        # NOTE: if you find that the robot's base is moving too much, its likely that the ik solver is
        # struggling to find solutions without the base doing most of the work to achieve the waypoint pose.
        # you can adjust the `self.delta` variable to be smaller so that the displacements are smaller, and
        #   there is a valid solution without excessive base movement
        # you can also set your own triggers manually (keep delta large but use an if/else on move_to_pose()
        #   so the base only moves above a certain distance threshold
        # you can also try adjusting joint limits of the base trans/rot in `ik_ros_utils.py` to be much smaller
        # one or some combination of these should help!

        ik.print_q(q_soln)
        if q_soln is not None:
            current_time = self.get_clock().now()
            
            if self.last_command_time is None or (current_time - self.last_command_time).nanoseconds > 0.5 * 1e9:
                ik.move_to_configuration(self, q_soln)
                self.last_command_time = current_time
        

    def execute_grasp(self, goal_pos):
          
        # TODO: New function to move arm to object position and grasp the object 
        
        # Open gripper
        self.move_to_pose({'gripper_aperture': 0.2}, blocking=True)

        # Reach for the object
        
        with self.joint_states_lock:
            q_init = ik.get_current_configuration(self.joint_state)

        # Use current FK orientation (wrist already pointing down from grasp-start pose)
        waypoint_orient = ik.chain.forward_kinematics(q_init)[:3, :3]

        q_soln = ik.get_grasp_goal(goal_pos, waypoint_orient, q_init)

        if q_soln is not None:
            # Move arm to the object
            ik.move_to_configuration(self, q_soln)

            # Close the gripper
            self.move_to_pose({'gripper_aperture': -0.15}, blocking=True)

            # Lift up the object
            with self.joint_states_lock:
                current_lift = self.joint_state['joint_lift']

            self.move_to_pose({'joint_lift': current_lift + 0.15}, blocking=True)

            # Set travel pose: lift at 0.8 m, arm fully retracted so the
            # robot can drive safely back to the home location.
            self.move_to_pose({
                'wrist_extension': 0.0,
                'joint_lift': 0.8,
                'joint_wrist_pitch': 0.0,
            }, blocking=True)

            self.grasp_done_pub.publish(Bool(data=True))
        else:
            self.is_grasping = False


    def compute_waypoint_to_goal(self, goal_pos, gripper_pos):

        # TODO: ------------- start --------------
        # fill with your response
        #   find the distance between the published goal position and the gripper position
        #   if its above some threshold (delta), consider the goal to be too far (since we're trying to track the object
        #   at least 2Hz) to reach before the next goal is published
        #   in this case, find a waypoint toward the goal position that is delta away from the gripper position (make some progress towards the goal)
        #   otherwise, the goal is close and we can move there directly
        goal_xyz = goal_pos[:3]
        gripper_xyz = gripper_pos[:3]
        distance = np.linalg.norm(goal_xyz - gripper_xyz)
        if distance > self.delta:
            goal_unit_vector = (goal_xyz - gripper_xyz) / distance
            waypoint_pos = gripper_xyz + self.delta * goal_unit_vector
        else:
            waypoint_pos = goal_xyz
        # Get the current joint configuration
        with self.joint_states_lock:
            q_init = ik.get_current_configuration(self.joint_state)
            
        # Use Forward Kinematics to extract the CURRENT rotation matrix
        # This guarantees the wrist stays exactly where it is!
        current_fk = self.ik_chain.forward_kinematics(q_init)
        waypoint_orient = current_fk[:3, :3]

        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        #waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, -0.1, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        # TODO: minor - uncomment the correct ready pose for part 1 or 2!
        #   part 1: 
        self.switch_to_position_mode()
        #self.move_to_pose(ik.READY_POSE_P1, blocking=True)
        
        #   part 2: READY_POSE_P2
        self.move_to_pose(ik.READY_POSE_P2, blocking=True)
        self.switch_to_position_mode()

    def main(self):
        HelloNode.main(self, 'follow_target', 'follow_target', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.joint_states_subscriber = self.create_subscription(JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1)

        # Do NOT stow — pre_grasp_approach already positioned the robot with
        # lift high and camera looking down.  Just ensure position mode and
        # confirm the grasp-start pose: lift at max, camera facing down,
        # arm slightly extended so the gripper camera sees the object clearly.
        self.switch_to_position_mode()
        # D405 range is 70–500 mm. At lift=1.0 m the floor is ~1 m away — out of range.
        # Keep lift at ~0.4 m and extend arm so the camera is ~400 mm above the ball.
        self.move_to_pose({
            'joint_lift':        0.3}, blocking =True)
        self.move_to_pose({
            # 'joint_lift':        0.3,
            'wrist_extension':   0.1,   
            'joint_wrist_yaw':   0.0,
            'joint_wrist_pitch': -np.pi/4,
            'joint_wrist_roll': 0.0,
            'gripper_aperture':  0.5,
        }, blocking=True)
        print("At grasp-start pose")
        print("[main] Setting up subscribers and waiting for goal poses on /gripper_detector/goal_pose ...")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/gripper_detector/goal_pose',
            callback=self.goal_callback,
            qos_profile=10,
        )

        self.debug_goal_pub = self.create_publisher(
            PoseStamped, '/debug/transformed_goal_pose', 10)

        self.grasp_done_pub = self.create_publisher(Bool, '/task/grasp_complete', 10)






if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()
