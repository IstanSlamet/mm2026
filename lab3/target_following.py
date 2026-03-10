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
import ik_ros_utils as ik
import ikpy

# Make sure to run:
#   ros2 launch stretch_core stretch_driver.launch.py

class IKTargetFollowing(HelloNode):
    def __init__(self):
        HelloNode.__init__(self)

        self.delta = 0.02 # cm
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
                self.target_frame,                      # Target Frame (where we want to move)
                goal_msg.header.frame_id,
                goal_msg.header.stamp,                          # Source Frame (where the sensor is)
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
            t = self.tf_buffer.lookup_transform(
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
            return None
        # TODO: -------------- end ---------------

        return gripper_transformed

    def goal_callback(self, goal_msg):
        # print(msg)
        # self.get_logger().info(f'Received goal pose: {msg.pose}')
        #test_pose = {'joint_lift': 0.6}
        #self.move_to_pose(test_pose, blocking=False)

        goal_transformed = self.get_goal_pose_in_base_frame(goal_msg)
        gripper_transformed = self.get_gripper_pose_in_base_frame()

        # Check if the TF lookups actually succeeded before proceeding
        if goal_transformed is None or gripper_transformed is None:
            self.get_logger().warn("Waiting for TF  tree to populate..", throttle_duration_sec=1.0)
            return
        #Wrap the extraction in a targeted try/except to catch real bugs
        try:
            goal_pos = ik.get_xyz_from_msg(goal_transformed)
            gripper_pos = ik.get_xyz_from_msg(gripper_transformed)
        except Exception as e:
            self.get_logger().error(f"Failed to extraxtr XYZ: {e}")
            print("Error getting transforms")
            return

        waypoint_pos, waypoint_orient = self.compute_waypoint_to_goal(goal_pos, gripper_pos)

        # TODO: ------------- start --------------
        # fill with your response
        #   use the same functions you used for IK in Lab 2, now in `ik_ros_utils.py`, 
        #   to move the robot to the transformed goal point.
        with self.joint_states_lock:
        	q_init = ik.get_current_configuration(self.joint_state)
        for i, link in enumerate(self.ik_chain.links):
        	if link.joint_type != "fixed":
        		val = q_init[i]
        		low, high = link.bounds
        		if val < low or val > high:
        			print(f"!!! JOINT OUT OF BOUNDS: {link.name} !!!")
        			print(f"Value: {val}, Limits: [{low}, {high}]")
        
        q_soln = ik.get_grasp_goal(waypoint_pos, waypoint_orient, q_init)
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
        
        # Distance we want to maintain
        standoff_distance = 0.025
        error_distance = distance - standoff_distance
        if error_distance > self.delta:
            goal_unit_vector = (goal_xyz - gripper_xyz) / distance
            waypoint_pos = gripper_xyz + self.delta * goal_unit_vector
        elif error_distance > 0:
            goal_unit_vector = (goal_xyz - gripper_xyz) / distance
            waypoint_pos = gripper_xyz + error_distance * goal_unit_vector
        else:
            waypoint_pos = gripper_xyz
        # TODO: -------------- end ---------------

        # use an zero rotation for the waypoint (its a point so we don't need to worry about orientation)
        waypoint_orient = ikpy.utils.geometry.rpy_matrix(0.0, 0.0, 0.0) # [roll, pitch, yaw]

        return waypoint_pos, waypoint_orient


    def move_to_ready_pose(self):
        # TODO: minor - uncomment the correct ready pose for part 1 or 2!
        #   part 1: 
        self.switch_to_position_mode()
        self.move_to_pose(ik.READY_POSE_P1, blocking=True)
        self.switch_to_position_mode()
        #   part 2: READY_POSE_P2
        # self.move_to_pose(ik.READY_POSE_P2, blocking=True)

    def main(self):
        HelloNode.main(self, 'follow_target', 'follow_target', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()
        self.joint_states_subscriber = self.create_subscription(JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1)

        self.stow_the_robot()
        self.move_to_ready_pose()
        print("At Ready Pose")


        # TODO: ------------- start --------------
        # 1. create a tf2 buffer and listener
        ## Create the buffer(the storage unit)
        self.tf_buffer = tf2_ros.Buffer()
        ## Create the listener (the data collector)
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        ## A TF buffer in ROS is a data storage container that records coordinate frame transforms over time, allowing nodes
        ## to look up relationships between frames at specific timestamps

        # 2. create a subscriber to the goal pose published by your object detector
        self.goal_pose_subscriber = self.create_subscription(
            PoseStamped,                                # Message Type
            'object_detector/goal_pose',                # Topic Name
            callback=self.goal_callback,  		# Function to trigger
            qos_profile=10                              # Queue Size
            )

        # TODO: -------------- end ---------------






if __name__ == '__main__':
    target_follower = IKTargetFollowing()
    target_follower.main()
    target_follower.new_thread.join()
