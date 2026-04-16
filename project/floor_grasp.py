#!/usr/bin/env python3
"""
Floor Grasp Node — switches to the D405 gripper camera once the patrol
detector has brought the robot to pre-grasp range (~15-20 cm), then uses
YOLO segmentation to localise the object, generate a 3D grasp point, and
execute the grasp sequence.

Subscribes:
  /patrol_detector/object_found  (std_msgs/Bool)
      Rising-edge triggers the pre-grasp pose and detection loop.
  /patrol_detector/goal_pose     (geometry_msgs/PoseStamped)
      Used only to know which object class to look for (passed through
      object_queries.yaml — same file as the patrol detector).

  Gripper camera topics (D405):
    /gripper_camera/color/image_raw
    /gripper_camera/aligned_depth_to_color/image_raw
    /gripper_camera/color/camera_info

Published:
  /floor_grasp/status  (std_msgs/String)
      State machine status: IDLE | PREGRASP | DETECTING | GRASPING | DONE | FAILED

How to run (after stretch_driver + d435i launch are already up):
  ros2 launch stretch_core d405_basic.launch.py   # gripper cam
  python3 floor_grasp_node.py
"""

import os
import threading

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from hello_helpers.hello_misc import HelloNode
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from ultralytics import YOLO
import message_filters
import tf2_ros
import tf2_geometry_msgs
import ikpy.utils.geometry

import detection_utils
import ik_ros_utils as ik

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = '/home/hello-robot/models/yoloe-26s-seg.pt'
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'object_queries.yaml')

# The D405 is mounted looking downward on the wrist; its image is already
# right-way-up for objects on the floor — no rotation needed.
CAMERA_ROTATE = None   # set to cv2.ROTATE_* if your mount differs

# Pre-grasp joint configuration — arm raised, extended slightly, wrist pitched
# down so the D405 is looking at the floor object.
PRE_GRASP_POSE = {
    'joint_lift':        0.20,   # low — close to floor object
    'wrist_extension':   0.30,   # extend enough to see the object
    'joint_wrist_yaw':   0.0,
    'joint_wrist_pitch': -1.0,   # pitch down toward floor
    'joint_wrist_roll':  0.0,
    'gripper_aperture':  0.5,    # open ready to grasp
}

# How many consecutive "not found" detection cycles before we give up
MAX_RETRIES = 8

# Detection timer period while in DETECTING state
DETECTION_PERIOD = 0.4   # seconds  (2.5 Hz)

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class GraspState:
    IDLE       = 'IDLE'
    PREGRASP   = 'PREGRASP'
    DETECTING  = 'DETECTING'
    GRASPING   = 'GRASPING'
    DONE       = 'DONE'
    FAILED     = 'FAILED'


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class FloorGraspNode(HelloNode):
    def __init__(self, obj_queries: list[str]):
        HelloNode.__init__(self)

        self.obj_queries = obj_queries
        self.state       = GraspState.IDLE
        self._state_lock = threading.Lock()

        # --- YOLO-E ---
        self.model = YOLO(MODEL_PATH)
        self.model.set_classes(obj_queries)

        # --- camera state (gripper D405) ---
        self.bridge              = CvBridge()
        self.latest_color        = None   # already oriented for YOLO
        self.latest_depth_raw    = None
        self.latest_cam_info     = None
        self.latest_stamp        = None
        self._cam_lock           = threading.Lock()

        self._retry_count        = 0
        self._detection_timer    = None   # created on demand

        # --- IK joint state (same pattern as target_following.py) ---
        self.joint_state         = None
        self.joint_states_lock   = threading.Lock()

    # ------------------------------------------------------------------
    # ROS initialisation  (called inside main after HelloNode.main())
    # ------------------------------------------------------------------

    def _ros_init(self):
        self.callback_group = ReentrantCallbackGroup()

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Joint states (for IK)
        self.create_subscription(
            __import__('sensor_msgs.msg', fromlist=['JointState']).JointState,
            '/stretch/joint_states',
            self._joint_states_cb,
            qos_profile=1,
        )

        # Patrol detector triggers
        self.create_subscription(
            Bool,
            '/patrol_detector/object_found',
            self._found_cb,
            qos_profile=10,
        )

        # Gripper camera — synchronised RGB-D
        color_sub    = message_filters.Subscriber(
            self, Image, '/gripper_camera/color/image_raw')
        depth_sub    = message_filters.Subscriber(
            self, Image, '/gripper_camera/aligned_depth_to_color/image_raw')
        cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/gripper_camera/color/camera_info')

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, cam_info_sub],
            queue_size=10,
            slop=0.02,
        )
        self._sync.registerCallback(self._image_cb)

        # Status publisher
        self.status_pub = self.create_publisher(
            String, '/floor_grasp/status', 10)

        self.get_logger().info('FloorGraspNode ready — waiting for patrol trigger.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_states_cb(self, msg):
        """Mirror of target_following.py — stores relevant joints as dict."""
        with self.joint_states_lock:
            joint_names = [
                'joint_lift', 'joint_arm_l0',
                'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll',
            ]
            self.joint_state = {}
            for name in joint_names:
                if name in msg.name:
                    i = msg.name.index(name)
                    self.joint_state[name] = msg.position[i]

    def _image_cb(self,
                  color_msg: Image,
                  depth_msg: Image,
                  cam_info_msg: CameraInfo):
        """Store the latest synchronised gripper-cam frame pair."""
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if CAMERA_ROTATE is not None:
            color = cv2.rotate(color, CAMERA_ROTATE)

        with self._cam_lock:
            self.latest_color     = color
            self.latest_depth_raw = depth
            self.latest_cam_info  = cam_info_msg
            self.latest_stamp     = color_msg.header.stamp

    def _found_cb(self, msg: Bool):
        """
        Rising-edge detector on the patrol found flag.
        Only acts when IDLE — avoids re-triggering mid-grasp.
        """
        with self._state_lock:
            if msg.data and self.state == GraspState.IDLE:
                self.get_logger().info('Patrol trigger received — entering pre-grasp.')
                self._set_state(GraspState.PREGRASP)
                # Hand off to a separate thread so we don't block the subscriber
                threading.Thread(target=self._run_grasp_pipeline,
                                 daemon=True).start()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_state(self, new_state: str):
        self.state = new_state
        msg = String()
        msg.data = new_state
        self.status_pub.publish(msg)
        self.get_logger().info(f'State → {new_state}')

    # ------------------------------------------------------------------
    # Main grasp pipeline  (runs in its own thread)
    # ------------------------------------------------------------------

    def _run_grasp_pipeline(self):
        # 1. Move to pre-grasp arm configuration
        self._set_state(GraspState.PREGRASP)
        self.switch_to_position_mode()
        self.move_to_pose(PRE_GRASP_POSE, blocking=True)
        self.get_logger().info('Pre-grasp pose reached.')

        # 2. Detection loop
        self._set_state(GraspState.DETECTING)
        self._retry_count = 0
        pose_msg = None

        while self._retry_count < MAX_RETRIES:
            pose_msg = self._detect_once()
            if pose_msg is not None:
                break
            self._retry_count += 1
            self.get_logger().info(
                f'Object not detected ({self._retry_count}/{MAX_RETRIES}), retrying…')
            # small sleep to wait for next camera frame
            import time; time.sleep(DETECTION_PERIOD)

        if pose_msg is None:
            self.get_logger().error('Object not found after max retries — aborting.')
            self._set_state(GraspState.FAILED)
            self._reset_to_idle()
            return

        # 3. Execute grasp
        self._set_state(GraspState.GRASPING)
        success = self._execute_grasp(pose_msg)

        if success:
            self._set_state(GraspState.DONE)
        else:
            self._set_state(GraspState.FAILED)

        self._reset_to_idle()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_once(self) -> PoseStamped | None:
        """
        Run YOLO on the latest gripper-cam frame and return a PoseStamped
        (in gripper_camera_color_optical_frame) for the best detection,
        or None if nothing is found.
        """
        with self._cam_lock:
            if self.latest_color is None:
                self.get_logger().warn('No gripper camera frame available yet.')
                return None
            color     = self.latest_color.copy()
            depth_raw = self.latest_depth_raw.copy()
            cam_info  = self.latest_cam_info
            stamp     = self.latest_stamp

        results    = self.model(color, verbose=False)
        detections = detection_utils.parse_results(results)

        # Optional live visualisation (mirrors patrol_object_detector.py)
        vis_depth = cv2.rotate(depth_raw, cv2.ROTATE_90_CLOCKWISE) \
            if CAMERA_ROTATE else depth_raw
        detection_utils.visualize_detections_masks(
            part=1,
            detections=detections,
            rgb_image=color,
            depth_image=vis_depth,
        )

        if not detections:
            return None

        return self._project_to_3d(detections, depth_raw, cam_info, stamp)

    # ------------------------------------------------------------------
    # 3D projection
    # ------------------------------------------------------------------

    def _project_to_3d(self,
                       detections: list,
                       depth_raw: np.ndarray,
                       cam_info: CameraInfo,
                       stamp,
                       target_idx: int = 0) -> PoseStamped | None:
        """
        Convert the segmentation mask of the best detection into a 3D centroid.

        The D405 image is NOT rotated (CAMERA_ROTATE=None), so pixel coords
        map directly onto the depth image without any coordinate transform.
        If you do rotate, mirror the coord-unwrap logic from
        patrol_object_detector.py._get_goal_pose().
        """
        mask_polygon = detections[target_idx]['mask']  # Nx2 [x, y]

        h, w = depth_raw.shape[:2]

        # Rasterise polygon mask
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [mask_polygon], 1)
        rows, cols = np.where(binary_mask > 0)

        if len(rows) == 0:
            return None

        # Sample depth at every mask pixel; discard zeros (invalid depth)
        points_3d = []
        for r, c in zip(rows, cols):
            depth_val = depth_raw[r, c]
            if depth_val > 0:
                pt = detection_utils.pixel_to_3d(
                    (c, r), depth_val, cam_info)
                points_3d.append(pt)

        if not points_3d:
            self.get_logger().warn('Mask found but no valid depth pixels.')
            return None

        goal_xyz = np.mean(np.array(points_3d), axis=0)
        self.get_logger().info(
            f'3D centroid (gripper cam frame): '
            f'x={goal_xyz[0]:.3f} y={goal_xyz[1]:.3f} z={goal_xyz[2]:.3f} m')

        return detection_utils.get_pose_msg(
            stamp,
            'gripper_camera_color_optical_frame',
            goal_xyz,
        )

    # ------------------------------------------------------------------
    # Grasp execution
    # ------------------------------------------------------------------

    def _execute_grasp(self, pose_cam: PoseStamped) -> bool:
        """
        Transform the detected pose from the gripper-camera frame to
        base_link, then use IK (from ik_ros_utils) to reach and close
        the gripper.  Returns True on success.
        """
        # --- transform pose to base_link ---
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                pose_cam.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            pose_base = tf2_geometry_msgs.do_transform_pose(
                pose_cam.pose, transform)

            goal_transformed        = PoseStamped()
            goal_transformed.header.frame_id = 'base_link'
            goal_transformed.pose   = pose_base
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF lookup failed: {e}')
            return False

        goal_pos = ik.get_xyz_from_msg(goal_transformed)

        # --- get current IK configuration ---
        with self.joint_states_lock:
            if self.joint_state is None:
                self.get_logger().error('No joint states received yet.')
                return False
            q_init = ik.get_current_configuration(self.joint_state)

        # --- open gripper before approach ---
        self.move_to_pose({'gripper_aperture': 0.5}, blocking=True)

        # --- IK solve with downward wrist orientation ---
        # Pitch the wrist ~90° down so the gripper approaches from above.
        # rpy_matrix(roll, pitch, yaw): pitch = -pi/2 → pointing straight down.
        grasp_orientation = ikpy.utils.geometry.rpy_matrix(0.0, -np.pi / 2, 0.0)

        q_soln = ik.get_grasp_goal(goal_pos, grasp_orientation, q_init)

        if q_soln is None:
            self.get_logger().error('IK found no solution for grasp pose.')
            return False

        ik.print_q(q_soln)

        # --- move to grasp position ---
        ik.move_to_configuration(self, q_soln)

        # --- close gripper ---
        self.move_to_pose({'gripper_aperture': -0.15}, blocking=True)
        self.get_logger().info('Gripper closed — object grasped.')

        # --- lift object ---
        with self.joint_states_lock:
            current_lift = self.joint_state.get('joint_lift', 0.3)
        self.move_to_pose({'joint_lift': current_lift + 0.20}, blocking=True)

        # --- retract arm to safe carry position ---
        self.move_to_pose({
            'wrist_extension':   0.0,
            'joint_wrist_pitch': 0.0,
        }, blocking=True)

        self.get_logger().info('Grasp complete — object lifted and arm retracted.')
        return True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_to_idle(self):
        """Return to a neutral stow and mark IDLE so the next patrol cycle works."""
        self.get_logger().info('Stowing robot and returning to IDLE.')
        self.stow_the_robot()
        with self._state_lock:
            self._set_state(GraspState.IDLE)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def main(self):
        HelloNode.main(
            self,
            'floor_grasp_node',
            'floor_grasp_node',
            wait_for_first_pointcloud=False,
        )
        self._ros_init()
        self.stow_the_robot()
        self.get_logger().info('Robot stowed — ready for patrol trigger.')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rclpy.init()

    with open(CONFIG_PATH, 'r') as f:
        obj_queries = yaml.safe_load(f)['queries']

    node = FloorGraspNode(obj_queries)
    node.main()
    node.new_thread.join()