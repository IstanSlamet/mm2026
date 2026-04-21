#!/usr/bin/env python3
"""
Gripper Object Detector — D405 gripper-camera YOLO-E detection node for Stretch 3.

Subscribes to the wrist-mounted D405 RGB-D camera, runs YOLO-E open-vocabulary
detection for the target objects, and publishes the 3D pose of the best detection.

How to run:
  # Terminal 1
  ros2 launch stretch_core stretch_driver.launch.py
  # Terminal 2  (D405 gripper camera)
  ros2 launch stretch_core d405_basic.launch.py
  # Terminal 3
  python3 gripper_object_detector.py
  # Terminal 4
  python3 text_input.py
"""

import os
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from ultralytics import YOLO
import message_filters
import detection_utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = '/home/hello-robot/models/yoloe-26s-seg.pt'
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'object_queries.yaml')

# Detection timer period — 2 Hz gives the visual-servoing loop time to act
DETECTION_PERIOD = 0.5  # seconds

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GripperObjectDetector(Node):
    def __init__(self, obj_queries: list[str]):
        super().__init__('gripper_object_detector')

        self.visualize = True  # Set to True for testing vision-only performance

        # --- YOLO-E model ---
        self.model = YOLO(MODEL_PATH)
        # Initial set of classes from YAML (default behavior)
        self.model.set_classes(obj_queries)
        self.get_logger().info(f'Initial Targets from YAML: {obj_queries}')

        # --- camera state ---
        self.bridge = CvBridge()
        self.latest_color    = None   
        self.latest_depth    = None   
        self.latest_cam_info = None   
        self.latest_stamp    = None

        # --- synchronized RGB-D + CameraInfo subscribers ---
        color_sub    = message_filters.Subscriber(
            self, Image, '/gripper_camera/color/image_rect_raw')
        depth_sub    = message_filters.Subscriber(
            self, Image, '/gripper_camera/aligned_depth_to_color/image_raw')
        cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/gripper_camera/color/camera_info')

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, cam_info_sub],
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer.registerCallback(self._image_callback)

        # --- publishers ---
        self.goal_pub  = self.create_publisher(
            PoseStamped, '/gripper_detector/goal_pose', 10)
        self.found_pub = self.create_publisher(
            Bool, '/gripper_detector/object_found', 10)

        # --- subscribers for text-input command ---
        self.target_sub = self.create_subscription(
            String, 
            '/task/target_object', 
            self._target_callback, 
            10
        )
        self.current_target = None # Start with no target; wait for text_input.py

        # --- detection timer ---
        self.create_timer(DETECTION_PERIOD, self._detect_callback)

    # ------------------------------------------------------------------
    # Target command callback
    # ------------------------------------------------------------------
    def _target_callback(self, msg):
        new_target = msg.data
        self.get_logger().info(f'RECEIVED COMMAND: "{new_target}"')
        
        # Dynamically update the YOLO-E open-vocabulary classes for just this object
        self.model.set_classes([new_target])
        self.current_target = new_target
        self.get_logger().info(f'YOLO-E now searching specifically for: "{new_target}"')

    # ------------------------------------------------------------------
    # Camera callback
    # ------------------------------------------------------------------
    def _image_callback(self, color_msg, depth_msg, cam_info_msg):
        if self.latest_color is None:
            self.get_logger().info("[detector] First synchronized frame received!")
        self.latest_color    = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        self.latest_depth    = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self.latest_cam_info = cam_info_msg
        self.latest_stamp    = color_msg.header.stamp

    # ------------------------------------------------------------------
    # Detection timer callback
    # ------------------------------------------------------------------
    def _detect_callback(self):
        # Only run if we have images AND a target string has been entered
        if self.latest_color is None or self.current_target is None:
            self.get_logger().info('Waiting for camera frames and target text command...', 
                               throttle_duration_sec=10)
            return

        try:
            results    = self.model(self.latest_color)
            detections = detection_utils.parse_results(results)
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            return

        if self.visualize:
            detection_utils.visualize_detections_masks(
                part=1,   # D405 depth range: 70–500 mm
                detections=detections,
                rgb_image=self.latest_color,
                depth_image=self.latest_depth,
            )

        pose_msg = self._get_goal_pose(detections)

        if pose_msg is None:
            self.get_logger().info(f'Target "{self.current_target}" not in view.', throttle_duration_sec=2)
            self.found_pub.publish(Bool(data=False))
            return

        try:
            self.found_pub.publish(Bool(data=True))
            self.goal_pub.publish(pose_msg)
            self.get_logger().info(f'Goal pose published for: {self.current_target}')
        except Exception as e:
            self.get_logger().error(f"Publishing error: {e}")

    # ------------------------------------------------------------------
    # 3D projection
    # ------------------------------------------------------------------
    def _get_goal_pose(self, detections, target_idx: int = 0) -> PoseStamped | None:
        if not detections:
            return None

        detection = detections[target_idx]
        centroid   = detection['centroid']   
        mask_poly  = detection['mask']       

        depth_val = detection_utils.mask_median_depth(
            mask_poly, self.latest_depth, min_mm=70, max_mm=2000)
        
        if depth_val is None:
            return None

        x_pix, y_pix = centroid
        h, w = self.latest_depth.shape[:2]
        x_idx = min(max(int(x_pix), 0), w - 1)
        y_idx = min(max(int(y_pix), 0), h - 1)

        goal_xyz = detection_utils.pixel_to_3d(
            (x_idx, y_idx), depth_val, self.latest_cam_info)

        frame_id = self.latest_cam_info.header.frame_id
        return detection_utils.get_pose_msg(self.latest_stamp, frame_id, goal_xyz)


if __name__ == '__main__':
    rclpy.init()

    # Fallback to load initial queries if text_input hasn't sent a message yet
    try:
        with open(CONFIG_PATH, 'r') as f:
            obj_queries = yaml.safe_load(f)['queries']
    except Exception:
        obj_queries = ['object'] # Generic fallback

    node = GripperObjectDetector(obj_queries)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()