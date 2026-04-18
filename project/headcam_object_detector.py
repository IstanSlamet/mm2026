#!/usr/bin/env python3
"""
Patrol Object Detector — head-camera YOLO-E detection node for Stretch 3.

Subscribes to the head RGB-D camera, runs YOLO-E open-vocabulary detection
for the target objects listed in object_queries.yaml, and publishes the 3D
pose of the best detection together with a boolean found/not-found flag.

Intended to run while a separate patrol node drives the robot; the patrol
node stops and hands off to the pre-grasp approach once object_found goes True.

How to run:
  # Terminal 1
  ros2 launch stretch_core stretch_driver.launch.py
  # Terminal 2
  ros2 launch stretch_core d435i_low_resolution.launch.py
  # Terminal 3
  python3 patrol_object_detector.py

Published topics:
  /patrol_detector/goal_pose    (geometry_msgs/PoseStamped)
      3D centroid of the detected object in camera_color_optical_frame.
  /patrol_detector/object_found (std_msgs/Bool)
      True each detection cycle the target is visible, False otherwise.
"""

import os

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from ultralytics import YOLO
import message_filters

import detection_utils


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = '/home/hello-robot/models/yoloe-26s-seg.pt'
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'object_queries.yaml')

# Detection timer period — 2 Hz gives the robot time to move between poses
DETECTION_PERIOD = 0.5  # seconds

# The Stretch 3 head camera (D435i) is mounted 90° clockwise relative to the
# natural viewing orientation.  Only the color image is rotated so YOLO sees
# objects right-side up; depth is left unrotated and we unrotate pixel coords
# before the depth lookup (same approach used in object_detector_pcd.py).
CAMERA_ROTATE = cv2.ROTATE_90_CLOCKWISE


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class PatrolObjectDetector(Node):
    def __init__(self, obj_queries: list[str]):
        super().__init__('patrol_object_detector')

        self.visualize = True

        # --- YOLO-E model ---
        self.model = YOLO(MODEL_PATH)
        self.model.set_classes(obj_queries)
        self.get_logger().info(f'Targets: {obj_queries}')

        # --- camera state ---
        self.bridge = CvBridge()
        self.latest_color_rotated = None   # 90° CW rotated color (fed to YOLO)
        self.latest_depth_raw     = None   # unrotated depth (used for 3D projection)
        self.latest_cam_info      = None   # intrinsics from unrotated frame
        self.latest_stamp         = None

        # --- synchronized RGB-D subscribers ---
        color_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(
            self, Image, '/camera/aligned_depth_to_color/image_raw')
        cam_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/color/camera_info')

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, cam_info_sub],
            queue_size=10,
            slop=0.01,
        )
        self.synchronizer.registerCallback(self._image_callback)

        # --- publishers ---
        self.goal_pub  = self.create_publisher(
            PoseStamped, '/patrol_detector/goal_pose', 10)
        self.found_pub = self.create_publisher(
            Bool, '/patrol_detector/object_found', 10)

        # --- detection timer ---
        self.create_timer(DETECTION_PERIOD, self._detect_callback)

    # ------------------------------------------------------------------
    # Camera callback
    # ------------------------------------------------------------------

    def _image_callback(self,
                        color_msg: Image,
                        depth_msg: Image,
                        cam_info_msg: CameraInfo):
        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        self.latest_color_rotated = cv2.rotate(color, CAMERA_ROTATE)
        self.latest_depth_raw     = depth          # keep unrotated
        self.latest_cam_info      = cam_info_msg
        self.latest_stamp         = color_msg.header.stamp

    # ------------------------------------------------------------------
    # Detection timer callback
    # ------------------------------------------------------------------

    def _detect_callback(self):
        if self.latest_color_rotated is None:
            self.get_logger().info('Waiting for camera frames...', throttle_duration_sec=5)
            return

        results    = self.model(self.latest_color_rotated)
        detections = detection_utils.parse_results(results)

        if self.visualize:
            vis_depth = cv2.rotate(self.latest_depth_raw, CAMERA_ROTATE)
            detection_utils.visualize_detections_masks(
                part=2,
                detections=detections,
                rgb_image=self.latest_color_rotated,
                depth_image=vis_depth,
            )

        pose_msg = self._get_goal_pose(detections)

        if pose_msg is None:
            self.get_logger().info('Object not detected.', throttle_duration_sec=2)
            self.found_pub.publish(Bool(data=False))
            return

        self.found_pub.publish(Bool(data=True))
        self.goal_pub.publish(pose_msg)
        self.get_logger().info('---------- Published Goal Pose ----------')

    # ------------------------------------------------------------------
    # 3D projection
    # ------------------------------------------------------------------

    def _get_goal_pose(self, detections, target_idx: int = 0) -> PoseStamped | None:
        """
        Project the best detection's segmentation mask to 3D and return a
        PoseStamped at the point-cloud centroid of the mask.

        The mask polygon is in rotated-image coordinates; each pixel is mapped
        back to the original (unrotated) frame before the depth lookup so that
        the original camera intrinsics remain valid.

        90° CW rotation mapping:
          rotated  (col=c, row=r)  →  original (col=r, row=original_h - 1 - c)
          original_h = rotated_w   (dimensions swap on 90° rotation)
        """
        if not detections:
            return None

        mask_polygon = detections[target_idx]['mask']  # Nx2 array of [x, y] pixel coords

        rotated_h, rotated_w = self.latest_color_rotated.shape[:2]
        original_h = rotated_w  # original image height before rotation

        # Rasterise the polygon in rotated-image space
        binary_mask = np.zeros((rotated_h, rotated_w), dtype=np.uint8)
        cv2.fillPoly(binary_mask, [mask_polygon], 1)
        rows, cols = np.where(binary_mask > 0)  # row=y, col=x in rotated image

        # Project each mask pixel to 3D using the unrotated depth + intrinsics
        points_3d = []
        for r, c in zip(rows, cols):
            orig_c = r
            orig_r = original_h - 1 - c
            depth_val = self.latest_depth_raw[orig_r, orig_c]
            if depth_val > 0:
                point = detection_utils.pixel_to_3d(
                    (orig_c, orig_r), depth_val, self.latest_cam_info)
                points_3d.append(point)

        if not points_3d:
            return None

        goal_xyz = np.mean(np.array(points_3d), axis=0)

        return detection_utils.get_pose_msg(
            self.latest_stamp,
            'camera_color_optical_frame',
            goal_xyz,
        )


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rclpy.init()

    with open(CONFIG_PATH, 'r') as f:
        obj_queries = yaml.safe_load(f)['queries']

    node = PatrolObjectDetector(obj_queries)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
