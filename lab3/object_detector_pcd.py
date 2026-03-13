import cv2
import yaml
import rclpy
import os.path as osp
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import detection_utils
import message_filters
import numpy as np


# Don't forget to start the camera before starting this node!
# Part 1: using in-gripper camera
#    ros2 launch stretch_core d405_basic.launch.py
# Part 2: using head camera
#    ros2 launch stretch_core d435i_low_resolution.launch.py
#
# ros2 run rviz2 rviz2 -d `ros2 pkg prefix --share stretch_calibration`/rviz/stretch_simple_test.rviz


class YOLOEObjectDetector(Node):
    def __init__(self, obj_queries):
        super().__init__('yoloe_object_detector')
        self.visualize = True

        # ----------- Camera Streaming Setup -----------

        # subscribe to the robot's color and aligned depth camera image topics from the gripper camera
        # using message_filters, instead of self.create_subscription() to allow us
        #   to synchronize the two camera streams can use a single callback that triggers when both come in
        # TODO: ------------- start --------------
        # leave as is for part 1, 
        # change for part 2 to use the head camera
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw') # changed from 'gripper_camera' to 'camera' and 'image_rect_raw' to 'image_raw'
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        self.color_cam_info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/color/camera_info')
        # DONE: -------------- end ---------------

        # Use ApproximateTimeSynchronizer and register a callback function that runs within some time tolerance of when both images are received
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.color_cam_info_sub],
            queue_size=10,
            slop=0.01  # 10ms tolerance
        )
        self.synchronizer.registerCallback(self.image_callback)

        # bridge to convert ROS2 image messages to OpenCV images
        self.bridge = CvBridge()

        # -----------------------------------------------------

        # ----------- YOLO-E Object Detection SetuP -----------

        # Load the YOLOE model, which should already saved to common models directory on the robot
        #   we use yolo-e-v26-small for its high performance and low latency on limited compute
        model_path = '/home/hello-robot/models'
        model_name = 'yoloe-26s-seg.pt'
        self.model = YOLO(osp.join(model_path, model_name))

        # pass prompt for the object/s you want to detect
        self.obj_queries = obj_queries
        self.model.set_classes(self.obj_queries)

        # Run the detector and goals at a fixed frequency to reduce latency introduced by the detector
        #   and give the robot time to move between poses
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_goals_callback)
        self.goal_pub = self.create_publisher(PoseStamped, '/object_detector/goal_pose', 10)
        self.goal_pose_msg = None
        self.latest_color = None
        self.latest_depth = None
        self.latest_color_cam_info = None

        # -----------------------------------------------------

    def image_callback(self, color_msg, depth_msg, color_cam_info_msg):
        # convert the color and depth ROS2 image messages to OpenCV images
        # TODO: ------------- start --------------
        # in part 1,fill with your response
        #   you may need to nest things in a try, except in case frames are missing
        #.  if you are unpacking frames correctly, you should see the live color and depth output
        #   plotted in a cv2 window by detection_utils.visualize_detection_masks()
        # in part 2, you may need to make changes to the code to handle the head camera orientation

        # https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython 
          
        # Convert ROS2 messages to OpenCV
        color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Rotate images and set as latest color and depth
        self.latest_color = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
        #self.latest_depth = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
        
        self.latest_color_cam_info = color_cam_info_msg #CameraInfo is not an image, so donot use Cv_bridge. It is metadata message.

        # save the timestamp from the header of the color image
        self.latest_stamp = color_msg.header.stamp

        pass
        # TODO: -------------- end ---------------


    def publish_goals_callback(self):
        # run object detection on the RGB image
        # TODO: ------------- start --------------
        # fill with your 
        if self.latest_color is None:
            self.get_logger().info("No images received yet, skipping detection...")
            return
        
        results = self.model(self.latest_color)
        #   pass the color frame to YOLO-E, parse the results using detection_utils.parse_results()
        detections = detection_utils.parse_results(results)
        # TODO: -------------- end ---------------

        # create visualizations from the detections
        if self.visualize:
            # Rotate a copy of the depth image 90 degrees just so it matches the color image for the viz window
            vis_depth = cv2.rotate(self.latest_depth, cv2.ROTATE_90_CLOCKWISE)
            detection_utils.visualize_detections_masks(
                # TODO: minor - change the part= arg when you edit your code for part 2! 
                #   adjusts the color scaling of the depth image display to match the camera range
                part=2, detections=detections, rgb_image=self.latest_color, depth_image=vis_depth)

        # get the goal pose and publish it, if it exists
        self.pose_msg = self.get_goal_pose(detections)

        if self.pose_msg is None:
            print("OBJECT NOT DETECTED, no pose to publish")
            return
        else:
            #self.goal_pub.publish(self.goal_pose_msg)
            self.goal_pub.publish(self.pose_msg)
            print()
            print("---------- Published Goal Pose ----------")


    def get_goal_pose(self, detections, target_idx=0):
            if detections is None or len(detections) == 0:
                return None

            target_mask = detections[target_idx]
            mask_polygon = target_mask["mask"] # This is the Nx2 array of [x, y] coordinates

            # 1. Create a blank 2D image canvas matching the rotated color image dimensions
            rotated_h, rotated_w = self.latest_color.shape[:2]
            binary_mask = np.zeros((rotated_h, rotated_w), dtype=np.uint8)

            # 2. Fill the polygon shape with 1s to create a true pixel mask
            cv2.fillPoly(binary_mask, [mask_polygon], 1)

            # 3. NOW we can safely find the row/col coordinates of the filled pixels!
            rows, cols = np.where(binary_mask > 0)
            
            original_h = rotated_w # The original unrotated image height
            
            points_3d = []
            for r, c in zip(rows, cols):
                # Map rotated mask pixel (c, r) back to the original image pixel (orig_c, orig_r)
                orig_c = r
                orig_r = original_h - 1 - c
                
                # Grab depth from the UNROTATED depth image
                depth = self.latest_depth[orig_r, orig_c]
                
                if depth > 0:
                    # Use pixel_to_3d (which already handles the mm to meters conversion!)
                    point = detection_utils.pixel_to_3d((orig_c, orig_r), depth, self.latest_color_cam_info)
                    points_3d.append(point)

            if len(points_3d) == 0: 
                return None
            
            # get mean across the columns to find the centroid of the mask
            points_array = np.array(points_3d)
            goal_pos = np.mean(points_array, axis=0)

            # package into the final PoseStamped message
            self.goal_pose_msg = detection_utils.get_pose_msg(
                self.latest_stamp, 
                "camera_color_optical_frame", 
                goal_pos
            )
            
            return self.goal_pose_msg
    # def get_goal_pose(self, detections, target_idx=0):
    #     if detections is None or len(detections) == 0:
    #         return None

    #     # TODO: ------------- start --------------        
    #     # in part 2, edit the code you wrote for part 1 to now project all points in the mask to 3D,
    #     target_mask = detections[target_idx]
    #     goal_msg = PoseStamped()
    #     goal_msg.header.stamp = self.latest_stamp
    #     goal_msg.header.frame_id = "camera_color_optical_frame"

    #     mask = target_mask["mask"] # key change: target entire mask instead of its centroid
    #     rows, cols = np.where(mask > 0)
    #     # The original image's height is now the rotated mask's width
    #     original_h = mask.shape[1]

    #     # loop through mask, unrotate the 2D pixels and project all points to 3d
    #     points_3d = []
    #     for r, c in zip(rows, cols):
    #         # Map the rotated mask pixel (c, r) back to the original image pixel (orig_c, orig_r)
    #         orig_c = r
    #         orig_r = original_h - 1 - c
    #         # Grab depth from the UNROTATED depth image using the unrotated coordinates
    #         depth = self.latest_depth[orig_r, orig_c]
            
    #         if depth > 0:
    #             # Use the original coordinates and the original CameraInfo to get the accurate 3D point
    #             point = detection_utils.pixel_to_3d((orig_c, orig_r), depth, self.latest_color_cam_info)
    #             points_3d.append(point)

    #     #   then get the centroid of the resulting pointcloud to use as the goal pose (instead of the 2D centroid in part 1)
    #     if len(points_3d) == 0: 
    #         return None
        
    #     # get mean across the columns to find the centroid of the mask
    #     points_array = np.array(points_3d)
    #     goal_pos = np.mean(points_array, axis=0)

    #     # package into the final PoseStamped message
    #     self.goal_pose_msg = detection_utils.get_pose_msg(
    #         self.latest_stamp, 
    #         "camera_color_optical_frame", 
    #         goal_pos
    #     )
    #     # TODO: -------------- end ---------------
    #     return self.goal_pose_msg


if __name__ == '__main__':
    rclpy.init()

    # load in the full list of object queries from the yaml file, as well as a target (if specified)
    with open('object_queries.yaml', 'r') as file:
        config = yaml.safe_load(file)
        obj_queries = config['queries']

    yolo_object_detector = YOLOEObjectDetector(obj_queries)
    rclpy.spin(yolo_object_detector)
    yolo_object_detector.destroy_node()
    rclpy.shutdown()
