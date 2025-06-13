#!/usr/bin/env python3

import sys
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, CompressedImage

class ImageSegmentation(Node):
    """
    Performs image segmentation using color and depth filtering.
    Segments objects from RGB-D camera data using HSV color ranges and depth thresholds.
    Publishes various masks and segmented images for downstream processing.
    """
    def __init__(self):
        super().__init__('image_segmentation')

        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                     # Hz
        self.declare_parameter('debug_view', True)

        # Input topic selection
        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        self.declare_parameter('use_compressed', False)
        
        # HSV filtering parameters 
        self.declare_parameter('hsv_hue_low', 0)                        # For black objects
        self.declare_parameter('hsv_hue_high', 179)                     # Full hue range for black
        self.declare_parameter('hsv_saturation_low', 0)
        self.declare_parameter('hsv_saturation_high', 255)
        self.declare_parameter('hsv_value_low', 0)
        self.declare_parameter('hsv_value_high', 50)                    # Low value for black
        
        # Depth filtering parameters 
        self.declare_parameter('depth_low', 250)                        # mm                        
        self.declare_parameter('depth_high', 800)                       # mm
        self.declare_parameter('depth_scale', 1.0)                      # Scale factor for depth values
        
        # Connected components parameter
        self.declare_parameter('connectivity', 4) 

        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.debug_view = self.get_parameter('debug_view').value
        
        self.hsv_hue_low = self.get_parameter('hsv_hue_low').value
        self.hsv_hue_high = self.get_parameter('hsv_hue_high').value
        self.hsv_saturation_low = self.get_parameter('hsv_saturation_low').value
        self.hsv_saturation_high = self.get_parameter('hsv_saturation_high').value
        self.hsv_value_low = self.get_parameter('hsv_value_low').value
        self.hsv_value_high = self.get_parameter('hsv_value_high').value
        
        self.depth_low = self.get_parameter('depth_low').value
        self.depth_high = self.get_parameter('depth_high').value
        self.depth_scale = self.get_parameter('depth_scale').value
        
        self.connectivity = self.get_parameter('connectivity').value
        
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        
        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',                Parameter.Type.DOUBLE,  self.update_rate),
            Parameter('debug_view',                 Parameter.Type.BOOL,    self.debug_view),
            Parameter('hsv_hue_low',                Parameter.Type.INTEGER, self.hsv_hue_low),
            Parameter('hsv_hue_high',               Parameter.Type.INTEGER, self.hsv_hue_high),
            Parameter('hsv_saturation_low',         Parameter.Type.INTEGER, self.hsv_saturation_low),
            Parameter('hsv_saturation_high',        Parameter.Type.INTEGER, self.hsv_saturation_high),
            Parameter('hsv_value_low',              Parameter.Type.INTEGER, self.hsv_value_low),
            Parameter('hsv_value_high',             Parameter.Type.INTEGER, self.hsv_value_high),
            Parameter('depth_low',                  Parameter.Type.INTEGER, self.depth_low),
            Parameter('depth_high',                 Parameter.Type.INTEGER, self.depth_high),
            Parameter('depth_scale',                Parameter.Type.DOUBLE,  self.depth_scale),
            Parameter('connectivity',               Parameter.Type.INTEGER, self.connectivity),
            Parameter('rgb_topic',                  Parameter.Type.STRING,  self.rgb_topic),
            Parameter('depth_topic',                Parameter.Type.STRING,  self.depth_topic),
            Parameter('use_compressed',             Parameter.Type.BOOL,    self.use_compressed),
        ]
        
        result: SetParametersResult = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.rgb_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        
        # Create publishers for various outputs
        self.hsv_mask_pub = self.create_publisher(
            Image,
            'segmentation/hsv_mask',
            10
        )
        
        self.depth_mask_pub = self.create_publisher(
            Image,
            'segmentation/depth_mask',
            10
        )
        
        self.combined_mask_pub = self.create_publisher(
            Image,
            'segmentation/combined_mask',
            10
        )
        
        self.cleaned_mask_pub = self.create_publisher(
            Image,
            'segmentation/cleaned_mask',
            10
        )
        
        self.result_rgb_pub = self.create_publisher(
            Image,
            'segmentation/result_rgb',
            10
        )
        
        self.result_depth_pub = self.create_publisher(
            Image,
            'segmentation/result_depth',
            10
        )
        
        # Create subscribers based on compression setting
        if self.use_compressed:
            self.create_subscription(
                CompressedImage,
                f"{self.rgb_topic}/compressed",
                self.rgb_image_callback,
                qos.qos_profile_sensor_data
            )
            self.create_subscription(
                CompressedImage,
                f"{self.depth_topic}/compressed",
                self.depth_image_callback,
                qos.qos_profile_sensor_data
            )
        else:
            self.create_subscription(
                Image,
                self.rgb_topic,
                self.rgb_image_callback,
                qos.qos_profile_sensor_data
            )
            self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_image_callback,
                qos.qos_profile_sensor_data
            )
        
        self.get_logger().info("ImageSegmentation Node Started.")

    def rgb_image_callback(self, msg) -> None:
        """Callback to convert RGB image from ROS format to OpenCV."""
        try:
            if self.use_compressed:
                self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"RGB CvBridgeError: {e}")
            return

    def depth_image_callback(self, msg) -> None:
        """Callback to convert depth image from ROS format to OpenCV."""
        try:
            if self.use_compressed:
                # For compressed depth, we need to handle the encoding properly
                self.depth_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Check depth image format and convert if necessary
            if self.depth_image.dtype == np.float32:
                # Image is in meters as float32, convert to millimeters as uint16
                self.depth_image = (self.depth_image * 1000.0).astype(np.uint16)
            elif self.depth_image.dtype != np.uint16:
                self.get_logger().warn(f"Unexpected depth format: {self.depth_image.dtype}.")
                
        except CvBridgeError as e:
            self.get_logger().error(f"Depth CvBridgeError: {e}")
            return

    def timer_callback(self) -> None:
        """Main processing loop for segmentation."""
        # Check if both images have been received
        if self.rgb_image is None or self.depth_image is None:
            return
        
        try:
            # Apply depth scale if needed
            scaled_depth = (self.depth_image * self.depth_scale).astype(np.uint16)
            
            # Convert RGB to HSV for color filtering
            hsv_image = cv.cvtColor(self.rgb_image, cv.COLOR_BGR2HSV)
            
            # Create HSV bounds from parameters
            lower_hsv = np.array([self.hsv_hue_low, self.hsv_saturation_low, self.hsv_value_low])
            upper_hsv = np.array([self.hsv_hue_high, self.hsv_saturation_high, self.hsv_value_high])
            
            # Create color mask
            hsv_mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)
            
            # Create depth mask with scaled values
            depth_mask = cv.inRange(scaled_depth, self.depth_low, self.depth_high)

            # Combine masks
            combined_mask = cv.bitwise_and(hsv_mask, depth_mask)
            
            # Clean mask using connected components
            cleaned_mask = self.clean_mask_with_components(combined_mask)
            
            # Apply mask to get segmented results
            result_rgb = cv.bitwise_and(self.rgb_image, self.rgb_image, mask=cleaned_mask)
            result_depth = cv.bitwise_and(self.depth_image, self.depth_image, mask=cleaned_mask)
            
            # Publish all masks and results
            self.publish_mask(hsv_mask, self.hsv_mask_pub)
            self.publish_mask(depth_mask, self.depth_mask_pub)
            self.publish_mask(combined_mask, self.combined_mask_pub)
            self.publish_mask(cleaned_mask, self.cleaned_mask_pub)
            self.publish_image(result_rgb, self.result_rgb_pub, 'bgr8')
            self.publish_depth(result_depth, self.result_depth_pub)
            
            # Debug visualization
            if self.debug_view:
                self.visualize_segmentation(
                    self.rgb_image, hsv_mask, depth_mask, 
                    combined_mask, cleaned_mask, result_rgb
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def clean_mask_with_components(self, mask):
        """Clean mask using connected components analysis."""
        # Compute connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask, self.connectivity, cv.CV_32S
        )
        
        if num_labels <= 1:  # Only background
            return np.zeros_like(mask)
        
        # Find largest component (excluding background at index 0)
        largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        cleaned_mask = np.where(labels == largest_label, 255, 0).astype('uint8')

        return cleaned_mask

    def publish_mask(self, mask, publisher):
        """Publish a grayscale mask as ROS Image message."""
        try:
            msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            publisher.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing mask: {e}")

    def publish_image(self, image, publisher, encoding):
        """Publish a color image as ROS Image message."""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            publisher.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing image: {e}")

    def publish_depth(self, depth, publisher):
        """Publish a depth image as ROS Image message."""
        try:
            # Ensure depth is uint16 before publishing
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
            publisher.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing depth: {e}")

    def visualize_segmentation(self, rgb, hsv_mask, depth_mask, combined_mask, cleaned_mask, result):
        """Visualize segmentation results for debugging."""
        # Create visualization layout
        viz = np.zeros((480*2, 640*3, 3), dtype=np.uint8)
        
        # Resize images if needed
        def resize_to_viz(img, is_mask=False):
            if len(img.shape) == 2 and not is_mask:
                # Depth image - normalize for visualization
                normalized = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                img = cv.cvtColor(normalized, cv.COLOR_GRAY2BGR)
            elif is_mask:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            return cv.resize(img, (640, 480))
        
        # Top row
        viz[0:480, 0:640] = resize_to_viz(rgb)
        viz[0:480, 640:1280] = resize_to_viz(hsv_mask, True)
        viz[0:480, 1280:1920] = resize_to_viz(depth_mask, True)
        
        # Bottom row
        viz[480:960, 0:640] = resize_to_viz(combined_mask, True)
        viz[480:960, 640:1280] = resize_to_viz(cleaned_mask, True)
        viz[480:960, 1280:1920] = resize_to_viz(result)
        
        # Add labels
        cv.putText(viz, "Original RGB", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "HSV Mask", (650, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Depth Mask", (1290, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Combined Mask", (10, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Cleaned Mask", (650, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Result", (1290, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv.namedWindow('Segmentation Debug', cv.WINDOW_NORMAL)
        cv.imshow('Segmentation Debug', viz)
        cv.waitKey(1)

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Validates and applies updated node parameters."""
        for param in params:
            name = param.name
            value = param.value
            
            if name == 'update_rate':
                if not isinstance(value, (int, float)) or value <= 0.0:
                    return SetParametersResult(successful=False, reason="update_rate must be > 0.")
                self.update_rate = float(value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"update_rate updated: {self.update_rate} Hz.")
            
            elif name == 'debug_view':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="debug_view must be a boolean.")
                self.debug_view = value
                if not self.debug_view:
                    cv.destroyAllWindows()
                self.get_logger().info(f"debug_view updated: {self.debug_view}.")
            
            elif name in ('hsv_hue_low', 'hsv_hue_high'):
                if not isinstance(value, int) or not (0 <= value <= 179):
                    return SetParametersResult(successful=False, reason=f"{name} must be an integer between 0-179.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
            
            elif name in ('hsv_saturation_low', 'hsv_saturation_high', 'hsv_value_low', 'hsv_value_high'):
                if not isinstance(value, int) or not (0 <= value <= 255):
                    return SetParametersResult(successful=False, reason=f"{name} must be an integer between 0-255.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
            
            elif name in ('depth_low', 'depth_high'):
                if not isinstance(value, int) or value < 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a non-negative integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value} mm.")
            
            elif name == 'depth_scale':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="depth_scale must be > 0.")
                self.depth_scale = float(value)
                self.get_logger().info(f"depth_scale updated: {value}.")
            
            elif name == 'connectivity':
                if value not in [4, 8]:
                    return SetParametersResult(successful=False, reason="connectivity must be 4 or 8.")
                self.connectivity = value
                self.get_logger().info(f"connectivity updated: {value}.")
            
            elif name in ('rgb_topic', 'depth_topic'):
                if not isinstance(value, str) or not value:
                    return SetParametersResult(successful=False, reason=f"{name} must be a non-empty string.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
                self.get_logger().warn("Topic changes require node restart to take effect.")
            
            elif name == 'use_compressed':
                if not isinstance(value, bool):
                    return SetParametersResult(successful=False, reason="use_compressed must be a boolean.")
                self.use_compressed = value
                self.get_logger().info(f"use_compressed updated: {value}.")
                self.get_logger().warn("Compression setting changes require node restart to take effect.")
        
        return SetParametersResult(successful=True)

    def destroy_node(self):
        if self.debug_view:
            cv.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ImageSegmentation()
    except Exception as e:
        print(f"[FATAL] ImageSegmentation failed to initialize: {e}.", file=sys.stderr)
        rclpy.shutdown()
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted with Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()