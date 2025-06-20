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

from sensor_msgs.msg import Image

class ImageSegmentation(Node):
    """
    Performs image segmentation using color and depth filtering.
    """
    def __init__(self):
        super().__init__('image_segmentation')

        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                     # Hz

        # Input topic selection
        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        
        # HSV filtering parameters 
        self.declare_parameter('hsv_hue_low', 0)                        # For black objects
        self.declare_parameter('hsv_hue_high', 179)                     # Full hue range for black
        self.declare_parameter('hsv_saturation_low', 0)
        self.declare_parameter('hsv_saturation_high', 255)
        self.declare_parameter('hsv_value_low', 0)
        self.declare_parameter('hsv_value_high', 100)                   # Low value for black
        
        # Depth filtering parameters 
        self.declare_parameter('depth_low', 0)                          # mm                        
        self.declare_parameter('depth_high', 2000)                      # mm
        
        # Connected components parameter
        self.declare_parameter('connectivity', 4) 
        
        # Area filtering parameters for connected components
        self.declare_parameter('min_component_area', 150000)             # Minimum pixels for valid component
        self.declare_parameter('max_component_area', 200000)             # Maximum pixels for valid component

        # Gaussian blur parameters
        self.declare_parameter('gaussian_kernel_size_width', 5)
        self.declare_parameter('gaussian_kernel_size_height', 5)
        self.declare_parameter('gaussian_sigma', 5.0)

        # Morphological operations parameters for HSV mask
        self.declare_parameter('hsv_morph_kernel_size_width', 3)
        self.declare_parameter('hsv_morph_kernel_size_height', 3)
        self.declare_parameter('hsv_morph_erode_iterations', 3)
        self.declare_parameter('hsv_morph_dilate_iterations', 5)

        # Morphological operations parameters for depth mask
        self.declare_parameter('depth_morph_kernel_size_width', 5)
        self.declare_parameter('depth_morph_kernel_size_height', 5)
        self.declare_parameter('depth_morph_erode_iterations', 5)
        self.declare_parameter('depth_morph_dilate_iterations', 0)

        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        
        self.hsv_hue_low = self.get_parameter('hsv_hue_low').value
        self.hsv_hue_high = self.get_parameter('hsv_hue_high').value
        self.hsv_saturation_low = self.get_parameter('hsv_saturation_low').value
        self.hsv_saturation_high = self.get_parameter('hsv_saturation_high').value
        self.hsv_value_low = self.get_parameter('hsv_value_low').value
        self.hsv_value_high = self.get_parameter('hsv_value_high').value
        
        self.depth_low = self.get_parameter('depth_low').value
        self.depth_high = self.get_parameter('depth_high').value
        
        self.connectivity = self.get_parameter('connectivity').value
        self.min_component_area = self.get_parameter('min_component_area').value
        self.max_component_area = self.get_parameter('max_component_area').value
        
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value

        self.gaussian_kernel_size_width = self.get_parameter('gaussian_kernel_size_width').value
        self.gaussian_kernel_size_height = self.get_parameter('gaussian_kernel_size_height').value
        self.gaussian_sigma = self.get_parameter('gaussian_sigma').value

        self.hsv_morph_kernel_size_width = self.get_parameter('hsv_morph_kernel_size_width').value
        self.hsv_morph_kernel_size_height = self.get_parameter('hsv_morph_kernel_size_height').value
        self.hsv_morph_erode_iterations = self.get_parameter('hsv_morph_erode_iterations').value
        self.hsv_morph_dilate_iterations = self.get_parameter('hsv_morph_dilate_iterations').value

        self.depth_morph_kernel_size_width = self.get_parameter('depth_morph_kernel_size_width').value
        self.depth_morph_kernel_size_height = self.get_parameter('depth_morph_kernel_size_height').value
        self.depth_morph_erode_iterations = self.get_parameter('depth_morph_erode_iterations').value
        self.depth_morph_dilate_iterations = self.get_parameter('depth_morph_dilate_iterations').value
        
        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',                    Parameter.Type.DOUBLE,  self.update_rate),
            Parameter('hsv_hue_low',                    Parameter.Type.INTEGER, self.hsv_hue_low),
            Parameter('hsv_hue_high',                   Parameter.Type.INTEGER, self.hsv_hue_high),
            Parameter('hsv_saturation_low',             Parameter.Type.INTEGER, self.hsv_saturation_low),
            Parameter('hsv_saturation_high',            Parameter.Type.INTEGER, self.hsv_saturation_high),
            Parameter('hsv_value_low',                  Parameter.Type.INTEGER, self.hsv_value_low),
            Parameter('hsv_value_high',                 Parameter.Type.INTEGER, self.hsv_value_high),
            Parameter('depth_low',                      Parameter.Type.INTEGER, self.depth_low),
            Parameter('depth_high',                     Parameter.Type.INTEGER, self.depth_high),
            Parameter('connectivity',                   Parameter.Type.INTEGER, self.connectivity),
            Parameter('min_component_area',             Parameter.Type.INTEGER, self.min_component_area),
            Parameter('max_component_area',             Parameter.Type.INTEGER, self.max_component_area),
            Parameter('rgb_topic',                      Parameter.Type.STRING,  self.rgb_topic),
            Parameter('depth_topic',                    Parameter.Type.STRING,  self.depth_topic),
            Parameter('gaussian_kernel_size_width',     Parameter.Type.INTEGER, self.gaussian_kernel_size_width),
            Parameter('gaussian_kernel_size_height',    Parameter.Type.INTEGER, self.gaussian_kernel_size_height),
            Parameter('gaussian_sigma',                 Parameter.Type.DOUBLE,  self.gaussian_sigma),
            Parameter('hsv_morph_kernel_size_width',    Parameter.Type.INTEGER, self.hsv_morph_kernel_size_width),
            Parameter('hsv_morph_kernel_size_height',   Parameter.Type.INTEGER, self.hsv_morph_kernel_size_height),
            Parameter('hsv_morph_erode_iterations',     Parameter.Type.INTEGER, self.hsv_morph_erode_iterations),
            Parameter('hsv_morph_dilate_iterations',    Parameter.Type.INTEGER, self.hsv_morph_dilate_iterations),
            Parameter('depth_morph_kernel_size_width',  Parameter.Type.INTEGER, self.depth_morph_kernel_size_width),
            Parameter('depth_morph_kernel_size_height', Parameter.Type.INTEGER, self.depth_morph_kernel_size_height),
            Parameter('depth_morph_erode_iterations',   Parameter.Type.INTEGER, self.depth_morph_erode_iterations),
            Parameter('depth_morph_dilate_iterations',  Parameter.Type.INTEGER, self.depth_morph_dilate_iterations),
        ]
        
        result: SetParametersResult = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.rgb_image = None
        self.depth_image = None
        self.bridge = CvBridge()
        
        # Create publishers for various outputs
        self.result_rgb_pub = self.create_publisher(
            Image,
            'segmentation/result_rgb',
            qos.qos_profile_sensor_data
        )
        
        self.result_depth_pub = self.create_publisher(
            Image,
            'segmentation/result_depth',
            qos.qos_profile_sensor_data
        )

        self.debug_viz_pub = self.create_publisher(
            Image,
            'segmentation/debug_visualization',
            10
        )

        # Create subscribers based on compression setting
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
        
        self.get_logger().info("ImageSegmentation Node Start.")

    def rgb_image_callback(self, msg) -> None:
        """Callback to convert RGB image from ROS format to OpenCV."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"RGB CvBridgeError: {e}")
            return

    def depth_image_callback(self, msg) -> None:
        """Callback to convert depth image from ROS format to OpenCV."""
        try:
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
            # Apply Gaussian blur to reduce noise in RGB image
            blurred_rgb = cv.GaussianBlur(
                self.rgb_image,
                (self.gaussian_kernel_size_width, self.gaussian_kernel_size_height),
                self.gaussian_sigma
            )

            # Convert RGB to HSV for color filtering
            hsv_image = cv.cvtColor(blurred_rgb, cv.COLOR_BGR2HSV)
            
            # Create HSV bounds from parameters
            lower_hsv = np.array([self.hsv_hue_low, self.hsv_saturation_low, self.hsv_value_low])
            upper_hsv = np.array([self.hsv_hue_high, self.hsv_saturation_high, self.hsv_value_high])
            
            # Create color mask
            hsv_mask = cv.inRange(hsv_image, lower_hsv, upper_hsv)

            # Apply morphological operations to HSV mask
            hsv_kernel = np.ones((self.hsv_morph_kernel_size_width, self.hsv_morph_kernel_size_height), np.uint8)
            hsv_mask = cv.erode(hsv_mask, hsv_kernel, iterations=self.hsv_morph_erode_iterations)
            hsv_mask = cv.dilate(hsv_mask, hsv_kernel, iterations=self.hsv_morph_dilate_iterations)
            
            # Create depth mask with scaled values
            depth_mask = cv.inRange(self.depth_image, self.depth_low, self.depth_high)

            # Apply morphological operations to depth mask
            depth_kernel = np.ones((self.depth_morph_kernel_size_width, self.depth_morph_kernel_size_height), np.uint8)
            depth_mask = cv.erode(depth_mask, depth_kernel, iterations=self.depth_morph_erode_iterations)
            depth_mask = cv.dilate(depth_mask, depth_kernel, iterations=self.depth_morph_dilate_iterations)

            # Combine masks
            combined_mask = cv.bitwise_and(hsv_mask, depth_mask)
            
            # Clean mask using connected components with area filtering
            cleaned_mask = self.clean_mask_with_area_filter(combined_mask)
            
            # Apply mask to get segmented results
            result_rgb = cv.bitwise_and(self.rgb_image, self.rgb_image, mask=cleaned_mask)
            result_depth = cv.bitwise_and(self.depth_image, self.depth_image, mask=cleaned_mask)
            
            # Publish all masks and results
            self.publish_image(result_rgb, self.result_rgb_pub, 'bgr8')
            self.publish_depth(result_depth, self.result_depth_pub)
            
            self.visualize_segmentation(
                self.rgb_image, hsv_mask, depth_mask, 
                combined_mask, cleaned_mask, result_rgb
            )
                
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def clean_mask_with_area_filter(self, mask):
        """Clean mask using connected components analysis with area range filtering."""
        # Compute connected components
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            mask, self.connectivity, cv.CV_32S
        )
        
        if num_labels <= 1:  # Only background
            return np.zeros_like(mask)
        
        # Filter components by area range (exclude background at index 0)
        valid_labels = []
        for i in range(1, num_labels):
            component_area = stats[i, cv.CC_STAT_AREA]
            if self.min_component_area <= component_area <= self.max_component_area:
                valid_labels.append(i)
        
        if not valid_labels:
            self.get_logger().debug(f"No components found in area range [{self.min_component_area}, {self.max_component_area}].")
            return np.zeros_like(mask)
        
        # Create cleaned mask
        cleaned_mask = np.zeros_like(mask)
        
        # Keep only the largest component within the area range
        areas = [stats[label, cv.CC_STAT_AREA] for label in valid_labels]
        largest_valid_idx = valid_labels[np.argmax(areas)]
        cleaned_mask = np.where(labels == largest_valid_idx, 255, 0)
        self.get_logger().debug(f"Kept largest valid component with area {max(areas)}.")

        return cleaned_mask.astype('uint8')

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
        
        # Create white background for result square and overlay the result
        result_resized = resize_to_viz(result)
        result_area = viz[480:960, 1280:1920]
        result_area.fill(255)
        mask = np.any(result_resized != [0, 0, 0], axis=2)
        result_area[mask] = result_resized[mask]
        
        # Add labels
        cv.putText(viz, "Original RGB", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "HSV Mask (with morph.)", (650, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv.putText(viz, "Depth Mask (with morph.)", (1290, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv.putText(viz, "Combined Mask", (10, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(viz, "Cleaned Mask", (650, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Result", (1290, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Publish the debug visualization
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
            self.debug_viz_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing debug visualization: {e}")    

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
            
            elif name == 'connectivity':
                if value not in [4, 8]:
                    return SetParametersResult(successful=False, reason="connectivity must be 4 or 8.")
                self.connectivity = value
                self.get_logger().info(f"connectivity updated: {value}.")
            
            elif name == 'min_component_area':
                if not isinstance(value, int) or value < 0:
                    return SetParametersResult(successful=False, reason="min_component_area must be a non-negative integer.")
                self.min_component_area = value
                self.get_logger().info(f"min_component_area updated: {value} pixels.")
            
            elif name == 'max_component_area':
                if not isinstance(value, int) or value <= 0:
                    return SetParametersResult(successful=False, reason="max_component_area must be a positive integer.")
                self.max_component_area = value
                self.get_logger().info(f"max_component_area updated: {value} pixels.")
            
            elif name in ('rgb_topic', 'depth_topic'):
                if not isinstance(value, str) or not value:
                    return SetParametersResult(successful=False, reason=f"{name} must be a non-empty string.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
                self.get_logger().warn("Topic changes require node restart to take effect.")

            elif name in ('gaussian_kernel_size_width', 'gaussian_kernel_size_height'):
                if not isinstance(value, int) or value < 1 or value % 2 == 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a positive odd integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")

            elif name == 'gaussian_sigma':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="gaussian_sigma must be > 0.")
                self.gaussian_sigma = float(value)
                self.get_logger().info(f"gaussian_sigma updated: {value}.")

            elif name in ('hsv_morph_kernel_size_width', 'hsv_morph_kernel_size_height',
                         'depth_morph_kernel_size_width', 'depth_morph_kernel_size_height'):
                if not isinstance(value, int) or value < 1:
                    return SetParametersResult(successful=False, reason=f"{name} must be a positive integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")

            elif name in ('hsv_morph_erode_iterations', 'hsv_morph_dilate_iterations',
                         'depth_morph_erode_iterations', 'depth_morph_dilate_iterations'):
                if not isinstance(value, int) or value < 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a non-negative integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
                
        return SetParametersResult(successful=True)

    def destroy_node(self):
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