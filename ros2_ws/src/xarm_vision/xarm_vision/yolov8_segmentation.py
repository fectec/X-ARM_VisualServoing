
#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image

class YoloV8Segmentation(Node):
    """
    Performs image segmentation using YOLOv8 and depth filtering.
    """
    def __init__(self):
        super().__init__('yolov8_segmentation')

        # Declare parameters
        self.declare_parameter('update_rate', 30.0)                     # Hz

        # Input topic selection
        self.declare_parameter('rgb_topic', '/k4a/rgb/image_raw')
        self.declare_parameter('depth_topic', '/k4a/depth_to_rgb/image_raw')
        
        # YOLOv8 parameters
        self.declare_parameter('model_name', 'oil_pan_segmentation.pt')
        self.declare_parameter('confidence_threshold', 0.7)
        
        # Depth filtering parameters 
        self.declare_parameter('depth_low', 0)                          # mm                        
        self.declare_parameter('depth_high', 2000)                      # mm
        
        # Connected components parameter
        self.declare_parameter('connectivity', 4) 
        
        # Area filtering parameters for connected components
        self.declare_parameter('min_component_area', 150000)               
        self.declare_parameter('max_component_area', 200000)             

        # Gaussian blur parameters
        self.declare_parameter('gaussian_kernel_size_width', 5)
        self.declare_parameter('gaussian_kernel_size_height', 5)
        self.declare_parameter('gaussian_sigma', 5.0)

        # Morphological operations parameters for YOLOv8 mask
        self.declare_parameter('yolo_morph_kernel_size_width', 3)
        self.declare_parameter('yolo_morph_kernel_size_height', 3)
        self.declare_parameter('yolo_morph_erode_iterations', 2)
        self.declare_parameter('yolo_morph_dilate_iterations', 2)

        # Morphological operations parameters for depth mask
        self.declare_parameter('depth_morph_kernel_size_width', 3)
        self.declare_parameter('depth_morph_kernel_size_height', 3)
        self.declare_parameter('depth_morph_erode_iterations', 8)
        self.declare_parameter('depth_morph_dilate_iterations', 8)

        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        
        self.model_name = self.get_parameter('model_name').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
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

        self.yolo_morph_kernel_size_width = self.get_parameter('yolo_morph_kernel_size_width').value
        self.yolo_morph_kernel_size_height = self.get_parameter('yolo_morph_kernel_size_height').value
        self.yolo_morph_erode_iterations = self.get_parameter('yolo_morph_erode_iterations').value
        self.yolo_morph_dilate_iterations = self.get_parameter('yolo_morph_dilate_iterations').value

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
            Parameter('model_name',                     Parameter.Type.STRING,  self.model_name),
            Parameter('confidence_threshold',           Parameter.Type.DOUBLE,  self.confidence_threshold),
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
            Parameter('yolo_morph_kernel_size_width',   Parameter.Type.INTEGER, self.yolo_morph_kernel_size_width),
            Parameter('yolo_morph_kernel_size_height',  Parameter.Type.INTEGER, self.yolo_morph_kernel_size_height),
            Parameter('yolo_morph_erode_iterations',    Parameter.Type.INTEGER, self.yolo_morph_erode_iterations),
            Parameter('yolo_morph_dilate_iterations',   Parameter.Type.INTEGER, self.yolo_morph_dilate_iterations),
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
        self.model = None
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Create publishers
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

        # Create subscribers
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
        
        self.get_logger().info("YoloV8Segmentation Node Start.")

    def _load_yolo_model(self):
        """Load YOLO segmentation model from package share directory."""
        try:
            package_share_dir = get_package_share_directory('xarm_vision')
            model_path = os.path.join(package_share_dir, 'models', self.model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}.")
                
            self.model = YOLO(model_path)
            self.get_logger().info(f'YOLO segmentation model loaded: {self.model_name}.')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

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

    def get_yolo_mask(self, image):
        """Get segmentation mask from YOLOv8 inference."""
        if self.model is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), image.copy()

        try:
            # Run YOLO inference
            results = self.model(image)
            
            # Initialize masks
            h, w = image.shape[:2]
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            annotated_frame = image.copy()
            
            # Find best detection
            best_detection = None
            best_confidence = 0.0
            
            for r in results:
                boxes = r.boxes
                masks = r.masks
                
                if boxes is not None and masks is not None:
                    for i, (box, mask) in enumerate(zip(boxes, masks)):
                        # Get confidence score
                        conf = float(box.conf[0].to('cpu').detach().numpy())
                        
                        # Only consider detections above confidence threshold
                        if conf >= self.confidence_threshold and conf > best_confidence:
                            # Get bounding box coordinates
                            b = box.xyxy[0].to('cpu').detach().numpy().copy()
                            
                            # Get class information
                            c = int(box.cls[0].to('cpu').detach().numpy())
                            class_name = self.model.names[c]
                            
                            # Get mask data
                            mask_data = mask.data[0].to('cpu').detach().numpy()
                            
                            best_detection = {
                                'coords': b,
                                'class_id': c,
                                'confidence': conf,
                                'class_name': class_name,
                                'mask': mask_data
                            }
                            best_confidence = conf
            
            # Process the best detection
            if best_detection is not None:
                # Process mask
                mask_resized = cv.resize(best_detection['mask'], (w, h), interpolation=cv.INTER_NEAREST)
                binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Create annotated frame
                coords = best_detection['coords']
                confidence = best_detection['confidence']
                class_name = best_detection['class_name']
                
                # Color for visualization (green for detected objects)
                color = (0, 255, 0)
                
                # Draw bounding box
                cv.rectangle(annotated_frame, 
                            (int(coords[0]), int(coords[1])), 
                            (int(coords[2]), int(coords[3])), 
                            color, 2)
                
                # Draw label with confidence
                label = f"{class_name} {confidence:.2f}"
                font_scale = 0.6
                thickness = 2
                label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Draw label background
                cv.rectangle(annotated_frame,
                            (int(coords[0]), int(coords[1]) - label_size[1] - 10),
                            (int(coords[0]) + label_size[0] + 10, int(coords[1])),
                            color, -1)
                
                # Draw label text
                cv.putText(annotated_frame, label,
                          (int(coords[0]) + 5, int(coords[1]) - 5),
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Apply mask overlay
                mask_overlay = annotated_frame.copy()
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                mask_overlay[mask_binary == 1] = color
                annotated_frame = cv.addWeighted(annotated_frame, 0.7, mask_overlay, 0.3, 0)
                
                self.get_logger().debug(f"YOLOv8 detection: {class_name} confidence {confidence:.2f}")
            
            return binary_mask, annotated_frame
            
        except Exception as e:
            self.get_logger().error(f"Error in YOLOv8 inference: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), image.copy()

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

            # Get YOLOv8 segmentation mask and annotated image
            yolo_mask, yolo_annotated = self.get_yolo_mask(blurred_rgb)

            # Apply morphological operations to YOLOv8 mask
            yolo_kernel = np.ones((self.yolo_morph_kernel_size_width, self.yolo_morph_kernel_size_height), np.uint8)
            yolo_mask = cv.erode(yolo_mask, yolo_kernel, iterations=self.yolo_morph_erode_iterations)
            yolo_mask = cv.dilate(yolo_mask, yolo_kernel, iterations=self.yolo_morph_dilate_iterations)
            
            # Create depth mask with scaled values
            depth_mask = cv.inRange(self.depth_image, self.depth_low, self.depth_high)

            # Apply morphological operations to depth mask
            depth_kernel = np.ones((self.depth_morph_kernel_size_width, self.depth_morph_kernel_size_height), np.uint8)
            depth_mask = cv.erode(depth_mask, depth_kernel, iterations=self.depth_morph_erode_iterations)
            depth_mask = cv.dilate(depth_mask, depth_kernel, iterations=self.depth_morph_dilate_iterations)

            # Combine masks
            combined_mask = cv.bitwise_and(yolo_mask, depth_mask)
            
            # Clean mask using connected components with area filtering
            cleaned_mask = self.clean_mask_with_area_filter(combined_mask)
            
            # Apply mask to get segmented results
            result_rgb = cv.bitwise_and(self.rgb_image, self.rgb_image, mask=cleaned_mask)
            result_depth = cv.bitwise_and(self.depth_image, self.depth_image, mask=cleaned_mask)
            
            # Publish all masks and results
            self.publish_image(result_rgb, self.result_rgb_pub, 'bgr8')
            self.publish_depth(result_depth, self.result_depth_pub)
            
            self.visualize_segmentation(
                self.rgb_image, yolo_mask, depth_mask, 
                combined_mask, cleaned_mask, result_rgb, yolo_annotated
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

    def visualize_segmentation(self, rgb, yolo_mask, depth_mask, combined_mask, cleaned_mask, result, yolo_annotated):
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
        
        # Top row: Original RGB, YOLOv8 annotated result, YOLOv8 mask
        viz[0:480, 0:640] = resize_to_viz(rgb)
        viz[0:480, 640:1280] = resize_to_viz(yolo_annotated)
        viz[0:480, 1280:1920] = resize_to_viz(yolo_mask, True)
        
        # Bottom row: Depth mask, Combined mask, Final result
        viz[480:960, 0:640] = resize_to_viz(depth_mask, True)
        viz[480:960, 640:1280] = resize_to_viz(combined_mask, True)
        
        # Create white background for result square and overlay the result
        result_resized = resize_to_viz(result)
        result_area = viz[480:960, 1280:1920]
        result_area.fill(255)
        mask = np.any(result_resized != [0, 0, 0], axis=2)
        result_area[mask] = result_resized[mask]
        
        # Add labels
        cv.putText(viz, "Original RGB", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "YOLOv8 Detection", (650, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(viz, "YOLOv8 Mask", (1290, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(viz, "Depth Mask", (10, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv.putText(viz, "Combined Mask", (650, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(viz, "Final Result", (1290, 510), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Publish the debug visualization
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
            self.debug_viz_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Error publishing debug visualization: {e}")    

    def parameter_callback(self, params: list[Parameter]) -> SetParametersResult:
        """Validates and applies updated node parameters."""
        model_changed = False
        
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
                     
            elif name == 'model_name':
                if not isinstance(value, str) or len(value.strip()) == 0:
                    return SetParametersResult(successful=False, reason="model_name must be a non-empty string.")
                if value != self.model_name:
                    self.model_name = value
                    model_changed = True
                    self.get_logger().info(f"model_name updated: {self.model_name}.")
            
            elif name == 'confidence_threshold':
                if not isinstance(value, (int, float)) or value < 0.0 or value > 1.0:
                    return SetParametersResult(successful=False, reason="confidence_threshold must be between 0.0 and 1.0.")
                self.confidence_threshold = float(value)
                self.get_logger().info(f"confidence_threshold updated: {self.confidence_threshold}.")
            
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
            
            elif name == 'rgb_topic':
                if not isinstance(value, str) or len(value.strip()) == 0:
                    return SetParametersResult(successful=False, reason="rgb_topic must be a non-empty string.")
                self.rgb_topic = value
                self.get_logger().info(f"rgb_topic updated: {value}.")
            
            elif name == 'depth_topic':
                if not isinstance(value, str) or len(value.strip()) == 0:
                    return SetParametersResult(successful=False, reason="depth_topic must be a non-empty string.")
                self.depth_topic = value
                self.get_logger().info(f"depth_topic updated: {value}.")
            
            elif name in ('gaussian_kernel_size_width', 'gaussian_kernel_size_height'):
                if not isinstance(value, int) or value <= 0 or value % 2 == 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a positive odd integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
            
            elif name == 'gaussian_sigma':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="gaussian_sigma must be > 0.")
                self.gaussian_sigma = float(value)
                self.get_logger().info(f"gaussian_sigma updated: {value}.")
            
            elif name in ('yolo_morph_kernel_size_width', 'yolo_morph_kernel_size_height', 
                         'depth_morph_kernel_size_width', 'depth_morph_kernel_size_height'):
                if not isinstance(value, int) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a positive integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
            
            elif name in ('yolo_morph_erode_iterations', 'yolo_morph_dilate_iterations',
                         'depth_morph_erode_iterations', 'depth_morph_dilate_iterations'):
                if not isinstance(value, int) or value < 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be a non-negative integer.")
                setattr(self, name, value)
                self.get_logger().info(f"{name} updated: {value}.")
        
        # Validate depth range
        if hasattr(self, 'depth_low') and hasattr(self, 'depth_high'):
            if self.depth_low >= self.depth_high:
                return SetParametersResult(successful=False, reason="depth_low must be < depth_high.")
        
        # Validate component area range
        if hasattr(self, 'min_component_area') and hasattr(self, 'max_component_area'):
            if self.min_component_area >= self.max_component_area:
                return SetParametersResult(successful=False, reason="min_component_area must be < max_component_area.")
        
        # Reload YOLO model if model_name changed
        if model_changed:
            try:
                self._load_yolo_model()
            except Exception as e:
                return SetParametersResult(successful=False, reason=f"Failed to load new model: {e}")
        
        return SetParametersResult(successful=True)
    
    def destroy_node(self):
        cv.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = YoloV8Segmentation()
    except Exception as e:
        print(f"[FATAL] YoloV8Segmentation failed to initialize: {e}.", file=sys.stderr)
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