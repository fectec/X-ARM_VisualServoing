#!/usr/bin/env python3

import sys
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import json
import os

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point
import sensor_msgs_py.point_cloud2 as pc2

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped



# Open3D imports
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Point cloud generation will be limited.")

# YAML import for OpenCV calibration files
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available. YAML calibration files won't be supported.")

class PointCloudGeneratorNode(Node):
    """
    Generates point clouds from segmented RGB-D images using Open3D.
    Provides a service to generate point clouds and publishes them for visualization.
    """
    
    def __init__(self):
        super().__init__('pointcloud_generator_node')
        
        # Declare parameters
        self.declare_parameter('camera_intrinsics_file', '')
        self.declare_parameter('depth_scale', 1000.0)  # Conversion factor from depth units to meters
        self.declare_parameter('auto_generate', False)  # Auto-generate on new images
        self.declare_parameter('publish_rate', 1.0)     # Hz for auto publishing
        self.declare_parameter('max_depth', 2.0)        # Max depth in meters
        self.declare_parameter('min_depth', 0.1)        # Min depth in meters
        self.declare_parameter('voxel_size', 0.002)     # Voxel size for downsampling
        self.declare_parameter('outlier_removal', True) # Remove outliers
        self.declare_parameter('statistical_outlier_nb_neighbors', 20)
        self.declare_parameter('statistical_outlier_std_ratio', 2.0)
        
        # Camera intrinsic parameters (as fallback or direct use)
        self.declare_parameter('use_intrinsics_from_params', False)
        self.declare_parameter('image_width', 2048)
        self.declare_parameter('image_height', 1536)
        self.declare_parameter('fx', 897.43423)
        self.declare_parameter('fy', 907.31646)
        self.declare_parameter('cx', 1048.95001)
        self.declare_parameter('cy', 765.7388)
        self.declare_parameter('camera_frame_id', 'camera_color_optical_frame')
        
        # Distortion parameters (optional)
        self.declare_parameter('k1', 0.017184)
        self.declare_parameter('k2', -0.019966)
        self.declare_parameter('p1', -0.000088)
        self.declare_parameter('p2', 0.002199)
        self.declare_parameter('k3', 0.000000)
        
        # Get parameters
        self.intrinsics_file = self.get_parameter('camera_intrinsics_file').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.auto_generate = self.get_parameter('auto_generate').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.outlier_removal = self.get_parameter('outlier_removal').value
        self.stat_nb_neighbors = self.get_parameter('statistical_outlier_nb_neighbors').value
        self.stat_std_ratio = self.get_parameter('statistical_outlier_std_ratio').value
        
        # Initialize variables
        self.rgb_image = None
        self.depth_image = None
        self.cleaned_mask = None
        self.bridge = CvBridge()
        self.camera_intrinsic = None
        self.latest_pointcloud = None
        
        # Load camera intrinsics
        self.load_camera_intrinsics()
        
        # Create subscribers to segmented images
        self.create_subscription(
            Image,
            'segmentation/result_rgb',
            self.rgb_callback,
            qos.qos_profile_sensor_data
        )
        
        self.create_subscription(
            Image,
            'segmentation/result_depth',
            self.depth_callback,
            qos.qos_profile_sensor_data
        )
        
        self.create_subscription(
            Image,
            'segmentation/cleaned_mask',
            self.mask_callback,
            qos.qos_profile_sensor_data
        )
        
        # Create point cloud publisher
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            'pointcloud/segmented_object',
            10
        )
        
        # Create point cloud timer publisher
        self.pointcloud_timer = self.create_timer(
            1.0 / self.publish_rate,
            self.generate_pointcloud
        )

        # Create service for manual point cloud generation
        self.generate_service = self.create_service(
            Trigger,
            'generate_pointcloud',
            self.generate_pointcloud_callback
        )

        # Create TF broadcaster for point cloud frame
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for auto generation
        if self.auto_generate:
            self.timer = self.create_timer(
                1.0 / self.publish_rate,
                self.auto_generate_callback
            )
        
        # Parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info("PointCloud Generator Node Started.")
        if not OPEN3D_AVAILABLE:
            self.get_logger().warn("Open3D not available - using basic point cloud generation")

    def publish_camera_tf(self):
        """Publish camera transform."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'camera_link'  # Frame padre
        t.child_frame_id = 'camera_color_optical_frame'
        
        # Transform de identidad (sin rotación ni traslación)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)
    
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file or parameters."""
        # Check if we should use parameters directly
        use_params = self.get_parameter('use_intrinsics_from_params').value
        
        if use_params:
            # Use intrinsic parameters directly from launch file
            try:
                width = self.get_parameter('image_width').value
                height = self.get_parameter('image_height').value
                fx = self.get_parameter('fx').value
                fy = self.get_parameter('fy').value
                cx = self.get_parameter('cx').value
                cy = self.get_parameter('cy').value
                
                # Create Open3D camera intrinsic object
                if OPEN3D_AVAILABLE:
                    self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                
                # Store parameters for manual calculation
                self.fx, self.fy = fx, fy
                self.cx, self.cy = cx, cy
                self.image_width, self.image_height = width, height
                self.camera_frame_id = self.get_parameter('camera_frame_id').value
                
                self.get_logger().info("Using camera intrinsics from parameters:")
                self.get_logger().info(f"  Resolution: {width}x{height}")
                self.get_logger().info(f"  fx: {fx:.2f}, fy: {fy:.2f}")
                self.get_logger().info(f"  cx: {cx:.2f}, cy: {cy:.2f}")
                return
                
            except Exception as e:
                self.get_logger().error(f"Failed to load intrinsics from parameters: {e}")
                self.get_logger().warn("Falling back to file or defaults")
        
        # Try to load from file
        if self.intrinsics_file and os.path.exists(self.intrinsics_file):
            try:
                # Determine file type by extension
                file_ext = os.path.splitext(self.intrinsics_file)[1].lower()
                
                if file_ext in ['.yaml', '.yml']:
                    # Load YAML format (OpenCV calibration format)
                    if not YAML_AVAILABLE:
                        self.get_logger().error("PyYAML not available, cannot load YAML file")
                        self.set_default_intrinsics()
                        return
                        
                    import yaml
                    with open(self.intrinsics_file, 'r') as f:
                        calibration_data = yaml.safe_load(f)
                    
                    # Extract parameters from OpenCV calibration format
                    if 'camera_matrix' in calibration_data:
                        camera_matrix_data = calibration_data['camera_matrix']['data']
                        width = calibration_data['image_width']
                        height = calibration_data['image_height']
                        
                        # Camera matrix is stored as [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                        fx = camera_matrix_data[0]  # [0,0]
                        fy = camera_matrix_data[4]  # [1,1] 
                        cx = camera_matrix_data[2]  # [0,2]
                        cy = camera_matrix_data[5]  # [1,2]
                        
                        self.get_logger().info(f"Loaded OpenCV calibration format:")
                        self.get_logger().info(f"  Resolution: {width}x{height}")
                        self.get_logger().info(f"  fx: {fx:.2f}, fy: {fy:.2f}")
                        self.get_logger().info(f"  cx: {cx:.2f}, cy: {cy:.2f}")
                        
                    else:
                        raise ValueError("Invalid YAML format: missing 'camera_matrix'")
                        
                elif file_ext == '.json':
                    # Load JSON format (your original format)
                    with open(self.intrinsics_file, 'r') as f:
                        intrinsic_json = json.load(f)
                    
                    # Convert flat list to 3x3 matrix if needed
                    if isinstance(intrinsic_json['intrinsic_matrix'], list):
                        intrinsic_matrix_flat = intrinsic_json['intrinsic_matrix']
                        if len(intrinsic_matrix_flat) == 9:
                            intrinsic_matrix = [
                                intrinsic_matrix_flat[0:3],
                                intrinsic_matrix_flat[3:6],
                                intrinsic_matrix_flat[6:9],
                            ]
                        else:
                            intrinsic_matrix = intrinsic_matrix_flat
                    else:
                        intrinsic_matrix = intrinsic_json['intrinsic_matrix']
                    
                    width = intrinsic_json['width']
                    height = intrinsic_json['height']
                    fx = intrinsic_matrix[0][0]
                    fy = intrinsic_matrix[1][1]
                    cx = intrinsic_matrix[0][2]
                    cy = intrinsic_matrix[1][2]
                    
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                # Create Open3D camera intrinsic object
                if OPEN3D_AVAILABLE:
                    self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                
                # Store parameters for manual calculation
                self.fx, self.fy = fx, fy
                self.cx, self.cy = cx, cy
                self.image_width, self.image_height = width, height
                self.camera_frame_id = self.get_parameter('camera_frame_id').value
                
                self.get_logger().info(f"Successfully loaded camera intrinsics from {self.intrinsics_file}")
                
            except Exception as e:
                self.get_logger().error(f"Failed to load intrinsics: {e}")
                self.get_logger().error(f"File format should be YAML (OpenCV) or JSON")
                self.set_default_intrinsics()
        else:
            # Use parameter values as fallback
            self.get_logger().warn("No intrinsics file found, using parameter values as fallback")
            try:
                width = self.get_parameter('image_width').value
                height = self.get_parameter('image_height').value
                fx = self.get_parameter('fx').value
                fy = self.get_parameter('fy').value
                cx = self.get_parameter('cx').value
                cy = self.get_parameter('cy').value
                
                if OPEN3D_AVAILABLE:
                    self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        width, height, fx, fy, cx, cy
                    )
                
                self.fx, self.fy = fx, fy
                self.cx, self.cy = cx, cy
                self.image_width, self.image_height = width, height
                self.camera_frame_id = self.get_parameter('camera_frame_id').value
                
                self.get_logger().info("Using fallback intrinsics from parameters")
                
            except Exception as e:
                self.get_logger().error(f"Failed to load fallback parameters: {e}")
                self.set_default_intrinsics()

    def set_default_intrinsics(self):
        """Set default Azure Kinect DK intrinsics."""
        # Default Azure Kinect DK color camera intrinsics (approximate)
        width, height = 1280, 720
        fx, fy = 977.0, 977.0
        cx, cy = 640.0, 360.0
        
        if OPEN3D_AVAILABLE:
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, fx, fy, cx, cy
            )
        
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        
        self.get_logger().info("Using default Azure Kinect intrinsics")

    def rgb_callback(self, msg):
        """Callback for segmented RGB image."""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"RGB CvBridgeError: {e}")

    def depth_callback(self, msg):
        """Callback for segmented depth image."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if self.depth_image.dtype != np.uint16:
                self.depth_image = self.depth_image.astype(np.uint16)
        except CvBridgeError as e:
            self.get_logger().error(f"Depth CvBridgeError: {e}")

    def mask_callback(self, msg):
        """Callback for cleaned mask."""
        try:
            self.cleaned_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            self.get_logger().error(f"Mask CvBridgeError: {e}")

    def generate_pointcloud_callback(self, request, response):
        """Service callback to generate point cloud."""
        try:
            if self.generate_pointcloud():
                response.success = True
                response.message = "Point cloud generated successfully"
                self.get_logger().info("Point cloud generated via service")
            else:
                response.success = False
                response.message = "Failed to generate point cloud - missing data"
        except Exception as e:
            response.success = False
            response.message = f"Error generating point cloud: {str(e)}"
            self.get_logger().error(f"Service error: {e}")
        
        return response

    def auto_generate_callback(self):
        """Timer callback for automatic point cloud generation."""
        if self.generate_pointcloud():
            self.get_logger().debug("Auto-generated point cloud")

    def generate_pointcloud(self):
        """Generate point cloud from current segmented images."""
        if self.rgb_image is None or self.depth_image is None or self.cleaned_mask is None:
            return False
        
        try:
            if OPEN3D_AVAILABLE:
                pointcloud = self.generate_with_open3d()
            else:
                pointcloud = self.generate_manual()
            
            if pointcloud is not None:
                self.latest_pointcloud = pointcloud
                self.publish_pointcloud(pointcloud)
                return True
            
        except Exception as e:
            self.get_logger().error(f"Error generating point cloud: {e}")
        
        return False

    def generate_with_open3d(self):
        """Generate point cloud using Open3D (preferred method)."""
        # Apply mask to get final segmented images
        result_rgb = cv.bitwise_and(self.rgb_image, self.rgb_image, mask=self.cleaned_mask)
        result_depth = cv.bitwise_and(self.depth_image, self.depth_image, mask=self.cleaned_mask)
        
        # Convert to Open3D images
        o3d_color = o3d.geometry.Image(result_rgb)
        o3d_depth = o3d.geometry.Image(result_depth)
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth, 
            depth_scale=self.depth_scale,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, 
            self.camera_intrinsic
        )
        
        # Transform coordinate system (flip Y and Z)
        pcd.transform([[1, 0, 0, 0], 
                       [0, -1, 0, 0], 
                       [0, 0, -1, 0], 
                       [0, 0, 0, 1]])
        
        # Apply filters
        if len(pcd.points) > 0:
            # Remove points outside depth range
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Filter by depth (Z coordinate after transformation)
            valid_depth = (np.abs(points[:, 2]) >= self.min_depth) & (np.abs(points[:, 2]) <= self.max_depth)
            
            if np.any(valid_depth):
                pcd.points = o3d.utility.Vector3dVector(points[valid_depth])
                pcd.colors = o3d.utility.Vector3dVector(colors[valid_depth])
                
                # Downsample
                if self.voxel_size > 0:
                    pcd = pcd.voxel_down_sample(self.voxel_size)
                
                # Remove outliers
                if self.outlier_removal and len(pcd.points) > self.stat_nb_neighbors:
                    pcd, _ = pcd.remove_statistical_outlier(
                        nb_neighbors=self.stat_nb_neighbors,
                        std_ratio=self.stat_std_ratio
                    )
                
                return pcd
        
        return None

    def generate_manual(self):
        """Generate point cloud manually without Open3D."""
        # Apply mask to get valid pixels
        mask_indices = np.where(self.cleaned_mask > 0)
        
        if len(mask_indices[0]) == 0:
            return None
        
        # Get depth values
        depth_values = self.depth_image[mask_indices] / self.depth_scale
        
        # Filter by depth range
        valid_depth = (depth_values >= self.min_depth) & (depth_values <= self.max_depth)
        
        if not np.any(valid_depth):
            return None
        
        # Get valid pixel coordinates
        u = mask_indices[1][valid_depth]  # x coordinates
        v = mask_indices[0][valid_depth]  # y coordinates
        z = depth_values[valid_depth]
        
        # Convert to 3D points using camera intrinsics
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Get corresponding colors
        colors = self.rgb_image[v, u]  # BGR format
        colors = colors[:, [2, 1, 0]]  # Convert to RGB
        
        # Create point cloud data
        points = np.column_stack((x, -y, -z))  # Apply coordinate transform
        colors_normalized = colors.astype(np.float32) / 255.0
        
        return {
            'points': points,
            'colors': colors_normalized
        }

    def publish_pointcloud(self, pointcloud):
        """Publish point cloud as ROS2 PointCloud2 message."""
        if pointcloud is None:
            return
        
        try:
            if OPEN3D_AVAILABLE and isinstance(pointcloud, o3d.geometry.PointCloud):
                points = np.asarray(pointcloud.points)
                colors = np.asarray(pointcloud.colors)
            else:
                points = pointcloud['points']
                colors = pointcloud['colors']
            
            if len(points) == 0:
                return
            
            # Create proper header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = getattr(self, 'camera_frame_id', 'camera_color_optical_frame')
            
            # Create point cloud data with RGB
            cloud_data = []
            for i in range(len(points)):
                point = [
                    float(points[i][0]),  # x
                    float(points[i][1]),  # y
                    float(points[i][2]),  # z
                    int(colors[i][0] * 255) << 16 |  # r
                    int(colors[i][1] * 255) << 8 |   # g
                    int(colors[i][2] * 255)          # b
                ]
                cloud_data.append(point)
            
            # Define point fields
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            ]
            
            # Create PointCloud2 message with proper header
            pc2_msg = pc2.create_cloud(
                header=header,
                fields=fields,
                points=cloud_data
            )
            
            self.pointcloud_pub.publish(pc2_msg)

            self.publish_camera_tf()
            
            self.get_logger().debug(f"Published point cloud with {len(points)} points")
            
        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")

    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value
            
            if name == 'depth_scale':
                if value > 0:
                    self.depth_scale = float(value)
                    self.get_logger().info(f"Updated depth_scale: {value}")
            
            elif name == 'auto_generate':
                self.auto_generate = bool(value)
                if self.auto_generate and not hasattr(self, 'timer'):
                    self.timer = self.create_timer(1.0 / self.publish_rate, self.auto_generate_callback)
                elif not self.auto_generate and hasattr(self, 'timer'):
                    self.timer.cancel()
                    delattr(self, 'timer')
                self.get_logger().info(f"Updated auto_generate: {value}")
            
            elif name == 'publish_rate':
                if value > 0:
                    self.publish_rate = float(value)
                    if hasattr(self, 'timer'):
                        self.timer.cancel()
                        self.timer = self.create_timer(1.0 / self.publish_rate, self.auto_generate_callback)
                    self.get_logger().info(f"Updated publish_rate: {value} Hz")
            
            elif name in ['max_depth', 'min_depth', 'voxel_size']:
                if value >= 0:
                    setattr(self, name, float(value))
                    self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'outlier_removal':
                self.outlier_removal = bool(value)
                self.get_logger().info(f"Updated outlier_removal: {value}")
            
            elif name in ['statistical_outlier_nb_neighbors']:
                if value > 0:
                    setattr(self, name.replace('statistical_', 'stat_').replace('_neighbors', '_nb_neighbors'), int(value))
                    self.get_logger().info(f"Updated {name}: {value}")
            
            elif name == 'statistical_outlier_std_ratio':
                if value > 0:
                    self.stat_std_ratio = float(value)
                    self.get_logger().info(f"Updated {name}: {value}")
        
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudGeneratorNode()
    except Exception as e:
        print(f"[FATAL] PointCloudGenerator failed to initialize: {e}", file=sys.stderr)
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