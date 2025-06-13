#!/usr/bin/env python3

import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

# Open3D import
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("ERROR: Open3D is required for point cloud alignment. Please install it.")
    sys.exit(1)

class PointCloudsAlignmentScaling(Node):
    """
    Aligns and scales a canonical model point cloud with incoming segmented point cloud.
    Publishes aligned & scaled model, and TF transforms for visualization in RViz.
    """
    def __init__(self):
        super().__init__('point_clouds_alignment_scaling')
        
        # Declare parameters
        self.declare_parameter('update_rate', 10.0)                    # Hz
        self.declare_parameter('canonical_model_path', '')             # Path to canonical model PLY file
        self.declare_parameter('voxel_size', 0.002)                   
        self.declare_parameter('alignment_percentile', 98.0)           # Percentile for feature identification

        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.canonical_model_path = self.get_parameter('canonical_model_path').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.alignment_percentile = self.get_parameter('alignment_percentile').value
        
        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',            Parameter.Type.DOUBLE, self.update_rate),
            Parameter('canonical_model_path',   Parameter.Type.STRING, self.canonical_model_path),
            Parameter('voxel_size',             Parameter.Type.DOUBLE, self.voxel_size),
            Parameter('alignment_percentile',   Parameter.Type.DOUBLE, self.alignment_percentile),
        ]

        result = self.parameter_callback(init_params)
        if not result.successful:
            raise RuntimeError(f"Parameter validation failed: {result.reason}")
        
        # Initialize variables
        self.scan_pointcloud = None
        self.canonical_model = None
        self.scan_centroid = None
        
        # Load canonical model
        self.load_canonical_model()
        
        # Create subscriber for segmented point cloud
        self.create_subscription(
            PointCloud2,
            'pointcloud/segmented_object',
            self.pointcloud_callback,
            qos.qos_profile_sensor_data
        )
        
        # Fused aligned+scaled model publisher
        self.aligned_scaled_pub = self.create_publisher(
            PointCloud2,
            'pointcloud/aligned_scaled_model',
            qos.qos_profile_sensor_data
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info("PointCloudsAlignmentScaling Start.")

    def load_canonical_model(self):
        """Load canonical model point cloud."""
        if not self.canonical_model_path:
            raise RuntimeError("canonical_model_path parameter is required.")
            
        try:
            self.canonical_model = o3d.io.read_point_cloud(self.canonical_model_path)
            
            if len(self.canonical_model.points) == 0:
                raise RuntimeError("Canonical model is empty.")
                
            self.get_logger().info(
                f"Loaded canonical model: {len(self.canonical_model.points)} points from {self.canonical_model_path}."
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load canonical model: {e}")

    def pointcloud_callback(self, msg):
        """Store the incoming segmented point cloud."""
        try:
            # Convert PointCloud2 to Open3D format
            points = []
            colors = []
            
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
                x, y, z, rgb = point
                points.append([x, y, z])
                
                # Extract RGB from packed integer
                r = (rgb >> 16) & 0xFF
                g = (rgb >> 8) & 0xFF
                b = rgb & 0xFF
                colors.append([r/255.0, g/255.0, b/255.0])
            
            if len(points) > 0:
                self.scan_pointcloud = o3d.geometry.PointCloud()
                self.scan_pointcloud.points = o3d.utility.Vector3dVector(points)
                self.scan_pointcloud.colors = o3d.utility.Vector3dVector(colors)
                
                # Compute and store scan centroid for TF
                self.scan_centroid = self.compute_centroid(self.scan_pointcloud)
                
        except Exception as e:
            self.get_logger().error(f"Point cloud conversion error: {e}")

    def timer_callback(self):
        """Main processing loop - align and scale models."""
        if self.scan_pointcloud is None or self.canonical_model is None:
            return
            
        try:
            # Translate scan to center origin at centroid
            translated_scan = self.translate_to_centroid(self.scan_pointcloud)
            
            # Align and scale canonical model with translated scan
            aligned_model = self.align_models(self.canonical_model, translated_scan)
            scaled_model = self.scale_model(aligned_model, translated_scan)
            
            # Publish final aligned+scaled model
            self.publish_pointcloud(scaled_model, self.aligned_scaled_pub, 'object_center_frame')
            
            # Publish only object center TF (point cloud generator handles camera TF)
            self.publish_object_center_tf()
                    
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def compute_centroid(self, point_cloud):
        """Compute centroid of point cloud."""
        return np.mean(np.asarray(point_cloud.points), axis=0)

    def translate_to_centroid(self, point_cloud):
        """Translate point cloud so centroid is at origin."""
        centroid = self.compute_centroid(point_cloud)
        translated_points = np.asarray(point_cloud.points) - centroid
        
        translated_pcd = o3d.geometry.PointCloud()
        translated_pcd.points = o3d.utility.Vector3dVector(translated_points)
        translated_pcd.colors = point_cloud.colors
        
        return translated_pcd

    def identify_feature_point(self, pcd, axis=2):
        """Identify defining feature point using percentile on specified axis."""
        coordinates = np.asarray(pcd.points)[:, axis]
        threshold = np.percentile(coordinates, self.alignment_percentile)
        top_points = np.asarray(pcd.points)[coordinates > threshold]
        return np.mean(top_points, axis=0)

    def identify_feature_point_scan(self, pcd, axis=0):
        """Identify defining feature point for scan using different axis."""
        coordinates = np.asarray(pcd.points)[:, axis]
        threshold = np.percentile(coordinates, self.alignment_percentile)
        top_points = np.asarray(pcd.points)[coordinates > threshold]
        return np.mean(top_points, axis=0)

    def compute_feature_line(self, pcd, use_scan_axis=False):
        """Compute feature line from centroid to defining feature."""
        centroid = self.compute_centroid(pcd)
        
        if use_scan_axis:
            feature_point = self.identify_feature_point_scan(pcd)
        else:
            feature_point = self.identify_feature_point(pcd)
        
        direction_vector = feature_point - centroid
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        return normalized_vector, centroid, feature_point

    def compute_rotation_matrix(self, v1, v2):
        """Compute rotation matrix to align v1 with v2."""
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute rotation axis and angle
        rotation_axis = np.cross(v1, v2)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:  # Vectors are parallel
            return np.eye(3)
            
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.dot(v1, v2)
        rotation_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        I = np.eye(3)
        
        R = I + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
        return R

    def rotation_matrix_around_z(self, angle_rad):
        """Create rotation matrix around Z axis."""
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    def align_models(self, source_model, target_scan):
        """Align source model with target scan using feature lines."""
        # Compute feature lines
        source_direction, _, _ = self.compute_feature_line(source_model)
        target_direction, _, _ = self.compute_feature_line(target_scan, use_scan_axis=True)
        
        # Initial rotation to align feature directions
        R_initial = self.compute_rotation_matrix(source_direction, target_direction)
        source_points = np.asarray(source_model.points)
        rotated_points = np.dot(source_points, R_initial.T)
        
        # Additional 180-degree rotation around Z if needed (common case)
        R_180 = self.rotation_matrix_around_z(np.pi)
        final_points = np.dot(rotated_points, R_180.T)
        
        # Create aligned point cloud
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(final_points)
        if source_model.colors:
            aligned_pcd.colors = source_model.colors
        
        return aligned_pcd

    def get_dimensions(self, pcd):
        """Get bounding box dimensions of point cloud."""
        bounding_box = pcd.get_axis_aligned_bounding_box()
        return bounding_box.get_extent()

    def scale_model(self, source_pcd, target_pcd):
        """Scale source point cloud to match target dimensions."""
        source_dimensions = self.get_dimensions(source_pcd)
        target_dimensions = self.get_dimensions(target_pcd)
        
        scale_factors = [
            target_dimensions[i] / source_dimensions[i]
            for i in range(3)
        ]
        
        scaled_points = [
            [scale_factors[j] * pt[j] for j in range(3)]
            for pt in source_pcd.points
        ]
        
        scaled_pcd = o3d.geometry.PointCloud()
        scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        if source_pcd.colors:
            scaled_pcd.colors = source_pcd.colors
        
        return scaled_pcd

    def publish_pointcloud(self, pcd, publisher, frame_id):
        """Publish point cloud as PointCloud2 message."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.colors else None
        
        if len(points) == 0:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # Prepare point cloud data
        cloud_data = []
        for i in range(len(points)):
            if colors is not None and len(colors) > i:
                rgb = (int(colors[i][0] * 255) << 16 | 
                       int(colors[i][1] * 255) << 8 | 
                       int(colors[i][2] * 255))
            else:
                rgb = 0xFFFFFF  # White default
                
            cloud_data.append([
                float(points[i][0]), 
                float(points[i][1]), 
                float(points[i][2]), 
                int(rgb)
            ])
        
        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # Create and publish message
        pc2_msg = pc2.create_cloud(header, fields, cloud_data)
        publisher.publish(pc2_msg)

    def publish_object_center_tf(self):
        """Publish TF transform: camera_color_optical_frame -> object_center_frame."""
        current_time = self.get_clock().now().to_msg()
        
        # TF: camera_color_optical_frame -> object_center_frame
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'camera_color_optical_frame'
        t.child_frame_id = 'object_center_frame'
        
        if self.scan_centroid is not None:
            t.transform.translation.x = float(self.scan_centroid[0])
            t.transform.translation.y = float(self.scan_centroid[1])
            t.transform.translation.z = float(self.scan_centroid[2])
        else:
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
        
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)

    def parameter_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            name = param.name
            value = param.value

            if name == 'update_rate':
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason="update_rate must be > 0.")
                self.update_rate = float(value)
                self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
                self.get_logger().info(f"Updated update_rate: {self.update_rate} Hz.")

            elif name in ['voxel_size', 'alignment_percentile']:
                if not isinstance(value, (int, float)) or value <= 0:
                    return SetParametersResult(successful=False, reason=f"{name} must be > 0.")
                setattr(self, name.replace('alignment_', ''), float(value))
                self.get_logger().info(f"Updated {name}: {value}")

            elif name == 'canonical_model_path':
                if not isinstance(value, str):
                    return SetParametersResult(successful=False, reason="canonical_model_path must be a string.")
                self.canonical_model_path = value
                self.get_logger().info(f"canonical_model_path updated: {value}.")
                self.load_canonical_model()

        return SetParametersResult(successful=True)

    def destroy_node(self):
        """Clean up before shutting down."""
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudsAlignmentScaling()
    except Exception as e:
        print(f"[FATAL] PointCloudsAlignmentScaling failed to initialize: {e}", file=sys.stderr)
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