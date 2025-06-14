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
    print("ERROR: Open3D is required for point cloud processing. Please install it.")
    sys.exit(1)

class PointCloudScaling(Node):
    """
    Scales canonical model point cloud to match incoming segmented point cloud.
    """
    def __init__(self):
        super().__init__('point_cloud_scaling')
        
        # Declare parameters
        self.declare_parameter('update_rate', 5.0)                     # Hz
        self.declare_parameter('canonical_model_path', '')             # Path to canonical model PLY file
        self.declare_parameter('scaling_mode', 'uniform')              # 'uniform', 'per_axis', 'largest_dim'

        # Retrieve parameters
        self.update_rate = self.get_parameter('update_rate').value
        self.canonical_model_path = self.get_parameter('canonical_model_path').value
        self.scaling_mode = self.get_parameter('scaling_mode').value
        
        # Timer for periodic processing
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        # Register the on-set-parameters callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Immediately validate the initial values
        init_params = [
            Parameter('update_rate',            Parameter.Type.DOUBLE, self.update_rate),
            Parameter('canonical_model_path',   Parameter.Type.STRING, self.canonical_model_path),
            Parameter('scaling_mode',           Parameter.Type.STRING, self.scaling_mode),
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
            qos.QoSProfile(
                reliability=qos.ReliabilityPolicy.RELIABLE,
                durability=qos.DurabilityPolicy.VOLATILE,
                history=qos.HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        
        # Publisher for scaled model at object center
        self.scaled_model_pub = self.create_publisher(
            PointCloud2,
            'pointcloud/scaled_canonical_model',
            qos.QoSProfile(
                reliability=qos.ReliabilityPolicy.RELIABLE,
                durability=qos.DurabilityPolicy.VOLATILE,
                history=qos.HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info(f"PointCloudsScaling Start.")

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
                
                # Compute and store scan centroid for positioning
                self.scan_centroid = self.compute_centroid(self.scan_pointcloud)
                
        except Exception as e:
            self.get_logger().error(f"Point cloud conversion error: {e}")

    def timer_callback(self):
        """Main processing loop - scale and position canonical model."""
        if self.scan_pointcloud is None or self.canonical_model is None or self.scan_centroid is None:
            return
            
        try:
            # Scale canonical model to match scan dimensions
            scaled_model = self.scale_model(self.canonical_model, self.scan_pointcloud)
            
            # Position scaled model at scan centroid
            positioned_model = self.position_model_at_centroid(scaled_model, self.scan_centroid)
            
            # Publish final scaled+positioned model
            self.publish_pointcloud(positioned_model, self.scaled_model_pub, 'camera_color_optical_frame')
            
            # Publish object center TF
            self.publish_object_center_tf()
                    
        except Exception as e:
            self.get_logger().error(f"Error in timer_callback: {e}")

    def compute_centroid(self, point_cloud):
        """Compute centroid of point cloud."""
        return np.mean(np.asarray(point_cloud.points), axis=0)

    def get_dimensions(self, pcd):
        """Get bounding box dimensions of point cloud."""
        bounding_box = pcd.get_axis_aligned_bounding_box()
        return bounding_box.get_extent()

    def scale_model(self, source_pcd, target_pcd):
        """Scale source point cloud to match target dimensions based on scaling mode."""
        source_dimensions = self.get_dimensions(source_pcd)
        target_dimensions = self.get_dimensions(target_pcd)
        
        if self.scaling_mode == 'uniform':
            # Use average scale factor for uniform scaling
            scale_factors = [
                target_dimensions[i] / source_dimensions[i] 
                for i in range(3) if source_dimensions[i] > 0
            ]
            avg_scale = np.mean(scale_factors) if scale_factors else 1.0
            scale_factors = [avg_scale, avg_scale, avg_scale]
            
        elif self.scaling_mode == 'largest_dim':
            # Scale based on largest dimension match
            source_max = np.max(source_dimensions)
            target_max = np.max(target_dimensions)
            scale_factor = target_max / source_max if source_max > 0 else 1.0
            scale_factors = [scale_factor, scale_factor, scale_factor]
            
        else:  # 'per_axis'
            # Scale each axis independently
            scale_factors = [
                target_dimensions[i] / source_dimensions[i] if source_dimensions[i] > 0 else 1.0
                for i in range(3)
            ]
        
        # Apply scaling
        source_points = np.asarray(source_pcd.points)
        scaled_points = source_points * np.array(scale_factors)
        
        scaled_pcd = o3d.geometry.PointCloud()
        scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
        if source_pcd.colors:
            scaled_pcd.colors = source_pcd.colors
        
        self.get_logger().debug(f"Scale factors: {scale_factors}")
        return scaled_pcd

    def position_model_at_centroid(self, model, target_centroid):
        """Position the model so its centroid aligns with target centroid."""
        # Get current model centroid
        model_centroid = self.compute_centroid(model)
        
        # Calculate translation vector
        translation = target_centroid - model_centroid
        
        # Apply translation
        translated_points = np.asarray(model.points) + translation
        
        positioned_model = o3d.geometry.PointCloud()
        positioned_model.points = o3d.utility.Vector3dVector(translated_points)
        if model.colors:
            positioned_model.colors = model.colors
        
        return positioned_model

    def publish_pointcloud(self, pcd, publisher, frame_id):
        """Publish point cloud as PointCloud2 message."""
        points = np.asarray(pcd.points)
        
        colors = np.zeros((len(points), 3))  
        colors[:, 0] = 0.0  
        colors[:, 1] = 0.0  
        colors[:, 2] = 1.0 
        
        if len(points) == 0:
            return
            
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        
        # Prepare point cloud data
        cloud_data = []
        for i in range(len(points)):
            rgb = (int(colors[i][0] * 255) << 16 | 
                int(colors[i][1] * 255) << 8 | 
                int(colors[i][2] * 255))
                
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

            elif name == 'scaling_mode':
                if value not in ['uniform', 'per_axis', 'largest_dim']:
                    return SetParametersResult(successful=False, reason="scaling_mode must be 'uniform', 'per_axis', or 'largest_dim'.")
                self.scaling_mode = value
                self.get_logger().info(f"Updated scaling_mode: {value}")

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
        node = PointCloudScaling()
    except Exception as e:
        print(f"[FATAL] PointCloudScaling failed to initialize: {e}", file=sys.stderr)
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