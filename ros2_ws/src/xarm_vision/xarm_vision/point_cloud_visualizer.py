#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy import qos
import numpy as np
import cv2 as cv
import threading

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class PointCloudVisualizerNode(Node):
    """
    Visualizes point clouds in real-time using Open3D or Matplotlib.
    Subscribes to PointCloud2 messages and displays them in a 3D viewer.
    """
    
    def __init__(self):
        super().__init__('point_cloud_visualizer')
        
        # Declare parameters
        self.declare_parameter('visualizer_backend', 'open3d')  # 'open3d', 'matplotlib', or 'both'
        self.declare_parameter('update_rate', 10.0)  # Hz
        self.declare_parameter('max_points_display', 10000)  # Limit for performance
        self.declare_parameter('point_size', 1.0)
        self.declare_parameter('background_color', [0.0, 0.0, 0.0])  # RGB
        self.declare_parameter('auto_rotate', False)
        self.declare_parameter('save_snapshots', False)
        self.declare_parameter('snapshot_directory', '/tmp/pointcloud_snapshots')
        
        # Get parameters
        self.backend = self.get_parameter('visualizer_backend').value
        self.update_rate = self.get_parameter('update_rate').value
        self.max_points = self.get_parameter('max_points_display').value
        self.point_size = self.get_parameter('point_size').value
        self.bg_color = self.get_parameter('background_color').value
        self.auto_rotate = self.get_parameter('auto_rotate').value
        self.save_snapshots = self.get_parameter('save_snapshots').value
        self.snapshot_dir = self.get_parameter('snapshot_directory').value
        
        # Initialize variables
        self.latest_pointcloud = None
        self.pointcloud_lock = threading.Lock()
        self.frame_count = 0
        
        # Setup visualization based on backend
        self.setup_visualization()
        
        # Create subscriber
        self.create_subscription(
            PointCloud2,
            'pointcloud/segmented_object',
            self.pointcloud_callback,
            qos.qos_profile_sensor_data
        )
        
        # Create timer for visualization updates
        self.timer = self.create_timer(
            1.0 / self.update_rate,
            self.update_visualization
        )
        
        self.get_logger().info(f"PointCloud Visualizer started with backend: {self.backend}")
        
        if self.save_snapshots:
            import os
            os.makedirs(self.snapshot_dir, exist_ok=True)
            self.get_logger().info(f"Saving snapshots to: {self.snapshot_dir}")

    def setup_visualization(self):
        """Initialize the visualization backend."""
        if self.backend in ['open3d', 'both'] and OPEN3D_AVAILABLE:
            self.setup_open3d()
        
        if self.backend in ['matplotlib', 'both'] and MATPLOTLIB_AVAILABLE:
            self.setup_matplotlib()
        
        if not OPEN3D_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            self.get_logger().error("No visualization backends available!")
            raise RuntimeError("No visualization libraries found")

    def setup_open3d(self):
        """Setup Open3D visualization."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name="Point Cloud Viewer",
            width=1024,
            height=768
        )
        
        # Set background color
        opt = self.vis.get_render_option()
        opt.background_color = np.array(self.bg_color)
        opt.point_size = self.point_size
        
        # Add coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coordinate_frame)
        
        # Initialize empty point cloud
        self.o3d_pointcloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.o3d_pointcloud)
        
        self.get_logger().info("Open3D visualizer initialized")

    def setup_matplotlib(self):
        """Setup Matplotlib 3D visualization."""
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Point Cloud Visualization')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        self.get_logger().info("Matplotlib visualizer initialized")

    def pointcloud_callback(self, msg):
        """Callback for point cloud messages."""
        try:
            # Extract points and colors from PointCloud2
            points = []
            colors = []
            
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
                if len(points) >= self.max_points:
                    break
                
                x, y, z, rgb = point
                points.append([x, y, z])
                
                # Extract RGB from packed integer
                rgb_int = int(rgb) if not np.isnan(rgb) else 0
                r = (rgb_int >> 16) & 0xFF
                g = (rgb_int >> 8) & 0xFF
                b = rgb_int & 0xFF
                colors.append([r/255.0, g/255.0, b/255.0])
            
            if len(points) > 0:
                with self.pointcloud_lock:
                    self.latest_pointcloud = {
                        'points': np.array(points),
                        'colors': np.array(colors),
                        'timestamp': msg.header.stamp
                    }
                
                self.get_logger().debug(f"Received point cloud with {len(points)} points")
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    def update_visualization(self):
        """Update the visualization with latest point cloud."""
        with self.pointcloud_lock:
            if self.latest_pointcloud is None:
                return
            
            pointcloud_data = self.latest_pointcloud.copy()
        
        try:
            if self.backend in ['open3d', 'both'] and OPEN3D_AVAILABLE:
                self.update_open3d(pointcloud_data)
            
            if self.backend in ['matplotlib', 'both'] and MATPLOTLIB_AVAILABLE:
                self.update_matplotlib(pointcloud_data)
            
            self.frame_count += 1
            
            # Save snapshot if enabled
            if self.save_snapshots and self.frame_count % 30 == 0:  # Every 30 frames
                self.save_snapshot(pointcloud_data)
                
        except Exception as e:
            self.get_logger().error(f"Error updating visualization: {e}")

    def update_open3d(self, pointcloud_data):
        """Update Open3D visualization."""
        points = pointcloud_data['points']
        colors = pointcloud_data['colors']
        
        # Update point cloud
        self.o3d_pointcloud.points = o3d.utility.Vector3dVector(points)
        self.o3d_pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Update geometry
        self.vis.update_geometry(self.o3d_pointcloud)
        
        # Auto-rotate if enabled
        if self.auto_rotate:
            ctr = self.vis.get_view_control()
            ctr.rotate(2.0, 0.0)  # Rotate 2 degrees around Y axis
        
        # Poll events and update renderer
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_matplotlib(self, pointcloud_data):
        """Update Matplotlib visualization."""
        points = pointcloud_data['points']
        colors = pointcloud_data['colors']
        
        # Clear previous plot
        self.ax.clear()
        
        # Plot points
        self.ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            s=self.point_size,
            alpha=0.8
        )
        
        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(f'Point Cloud ({len(points)} points)')
        
        # Set equal aspect ratio and limits
        if len(points) > 0:
            # Calculate bounds
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            
            # Add some padding
            padding = 0.1
            x_range = max(x_max - x_min, 0.1)
            y_range = max(y_max - y_min, 0.1)
            z_range = max(z_max - z_min, 0.1)
            
            self.ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
            self.ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
            self.ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])
        
        # Auto-rotate if enabled
        if self.auto_rotate:
            self.ax.view_init(elev=20, azim=self.frame_count * 2)
        
        # Update plot
        plt.draw()
        plt.pause(0.001)

    def save_snapshot(self, pointcloud_data):
        """Save a snapshot of the current point cloud."""
        try:
            import os
            timestamp = self.get_clock().now().nanoseconds
            
            if self.backend in ['open3d', 'both'] and OPEN3D_AVAILABLE:
                # Save Open3D screenshot
                filename = os.path.join(self.snapshot_dir, f"pointcloud_o3d_{timestamp}.png")
                self.vis.capture_screen_image(filename)
                self.get_logger().info(f"Saved Open3D snapshot: {filename}")
            
            if self.backend in ['matplotlib', 'both'] and MATPLOTLIB_AVAILABLE:
                # Save Matplotlib screenshot
                filename = os.path.join(self.snapshot_dir, f"pointcloud_mpl_{timestamp}.png")
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                self.get_logger().info(f"Saved Matplotlib snapshot: {filename}")
            
            # Also save raw point cloud data
            if OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud_data['points'])
                pcd.colors = o3d.utility.Vector3dVector(pointcloud_data['colors'])
                
                ply_filename = os.path.join(self.snapshot_dir, f"pointcloud_{timestamp}.ply")
                o3d.io.write_point_cloud(ply_filename, pcd)
                self.get_logger().info(f"Saved point cloud data: {ply_filename}")
                
        except Exception as e:
            self.get_logger().error(f"Error saving snapshot: {e}")

    def destroy_node(self):
        """Clean up visualization windows."""
        try:
            if hasattr(self, 'vis') and OPEN3D_AVAILABLE:
                self.vis.destroy_window()
            
            if MATPLOTLIB_AVAILABLE:
                plt.close('all')
                
        except Exception as e:
            self.get_logger().error(f"Error cleaning up: {e}")
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PointCloudVisualizerNode()
    except Exception as e:
        print(f"[FATAL] PointCloud Visualizer failed to initialize: {e}", file=sys.stderr)
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