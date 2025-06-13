import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_vision = get_package_share_directory('xarm_vision')
    
    config_file = os.path.join(pkg_vision, 'config', 'segmentation_point_cloud_config.yaml')
    intrinsics_file = os.path.join(pkg_vision, 'config', 'kinect_calibration.yaml')
    rviz_config_file = os.path.join(pkg_vision, 'config', 'segmentation_point_cloud.rviz')
    canonical_model_file = os.path.join(pkg_vision, 'models', 'canonical_model.ply')
    
    image_segmentation_node = Node(
        package='xarm_vision',
        executable='image_segmentation',
        name='image_segmentation',
        parameters=[config_file],
        output='screen'
    )
    
    point_cloud_generator_node = Node(
        package='xarm_vision',
        executable='point_cloud_generator',
        name='point_cloud_generator',
        parameters=[
            config_file,
            {'camera_intrinsics_file': intrinsics_file}
        ],
        output='screen'
    )
    
    point_clouds_alignment_scaling_node = Node(
        package='xarm_vision',
        executable='point_clouds_alignment_scaling',
        name='point_clouds_alignment_scaling',
        parameters=[
            config_file,
            {'canonical_model_path': canonical_model_file}
        ],
        output='screen'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )
    
    return LaunchDescription([
        image_segmentation_node,
        point_cloud_generator_node,
        point_clouds_alignment_scaling_node,
        rviz_node
    ])