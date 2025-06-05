#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import os

def generate_launch_description():
    # Package name
    package_name = 'xarm_vision'
    
    # Paths to configuration files
    config_file_path = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'segmentation_pointcloud_config.yaml'
    ])
    
    intrinsics_file_path = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'ost.yaml'
    ])
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=config_file_path,
        description='Path to main configuration YAML file'
    )
    
    camera_intrinsics_arg = DeclareLaunchArgument(
        'camera_intrinsics_file',
        default_value=intrinsics_file_path,
        description='Path to camera intrinsics YAML file'
    )
    
    use_visualizer_arg = DeclareLaunchArgument(
        'use_visualizer',
        default_value='true',
        description='Launch point cloud visualizer'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_color_optical_frame',
        description='Camera optical frame ID'
    )
    
    # Log configuration
    log_config = LogInfo(
        msg=['Launching segmentation and point cloud pipeline...',
             '\n  Config file: ', LaunchConfiguration('config_file'),
             '\n  Intrinsics file: ', LaunchConfiguration('camera_intrinsics_file')]
    )
    
    # Image segmentation node
    segmentation_node = Node(
        package=package_name,
        executable='image_segmentation',  # Debe coincidir con tu entry_point en setup.py
        name='image_segmentation_node',
        parameters=[LaunchConfiguration('config_file')],  # Carga el YAML de configuración
        output='screen',
        emulate_tty=True,
    )
    
    # Point cloud generator node
    pointcloud_node = Node(
        package=package_name,
        executable='point_cloud_generator_node',
        name='point_cloud_generator_node',
        parameters=[
            LaunchConfiguration('config_file'),  # Carga el YAML de configuración principal
            {
                'camera_intrinsics_file': LaunchConfiguration('camera_intrinsics_file'),  # Override con ost.yaml
                'camera_frame_id': LaunchConfiguration('camera_frame'),
            }
        ],
        output='screen',
        emulate_tty=True,
    )
    
    # Point cloud visualizer node (conditional)
    visualizer_node = Node(
        package=package_name,
        executable='point_cloud_visualizer_node',
        name='point_cloud_visualizer_node',
        parameters=[LaunchConfiguration('config_file')],  # Carga el YAML de configuración
        condition=IfCondition(LaunchConfiguration('use_visualizer')),
        output='screen',
        emulate_tty=True,
    )
    
    # RViz2 para visualización adicional (opcional)
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare(package_name),
        'rviz',
        'pointcloud_visualization.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('use_visualizer')),
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        config_file_arg,
        camera_intrinsics_arg,
        use_visualizer_arg,
        camera_frame_arg,
        
        # Log info
        log_config,
        
        # Nodes
        segmentation_node,
        pointcloud_node,
        visualizer_node,
        # rviz_node,  # Descomenta cuando tengas el archivo rviz
    ])