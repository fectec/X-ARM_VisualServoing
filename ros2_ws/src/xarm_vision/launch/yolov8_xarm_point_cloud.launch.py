import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_vision = get_package_share_directory('xarm_vision')
    pkg_xarm_moveit = get_package_share_directory('xarm_moveit_config')
    
    config_file = os.path.join(pkg_vision, 'config', 'point_cloud_config.yaml')
    intrinsics_file = os.path.join(pkg_vision, 'config', 'kinect_calibration.yaml')
    canonical_model_file = os.path.join(pkg_vision, 'models', 'canonical_model.ply')
    
    yolov8_segmentation_node = Node(
        package='xarm_vision',
        executable='yolov8_segmentation',
        name='yolov8_segmentation',
        parameters=[config_file],
        output='screen'
    )
    
    point_cloud_generator_node = Node(
        package='xarm_vision',
        executable='point_cloud_generator',
        name='point_cloud_generator',
        parameters=[
            config_file,
            {'camera_intrinsics_file': intrinsics_file},
            {'xarm_integration': True}
        ],
        output='screen'
    )
    
    point_cloud_scaling_node = Node(
        package='xarm_vision',
        executable='point_cloud_scaling',
        name='point_cloud_scaling',
        parameters=[
            config_file,
            {'canonical_model_path': canonical_model_file}
        ],
        output='screen'
    )
    
    xarm_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_xarm_moveit, 'launch', 'xarm6_moveit_realmove.launch.py')
        ]),
        launch_arguments={
            'robot_ip': '192.168.1.208',
            'add_gripper': 'true',
        }.items()
    )
    
    return LaunchDescription([
        xarm_moveit_launch,
        yolov8_segmentation_node,
        point_cloud_generator_node,
        point_cloud_scaling_node,
    ])