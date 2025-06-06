#!/bin/bash

# X-ARM + Azure Kinect ROS2 Setup
# Run this script after cloning the repository

# Navigate to your repository root (adjust the path as needed)
cd ~/X-ARM_VisualServoing

# Fetch all submodules (xarm_ros2 + azure_kinect_ros2_driver)
git submodule update --init --recursive

# Checkout correct branches
git -C ros2_ws/src/xarm_ros2 checkout humble
git -C ros2_ws/src/azure_kinect_ros2_driver checkout main

# Update submodules to latest
git submodule update --remote

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Install ROS2 package dependencies
cd ros2_ws/src/
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro humble -y

# Navigate to the ROS2 workspace directory and build the workspace
cd ../
colcon build

# Source the workspace
source install/setup.bash
