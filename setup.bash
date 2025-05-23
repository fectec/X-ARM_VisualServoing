#!/bin/bash

# X-ARM ROS2 Submodule Setup
# Run this script after cloning the repository

# Navigate to your repository root (adjust the path as needed)
cd ~/X-ARM_VisualServoing

# Initialize and update submodules
# (this fetches the xarm_ros2 submodule located in ros2_ws/src/xarm_ros2)
git submodule update --init --recursive

# Ensure the xarm_ros2 submodule is on the correct branch for the ROS2 distro
git -C ros2_ws/src/xarm_ros2 checkout humble

# Update submodule to latest commit on the humble branch
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
