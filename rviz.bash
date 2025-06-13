#!/bin/bash

# X-ARM Robot RViz Connection Script

# Configuration variables
ROBOT_IP="192.168.1.208"
PC_IP="192.168.1.100"
ROBOT_TYPE="xarm6"  
INTERFACE="enp7s0"   

# Assign a static IP to your Ethernet interface
sudo ip addr add ${PC_IP}/24 dev ${INTERFACE}

# Bring the interface up
sudo ip link set ${INTERFACE} up

# Test the connection
ping -c 3 ${ROBOT_IP}

# Source ROS2 workspace (adjust the path as needed)
source /opt/ros/humble/setup.bash
source ~/X-ARM_VisualServoing/ros2_ws/install/setup.bash

# Launch xarm_driver_node
ros2 launch xarm_api ${ROBOT_TYPE}_driver.launch.py robot_ip:=${ROBOT_IP}

# Wait for driver to initialize
sleep 2

# Enable all joints
ros2 service call /xarm/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"

# Set proper mode (0) and state (0)
ros2 service call /xarm/set_mode xarm_msgs/srv/SetInt16 "{data: 0}"
ros2 service call /xarm/set_state xarm_msgs/srv/SetInt16 "{data: 0}"

# Launch MoveIt for robot control
ros2 launch xarm_moveit_config ${ROBOT_TYPE}_moveit_realmove.launch.py robot_ip:=${ROBOT_IP} add_gripper:=true
