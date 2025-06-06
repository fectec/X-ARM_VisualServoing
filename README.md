# X-ARM Visual Servoing

## Execution Guide

<p align="justify">
You will need <a href="https://docs.ros.org/en/humble/Installation.html" target="_blank">ROS2 Humble</a> installed on your system. This project was developed and tested on Ubuntu Linux 22.04 (Jammy Jellyfish).
</p>

### Setup

After cloning the repository, fetch the required submodules:

- **xarm_ros2**: ROS2 driver for UFACTORY xArm robotic manipulators.
- **azure_kinect_ros2_driver**: ROS2 driver for Azure Kinect sensor data.

```bash
source setup.bash
```

<p align="justify">
Then, to install the Azure Kinect SDK on Ubuntu 22.04:
</p>

```bash
source kinect_sdk_install.bash
```

### Running the System

#### Test Robot Mobility with RViz + MoveIt

To launch the robot visualization and motion planning interface:

```bash
source rviz.bash
```

#### Run Azure Kinect Node

To start streaming sensor data from the Azure Kinect:

```bash
ros2 run azure_kinect_ros2_driver azure_kinect_node
```
