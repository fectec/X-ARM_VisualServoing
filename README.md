# X-ARM Visual Servoing

<p align="justify">
Developed for the Computer Vision and Machine Learning modules in the undergraduate course <b>"Intelligent Robotics Implementation."</b> This project is based on the work of Gustavo De Los Ríos Alatorre at the Instituto Tecnológico y de Estudios Superiores de Monterrey, Campus Monterrey, School of Engineering and Sciences, titled <a href="https://github.com/GustavoDLRA/3D-SemiDeformable-ObjectTracking" target="_blank">"3-D Detection & Tracking for Semi-Deformable Objects"</a>, developed as a master's thesis submitted in partial fulfillment of the requirements for the degree of Master of Science in Computer Science, Monterrey, Nuevo León, May 2024.
</p>

## Execution Guide

<p align="justify">
You will need <a href="https://docs.ros.org/en/humble/Installation.html" target="_blank">ROS2 Humble</a> installed on your system. This project was developed and tested on Ubuntu Linux 22.04 (Jammy Jellyfish).
</p>

### Setup

After cloning the repository, fetch the required submodules:

- **xarm_ros2**: ROS2 driver for UFACTORY X-ARM robotic manipulators.
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
