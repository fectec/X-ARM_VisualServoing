import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/fectec/X-ARM_VisualServoing/ros2_ws/install/uf_ros_lib'
