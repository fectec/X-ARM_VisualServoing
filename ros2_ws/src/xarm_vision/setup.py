from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'xarm_vision'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'models'),
            glob(os.path.join('models', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fectec',
    maintainer_email='fectec151@gmail.com',
    description='Vision processing package for X-ARM robot with Azure Kinect.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_segmentation = xarm_vision.image_segmentation:main',
            'point_cloud_generator = xarm_vision.point_cloud_generator:main',
            'point_cloud_scaling = xarm_vision.point_cloud_scaling:main',
            'yolov8_segmentation = xarm_vision.yolov8_segmentation:main'   
        ],
    },
)