from setuptools import setup
from glob import glob
import os

package_name = 'distributed_reservoir'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=[
        'rclpy',
        'std_msgs',
        'numpy',
        'scipy',
        'scikit-learn',
        'torch',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            f'agent = {package_name}.agent:main',
            f'edge_server = {package_name}.edge_server:main',
        ],
    },
)
