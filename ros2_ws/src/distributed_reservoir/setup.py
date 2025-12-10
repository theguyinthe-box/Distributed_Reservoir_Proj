from setuptools import setup

package_name = 'distributed_reservoir'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['rclpy', 'std_msgs', 'numpy', 'scipy', 'scikit-learn', 'torch', 'pandas'],
    entry_points={
        'console_scripts': [
            'agent = ' + package_name + '.agent:main',
            'edge = ' + package_name + '.edge_server:main',
        ],
    },
)