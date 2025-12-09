from setuptools import setup

setup(
    name='reservoir_eval',
    version='0.0.1',
    packages=['distributed_reservoir'],
    package_dir={'': 'distributed_reservoir'},
    install_requires=['rclpy', 'std_msgs', 'numpy', 'scipy', 'scikit-learn', 'torch', 'pandas'],
    entry_points={
        'console_scripts': [
            'agent = distributed_reservoir.agent:main',
            'edge = distributed_reservoir.edge_server:main',
        ],
    },
)