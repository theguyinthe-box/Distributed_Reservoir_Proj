from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    func_arg = DeclareLaunchArgument(
        'func',
        default_value='lorenz',
        description='Dynamical system function to use (lorenz or rossler)'
    )

    dt_arg = DeclareLaunchArgument(
        'dt',
        default_value='0.01',
        description='Time step for ODE integration'
    )

    integrator_arg = DeclareLaunchArgument(
        'integrator',
        default_value='RK45',
        description='ODE integrator method (RK45, RK23, DOP853, etc.)'
    )

    training_length_arg = DeclareLaunchArgument(
        'training_length',
        default_value='100',
        description='Length of training phase'
    )

    eval_length_arg = DeclareLaunchArgument(
        'eval_length',
        default_value='100',
        description='Length of evaluation phase'
    )

    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value='2',
        description='Batch size for data generation'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='ROS2 logging level (debug, info, warn, error, fatal)'
    )

    # Create agent node with parameters
    agent_node = Node(
        package='distributed_reservoir',
        executable='agent',
        name='agent_ros_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'func': LaunchConfiguration('func')},
            {'dt': LaunchConfiguration('dt')},
            {'integrator': LaunchConfiguration('integrator')},
            {'training_length': LaunchConfiguration('training_length')},
            {'eval_length': LaunchConfiguration('eval_length')},
            {'batch_size': LaunchConfiguration('batch_size')},
        ]
    )

    return LaunchDescription([
        func_arg,
        dt_arg,
        integrator_arg,
        training_length_arg,
        eval_length_arg,
        batch_size_arg,
        log_level_arg,
        agent_node,
    ])
