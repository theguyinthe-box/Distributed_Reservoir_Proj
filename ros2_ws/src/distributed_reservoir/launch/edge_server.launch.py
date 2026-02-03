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
 
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='reservoir',
        description='Model type to use (reservoir or lstm)'
    )
 
    res_dim_arg = DeclareLaunchArgument(
        'res_dim',
        default_value='256',
        description='Reservoir dimension size'
    )

    spectral_radius_arg = DeclareLaunchArgument(
        'spectral_radius',
        default_value='1.1',
        description='Spectral radius of reservoir weight matrix'
    )

    leak_rate_arg = DeclareLaunchArgument(
        'leak_rate',
        default_value='0.15',
        description='Leak rate for reservoir neurons'
    )

    n_iterations_arg = DeclareLaunchArgument(
        'n_iterations',
        default_value='20',
        description='Number of iterations for processing'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='ROS2 logging level (debug, info, warn, error, fatal)'
    )

    # Create edge server node with parameters
    edge_server_node = Node(
        package='distributed_reservoir',
        executable='edge_server',
        name='edge_server_ros_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'func': LaunchConfiguration('func')},
            {'model_type': LaunchConfiguration('model_type')},
            {'res_dim': LaunchConfiguration('res_dim')},
            {'spectral_radius': LaunchConfiguration('spectral_radius')},
            {'leak_rate': LaunchConfiguration('leak_rate')},
            {'n_iterations': LaunchConfiguration('n_iterations')},
        ]
    )

    return LaunchDescription([
        func_arg,
        model_type_arg,
        res_dim_arg,
        spectral_radius_arg,
        leak_rate_arg,
        n_iterations_arg,
        log_level_arg,
        edge_server_node,
    ])