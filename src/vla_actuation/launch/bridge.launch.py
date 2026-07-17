import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Dynamically track down the absolute path of the YAML configuration inside installation share
    pkg_share = get_package_share_directory('vla_actuation')
    config_file_path = os.path.join(pkg_share, 'config', 'vla_bridge.yaml')

    return LaunchDescription([
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='vla_gz_bridge',
            output='screen',
            parameters=[{
                'config_file': config_file_path
            }]
        )
    ])