from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sbamp',
            executable='record_manual_wp_node.py',
            name='record_manual_wp_node',
            output='screen',
            parameters=[{
                'goal_pose_topic': '/goal_pose',
                'visualize_wp_topic': '/visualization/manual_waypoints',
                'waypoint_file_name': 'waypoints_manual.csv'
            }]
        )
    ])