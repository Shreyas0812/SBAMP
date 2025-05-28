from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    waypoint_file_name = "waypoints_manual_spline.csv"
    
    # Simulation
    pose_topic = "/ego_racecar/odom"

    # Car
    # pose_topic = "/pf/odom/pose"
    
    scan_topic = "/scan"
    next_wp_topic = "/next_waypoint"
    original_map_topic = "/map"
    occupancy_grid_topic = "/occupancy_grid"
    rrt_path_topic = "/rrt_path"
    drive_topic = "/drive"
    
    visualize_wp_topic = "/visualization/waypoints"
    visualize_rrt_path_topic = "/visualization/rrt_path"
    visualize_sed_path_topic = "/visualization/sed_path"

    lookahead_distance = 0.8
    y_ego_threshold = 1.2
    
    map_height = 713
    map_width = 727
    map_resolution = 0.05
    map_origin = [-20.2, -5.68]

    expand_occ_size = 2

    kp = 1.0
    kv = 1.0

    return LaunchDescription([
        # SBAMP node
        Node(
            package="sbamp",
            executable="sbamp_node.py",
            name="sbamp_node",
            output="screen",
            parameters=[
                {"rrt_path_topic": rrt_path_topic},
                {"pose_topic": pose_topic},
                {"drive_topic": drive_topic},
                {"kv": kv},
                {"kp": kp},
            ]
        ),
        # Visualize node
        Node(
            package="sbamp",
            executable="visualize_node.py",
            name="visualize_node",
            output="screen",
            parameters=[
                {"waypoint_file_name": waypoint_file_name},
                {"visualize_wp_topic": visualize_wp_topic},
                {"rrt_path_topic": rrt_path_topic},
                {"visualize_rrt_path_topic": visualize_rrt_path_topic},   
                {"visualize_sed_path_topic": visualize_sed_path_topic},
            ]
        ),
        # RRT node
        Node(
            package="sbamp",
            executable="rrt_node.py",
            name="rrt_node",
            output="screen",
            parameters=[
                {"pose_topic": pose_topic},
                {"occupancy_grid_topic": occupancy_grid_topic},
                {"next_wp_topic": next_wp_topic},
                {"rrt_path_topic": rrt_path_topic},
                {"map_height": map_height},
                {"map_width": map_width},
                {"map_resolution": map_resolution},
                {"map_origin": map_origin},
            ]
        ),
        # Occupancy grid node
        Node(
            package="sbamp",
            executable="occupancy_grid_node.py",
            name="occupancy_grid_node",
            output="screen",
            parameters=[
                {"scan_topic": scan_topic},
                {"pose_topic": pose_topic},
                {"original_map_topic": original_map_topic},
                {"occupancy_grid_topic": occupancy_grid_topic},
                {"map_height": map_height},
                {"map_width": map_width},
                {"map_resolution": map_resolution},
                {"map_origin": map_origin},
                {"expand_occ_size": expand_occ_size},
            ]
        ),
        # Next waypoint node
        Node(
            package="sbamp",
            executable="next_waypoint_node.py",
            name="next_waypoint_node",
            output="screen",
            parameters=[
                {"waypoint_file_name": waypoint_file_name},
                {"pose_topic": pose_topic},
                {"next_wp_topic": next_wp_topic},
                {"lookahead_distance": lookahead_distance},
                {"y_ego_threshold": y_ego_threshold}
            ]
        ),
    ])