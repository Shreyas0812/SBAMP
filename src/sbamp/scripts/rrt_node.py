#!/usr/bin/env python3
import numpy as np
from scipy.spatial import KDTree
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
from rclpy.node import Node
from scipy.ndimage import binary_dilation
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback


class NodeRRT:
    """
    Class representing a single node in the RRT tree.
    """
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        # ROS Topics
        pose_topic = "/ego_racecar/odom"

        scan_topic = "/scan"

        # ROS Subscribers
        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 10)





        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

        # ROS Publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.tree_viz_pub_ = self.create_publisher(MarkerArray, '/rrt_tree', 10)
        self.path_viz_pub_ = self.create_publisher(Marker, '/rrt_path', 10)
        




        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1  # Only store the last message
        )
        self.map_update_pub_ = self.create_publisher(OccupancyGrid, '/map_updated', qos_profile)
       

        # RRT Parameters
        self.max_iter = 2500
        self.goal_threshold = 0.15
        self.steer_range = 0.5
        self.sample_range_x = 5.0
        self.sample_range_y = 5.0

        # Occupancy Grid Setup
        self.map_ = None  # Global map (OccupancyGrid)
        self.map_updated_ = None  # Updated map with inflated obstacles
        self.margin = 0.15  # Margin to inflate obstacles (meters)

        # Visualization Markers
        self.tree_nodes_marker = Marker()
        self.tree_branches_marker = Marker()
        self.path_marker = Marker()


        self.timer = self.create_timer(1.0, self.publish_static_obstacle)
       

        # Initialize Visualization Markers and Occupancy Grid
        

        self.L = 1.3
        self.P = 0.5
        csv_data = np.genfromtxt("./levine1.csv", delimiter=",", skip_header=1, dtype=None, encoding='UTF-8')
        self.waypoints = csv_data[:, 0:2]
        self.kd_tree = KDTree(self.waypoints)
        self.path = []

        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_markers', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints_markers)
        self.publish_waypoints_markers()
        self.init_occupancy_grid()


        self.sample_viz_pub = self.create_publisher(
            MarkerArray, 
            '/sample_points', 
            10
        )
        self.samples = []  # Store sampled points for visualization
        self.max_samples = 100  # Maximum number of samples to display
        self.sample_test_timer = self.create_timer(0.2, self.test_sampling)


        self.planning_timer = self.create_timer(5.0, self.run_rrt_planning)




    def publish_waypoints_markers(self):
        marker_array = MarkerArray()
        self.get_logger().info("Publishing waypoints markers")
        
        for i, wp in enumerate(self.path):
            marker = Marker()
            marker.header.frame_id = "map"  # Make sure RViz uses the correct reference frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "waypoints"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.1  # Slightly above ground for visibility
            marker.scale.x = 0.2  # Adjust size for better visualization
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0  # Fully opaque
            marker.color.r = 1.0  # Red color
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)  


    def init_occupancy_grid(self):
        """
        Initialize the occupancy grid using the map topic.
        Inflate obstacles based on a margin.
        """
        self.map_received = False  # Flag to check if map is received
        self.get_logger().info("publishing map")

        # Create a one-time subscriber to the '/map' topic
        def map_callback(map_msg):
            self.get_logger().info("Map callback triggered")
            self.map_ = map_msg
            self.map_updated_ = map_msg

            resolution = map_msg.info.resolution
            width, height = map_msg.info.width, map_msg.info.height

            inflated_map_data = np.array(map_msg.data).reshape((height, width))

            # Inflate obstacles in the map using margin (binary occupancy grid)
            
            obstacle_mask = (inflated_map_data == 100)
            inflated_obstacle_mask = binary_dilation(obstacle_mask, iterations=int(self.margin / resolution))
            inflated_map_data[inflated_obstacle_mask] = 100

            map_msg.data = inflated_map_data.flatten().tolist()
            self.map_updated_ = map_msg

            self.map_received = True  # Set flag to indicate map is received
            self.map_update_pub_.publish(self.map_updated_)
            

        # Subscribe to the '/map' topic


        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1  # Store only one message
        )

        self.create_subscription(OccupancyGrid, '/map', map_callback, qos_profile)
        
       

        # Wait until the map is received
        while not self.map_received:
            rclpy.spin_once(self)



        

    def scan_callback(self, scan_msg):
        """
        LaserScan callback to update the occupancy grid based on scan data.
        
        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
            None
        """
        # Check if we have received the map
        self.get_logger().info("Received scan data with %d points" % len(scan_msg.ranges))
        if not hasattr(self, 'map_received') or not self.map_received:
            return

        # Extract scan parameters
        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        angle_increment = scan_msg.angle_increment
        
        # Get the car's current position from odometry
        if not hasattr(self, 'current_pose'):
            self.get_logger().warn("No pose information available yet")
            return
        
        car_x, car_y, car_theta = self.current_pose
        
        # Create a copy of the map data for updating
        map_data = np.array(self.map_updated_.data).reshape(
            (self.map_updated_.info.height, self.map_updated_.info.width))
        
        # Iterate through each laser scan range
        for i, r in enumerate(ranges):
            if np.isinf(r) or np.isnan(r):
                continue  # Skip invalid ranges
            
            # Calculate the angle of the laser beam
            angle = angle_min + i * angle_increment
            
            # Transform laser beam endpoint to map coordinates
            obstacle_x = car_x + r * np.cos(car_theta + angle)
            obstacle_y = car_y + r * np.sin(car_theta + angle)
            
            # Convert to grid coordinates
            grid_x = int((obstacle_x - self.map_updated_.info.origin.position.x) / self.map_updated_.info.resolution)
            grid_y = int((obstacle_y - self.map_updated_.info.origin.position.y) / self.map_updated_.info.resolution)
            
            # Update the occupancy grid (mark as occupied)
            if 0 <= grid_x < self.map_updated_.info.width and 0 <= grid_y < self.map_updated_.info.height:
                map_data[grid_y, grid_x] = 100
        
        # Update the map data and publish
        self.map_updated_.data = map_data.flatten().tolist()
        self.map_update_pub_.publish(self.map_updated_)
        self.get_logger().info("Updated occupancy grid and published")


    def pose_callback(self, pose_msg):
        # Extract pose information
        
        position = pose_msg.pose.pose.position
        orientation = pose_msg.pose.pose.orientation
        
        # Convert quaternion to Euler angles
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(quaternion)
        
        # Store current pose
        self.current_pose = (position.x, position.y, yaw)
        
        # Rest of your RRT implementation will go here


    def euler_from_quaternion(self, quaternion):
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw




    def publish_static_obstacle(self):
        """
        Publish a static obstacle to the occupancy grid
        """
        if hasattr(self, 'map_updated_') and self.map_updated_ is not None:
            # Create a copy of the map data
            map_data = np.array(self.map_updated_.data).reshape(
                (self.map_updated_.info.height, self.map_updated_.info.width))
            
            # Define obstacle position in map coordinates
            # You can adjust these coordinates based on where you want the obstacle
            obstacle_x, obstacle_y = 8.58, 3.14  # Position in the middle of the track
            
            # Convert to grid coordinates
            grid_x = int((obstacle_x - self.map_updated_.info.origin.position.x) / self.map_updated_.info.resolution)
            grid_y = int((obstacle_y - self.map_updated_.info.origin.position.y) / self.map_updated_.info.resolution)
            
            # Mark cells as occupied (with some radius)
            radius = int(0.3 / self.map_updated_.info.resolution)  # 30cm radius
            for i in range(-radius, radius+1):
                for j in range(-radius, radius+1):
                    if i*i + j*j <= radius*radius:  # Circle shape
                        if 0 <= grid_x+i < self.map_updated_.info.width and 0 <= grid_y+j < self.map_updated_.info.height:
                            map_data[grid_y+j, grid_x+i] = 100
            
            # Update and publish
            self.map_updated_.data = map_data.flatten().tolist()
            self.map_update_pub_.publish(self.map_updated_)
            self.get_logger().info("Published static obstacle at x=2.0, y=0.0")









    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point
        within corridor boundaries and a certain distance ahead of the car.

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point
        """
        # Maximum distance to sample ahead of the car
        max_forward_distance = self.sample_range_x
        
        # Current car position and orientation
        if not hasattr(self, 'current_pose'):
            return None, None
        
        car_x, car_y, car_theta = self.current_pose
        
        # Maximum attempts to find a valid point
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            # Sample a point within the forward distance (biased toward forward direction)
            forward_distance = np.random.uniform(0.7, max_forward_distance)
            
            # Sample lateral position within corridor boundaries
            # Use a narrower range for lateral sampling to stay within corridor
            lateral_distance = np.random.uniform(-self.sample_range_y, self.sample_range_y)
            
            # Transform to global frame
            x = car_x + forward_distance * np.cos(car_theta) - lateral_distance * np.sin(car_theta)
            y = car_y + forward_distance * np.sin(car_theta) + lateral_distance * np.cos(car_theta)
            
            # Check if the point is in free space using the occupancy grid
            if self.is_point_free(x, y):
                return (x, y)
            
            attempts += 1
        
        # If we couldn't find a valid point, try one more time with a smaller range
        x = car_x + np.random.uniform(0.3, 1.0) * np.cos(car_theta)
        y = car_y + np.random.uniform(0.3, 1.0) * np.sin(car_theta)
        
        return (x, y)

    def is_point_free(self, x, y):
        """
        Check if a point is in free space in the occupancy grid.
        
        Args:
            x, y (float): Coordinates to check
        Returns:
            bool: True if the point is in free space
        """

        if not hasattr(self, 'map_updated_') or self.map_updated_ is None:
            return False
        
        # Convert world coordinates to grid coordinates
        grid_x = int((x - self.map_updated_.info.origin.position.x) / self.map_updated_.info.resolution)
        grid_y = int((y - self.map_updated_.info.origin.position.y) / self.map_updated_.info.resolution)
        
        # Check if within grid bounds
        if 0 <= grid_x < self.map_updated_.info.width and 0 <= grid_y < self.map_updated_.info.height:
            # Check if cell is free (0 = free, 100 = occupied, -1 = unknown)
            idx = grid_y * self.map_updated_.info.width + grid_x
            return self.map_updated_.data[idx] == 0
        
        return False


    def test_sampling(self):
        """Test the sampling method and visualize results"""
        if not hasattr(self, 'map_updated_') or self.map_updated_ is None:
            self.get_logger().info("Waiting for map...")
            return
        
        if not hasattr(self, 'current_pose'):
            self.get_logger().info("Waiting for pose...")
            return
        
        # Sample a new point
        sample_point = self.sample()
        if sample_point[0] is not None:
            self.samples.append(sample_point)
            if len(self.samples) > self.max_samples:
                self.samples.pop(0)  # Remove oldest sample
            
            self.get_logger().info(f"Sampled point: ({sample_point[0]:.2f}, {sample_point[1]:.2f})")
            self.visualize_samples()

    def visualize_samples(self):
        """Visualize sampled points"""
        marker_array = MarkerArray()
        
        # Create marker for samples
        samples_marker = Marker()
        samples_marker.header.frame_id = "map"
        samples_marker.header.stamp = self.get_clock().now().to_msg()
        samples_marker.ns = "samples"
        samples_marker.id = 0
        samples_marker.type = Marker.POINTS
        samples_marker.action = Marker.ADD
        samples_marker.scale.x = 0.1
        samples_marker.scale.y = 0.1
        samples_marker.color.a = 1.0
        samples_marker.color.r = 0.0
        samples_marker.color.g = 1.0
        samples_marker.color.b = 0.0
        
        # Add points
        for sample in self.samples:
            p = Point()
            p.x = sample[0]
            p.y = sample[1]
            p.z = 0.1
            samples_marker.points.append(p)
        
        # Create marker for current pose
        pose_marker = Marker()
        pose_marker.header.frame_id = "map"
        pose_marker.header.stamp = self.get_clock().now().to_msg()
        pose_marker.ns = "current_pose"
        pose_marker.id = 1
        pose_marker.type = Marker.ARROW
        pose_marker.action = Marker.ADD
        pose_marker.scale.x = 0.5  # Arrow length
        pose_marker.scale.y = 0.1  # Arrow width
        pose_marker.scale.z = 0.1  # Arrow height
        pose_marker.color.a = 1.0
        pose_marker.color.r = 1.0
        pose_marker.color.g = 0.0
        pose_marker.color.b = 0.0
        
        # Set pose
        pose_marker.pose.position.x = self.current_pose[0]
        pose_marker.pose.position.y = self.current_pose[1]
        pose_marker.pose.position.z = 0.1
        
        # Convert yaw to quaternion (simplified)
        theta = self.current_pose[2]
        pose_marker.pose.orientation.z = np.sin(theta/2)
        pose_marker.pose.orientation.w = np.cos(theta/2)
        
        # Add markers to array
        marker_array.markers.append(samples_marker)
        marker_array.markers.append(pose_marker)
        
        # Publish
        self.sample_viz_pub.publish(marker_array)



    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        nearest_node = 0
        min_dist = float('inf')
        
        for i in range(len(tree)):
            # Calculate squared distance (avoiding sqrt for efficiency)
            dist = (tree[i].x - sampled_point[0])**2 + (tree[i].y - sampled_point[1])**2
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = i
        
        return nearest_node

        

    def steer(self, nearest_node, sampled_point, nearest_node_idx):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled point is.

        Args:
            nearest_node (Node): nearest node on the tree
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        # Calculate the distance between nearest node and sampled point
        dx = sampled_point[0] - nearest_node.x
        dy = sampled_point[1] - nearest_node.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # If distance is less than steer range, we can reach the sampled point directly
        if dist <= self.steer_range:
            new_x = sampled_point[0]
            new_y = sampled_point[1]
        # Otherwise, move in the direction of sampled point by steer_range distance
        else:
            # Normalize the direction vector and scale by steer_range
            new_x = nearest_node.x + (dx / dist) * self.steer_range
            new_y = nearest_node.y + (dy / dist) * self.steer_range
        
        # Get the index of the nearest node in the tree
        # This is what we need to fix - we need to pass the index, not the node itself
        
        
        # Create and return the new node with the index as its parent
        new_node = NodeRRT(new_x, new_y, nearest_node_idx)
        
        return new_node



    



    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest_node (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                            with the occupancy grid
        """
        # Calculate the number of points to check along the path
        x_cell_diff = abs(int(np.ceil((nearest_node.x - new_node.x) / self.map_updated_.info.resolution)))
        y_cell_diff = abs(int(np.ceil((nearest_node.y - new_node.y) / self.map_updated_.info.resolution)))
        
        # Determine step size for interpolation
        num_steps = max(x_cell_diff, y_cell_diff)
        if num_steps == 0:  # If nodes are very close
            return not self.is_point_free(new_node.x, new_node.y)
        
        dt = 1.0 / num_steps
        t = 0.0
        
        # Check points along the path
        for i in range(num_steps + 1):
            # Interpolate between nearest_node and new_node
            x = nearest_node.x + t * (new_node.x - nearest_node.x)
            y = nearest_node.y + t * (new_node.y - nearest_node.y)
            
            # Check if this point is in collision
            if not self.is_point_free(x, y):
                return True  # Collision detected
            
            t += dt
        
        return False  # No collision


    

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        # Calculate squared distance between the node and goal
        dist_squared = (latest_added_node.x - goal_x)**2 + (latest_added_node.y - goal_y)**2
        
        # Check if distance is less than the goal threshold squared
        return dist_squared < (self.goal_threshold**2)


    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        # Initialize path with the goal node
        path = [latest_added_node]
        current_node = latest_added_node
        
        # Traverse the tree backwards from the goal node to the root
        while current_node.parent is not None:
            # Add parent to the path
            current_node = tree[current_node.parent]
            path.append(current_node)
        
        # Reverse the path to get it from start to goal
        path.reverse()

        self.path = []
        for node in path:
            self.path.append([node.x, node.y])
        
        return path



    def plan_rrt(self, start_x, start_y, goal_x, goal_y):
        """
        RRT path planning algorithm implementation
        """
        # Initialize the tree with the start node
        start_node = NodeRRT(start_x, start_y)
        tree = [start_node]
        
        # Main RRT loop
        for i in range(self.max_iter):
            # Sample a random point in free space
            sampled_point = self.sample()
            if sampled_point[0] is None:
                continue
            
            # Find the nearest node in the tree
            nearest_node_idx = self.nearest(tree, sampled_point)
            
            # Steer from the nearest node toward the sampled point
            # Pass both the node and its index
            new_node = self.steer(tree[nearest_node_idx], sampled_point, nearest_node_idx)
            
            # Store the index as the parent, not the node itself
            new_node.parent = nearest_node_idx
            
            # Check if the path to the new node is collision-free
            if not self.check_collision(tree[nearest_node_idx], new_node):
                # Add the new node to the tree
                tree.append(new_node)
                
                # Check if we've reached the goal
                if self.is_goal(new_node, goal_x, goal_y):
                    self.get_logger().info(f"Path found after {i+1} iterations")
                    return self.find_path(tree, new_node), tree
        
        # If no path is found after max iterations
        self.get_logger().warn(f"No path found after {self.max_iter} iterations")
        return None, tree

    
    def visualize_rrt(self, path=None, tree=None):
        """
        Visualize the RRT tree and path
        
        Args:
            path: List of nodes forming the path from start to goal
            tree: List of all nodes in the RRT tree
        """
        if not tree:
            return
            
        marker_array = MarkerArray()
        
        # Tree nodes marker
        nodes_marker = Marker()
        nodes_marker.header.frame_id = "map"
        nodes_marker.header.stamp = self.get_clock().now().to_msg()
        nodes_marker.ns = "tree_nodes"
        nodes_marker.id = 0
        nodes_marker.type = Marker.POINTS
        nodes_marker.action = Marker.ADD
        nodes_marker.scale.x = 0.05
        nodes_marker.scale.y = 0.05
        nodes_marker.color.r = 0.0
        nodes_marker.color.g = 0.7
        nodes_marker.color.b = 0.0
        nodes_marker.color.a = 1.0
        
        # Tree branches marker
        branches_marker = Marker()
        branches_marker.header.frame_id = "map"
        branches_marker.header.stamp = self.get_clock().now().to_msg()
        branches_marker.ns = "tree_branches"
        branches_marker.id = 1
        branches_marker.type = Marker.LINE_LIST
        branches_marker.action = Marker.ADD
        branches_marker.scale.x = 0.02
        branches_marker.color.r = 0.0
        branches_marker.color.g = 0.0
        branches_marker.color.b = 0.7
        branches_marker.color.a = 0.5
        
        # Add nodes and branches to markers
        for i, node in enumerate(tree):
            # Add node
            p = Point()
            p.x = node.x
            p.y = node.y
            p.z = 0.1
            nodes_marker.points.append(p)
            
            # Add branch (line from parent to node)
            if i > 0 and node.parent is not None:
                # Start point (parent)
                p1 = Point()
                p1.x = tree[node.parent].x
                p1.y = tree[node.parent].y
                p1.z = 0.1
                branches_marker.points.append(p1)
                
                # End point (current node)
                p2 = Point()
                p2.x = node.x
                p2.y = node.y
                p2.z = 0.1
                branches_marker.points.append(p2)
        
        marker_array.markers.append(nodes_marker)
        marker_array.markers.append(branches_marker)
        self.tree_viz_pub_.publish(marker_array)
        
        # Visualize path if available
        
    # [existing code for tree visualization]
    
    # Visualize path if available
        if path:
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "path"
            path_marker.id = 2
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            
            # Make the path thicker and more visible
            path_marker.scale.x = 0.12  # Increased thickness
            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0
            
            # Add points to the path marker
            for node in path:
                p = Point()
                p.x = node.x
                p.y = node.y
                p.z = 0.15  # Slightly higher than the tree for visibility
                path_marker.points.append(p)
            
            # Add spheres at each waypoint for better visibility
            waypoints_marker = Marker()
            waypoints_marker.header.frame_id = "map"
            waypoints_marker.header.stamp = self.get_clock().now().to_msg()
            waypoints_marker.ns = "path_waypoints"
            waypoints_marker.id = 3
            waypoints_marker.type = Marker.SPHERE_LIST
            waypoints_marker.action = Marker.ADD
            waypoints_marker.scale.x = 0.15
            waypoints_marker.scale.y = 0.15
            waypoints_marker.scale.z = 0.15
            waypoints_marker.color.r = 1.0
            waypoints_marker.color.g = 0.5
            waypoints_marker.color.b = 0.0
            waypoints_marker.color.a = 1.0
            
            for node in path:
                p = Point()
                p.x = node.x
                p.y = node.y
                p.z = 0.15
                waypoints_marker.points.append(p)
            
            # Publish both markers
            self.path_viz_pub_.publish(path_marker)
            
            # Create a marker array for the waypoints
            waypoints_array = MarkerArray()
            waypoints_array.markers.append(waypoints_marker)
            self.tree_viz_pub_.publish(waypoints_array)



    def run_rrt_planning(self):
        # Make sure we have valid map and pose data
        if not hasattr(self, 'map_updated_') or self.map_updated_ is None:
            self.get_logger().info("Waiting for map...")
            return
        
        if not hasattr(self, 'current_pose'):
            self.get_logger().info("Waiting for pose...")
            return
        
        # Get current position
        start_x, start_y = self.current_pose[0], self.current_pose[1]
        car_theta = self.current_pose[2]
        
        # Find the furthest waypoint within 5 meters
        max_distance = 5.0
        car_position = np.array([start_x, start_y])
        
        # Calculate distances to all waypoints
        valid_waypoints = []
        for wp in self.waypoints:
            # Calculate distance from car to waypoint
            distance = np.linalg.norm(wp - car_position)
            
            # Check if waypoint is ahead of the car (in car's frame)
            dx = wp[0] - start_x
            dy = wp[1] - start_y
            
            # Transform to car's frame to check if it's ahead
            x_car_frame = dx * np.cos(-car_theta) - dy * np.sin(-car_theta)
            
            # Only consider waypoints that are ahead and within max_distance
            if x_car_frame > 0 and distance <= max_distance:
                valid_waypoints.append((distance, wp))
        
        # Set goal to furthest valid waypoint or default if none found
       
            # Find the furthest valid waypoint
        furthest_wp = max(valid_waypoints, key=lambda x: x[0])
        goal_x, goal_y = furthest_wp[1]
        self.get_logger().info(f"Selected waypoint at ({goal_x:.2f}, {goal_y:.2f}) as goal")
        # else:
        #     # Fallback to default behavior if no valid waypoints
        #     goal_x = start_x + 5.0 * np.cos(car_theta)
        #     goal_y = start_y + 5.0 * np.sin(car_theta)
        #     self.get_logger().info(f"No valid waypoints found, using default goal at ({goal_x:.2f}, {goal_y:.2f})")
        
        self.get_logger().info(f"Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})")
        
        # Run RRT planner
        path, tree = self.plan_rrt(start_x, start_y, goal_x, goal_y)
        
        # Visualize results
        self.visualize_rrt(path, tree)
        
        return path





   





    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()