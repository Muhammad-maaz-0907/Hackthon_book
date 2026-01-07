---
title: Nav2 Path Planning
sidebar_position: 5
---

# Nav2 Path Planning

Navigation 2 (Nav2) is the next-generation navigation stack for ROS 2 that enables robots to autonomously navigate in complex environments. This lesson covers Nav2's path planning capabilities, integration with VSLAM, and application to humanoid robotics.

## Introduction to Navigation 2 (Nav2)

Nav2 is a complete navigation stack for ROS 2 that provides:
- **Global Path Planning**: Long-term route planning from start to goal
- **Local Path Planning**: Short-term obstacle avoidance and path following
- **Map Management**: Occupancy grid and costmap management
- **Controller Integration**: Waypoint following and motion control
- **Recovery Behaviors**: Handling navigation failures and stuck situations

### Nav2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Nav2                             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────┐ │
│  │   Global        │ │   Local         │ │ Recovery  │ │
│  │   Planner       │ │   Planner       │ │ Behaviors │ │
│  │                 │ │                 │ │           │ │
│  │ • A*            │ │ • DWA           │ │ • Clear   │ │
│  │ • NavFn         │ │ • TEB           │ │ • Rotate  │ │
│  │ • Global Costmap│ │ • Local Costmap │ │ • Backup  │ │
│  └─────────────────┘ └─────────────────┘ └───────────┘ │
│         │                       │               │       │
│         ▼                       ▼               ▼       │
│  ┌─────────────────────────────────────────────────────┐│
│  │         Controller Layer                            ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  ││
│  │  │ Pure        │ │ MPC         │ │ Follow      │  ││
│  │  │ Pursuit     │ │ Controller  │ │ Path        │  ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘  ││
│  └─────────────────────────────────────────────────────┘│
│         │                       │               │       │
│         ▼                       ▼               ▼       │
│  ┌─────────────────────────────────────────────────────┐│
│  │           Robot Drivers                             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Nav2 Core Components

### Global Planner

The global planner computes a path from the start to the goal based on the global costmap:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionServer
import numpy as np
import heapq

class Nav2GlobalPlanner(Node):
    def __init__(self):
        super().__init__('nav2_global_planner')

        # Global costmap subscription
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            10
        )

        # Action server for path computation
        self.path_action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.compute_path_callback
        )

        self.global_costmap = None
        self.map_resolution = 0.05  # 5cm per cell

    def costmap_callback(self, msg):
        """Update global costmap"""
        self.global_costmap = msg
        self.map_resolution = msg.info.resolution

    def compute_path_callback(self, goal_handle):
        """Compute path from start to goal"""
        goal = goal_handle.request.goal
        start = goal_handle.request.start

        # Get current map
        if self.global_costmap is None:
            goal_handle.abort()
            return

        # Convert to map coordinates
        start_map = self.world_to_map(start.pose.position.x, start.pose.position.y)
        goal_map = self.world_to_map(goal.pose.position.x, goal.pose.position.y)

        # Plan path using A* algorithm
        path = self.a_star_planner(start_map, goal_map)

        if path is not None:
            # Convert path back to world coordinates
            world_path = self.path_to_world_coordinates(path)

            # Create result message
            result = ComputePathToPose.Result()
            result.path = world_path

            goal_handle.succeed()
            return result
        else:
            goal_handle.abort()
            return

    def a_star_planner(self, start, goal):
        """A* path planning algorithm"""
        # Convert occupancy grid to numpy array
        costmap_array = np.array(self.global_costmap.data).reshape(
            self.global_costmap.info.height,
            self.global_costmap.info.width
        )

        # A* implementation
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current, costmap_array):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, pos1, pos2):
        """Heuristic function for A* (Manhattan distance)"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos, costmap_array):
        """Get valid neighbors for current position"""
        neighbors = []
        rows, cols = costmap_array.shape

        # 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = pos[0] + dx, pos[1] + dy

                if 0 <= nx < rows and 0 <= ny < cols:
                    # Check if cell is traversable (cost < 50, which is 50% occupied)
                    if costmap_array[nx, ny] < 50:
                        neighbors.append((nx, ny))

        return neighbors

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates"""
        map_x = int((x - self.global_costmap.info.origin.position.x) / self.global_costmap.info.resolution)
        map_y = int((y - self.global_costmap.info.origin.position.y) / self.global_costmap.info.resolution)
        return (map_x, map_y)

    def path_to_world_coordinates(self, path):
        """Convert path in map coordinates to world coordinates"""
        world_path = Path()
        world_path.header.frame_id = 'map'

        for map_coord in path:
            world_x = map_coord[0] * self.global_costmap.info.resolution + self.global_costmap.info.origin.position.x
            world_y = map_coord[1] * self.global_costmap.info.resolution + self.global_costmap.info.origin.position.y

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            world_path.poses.append(pose)

        return world_path
```

### Local Planner

The local planner handles short-term path following and obstacle avoidance:

```python
class Nav2LocalPlanner(Node):
    def __init__(self):
        super().__init__('nav2_local_planner')

        # Local costmap subscription
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.local_costmap_callback,
            10
        )

        # Laser scan subscription for obstacle detection
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Velocity command publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Path following subscription
        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.local_costmap = None
        self.current_path = []
        self.current_waypoint_idx = 0
        self.robot_pose = None

    def local_costmap_callback(self, msg):
        """Update local costmap"""
        self.local_costmap = msg

    def path_callback(self, path_msg):
        """Receive global path to follow"""
        self.current_path = path_msg.poses
        self.current_waypoint_idx = 0

    def follow_path(self):
        """Follow the current path with obstacle avoidance"""
        if not self.current_path or self.current_waypoint_idx >= len(self.current_path):
            return

        # Get current waypoint
        target_waypoint = self.current_path[self.current_waypoint_idx].pose

        # Calculate distance to current waypoint
        dist_to_waypoint = self.calculate_distance_to_waypoint(target_waypoint)

        # Check if we reached the current waypoint
        if dist_to_waypoint < 0.5:  # 50cm threshold
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path):
                # Path completed
                self.stop_robot()
                return

        # Calculate velocity command to reach waypoint
        cmd_vel = self.calculate_velocity_to_waypoint(target_waypoint)

        # Check for obstacles in the local costmap
        if self.has_obstacles_in_front():
            # Obstacle detected - apply obstacle avoidance
            cmd_vel = self.avoid_obstacles(cmd_vel)

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd_vel)

    def calculate_velocity_to_waypoint(self, target_waypoint):
        """Calculate velocity command to reach target waypoint"""
        # Get robot position and orientation
        robot_pos = self.robot_pose.position
        robot_yaw = self.get_yaw_from_quaternion(self.robot_pose.orientation)

        # Calculate target vector
        target_vec_x = target_waypoint.position.x - robot_pos.x
        target_vec_y = target_waypoint.position.y - robot_pos.y

        # Calculate distance and angle to target
        dist_to_target = np.sqrt(target_vec_x**2 + target_vec_y**2)
        angle_to_target = np.arctan2(target_vec_y, target_vec_x) - robot_yaw

        # Normalize angle to [-pi, pi]
        angle_to_target = np.arctan2(np.sin(angle_to_target), np.cos(angle_to_target))

        # Create velocity command
        cmd_vel = Twist()

        # Proportional control for linear velocity
        cmd_vel.linear.x = min(0.5, dist_to_target * 0.5)  # Max 0.5 m/s

        # Proportional control for angular velocity
        cmd_vel.angular.z = angle_to_target * 0.8  # Turn rate

        return cmd_vel

    def has_obstacles_in_front(self):
        """Check if there are obstacles in front of the robot"""
        if self.local_costmap is None:
            return False

        # Check the front sector of the local costmap
        # Convert robot's front direction to map coordinates
        robot_map_x, robot_map_y = self.world_to_map(
            self.robot_pose.position.x, self.robot_pose.position.y
        )

        # Check a sector in front of the robot
        front_threshold = 0.5 / self.local_costmap.info.resolution  # 50cm in map cells

        for i in range(int(front_threshold)):
            check_x = int(robot_map_x + i * np.cos(self.get_yaw_from_quaternion(self.robot_pose.orientation)))
            check_y = int(robot_map_y + i * np.sin(self.get_yaw_from_quaternion(self.robot_pose.orientation)))

            if (0 <= check_x < self.local_costmap.info.width and
                0 <= check_y < self.local_costmap.info.height):
                cell_value = self.local_costmap.data[
                    check_y * self.local_costmap.info.width + check_x
                ]
                if cell_value > 50:  # Occupied cell
                    return True

        return False

    def avoid_obstacles(self, cmd_vel):
        """Modify velocity command to avoid obstacles"""
        # Simple obstacle avoidance: reduce forward speed and turn away
        cmd_vel.linear.x *= 0.3  # Reduce forward speed to 30%
        cmd_vel.angular.z += 0.5  # Add turning motion

        return cmd_vel
```

## Nav2 Configuration for Humanoid Robotics

### Humanoid-Specific Parameters

```yaml
# nav2_params_humanoid.yaml
amcl:
  ros__parameters:
    use_sim_time: False
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha_slowweight: 0.0
    base_frame_id: "base_link"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: False
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: True
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

bt_navigator:
  ros__parameters:
    use_sim_time: False
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Humanoid-specific behavior tree
    behavior_tree_xml_filename: "navigate_w_replanning_and_recovery.xml"

controller_server:
  ros__parameters:
    use_sim_time: False
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    # Humanoid-specific controllers
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # DWB Controller for humanoid
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICtrl"
      debug_trajectory_details: False
      min_vel_x: 0.0
      max_vel_x: 0.5  # Reduced for humanoid stability
      min_vel_y: -0.5
      max_vel_y: 0.5
      max_vel_theta: 0.6  # Slower turning for stability
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      min_speed_theta: 0.0
      acc_lim_x: 2.5
      acc_lim_y: 0.0
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_y: 0.0
      decel_lim_theta: -3.2
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      linear_granularity: 0.05
      angular_granularity: 0.025
      transform_tolerance: 0.2
      xy_goal_tolerance: 0.25  # Larger tolerance for humanoid
      yaw_goal_tolerance: 0.25
      stateful: True
      global_plan_overwrite_orientation: True
      prune_plan: True
      prune_distance: 1.0
      controller_frequency: 20.0
      publish_cost_grid_pc: False
      conservative_reset_dist: 3.0
      cost_scaling_dist: 0.6
      cost_scaling_gain: 1.0
      oscillation_reset_dist: 0.05
      forward_sampling_distance: 0.3
      rotate_to_heading_angular_vel: 1.8
      use_rotate_to_heading: False
      max_iterations: 1000
      max_on_approach_iterations: 0
      use_interpolation: True

global_costmap:
  ros__parameters:
    update_frequency: 1.0
    publish_frequency: 1.0
    global_frame: map
    robot_base_frame: base_link
    use_sim_time: False
    rolling_window: false
    width: 100
    height: 100
    resolution: 0.05  # 5cm resolution
    origin_x: -50.0
    origin_y: -50.0
    plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: True

    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0  # Humanoid height consideration
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0

    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      enabled: True
      cost_scaling_factor: 3.0  # Higher inflation for humanoid safety
      inflation_radius: 0.6  # Larger safety margin for humanoid

local_costmap:
  ros__parameters:
    update_frequency: 5.0
    publish_frequency: 2.0
    global_frame: odom
    robot_base_frame: base_link
    use_sim_time: False
    rolling_window: true
    width: 6
    height: 6
    resolution: 0.05  # 5cm resolution
    plugins: ["voxel_layer", "inflation_layer"]

    voxel_layer:
      plugin: "nav2_costmap_2d::VoxelLayer"
      enabled: True
      publish_voxel_map: True
      origin_z: 0.0
      z_resolution: 0.2
      z_voxels: 10
      max_obstacle_height: 2.0
      mark_threshold: 0
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
        raytrace_max_range: 3.0
        raytrace_min_range: 0.0
        obstacle_max_range: 2.5
        obstacle_min_range: 0.0

    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      enabled: True
      cost_scaling_factor: 3.0
      inflation_radius: 0.4  # Smaller than global for local planning

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5  # Larger tolerance for humanoid
      use_astar: false
      allow_unknown: true

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
      ideal_linear_velocity: 0.0
      max_angular_accel: 3.2
      max_angular_velocity: 1.0
      min_angular_velocity: 0.4
      tolerance: 1.5707963267948966
    backup:
      plugin: "nav2_recoveries/BackUp"
      backup_dist: -0.15  # Back up 15cm
      backup_speed: 0.025
      sim_time: 2.0
      trans_stopped_velocity: 0.0001
      rotational_stopped_velocity: 0.0001
    wait:
      plugin: "nav2_recoveries/Wait"
      wait_duration: 1.0
```

## VSLAM Integration with Nav2

### Using VSLAM for Localization

```python
class VSLAMNav2Integration(Node):
    def __init__(self):
        super().__init__('vslam_nav2_integration')

        # VSLAM subscription
        self.vslam_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/vslam/pose_with_covariance',
            self.vslam_pose_callback,
            10
        )

        # Nav2 initial pose publisher
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # AMCL pose subscription (for comparison)
        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10
        )

        # VSLAM pose publisher for Nav2
        self.vslam_to_tf_timer = self.create_timer(0.05, self.publish_vslam_pose_as_tf)

        # Current poses
        self.vslam_pose = None
        self.amcl_pose = None

    def vslam_pose_callback(self, msg):
        """Receive pose from VSLAM system"""
        self.vslam_pose = msg

        # Optionally send to Nav2 as initial pose
        if not self.nav2_initialized:
            self.initialize_nav2_with_vslam(msg)

    def initialize_nav2_with_vslam(self, vslam_pose_msg):
        """Initialize Nav2 with VSLAM pose"""
        # Publish VSLAM pose as initial pose for Nav2
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = 'map'
        initial_pose.pose = vslam_pose_msg.pose

        self.initial_pose_pub.publish(initial_pose)
        self.nav2_initialized = True

    def publish_vslam_pose_as_tf(self):
        """Publish VSLAM pose as TF transform for Nav2"""
        if self.vslam_pose is None:
            return

        # Create TF broadcaster
        tf_broadcaster = TransformBroadcaster(self)

        # Create transform from map to odom via VSLAM pose
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # Use VSLAM pose as the transform
        t.transform.translation.x = self.vslam_pose.pose.pose.position.x
        t.transform.translation.y = self.vslam_pose.pose.pose.position.y
        t.transform.translation.z = self.vslam_pose.pose.pose.position.z

        t.transform.rotation = self.vslam_pose.pose.pose.orientation

        tf_broadcaster.sendTransform(t)

    def amcl_pose_callback(self, msg):
        """Receive AMCL pose for comparison with VSLAM"""
        self.amcl_pose = msg

        # Compare VSLAM and AMCL poses
        if self.vslam_pose is not None:
            self.compare_poses(self.vslam_pose, msg)

    def compare_poses(self, vslam_pose, amcl_pose):
        """Compare VSLAM and AMCL poses"""
        # Calculate position difference
        pos_diff = np.sqrt(
            (vslam_pose.pose.pose.position.x - amcl_pose.pose.pose.position.x)**2 +
            (vslam_pose.pose.pose.position.y - amcl_pose.pose.pose.position.y)**2 +
            (vslam_pose.pose.pose.position.z - amcl_pose.pose.pose.position.z)**2
        )

        # Calculate orientation difference
        vslam_quat = [
            vslam_pose.pose.pose.orientation.x,
            vslam_pose.pose.pose.orientation.y,
            vslam_pose.pose.pose.orientation.z,
            vslam_pose.pose.pose.orientation.w
        ]
        amcl_quat = [
            amcl_pose.pose.pose.orientation.x,
            amcl_pose.pose.pose.orientation.y,
            amcl_pose.pose.pose.orientation.z,
            amcl_pose.pose.pose.orientation.w
        ]

        # Convert quaternions to Euler angles and compare
        vslam_euler = self.quaternion_to_euler(vslam_quat)
        amcl_euler = self.quaternion_to_euler(amcl_quat)

        orientation_diff = abs(vslam_euler[2] - amcl_euler[2])  # Yaw difference

        self.get_logger().info(f'Pose difference - Position: {pos_diff:.3f}m, Orientation: {orientation_diff:.3f}rad')
```

### Map Building with VSLAM

```python
class VSLAMMapBuilder(Node):
    def __init__(self):
        super().__init__('vslam_map_builder')

        # VSLAM pose subscription
        self.vslam_pose_sub = self.create_subscription(
            PoseStamped,
            '/vslam/pose',
            self.vslam_pose_callback,
            10
        )

        # Camera image subscription
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Map publisher
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/vslam_map',
            10
        )

        # VSLAM map
        self.vslam_map = None
        self.keyframes = []
        self.map_resolution = 0.05  # 5cm resolution

    def vslam_pose_callback(self, msg):
        """Process VSLAM pose for map building"""
        # Store keyframe poses
        self.keyframes.append({
            'pose': msg.pose,
            'timestamp': msg.header.stamp
        })

        # Update map based on current pose
        self.update_map_from_pose(msg.pose)

    def image_callback(self, msg):
        """Process camera images for map building"""
        # Extract features from image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        features = self.extract_features(cv_image)

        # Associate features with current pose
        current_pose = self.get_latest_vslam_pose()
        if current_pose:
            self.associate_features_with_pose(features, current_pose)

    def update_map_from_pose(self, pose):
        """Update occupancy map based on robot pose"""
        if self.vslam_map is None:
            # Initialize map
            self.initialize_map(pose)

        # Update map based on pose
        # This is a simplified approach - real implementation would use
        # more sophisticated mapping algorithms
        self.mark_robot_position_as_free(pose)
        self.update_observed_areas(pose)

    def initialize_map(self, initial_pose):
        """Initialize occupancy grid map"""
        # Create a large enough map based on expected exploration area
        map_width = 100  # meters
        map_height = 100  # meters

        self.vslam_map = OccupancyGrid()
        self.vslam_map.header.frame_id = 'map'
        self.vslam_map.info.resolution = self.map_resolution
        self.vslam_map.info.width = int(map_width / self.map_resolution)
        self.vslam_map.info.height = int(map_height / self.map_resolution)

        # Center the map around the initial pose
        self.vslam_map.info.origin.position.x = initial_pose.position.x - (map_width / 2)
        self.vslam_map.info.origin.position.y = initial_pose.position.y - (map_height / 2)

        # Initialize all cells as unknown (-1)
        self.vslam_map.data = [-1] * (self.vslam_map.info.width * self.vslam_map.info.height)

    def mark_robot_position_as_free(self, pose):
        """Mark robot's current position as free space"""
        # Convert world coordinates to map coordinates
        map_x = int((pose.position.x - self.vslam_map.info.origin.position.x) / self.vslam_map.info.resolution)
        map_y = int((pose.position.y - self.vslam_map.info.origin.position.y) / self.vslam_map.info.resolution)

        # Mark robot position and nearby area as free
        robot_radius_cells = int(0.3 / self.vslam_map.info.resolution)  # 30cm radius

        for dx in range(-robot_radius_cells, robot_radius_cells + 1):
            for dy in range(-robot_radius_cells, robot_radius_cells + 1):
                x_idx = map_x + dx
                y_idx = map_y + dy

                if (0 <= x_idx < self.vslam_map.info.width and
                    0 <= y_idx < self.vslam_map.info.height):
                    # Mark as free space (0)
                    self.vslam_map.data[y_idx * self.vslam_map.info.width + x_idx] = 0

    def update_observed_areas(self, pose):
        """Update map based on observed areas from sensors"""
        # In a real implementation, this would use sensor data
        # to update occupancy probabilities
        pass

    def publish_map(self):
        """Publish the current map"""
        if self.vslam_map:
            self.vslam_map.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.vslam_map)
```

## Humanoid Navigation Challenges

### Humanoid-Specific Considerations

```python
class HumanoidNavigationNode(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_node')

        # Humanoid-specific navigation parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('step_height', 0.1),      # Maximum step height
                ('step_length', 0.4),      # Typical step length
                ('footprint_radius', 0.3), # Humanoid footprint
                ('turn_radius', 0.5),      # Minimum turning radius
                ('walking_speed', 0.5),    # Walking speed (m/s)
                ('balance_margin', 0.2),   # Safety margin for balance
            ]
        )

        # Get parameters
        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.footprint_radius = self.get_parameter('footprint_radius').value
        self.turn_radius = self.get_parameter('turn_radius').value
        self.walking_speed = self.get_parameter('walking_speed').value
        self.balance_margin = self.get_parameter('balance_margin').value

        # Navigation action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def navigate_with_humanoid_constraints(self, goal_pose):
        """Navigate considering humanoid physical constraints"""
        # Pre-process goal to ensure it's reachable by humanoid
        adjusted_goal = self.adjust_goal_for_humanoid_constraints(goal_pose)

        # Check if path is feasible for humanoid
        if not self.is_path_feasible_for_humanoid(adjusted_goal):
            self.get_logger().warn('Path not feasible for humanoid robot')
            return False

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = adjusted_goal

        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)

        return future

    def adjust_goal_for_humanoid_constraints(self, original_goal):
        """Adjust navigation goal based on humanoid constraints"""
        # Check if goal is too close to walls (considering humanoid footprint)
        adjusted_goal = original_goal

        # Verify the goal position is not in an area too narrow for humanoid
        if not self.is_position_accessible_by_humanoid(original_goal.pose.position):
            # Find nearest accessible position
            adjusted_position = self.find_nearest_accessible_position(original_goal.pose.position)
            adjusted_goal.pose.position = adjusted_position

        return adjusted_goal

    def is_position_accessible_by_humanoid(self, position):
        """Check if position is accessible considering humanoid dimensions"""
        # Check if position is in an area wide enough for humanoid
        # This would involve checking local costmap for sufficient clearance
        return True  # Placeholder - implement actual check

    def find_nearest_accessible_position(self, target_position):
        """Find nearest position that's accessible to humanoid"""
        # Use costmap to find nearest accessible position
        # This would implement a search algorithm
        return target_position  # Placeholder

    def is_path_feasible_for_humanoid(self, goal_pose):
        """Check if path to goal is feasible for humanoid"""
        # Check if path avoids stairs, steps higher than capability
        # Check if path width is sufficient for humanoid
        # Check if obstacles are at appropriate heights
        return True  # Placeholder - implement actual feasibility check
```

### Balance-Aware Navigation

```python
class BalanceAwareNavigation:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.center_of_mass_estimator = CenterOfMassEstimator(robot_model)
        self.balance_controller = BalanceController(robot_model)

    def plan_balance_safe_path(self, start_pose, goal_pose, environment_map):
        """Plan path that maintains robot balance"""
        # Consider center of mass during path planning
        # Ensure path doesn't require extreme poses that could cause imbalance
        safe_path = self.find_balance_safe_path(start_pose, goal_pose, environment_map)

        return safe_path

    def find_balance_safe_path(self, start_pose, goal_pose, environment_map):
        """Find path that maintains balance throughout navigation"""
        # Use modified A* or other algorithm that considers balance
        # Add balance cost to path planning heuristic
        pass

    def evaluate_balance_along_path(self, path):
        """Evaluate balance requirements along the path"""
        balance_metrics = []
        for pose in path:
            balance_state = self.estimate_balance_state_at_pose(pose)
            balance_metrics.append(balance_state)

        return balance_metrics

    def estimate_balance_state_at_pose(self, pose):
        """Estimate balance state when robot is at given pose"""
        # Estimate center of mass position relative to support polygon
        # Consider foot placement and stability
        pass
```

## Nav2 Recovery Behaviors for Humanoid Robots

### Humanoid-Specific Recovery Strategies

```python
class HumanoidRecoveryBehaviors:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.balance_checker = BalanceChecker(robot_controller)
        self.obstacle_detector = ObstacleDetector()

    def clear_humanoid_local_costmap(self):
        """Clear local costmap with humanoid-specific strategy"""
        # Clear costmap while maintaining safety for humanoid balance
        # Consider that humanoid might need more space to maneuver
        pass

    def humanoid_spin_recovery(self):
        """Spin recovery behavior adapted for humanoid"""
        # Humanoid spin: rotate while maintaining balance
        # Use coordinated joint movements to maintain stability
        if self.balance_checker.is_balanced():
            self.robot_controller.rotate_in_place(90)  # Rotate 90 degrees
        else:
            self.balance_controller.restore_balance()
            return False
        return True

    def humanoid_backup_recovery(self):
        """Backup recovery adapted for humanoid"""
        # Move backward carefully while maintaining balance
        # Use controlled stepping if available
        if self.balance_checker.is_balanced():
            self.robot_controller.move_backward_safely(0.3)  # 30cm
        else:
            self.balance_controller.restore_balance()
            return False
        return True

    def humanoid_wait_recovery(self):
        """Wait recovery with balance monitoring"""
        # Wait while monitoring balance and environment
        # Resume when conditions improve
        start_time = time.time()
        wait_duration = 3.0  # 3 seconds

        while time.time() - start_time < wait_duration:
            if self.balance_checker.is_balanced():
                # Check if obstacle has moved
                if not self.obstacle_detector.has_obstacles_in_path():
                    return True  # Conditions improved
            else:
                self.balance_controller.restore_balance()

            time.sleep(0.1)  # Check every 100ms

        return False  # Wait timeout
```

## Isaac ROS Navigation Integration

### GPU-Accelerated Path Planning

```python
class IsaacROSGPUPathPlanner:
    def __init__(self):
        # Initialize GPU context for path planning
        self.gpu_context = self.initialize_gpu_context()
        self.gpu_path_planner = self.create_gpu_path_planner()

    def plan_path_gpu_accelerated(self, start, goal, costmap):
        """Plan path using GPU acceleration"""
        # Transfer costmap to GPU
        gpu_costmap = self.transfer_costmap_to_gpu(costmap)

        # Plan path on GPU
        gpu_path = self.gpu_path_planner.plan(gpu_costmap, start, goal)

        # Transfer path back to CPU
        cpu_path = self.transfer_path_from_gpu(gpu_path)

        return cpu_path

    def initialize_gpu_context(self):
        """Initialize CUDA context for path planning"""
        # Set up CUDA context for parallel path planning
        # This could involve setting up CUDA streams, memory pools, etc.
        pass

    def create_gpu_path_planner(self):
        """Create GPU-accelerated path planner"""
        # This would create a path planner that uses CUDA kernels
        # for parallel graph search or other path planning algorithms
        pass

    def transfer_costmap_to_gpu(self, costmap):
        """Transfer costmap data to GPU memory"""
        # Convert costmap to GPU-friendly format
        # Transfer to GPU memory
        pass

    def transfer_path_from_gpu(self, gpu_path):
        """Transfer path from GPU memory to CPU"""
        # Convert GPU path representation to standard format
        # Transfer from GPU memory
        pass
```

## Performance Optimization

### Multi-Threaded Navigation

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class MultiThreadedNavigation:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.global_planner = Nav2GlobalPlanner()
        self.local_planner = Nav2LocalPlanner()
        self.controller = ControllerBase()

    def run_navigation_pipeline(self, goal):
        """Run navigation pipeline using multiple threads"""
        # Submit global planning task
        global_plan_future = self.executor.submit(
            self.global_planner.compute_path_to_pose, goal
        )

        # Monitor local conditions while global planning
        local_monitoring_future = self.executor.submit(
            self.local_planner.monitor_local_conditions
        )

        # Wait for global plan
        global_plan = global_plan_future.result()

        # Execute local planning and control
        local_plan_future = self.executor.submit(
            self.local_planner.follow_path, global_plan
        )

        control_future = self.executor.submit(
            self.controller.execute_path, global_plan
        )

        return local_plan_future.result(), control_future.result()
```

## Navigation Quality Assessment

### Navigation Performance Metrics

```python
class NavigationMetrics:
    def __init__(self):
        self.path_efficiency = 0.0  # Ratio of straight-line to actual path length
        self.navigation_success_rate = 0.0
        self.average_completion_time = 0.0
        self.collision_rate = 0.0
        self.path_deviation = 0.0  # Average deviation from planned path

    def calculate_path_efficiency(self, planned_path, actual_path):
        """Calculate path efficiency metric"""
        planned_length = self.calculate_path_length(planned_path)
        actual_length = self.calculate_path_length(actual_path)

        if planned_length > 0:
            return planned_length / actual_length
        else:
            return 0.0

    def calculate_navigation_success_rate(self, attempts, successes):
        """Calculate navigation success rate"""
        if attempts > 0:
            return successes / attempts
        else:
            return 0.0

    def calculate_path_deviation(self, planned_path, actual_path):
        """Calculate average deviation from planned path"""
        deviations = []

        for actual_pose in actual_path:
            closest_planned_pose = self.find_closest_pose_on_path(actual_pose, planned_path)
            deviation = self.calculate_distance(actual_pose, closest_planned_pose)
            deviations.append(deviation)

        return sum(deviations) / len(deviations) if deviations else 0.0

    def calculate_path_length(self, path):
        """Calculate total length of path"""
        total_length = 0.0
        for i in range(1, len(path)):
            segment_length = self.calculate_distance(path[i-1], path[i])
            total_length += segment_length
        return total_length
```

## Troubleshooting Nav2

### Common Navigation Issues

#### 1. Local Planner Oscillation
**Problem**: Robot oscillates back and forth in front of obstacles
**Solutions**:
- Adjust local planner parameters (increase minimum turning radius)
- Increase inflation radius in costmaps
- Improve obstacle detection and filtering
- Use recovery behaviors more aggressively

#### 2. Global Planner Failure
**Problem**: Cannot find path to goal
**Solutions**:
- Check map quality and resolution
- Verify goal position is in free space
- Adjust planner parameters (tolerance, algorithm)
- Check for disconnected map regions

#### 3. Localization Drift
**Problem**: Robot position estimate becomes inaccurate
**Solutions**:
- Improve sensor quality (better LiDAR, camera calibration)
- Use VSLAM for better localization
- Add fiducial markers to environment
- Tune AMCL parameters

#### 4. Navigation Timeout
**Problem**: Navigation takes too long or fails
**Solutions**:
- Increase timeout parameters
- Use simpler recovery behaviors
- Improve path planning algorithms
- Optimize costmap update frequency

### Navigation Diagnostics

```python
class NavigationDiagnostics:
    def __init__(self, navigation_node):
        self.nav_node = navigation_node
        self.metrics = NavigationMetrics()
        self.diagnostic_timer = self.nav_node.create_timer(1.0, self.run_diagnostics)

    def run_diagnostics(self):
        """Run navigation diagnostics"""
        diagnostics = {}

        # Check navigation status
        diagnostics['navigation_active'] = self.nav_node.is_navigation_active()
        diagnostics['current_goal'] = self.nav_node.get_current_goal()
        diagnostics['robot_pose'] = self.nav_node.get_robot_pose()

        # Check costmaps
        diagnostics['global_costmap_valid'] = self.nav_node.is_global_costmap_valid()
        diagnostics['local_costmap_valid'] = self.nav_node.is_local_costmap_valid()

        # Check controllers
        diagnostics['controller_status'] = self.nav_node.get_controller_status()

        # Performance metrics
        diagnostics['path_efficiency'] = self.metrics.path_efficiency
        diagnostics['success_rate'] = self.metrics.navigation_success_rate

        # Log diagnostics
        self.log_diagnostics(diagnostics)

    def log_diagnostics(self, diagnostics):
        """Log diagnostic information"""
        self.nav_node.get_logger().info(f'Navigation Status: {diagnostics["navigation_active"]}')
        self.nav_node.get_logger().info(f'Path Efficiency: {diagnostics["path_efficiency"]}')
        self.nav_node.get_logger().info(f'Success Rate: {diagnostics["success_rate"]}')
```

## Best Practices for Humanoid Navigation

### 1. Safety First
- Always maintain safety margins in costmaps
- Implement emergency stop behaviors
- Monitor balance continuously during navigation
- Use conservative speed limits

### 2. Environmental Awareness
- Consider dynamic obstacles (people, moving objects)
- Account for environmental changes over time
- Handle different terrain types appropriately
- Plan for various lighting conditions

### 3. Human-Robot Interaction
- Plan paths that are predictable to humans
- Maintain appropriate social distances
- Consider human traffic patterns
- Signal navigation intentions clearly

### 4. Performance Optimization
- Use appropriate map resolutions
- Optimize costmap update frequencies
- Implement efficient path planning algorithms
- Use GPU acceleration where possible

## Integration with Higher-Level Systems

### Task and Motion Planning

```python
class TaskMotionPlanner:
    def __init__(self, nav2_system, task_planner):
        self.nav2_system = nav2_system
        self.task_planner = task_planner

    def plan_task_with_navigation(self, high_level_task):
        """Plan high-level task with required navigation"""
        # Decompose task into subtasks
        subtasks = self.task_planner.decompose_task(high_level_task)

        navigation_goals = []
        for subtask in subtasks:
            if subtask.requires_navigation():
                # Plan navigation to subtask location
                nav_goal = self.nav2_system.plan_to_location(subtask.location)
                navigation_goals.append(nav_goal)

        # Combine task and navigation plans
        combined_plan = self.merge_task_and_navigation_plans(subtasks, navigation_goals)

        return combined_plan
```

## Future of Navigation for Humanoid Robots

### Emerging Technologies

#### Semantic Navigation
- Use semantic maps for higher-level navigation
- Navigate based on object recognition and scene understanding
- Plan paths based on environmental context

#### Learning-Based Navigation
- Learn navigation behaviors from demonstration
- Adapt to new environments through experience
- Improve path planning through reinforcement learning

#### Collaborative Navigation
- Navigate in coordination with humans
- Share space and paths with multiple agents
- Communicate navigation intentions

## Nav2 Configuration for Humanoid Robots

### Humanoid-Specific Navigation Parameters

```yaml
# Example Nav2 configuration for humanoid robot
bt_navigator:
  ros__parameters:
    # Behavior tree configuration
    use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_are_error_positions_equal_condition_bt_node
      - nav2_would_a_controller_recovery_help_condition_bt_node
      - nav2_amr_is_path_valid_condition_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_is_battery_charging_condition_bt_node
      - nav2_is_battery_charged_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transformer_node_bt_node
      - nav2_get_costmap_node_bt_node
      - nav2_get_costmap_node_v2_bt_node
      - nav2_get_local_plan_node_bt_node
      - nav2_get_path_node_bt_node
      - nav2_get_dummy_path_node_bt_node
      - nav2_get_rotation_command_node_bt_node
      - nav2_get_tracking_point_node_bt_node
      - nav2_controller_selector_node_bt_node
      - nav2_goal_checker_selector_node_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_back_up_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node

# Humanoid-specific costmap configuration
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 10.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: false
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      # Humanoid-specific parameters
      footprint: "[[-0.3, -0.2], [-0.3, 0.2], [0.3, 0.2], [0.3, -0.2]]"  # Larger footprint for humanoid
      footprint_padding: 0.1
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0  # Humanoid can see obstacles up to 2m
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 10.0
          raytrace_min_range: 0.0
          obstacle_max_range: 5.0
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher inflation for safety
        inflation_radius: 0.55    # Larger safety buffer for humanoid

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: false
      robot_radius: 0.3  # Humanoid radius
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 10.0
          raytrace_min_range: 0.0
          obstacle_max_range: 5.0
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

## Humanoid Footstep Planning

### Footstep Planner Integration

```python
class HumanoidFootstepPlanner:
    def __init__(self):
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (foot lift)
        self.step_time = 1.0    # seconds per step

    def plan_footsteps(self, path, robot_pose):
        """Plan footstep sequence for humanoid navigation"""
        footsteps = []

        # Convert navigation path to footstep plan
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            # Calculate direction vector
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = np.sqrt(dx*dx + dy*dy)

            # Generate footsteps along the path
            num_steps = int(dist / self.step_length) + 1

            for j in range(num_steps):
                step_x = start[0] + (dx * j / num_steps)
                step_y = start[1] + (dy * j / num_steps)

                # Alternate feet (left/right)
                foot_type = "left" if j % 2 == 0 else "right"

                footsteps.append({
                    'position': (step_x, step_y),
                    'foot_type': foot_type,
                    'step_time': self.step_time * j
                })

        return footsteps

    def execute_footsteps(self, footsteps):
        """Execute footstep plan"""
        for step in footsteps:
            # Move to footstep position
            self.move_to_footstep(step)

            # Wait for step completion
            time.sleep(self.step_time)

    def move_to_footstep(self, step):
        """Move robot foot to specified position"""
        # This would interface with humanoid robot's leg controller
        pass
```

## Social Navigation for Humanoid Robots

### Human-Aware Navigation

```python
class SocialNavigationNode(Node):
    def __init__(self):
        super().__init__('social_navigation_node')

        # Human detection and tracking
        self.human_detector = self.create_subscription(
            Detection2DArray, '/human_detector/detections',
            self.human_detection_callback, 10
        )

        # Social navigation parameters
        self.social_params = {
            'personal_space_radius': 0.8,  # meters
            'social_zone_radius': 1.5,     # meters
            'group_zone_radius': 3.0,      # meters
            'avoidance_speed_factor': 0.7, # Reduce speed near humans
            'respectful_distance': 1.2     # Maintain distance
        }

        self.humans = []
        self.groups = []

    def human_detection_callback(self, msg):
        """Process human detections for social navigation"""
        self.humans = []

        for detection in msg.detections:
            human = {
                'position': (detection.bbox.center.x, detection.bbox.center.y),
                'velocity': self.estimate_human_velocity(detection),
                'intent': self.classify_human_intent(detection)
            }
            self.humans.append(human)

    def estimate_human_velocity(self, detection):
        """Estimate human velocity from tracking"""
        # Implementation for velocity estimation
        pass

    def classify_human_intent(self, detection):
        """Classify human intent (walking, stopping, etc.)"""
        # Implementation for intent classification
        return "walking"

    def compute_social_path(self, start, goal):
        """Compute path considering social constraints"""
        # Modify costmap to account for humans
        social_costmap = self.create_social_costmap()

        # Plan path with social constraints
        path = self.plan_path_with_costmap(start, goal, social_costmap)

        return path

    def create_social_costmap(self):
        """Create costmap with social constraints"""
        # Create costmap considering human positions and social zones
        costmap = np.zeros((100, 100))  # Example costmap

        for human in self.humans:
            # Add cost for personal space
            self.add_circular_cost(
                costmap,
                human['position'],
                self.social_params['personal_space_radius'],
                255  # High cost
            )

            # Add cost for social zone
            self.add_circular_cost(
                costmap,
                human['position'],
                self.social_params['social_zone_radius'],
                128  # Medium cost
            )

        return costmap

    def add_circular_cost(self, costmap, center, radius, cost):
        """Add circular cost to costmap"""
        center_x, center_y = int(center[0]), int(center[1])
        radius_cells = int(radius / 0.05)  # Assuming 0.05m resolution

        for x in range(max(0, center_x - radius_cells),
                      min(costmap.shape[0], center_x + radius_cells)):
            for y in range(max(0, center_y - radius_cells),
                          min(costmap.shape[1], center_y + radius_cells)):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius_cells:
                    costmap[x, y] = max(costmap[x, y], cost)
```

## Nav2 Performance Optimization

### Multi-Threading and Asynchronous Processing

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class OptimizedNav2Node(Node):
    def __init__(self):
        super().__init__('optimized_nav2_node')

        # Use thread pool for heavy computations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Async timers for different navigation components
        self.async_timer = self.create_timer(0.1, self.async_navigation_update)

        # Navigation components
        self.global_planner = GlobalPlannerAsync()
        self.local_planner = LocalPlannerAsync()
        self.controller = ControllerAsync()

    async def async_navigation_update(self):
        """Asynchronous navigation update"""
        # Run multiple navigation tasks concurrently
        tasks = [
            self.update_global_plan_async(),
            self.update_local_plan_async(),
            self.update_controller_async()
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def update_global_plan_async(self):
        """Asynchronously update global plan"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.global_planner.plan,
            self.current_goal
        )

    async def update_local_plan_async(self):
        """Asynchronously update local plan"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.local_planner.plan,
            self.global_path,
            self.local_costmap
        )

    async def update_controller_async(self):
        """Asynchronously update controller"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.controller.compute_velocity,
            self.local_plan,
            self.robot_state
        )
```

## Nav2 Troubleshooting for Humanoid Robots

### Common Issues and Solutions

#### 1. Navigation Failures
**Problem**: Robot fails to navigate successfully
**Solutions**:
- Check costmap configuration for humanoid dimensions
- Verify sensor data quality and range
- Adjust planner parameters for humanoid speed
- Implement proper recovery behaviors

#### 2. Balance Issues During Navigation
**Problem**: Robot loses balance while navigating
**Solutions**:
- Reduce navigation speed for stability
- Implement balance monitoring during navigation
- Use smaller lookahead distances
- Add balance recovery behaviors

#### 3. Path Planning Issues
**Problem**: Robot plans impossible paths
**Solutions**:
- Adjust robot footprint for humanoid dimensions
- Increase inflation radius for safety
- Use appropriate planner for humanoid kinematics
- Consider step constraints in path planning

#### 4. Social Navigation Problems
**Problem**: Robot doesn't respect human presence
**Solutions**:
- Implement proper human detection and tracking
- Configure social costmap parameters
- Add social navigation behaviors
- Test in human environments

### Debugging Tools

```bash
# Monitor Nav2 status
ros2 run nav2_util lifecycle_bringup

# Visualize costmaps
ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz

# Check behavior tree execution
ros2 run nav2_bt_navigator bt_navigator

# Monitor navigation performance
ros2 run nav2_msgs navigation_performance_analyzer
```

## Best Practices for Humanoid Navigation

### 1. Safety First
- Always maintain safety buffers in costmaps
- Implement emergency stop capabilities
- Monitor robot balance continuously
- Use conservative navigation parameters

### 2. Performance Optimization
- Optimize costmap resolution for your application
- Use appropriate planner for environment type
- Implement efficient path smoothing
- Monitor and optimize computation time

### 3. Human-Robot Interaction
- Implement social navigation behaviors
- Consider human comfort zones
- Add appropriate speed reduction near humans
- Test navigation in human environments

### 4. Robustness
- Implement comprehensive recovery behaviors
- Handle sensor failures gracefully
- Validate path feasibility before execution
- Monitor navigation performance continuously

## Integration with Other Systems

### VSLAM Integration

```python
class VSLAMNav2Integration:
    def __init__(self):
        self.vslam_pose = None
        self.nav2_localizer = None

    def update_localization(self, vslam_pose):
        """Update robot localization with VSLAM data"""
        self.vslam_pose = vslam_pose
        # Fuse with other localization sources
        fused_pose = self.fuse_localization_sources(vslam_pose)
        self.nav2_localizer.update_pose(fused_pose)
```

### Manipulation Integration

```python
class NavigationManipulationIntegration:
    def __init__(self):
        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, MoveToPose, 'move_to_pose')

    async def navigate_and_manipulate(self, nav_goal, manipulation_goal):
        """Execute navigation followed by manipulation"""
        # Navigate to manipulation location
        nav_result = await self.execute_navigation(nav_goal)

        if nav_result.success:
            # Execute manipulation task
            manip_result = await self.execute_manipulation(manipulation_goal)
            return manip_result
        else:
            return nav_result
```

## Next Steps

With a solid understanding of Nav2 path planning and its integration with VSLAM, continue to [Sim-to-Real Concepts](./sim-to-real.md) to learn about bridging the gap between simulation and real-world deployment for humanoid robotics systems.