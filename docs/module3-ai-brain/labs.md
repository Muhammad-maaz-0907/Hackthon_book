---
title: Module 3 Labs
sidebar_position: 8
---

# Module 3 Labs: AI-Robot Brain Implementation

This lab section provides hands-on exercises to reinforce the AI-Robot Brain concepts you've learned in Module 3. Each lab builds on the previous one to give you practical experience with Isaac Sim, Isaac ROS, VSLAM, and other AI systems in the context of humanoid robotics.

## Lab 3A: Isaac Sim Environment Setup

### Objective
Set up and configure Isaac Sim for humanoid robotics development, including proper installation, configuration, and basic scene creation.

### Prerequisites
- Ubuntu 22.04 LTS
- NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- ROS 2 Humble Hawksbill
- Basic understanding of 3D modeling concepts

### Steps

1. **Install Isaac Sim**:
   ```bash
   # Download Isaac Sim from NVIDIA Developer Portal
   # Extract to home directory
   cd ~/isaac-sim
   ./isaac-sim-headless.sh  # For headless operation
   # Or
   ./isaac-sim-gui.sh       # For GUI operation
   ```

2. **Verify Installation**:
   ```bash
   # Check if Isaac Sim launches properly
   python3 -c "import omni; print('Isaac Sim imported successfully')"

   # Launch Isaac Sim in headless mode
   python3 -c "
   import omni
   from omni.isaac.core import World
   world = World(stage_units_in_meters=1.0)
   print('Isaac Sim World created successfully')
   world.play()
   world.step()
   world.stop()
   "
   ```

3. **Create Basic Humanoid Scene**:
   ```python
   # Create a Python script to set up a basic humanoid scene
   # Save as basic_humanoid_scene.py
   import omni
   from omni.isaac.core import World
   from omni.isaac.core.utils.stage import add_reference_to_stage
   from omni.isaac.core.utils.prims import create_primitive
   from omni.isaac.core.robots import Robot
   import numpy as np

   # Initialize world
   my_world = World(stage_units_in_meters=1.0)

   # Add ground plane
   create_primitive(
       prim_path="/World/GroundPlane",
       primitive_props={"size": 1000, "color": np.array([0.5, 0.5, 0.5])}
   )

   # Add a simple humanoid model (using a basic robot for now)
   # In a real scenario, you would add a proper humanoid model
   add_reference_to_stage(
       usd_path="path/to/humanoid_model.usd",
       prim_path="/World/Humanoid"
   )

   # Play the simulation
   my_world.play()

   # Run simulation steps
   for i in range(100):
       my_world.step(render=True)

   my_world.stop()
   ```

4. **Test Basic Simulation**:
   ```bash
   # Run the basic scene
   python3 basic_humanoid_scene.py
   ```

### Expected Output
- Isaac Sim launches without errors
- Basic scene with ground plane loads successfully
- Simulation runs with basic physics

### Troubleshooting
- If Isaac Sim fails to launch, check NVIDIA GPU drivers
- If physics don't work, verify CUDA installation
- If models don't load, check USD file paths and permissions

## Lab 3B: Isaac ROS Integration

### Objective
Connect Isaac Sim with ROS 2 using Isaac ROS packages and verify communication between simulation and ROS nodes.

### Steps

1. **Install Isaac ROS Packages**:
   ```bash
   # Add NVIDIA package repository
   curl -sL https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -sL https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

   sudo apt update

   # Install Isaac ROS packages
   sudo apt install ros-humble-isaac-ros-common
   sudo apt install ros-humble-isaac-ros-perception
   sudo apt install ros-humble-isaac-ros-navigation
   sudo apt install ros-humble-isaac-ros-manipulation
   ```

2. **Create Isaac ROS Bridge Test**:
   ```python
   # Create isaac_ros_bridge_test.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, Imu, JointState
   from geometry_msgs.msg import Twist
   from std_msgs.msg import String
   import numpy as np

   class IsaacROSBridgeTest(Node):
       def __init__(self):
           super().__init__('isaac_ros_bridge_test')

           # Publishers for sending commands to Isaac Sim
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)

           # Subscribers for receiving data from Isaac Sim
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10
           )
           self.imu_sub = self.create_subscription(
               Imu, '/imu/data', self.imu_callback, 10
           )
           self.joint_state_sub = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10
           )

           # Timer for sending test commands
           self.timer = self.create_timer(0.1, self.send_test_commands)

           self.get_logger().info('Isaac ROS Bridge Test node initialized')

       def image_callback(self, msg):
           self.get_logger().info(f'Received image: {msg.width}x{msg.height}')

       def imu_callback(self, msg):
           self.get_logger().info(f'IMU: linear_acceleration=({msg.linear_acceleration.x:.3f}, {msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f})')

       def joint_state_callback(self, msg):
           self.get_logger().info(f'Received {len(msg.position)} joint positions')

       def send_test_commands(self):
           # Send a simple velocity command
           cmd = Twist()
           cmd.linear.x = 0.5  # Move forward at 0.5 m/s
           cmd.angular.z = 0.2  # Rotate at 0.2 rad/s
           self.cmd_vel_pub.publish(cmd)

   def main(args=None):
       rclpy.init(args=args)
       node = IsaacROSBridgeTest()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Launch Isaac Sim with ROS Bridge**:
   ```bash
   # Terminal 1: Launch Isaac Sim with ROS bridge
   cd ~/isaac-sim
   ./isaac-sim-gui.sh &

   # In Isaac Sim, run the ROS bridge example:
   # In the Isaac Sim UI, go to Window > Extensions > Isaac ROS Bridge
   # Or run from Python:
   python3 -c "
   import omni
   from omni.isaac.ros_bridge import create_ros_bridge
   # This would create a ROS bridge in Isaac Sim
   "
   ```

4. **Test Communication**:
   ```bash
   # Terminal 2: Run the ROS bridge test
   source /opt/ros/humble/setup.bash
   source /usr/local/cuda/bin/cuda-env.sh  # If needed
   python3 isaac_ros_bridge_test.py
   ```

5. **Verify Communication**:
   ```bash
   # Check if topics are being published
   ros2 topic list | grep -E "(camera|imu|joint)"

   # Echo sensor data
   ros2 topic echo /camera/image_raw --field header.frame_id
   ros2 topic echo /imu/data --field orientation
   ```

### Expected Output
- Isaac Sim and ROS 2 communicate successfully
- Sensor data flows from Isaac Sim to ROS
- Commands can be sent from ROS to Isaac Sim
- No communication errors or dropped messages

### Troubleshooting
- Check if ROS domain IDs match between Isaac Sim and ROS nodes
- Verify network configuration if running remotely
- Ensure Isaac Sim has ROS bridge extension enabled
- Check for firewall issues if running on different machines

## Lab 3C: VSLAM Implementation

### Objective
Implement and test Visual SLAM in Isaac Sim, connecting it to ROS 2 for real-time mapping and localization.

### Steps

1. **Set Up VSLAM Components**:
   ```bash
   # Install VSLAM packages
   sudo apt install ros-humble-isaac-ros-visual- slam
   sudo apt install ros-humble-rtabmap-ros
   sudo apt install ros-humble-openslam-gmapping
   ```

2. **Create VSLAM Test Script**:
   ```python
   # Create vslam_test.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, CameraInfo
   from nav_msgs.msg import Odometry
   from geometry_msgs.msg import PoseStamped
   from cv_bridge import CvBridge
   import numpy as np
   import cv2

   class VSLAMTestNode(Node):
       def __init__(self):
           super().__init__('vslam_test_node')

           # Initialize CV bridge
           self.cv_bridge = CvBridge()

           # Subscriptions
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10
           )
           self.camera_info_sub = self.create_subscription(
               CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
           )
           self.odom_sub = self.create_subscription(
               Odometry, '/visual_slam/odometry', self.odom_callback, 10
           )

           # Publishers
           self.pose_pub = self.create_publisher(PoseStamped, '/estimated_pose', 10)

           # VSLAM state
           self.camera_intrinsics = None
           self.latest_image = None
           self.position_estimate = np.zeros(3)

           self.get_logger().info('VSLAM Test Node initialized')

       def camera_info_callback(self, msg):
           """Store camera intrinsic parameters"""
           if self.camera_intrinsics is None:
               self.camera_intrinsics = np.array([
                   [msg.k[0], msg.k[1], msg.k[2]],
                   [msg.k[3], msg.k[4], msg.k[5]],
                   [msg.k[6], msg.k[7], msg.k[8]]
               ])
               self.get_logger().info('Camera intrinsics received')

       def image_callback(self, msg):
           """Process camera images for VSLAM"""
           try:
               # Convert ROS image to OpenCV
               cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

               # Store for processing
               self.latest_image = cv_image

               # In a real implementation, you would:
               # 1. Extract features from the image
               # 2. Match features with previous frames
               # 3. Estimate motion using triangulation
               # 4. Update pose estimate

               # For this test, just log that we received an image
               self.get_logger().info(f'Processed image: {cv_image.shape}')

           except Exception as e:
               self.get_logger().error(f'Error processing image: {e}')

       def odom_callback(self, msg):
           """Process odometry from VSLAM system"""
           self.position_estimate[0] = msg.pose.pose.position.x
           self.position_estimate[1] = msg.pose.pose.position.y
           self.position_estimate[2] = msg.pose.pose.position.z

           self.get_logger().info(f'VSLAM Position: {self.position_estimate}')

   def main(args=None):
       rclpy.init(args=args)
       node = VSLAMTestNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Launch VSLAM Pipeline**:
   ```bash
   # Create a launch file for VSLAM: vslam_pipeline.launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration

   def generate_launch_description():
       return LaunchDescription([
           # Declare launch arguments
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),

           # Visual SLAM node (Isaac ROS implementation)
           Node(
               package='isaac_ros_visual_slam',
               executable='visual_slam_node',
               name='visual_slam_node',
               parameters=[{
                   'use_sim_time': LaunchConfiguration('use_sim_time'),
                   'enable_rectified_pose': True,
                   'enable_fisheye_distortion': False,
                   'map_frame': 'map',
                   'odom_frame': 'odom',
                   'base_frame': 'base_link',
                   'sub_camera_info_topic_name': 'camera_info',
                   'sub_image_topic_name': 'image',
                   'use_odometry_input': False
               }],
               remappings=[
                   ('/visual_slam/image', '/camera/image_raw'),
                   ('/visual_slam/camera_info', '/camera/camera_info')
               ]
           ),

           # Pose broadcaster
           Node(
               package='tf2_ros',
               executable='static_transform_publisher',
               name='camera_broadcaster',
               arguments=['0', '0', '0.5', '0', '0', '0', 'base_link', 'camera_link']
           )
       ])
   ```

4. **Run VSLAM Test**:
   ```bash
   # Terminal 1: Launch Isaac Sim with camera
   # Run Isaac Sim with a camera-equipped robot model

   # Terminal 2: Launch VSLAM pipeline
   ros2 launch vslam_pipeline.launch.py

   # Terminal 3: Run VSLAM test node
   python3 vslam_test.py
   ```

5. **Test VSLAM Performance**:
   ```bash
   # Monitor VSLAM performance
   ros2 topic echo /visual_slam/track_odometry --field pose.pose.position
   ros2 topic echo /visual_slam/map --field info.width
   ```

### Expected Output
- VSLAM system processes camera images successfully
- Position estimates are published consistently
- Map is being built as robot moves through environment
- No tracking failures or drift in static environment

### Troubleshooting
- Check camera calibration parameters
- Verify sufficient visual features in environment
- Ensure proper lighting conditions
- Check for sufficient processing power

## Lab 3D: Isaac ROS Navigation Stack

### Objective
Set up and test the navigation stack using Isaac ROS packages for humanoid robot navigation in simulation.

### Steps

1. **Install Navigation Packages**:
   ```bash
   # Install navigation packages
   sudo apt install ros-humble-navigation2
   sudo apt install ros-humble-nav2-bringup
   sudo apt install ros-humble-isaac-ros-navigation
   ```

2. **Create Navigation Configuration**:
   ```yaml
   # Create nav2_params_isaac.yaml
   amcl:
     ros__parameters:
       use_sim_time: true
       alpha1: 0.2
       alpha2: 0.2
       alpha3: 0.2
       alpha4: 0.2
       alpha_slowweight: 0.0
       base_frame_id: "base_link"
       beam_skip_distance: 0.5
       beam_skip_error_threshold: 0.9
       beam_skip_threshold: 0.3
       do_beamskip: false
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
       tf_broadcast: true
       transform_tolerance: 1.0
       update_min_a: 0.2
       update_min_d: 0.25
       z_hit: 0.5
       z_max: 0.05
       z_rand: 0.5
       z_short: 0.05

   bt_navigator:
     ros__parameters:
       use_sim_time: true
       global_frame: map
       robot_base_frame: base_link
       odom_topic: /odom
       bt_loop_duration: 10
       default_server_timeout: 20
       enable_groot_monitoring: true
       groot_zmq_publisher_port: 1666
       groot_zmq_server_port: 1667
       # Humanoid-specific behavior tree
       default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
       plugin_lib_names:
       - nav2_compute_path_to_pose_action_bt_node
       - nav2_follow_path_action_bt_node
       - nav2_back_up_action_bt_node
       - nav2_spin_action_bt_node
       - nav2_wait_action_bt_node
       - nav2_clear_costmap_service_bt_node
       - nav2_is_stuck_condition_bt_node
       - nav2_goal_reached_condition_bt_node
       - nav2_goal_updated_condition_bt_node
       - nav2_initial_pose_received_condition_bt_node
       - nav2_reinitialize_global_localization_service_bt_node
       - nav2_rate_controller_bt_node
       - nav2_distance_controller_bt_node
       - nav2_speed_controller_bt_node
       - nav2_truncate_path_action_bt_node
       - nav2_goal_updater_node_bt_node
       - nav2_recovery_node_bt_node
       - nav2_pipeline_sequence_bt_node
       - nav2_round_robin_node_bt_node
       - nav2_transform_available_condition_bt_node
       - nav2_time_expired_condition_bt_node
       - nav2_path_expiring_timer_condition
       - nav2_distance_traveled_condition_bt_node
       - nav2_single_trigger_bt_node
       - nav2_is_battery_low_condition_bt_node
       - nav2_navigate_through_poses_action_bt_node
       - nav2_navigate_to_pose_action_bt_node
       - nav2_remove_passed_goals_action_bt_node
       - nav2_planner_selector_bt_node
       - nav2_controller_selector_bt_node
       - nav2_goal_checker_selector_bt_node

   controller_server:
     ros__parameters:
       use_sim_time: true
       controller_frequency: 20.0
       min_x_velocity_threshold: 0.001
       min_y_velocity_threshold: 0.5
       min_theta_velocity_threshold: 0.001
       progress_checker_plugin: "progress_checker"
       goal_checker_plugin: "goal_checker"
       controller_plugins: ["FollowPath"]

       # Humanoid-specific controller
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
       use_sim_time: true
       rolling_window: false
       width: 100
       height: 100
       resolution: 0.05
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
         cost_scaling_factor: 3.0
         inflation_radius: 0.6  # Larger safety margin for humanoid

   local_costmap:
     ros__parameters:
       update_frequency: 5.0
       publish_frequency: 2.0
       global_frame: odom
       robot_base_frame: base_link
       use_sim_time: true
       rolling_window: true
       width: 6
       height: 6
       resolution: 0.05
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
         inflation_radius: 0.4

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
         backup_dist: -0.15
         backup_speed: 0.025
         sim_time: 2.0
         trans_stopped_velocity: 0.0001
         rotational_stopped_velocity: 0.0001
       wait:
         plugin: "nav2_recoveries/Wait"
         wait_duration: 1.0
   ```

3. **Create Navigation Test Script**:
   ```python
   # Create navigation_test.py
   import rclpy
   from rclpy.node import Node
   from rclpy.action import ActionClient
   from nav2_msgs.action import NavigateToPose
   from geometry_msgs.msg import PoseStamped
   import time

   class NavigationTestNode(Node):
       def __init__(self):
           super().__init__('navigation_test_node')

           # Create action client for navigation
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

           # Timer to send navigation goals
           self.nav_timer = self.create_timer(10.0, self.send_navigation_goal)
           self.goal_count = 0

       def send_navigation_goal(self):
           """Send a navigation goal to the robot"""
           goal_msg = NavigateToPose.Goal()

           # Set goal pose
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
           goal_msg.pose.pose.position.x = float(self.goal_count % 5)  # Cycle through x positions
           goal_msg.pose.pose.position.y = float((self.goal_count // 5) % 5)  # Cycle through y positions
           goal_msg.pose.pose.orientation.w = 1.0  # No rotation

           self.get_logger().info(f'Sending navigation goal: ({goal_msg.pose.pose.position.x}, {goal_msg.pose.pose.position.y})')

           # Wait for action server
           if not self.nav_client.wait_for_server(timeout_sec=5.0):
               self.get_logger().error('Navigation action server not available')
               return

           # Send goal
           future = self.nav_client.send_goal_async(goal_msg)
           future.add_done_callback(self.goal_response_callback)

           self.goal_count += 1

       def goal_response_callback(self, future):
           """Handle navigation goal response"""
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Goal rejected')
               return

           self.get_logger().info('Goal accepted')
           result_future = goal_handle.get_result_async()
           result_future.add_done_callback(self.result_callback)

       def result_callback(self, future):
           """Handle navigation result"""
           result = future.result().result
           self.get_logger().info(f'Navigation result: {result}')

   def main(args=None):
       rclpy.init(args=args)
       node = NavigationTestNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           pass
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Launch Navigation System**:
   ```bash
   # Create navigation launch file: navigation_launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')
       params_file = LaunchConfiguration('params_file')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation clock if true'
           ),
           DeclareLaunchArgument(
               'params_file',
               default_value=PathJoinSubstitution([
                   FindPackageShare('my_robot_bringup'),
                   'config',
                   'nav2_params_isaac.yaml'
               ]),
               description='Full path to the ROS2 parameters file to use for all launched nodes'
           ),

           # Localization (AMCL)
           Node(
               package='nav2_amcl',
               executable='amcl',
               name='amcl',
               parameters=[params_file, {'use_sim_time': use_sim_time}],
               remappings=[('scan', 'scan')],
               output='screen'
           ),

           # Map server
           Node(
               package='nav2_map_server',
               executable='map_server',
               name='map_server',
               parameters=[params_file, {'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Planner server
           Node(
               package='nav2_planner',
               executable='planner_server',
               name='planner_server',
               parameters=[params_file, {'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Controller server
           Node(
               package='nav2_controller',
               executable='controller_server',
               name='controller_server',
               parameters=[params_file, {'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Behavior tree navigator
           Node(
               package='nav2_bt_navigator',
               executable='bt_navigator',
               name='bt_navigator',
               parameters=[params_file, {'use_sim_time': use_sim_time}],
               output='screen'
           ),

           # Lifecycle manager
           Node(
               package='nav2_lifecycle_manager',
               executable='lifecycle_manager',
               name='lifecycle_manager_navigation',
               parameters=[{'use_sim_time': use_sim_time},
                          {'autostart': True},
                          {'node_names': ['map_server', 'amcl', 'planner_server',
                                        'controller_server', 'bt_navigator']}],
           )
       ])
   ```

5. **Run Navigation Test**:
   ```bash
   # Terminal 1: Launch Isaac Sim with robot
   cd ~/isaac-sim
   ./isaac-sim-gui.sh

   # Terminal 2: Launch navigation system
   ros2 launch navigation_launch.py params_file:=/path/to/nav2_params_isaac.yaml

   # Terminal 3: Run navigation test
   python3 navigation_test.py
   ```

### Expected Output
- Navigation system initializes successfully
- Robot plans and executes paths to goals
- Costmaps update correctly with obstacles
- Recovery behaviors trigger when needed

### Troubleshooting
- Verify map server has a valid map
- Check costmap configuration for humanoid size
- Ensure proper TF tree with all required frames
- Validate sensor data is flowing correctly

## Lab 3E: Sim-to-Real Transfer Validation

### Objective
Validate the sim-to-real transfer by comparing simulation performance with real-world data and implementing domain randomization techniques.

### Steps

1. **Create Transfer Validation Framework**:
   ```python
   # Create transfer_validation.py
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.spatial.transform import Rotation as R
   import yaml

   class TransferValidationFramework:
       def __init__(self):
           self.simulation_data = []
           self.real_robot_data = []
           self.performance_metrics = {}
           self.transfer_success_rate = 0.0

       def collect_simulation_data(self, episodes=100):
           """Collect performance data from simulation"""
           for episode in range(episodes):
               # Simulate robot performing task
               episode_data = self.run_simulation_episode()
               self.simulation_data.append(episode_data)

           return self.simulation_data

       def collect_real_robot_data(self, episodes=10):
           """Collect performance data from real robot (simplified)"""
           # In practice, this would connect to real robot
           for episode in range(episodes):
               # This would be actual robot execution
               episode_data = self.run_real_robot_episode()
               self.real_robot_data.append(episode_data)

           return self.real_robot_data

       def run_simulation_episode(self):
           """Run one episode in simulation"""
           # Simulate robot performing a navigation task
           # This would include actual robot simulation
           return {
               'success': np.random.choice([True, False], p=[0.9, 0.1]),  # Simulated success rate
               'time': np.random.normal(10.0, 2.0),  # Time to complete task
               'path_efficiency': np.random.normal(0.8, 0.1),  # Path efficiency
               'collisions': np.random.poisson(0.5),  # Number of collisions
               'energy': np.random.normal(50.0, 10.0)  # Energy consumption
           }

       def run_real_robot_episode(self):
           """Run one episode with real robot (placeholder)"""
           # In practice, this would control real robot
           return {
               'success': np.random.choice([True, False], p=[0.7, 0.3]),  # Lower success rate
               'time': np.random.normal(12.0, 3.0),  # Longer time
               'path_efficiency': np.random.normal(0.7, 0.15),  # Lower efficiency
               'collisions': np.random.poisson(1.2),  # More collisions
               'energy': np.random.normal(60.0, 15.0)  # Higher energy
           }

       def calculate_transfer_metrics(self):
           """Calculate sim-to-real transfer metrics"""
           if not self.simulation_data or not self.real_robot_data:
               return {}

           # Calculate performance differences
           sim_success_rate = np.mean([d['success'] for d in self.simulation_data])
           real_success_rate = np.mean([d['success'] for d in self.real_robot_data])

           sim_avg_time = np.mean([d['time'] for d in self.simulation_data])
           real_avg_time = np.mean([d['time'] for d in self.real_robot_data])

           sim_path_efficiency = np.mean([d['path_efficiency'] for d in self.simulation_data])
           real_path_efficiency = np.mean([d['path_efficiency'] for d in self.real_robot_data])

           # Calculate transfer gap
           success_gap = real_success_rate - sim_success_rate
           time_gap = real_avg_time - sim_avg_time
           efficiency_gap = real_path_efficiency - sim_path_efficiency

           metrics = {
               'success_rate': {
                   'simulation': sim_success_rate,
                   'real': real_success_rate,
                   'gap': success_gap
               },
               'avg_time': {
                   'simulation': sim_avg_time,
                   'real': real_avg_time,
                   'gap': time_gap
               },
               'path_efficiency': {
                   'simulation': sim_path_efficiency,
                   'real': real_path_efficiency,
                   'gap': efficiency_gap
               },
               'transfer_success_rate': real_success_rate / sim_success_rate if sim_success_rate > 0 else 0
           }

           return metrics

       def visualize_transfer_comparison(self):
           """Create visualizations comparing sim vs real performance"""
           metrics = self.calculate_transfer_metrics()

           if not metrics:
               print("No data to visualize")
               return

           fig, axes = plt.subplots(2, 2, figsize=(12, 10))

           # Success rate comparison
           ax1 = axes[0, 0]
           categories = ['Simulation', 'Real Robot']
           values = [metrics['success_rate']['simulation'], metrics['success_rate']['real']]
           ax1.bar(categories, values, color=['blue', 'red'])
           ax1.set_title('Success Rate Comparison')
           ax1.set_ylabel('Success Rate')
           ax1.set_ylim(0, 1)

           # Time comparison
           ax2 = axes[0, 1]
           values = [metrics['avg_time']['simulation'], metrics['avg_time']['real']]
           ax2.bar(categories, values, color=['blue', 'red'])
           ax2.set_title('Average Time Comparison')
           ax2.set_ylabel('Time (seconds)')

           # Path efficiency comparison
           ax3 = axes[1, 0]
           values = [metrics['path_efficiency']['simulation'], metrics['path_efficiency']['real']]
           ax3.bar(categories, values, color=['blue', 'red'])
           ax3.set_title('Path Efficiency Comparison')
           ax3.set_ylabel('Efficiency')
           ax3.set_ylim(0, 1)

           # Transfer gap visualization
           ax4 = axes[1, 1]
           gaps = [metrics['success_rate']['gap'], metrics['avg_time']['gap'], metrics['path_efficiency']['gap']]
           gap_labels = ['Success Gap', 'Time Gap', 'Efficiency Gap']
           colors = ['green' if g >= 0 else 'red' for g in gaps]
           ax4.bar(gap_labels, gaps, color=colors)
           ax4.set_title('Sim-to-Real Gaps')
           ax4.set_ylabel('Gap Value')
           ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

           plt.tight_layout()
           plt.savefig('transfer_comparison.png')
           plt.show()

       def implement_domain_randomization(self):
           """Implement domain randomization to improve transfer"""
           # Define randomization ranges for simulation
           randomization_params = {
               'lighting': {
                   'intensity_range': [0.5, 1.5],  # 50% to 150% of normal
                   'temperature_range': [3000, 6500],  # Color temperature
                   'direction_variance': 0.1  # Radians
               },
               'physics': {
                   'friction_range': [0.7, 1.3],  # 70% to 130% of normal
                   'restitution_range': [0.8, 1.2],
                   'mass_variance': 0.1  # ±10% mass variation
               },
               'sensors': {
                   'noise_multiplier': [0.8, 1.2],
                   'bias_range': [-0.01, 0.01],
                   'delay_range': [0.0, 0.05]  # 0-50ms delay
               }
           }

           return randomization_params

       def adapt_control_for_real_world(self):
           """Adapt control strategies for real-world deployment"""
           # Implement adaptive control techniques
           adaptation_strategies = {
               'gain_scheduling': {
                   'enabled': True,
                   'conditions': ['terrain_type', 'load', 'environmental_conditions']
               },
               'robust_control': {
                   'enabled': True,
                   'method': 'H-infinity',
                   'uncertainty_bounds': [0.1, 0.2]  # 10-20% uncertainty
               },
               'adaptive_control': {
                   'enabled': True,
                   'algorithm': 'model_reference_adaptive_control',
                   'learning_rate': 0.01
               }
           }

           return adaptation_strategies
   ```

2. **Run Transfer Validation Test**:
   ```python
   # Create transfer_test_script.py
   from transfer_validation import TransferValidationFramework

   def main():
       # Initialize validation framework
       validator = TransferValidationFramework()

       print("Collecting simulation data...")
       sim_data = validator.collect_simulation_data(episodes=50)

       print("Collecting real robot data...")
       real_data = validator.collect_real_robot_data(episodes=10)

       print("Calculating transfer metrics...")
       metrics = validator.calculate_transfer_metrics()

       print("\nTransfer Validation Results:")
       print(f"Success Rate - Sim: {metrics['success_rate']['simulation']:.3f}, Real: {metrics['success_rate']['real']:.3f}")
       print(f"Average Time - Sim: {metrics['avg_time']['simulation']:.3f}s, Real: {metrics['avg_time']['real']:.3f}s")
       print(f"Path Efficiency - Sim: {metrics['path_efficiency']['simulation']:.3f}, Real: {metrics['path_efficiency']['real']:.3f}")
       print(f"Transfer Success Rate: {metrics['transfer_success_rate']:.3f}")

       print("\nImplementing domain randomization...")
       dr_params = validator.implement_domain_randomization()
       print(f"Randomization parameters: {dr_params}")

       print("\nAdapting control for real world...")
       adaptation_strategies = validator.adapt_control_for_real_world()
       print(f"Adaptation strategies: {adaptation_strategies}")

       print("\nGenerating comparison visualization...")
       validator.visualize_transfer_comparison()

       print("\nTransfer validation complete!")

   if __name__ == "__main__":
       main()
   ```

3. **Execute Transfer Validation**:
   ```bash
   # Run the transfer validation
   python3 transfer_test_script.py
   ```

### Expected Output
- Performance metrics comparing simulation vs real performance
- Visualizations showing the sim-to-real gap
- Domain randomization parameters for improved transfer
- Adaptation strategies for real-world deployment

### Troubleshooting
- If real robot data is unavailable, use simulated "real" data with added noise
- Ensure proper baselines are established for fair comparison
- Validate that metrics are meaningful for the specific task
- Consider multiple runs to account for variance

## Lab 3F: Hardware Integration and Optimization

### Objective
Optimize the AI-Robot brain implementation for specific hardware targets and validate performance across different platforms.

### Steps

1. **Hardware Profiling Script**:
   ```python
   # Create hardware_profiler.py
   import psutil
   import GPUtil
   import time
   import subprocess
   import json
   from pathlib import Path

   class HardwareProfiler:
       def __init__(self):
           self.profiler_data = {}
           self.monitoring = False
           self.start_time = None

       def get_system_info(self):
           """Get comprehensive system information"""
           system_info = {
               'cpu': {
                   'count': psutil.cpu_count(logical=False),
                   'logical_count': psutil.cpu_count(logical=True),
                   'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                   'architecture': subprocess.check_output(['uname', '-m']).decode().strip()
               },
               'memory': {
                   'total': psutil.virtual_memory().total,
                   'available': psutil.virtual_memory().available,
                   'percent_used': psutil.virtual_memory().percent
               },
               'gpus': []
           }

           # Get GPU information
           gpus = GPUtil.getGPUs()
           for gpu in gpus:
               system_info['gpus'].append({
                   'id': gpu.id,
                   'name': gpu.name,
                   'memory_total': gpu.memoryTotal,
                   'memory_used': gpu.memoryUsed,
                   'memory_percent': gpu.memoryUtil * 100,
                   'utilization': gpu.load * 100
               })

           return system_info

       def start_monitoring(self):
           """Start monitoring system resources"""
           self.monitoring = True
           self.start_time = time.time()
           self.monitoring_data = {
               'timestamps': [],
               'cpu_percent': [],
               'memory_percent': [],
               'gpu_utilization': [],
               'gpu_memory': [],
               'process_count': []
           }

       def capture_monitoring_data(self):
           """Capture current system resource usage"""
           if not self.monitoring:
               return

           current_time = time.time() - self.start_time
           self.monitoring_data['timestamps'].append(current_time)

           # CPU usage
           self.monitoring_data['cpu_percent'].append(psutil.cpu_percent(interval=1))

           # Memory usage
           self.monitoring_data['memory_percent'].append(psutil.virtual_memory().percent)

           # GPU usage
           gpus = GPUtil.getGPUs()
           if gpus:
               gpu = gpus[0]  # Use first GPU
               self.monitoring_data['gpu_utilization'].append(gpu.load * 100)
               self.monitoring_data['gpu_memory'].append(gpu.memoryUtil * 100)
           else:
               self.monitoring_data['gpu_utilization'].append(0)
               self.monitoring_data['gpu_memory'].append(0)

           # Process count
           self.monitoring_data['process_count'].append(len(list(psutil.process_iter())))

       def stop_monitoring(self):
           """Stop monitoring and return data"""
           self.monitoring = False
           return self.monitoring_data

       def profile_ros_nodes(self, node_names):
           """Profile specific ROS nodes"""
           profile_data = {}
           for node_name in node_names:
               # Get process information for ROS node
               try:
                   # Find process by name (simplified)
                   for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                       if node_name in proc.info['name']:
                           profile_data[node_name] = {
                               'pid': proc.info['pid'],
                               'cpu_percent': proc.info['cpu_percent'],
                               'memory_percent': proc.info['memory_percent']
                           }
                           break
               except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                   continue

           return profile_data

       def generate_optimization_report(self, monitoring_data):
           """Generate optimization recommendations"""
           report = {
               'system_summary': self.get_system_info(),
               'performance_analysis': self.analyze_performance(monitoring_data),
               'optimization_recommendations': self.get_recommendations(monitoring_data)
           }

           return report

       def analyze_performance(self, data):
           """Analyze performance data"""
           if not data['cpu_percent']:
               return {}

           analysis = {
               'average_cpu': sum(data['cpu_percent']) / len(data['cpu_percent']),
               'peak_cpu': max(data['cpu_percent']),
               'average_memory': sum(data['memory_percent']) / len(data['memory_percent']),
               'peak_memory': max(data['memory_percent']),
               'average_gpu_util': sum(data['gpu_utilization']) / len(data['gpu_utilization']) if data['gpu_utilization'] else 0,
               'average_gpu_memory': sum(data['gpu_memory']) / len(data['gpu_memory']) if data['gpu_memory'] else 0,
               'total_runtime': max(data['timestamps']) if data['timestamps'] else 0
           }

           return analysis

       def get_recommendations(self, data):
           """Get optimization recommendations based on profiling data"""
           recommendations = []

           avg_cpu = sum(data['cpu_percent']) / len(data['cpu_percent']) if data['cpu_percent'] else 0
           avg_gpu = sum(data['gpu_utilization']) / len(data['gpu_utilization']) if data['gpu_utilization'] else 0

           if avg_cpu > 80:
               recommendations.append("High CPU usage detected - consider optimizing algorithms or using multithreading")

           if avg_gpu > 85:
               recommendations.append("High GPU usage detected - consider model optimization or batch size reduction")

           if max(data['memory_percent']) > 90:
               recommendations.append("High memory usage detected - consider memory optimization techniques")

           return recommendations
   ```

2. **Create Optimization Test Script**:
   ```python
   # Create optimization_test.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from std_msgs.msg import String
   import time
   import threading
   from hardware_profiler import HardwareProfiler

   class OptimizationTestNode(Node):
       def __init__(self):
           super().__init__('optimization_test_node')

           # Initialize profiler
           self.profiler = HardwareProfiler()
           self.profiler.start_monitoring()

           # Create publisher and subscriber for testing
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.optimized_image_callback, 1
           )
           self.status_pub = self.create_publisher(String, '/optimization_status', 10)

           # Performance counters
           self.message_count = 0
           self.start_time = time.time()

           # Start monitoring thread
           self.monitoring_thread = threading.Thread(target=self.continuous_monitoring)
           self.monitoring_thread.daemon = True
           self.monitoring_thread.start()

       def optimized_image_callback(self, msg):
           """Optimized image processing callback"""
           # Simulate optimized processing
           # In real implementation, this would include:
           # - Efficient data structures
           # - GPU acceleration where possible
           # - Memory pooling
           # - Batch processing

           self.message_count += 1

           # Log performance periodically
           if self.message_count % 100 == 0:
               elapsed_time = time.time() - self.start_time
               rate = self.message_count / elapsed_time if elapsed_time > 0 else 0

               status_msg = String()
               status_msg.data = f'Processed {self.message_count} messages at {rate:.2f} Hz'
               self.status_pub.publish(status_msg)

               self.get_logger().info(f'Performance: {rate:.2f} Hz')

       def continuous_monitoring(self):
           """Continuously monitor system resources"""
           while rclpy.ok():
               self.profiler.capture_monitoring_data()
               time.sleep(0.1)  # Monitor every 100ms

       def finalize_profiling(self):
           """Finalize profiling and generate report"""
           monitoring_data = self.profiler.stop_monitoring()
           report = self.profiler.generate_optimization_report(monitoring_data)

           # Save report to file
           with open('optimization_report.json', 'w') as f:
               json.dump(report, f, indent=2)

           self.get_logger().info('Optimization report saved to optimization_report.json')
           return report

   def main(args=None):
       rclpy.init(args=args)
       node = OptimizationTestNode()

       try:
           # Run for 60 seconds to collect data
           start_time = time.time()
           while time.time() - start_time < 60:
               rclpy.spin_once(node, timeout_sec=0.1)
       except KeyboardInterrupt:
           pass
       finally:
           report = node.finalize_profiling()
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Run Hardware Optimization Test**:
   ```bash
   # Run the optimization test
   python3 optimization_test.py
   ```

4. **Create Hardware-Specific Configurations**:
   ```yaml
   # Create hardware_configs.yaml
   hardware_configurations:
     desktop_workstation:
       name: "High-End Desktop"
       cpu: "Intel i9-12900K or AMD Ryzen 9 5900X"
       gpu: "RTX 4090 or RTX 6000 Ada"
       ram: "64GB+ DDR4-3600"
       storage: "2TB+ NVMe Gen 4 SSD"
       optimization:
         use_gpu: true
         batch_size: 32
         precision: "fp16"
         multi_gpu: true
         tensor_cores: true

     edge_jetson:
       name: "NVIDIA Jetson AGX Orin"
       cpu: "ARM Cortex-A78AE Octa-core"
       gpu: "512-core Ada GPU"
       ram: "32GB LPDDR5"
       storage: "1TB NVMe SSD"
       optimization:
         use_gpu: true
         batch_size: 8
         precision: "int8"
         multi_gpu: false
         tensor_cores: true
         power_limit: "40W"
         thermal_management: "active"

     cloud_v100:
       name: "Cloud V100 Instance"
       cpu: "High-core count Intel/AMD"
       gpu: "Tesla V100 32GB"
       ram: "128GB+"
       storage: "High-performance cloud storage"
       optimization:
         use_gpu: true
         batch_size: 64
         precision: "fp16"
         multi_gpu: true
         tensor_cores: true
         distributed: true

     simulation_only:
       name: "Simulation-Only Laptop"
       cpu: "Intel i7 or AMD Ryzen 7"
       gpu: "RTX 3070 or equivalent"
       ram: "32GB DDR4-3200"
       storage: "1TB NVMe SSD"
       optimization:
         use_gpu: true
         batch_size: 16
         precision: "fp32"
         multi_gpu: false
         tensor_cores: false
         sim_priority: true
   ```

### Expected Output
- Comprehensive hardware profiling data
- Performance metrics for different components
- Optimization recommendations
- Hardware-specific configuration profiles

### Troubleshooting
- Ensure proper permissions for system monitoring
- Verify GPU monitoring tools are installed
- Check for sufficient system resources during profiling
- Validate that monitoring doesn't interfere with normal operation

## Lab Summary

In these labs, you've gained hands-on experience with:

1. **Isaac Sim Setup**: Learned to configure and run Isaac Sim for humanoid robotics
2. **Isaac ROS Integration**: Connected simulation with ROS 2 for bidirectional communication
3. **VSLAM Implementation**: Set up visual SLAM systems for mapping and localization
4. **Navigation Stack**: Configured and tested the Nav2 stack for humanoid navigation
5. **Sim-to-Real Transfer**: Validated the transfer from simulation to reality
6. **Hardware Optimization**: Profiled and optimized systems for different hardware targets

These skills form the foundation for implementing AI-robot brains that can operate effectively in both simulation and real-world environments. Each lab builds on the previous one to give you a comprehensive understanding of the challenges and solutions in humanoid robotics AI systems.

## Next Steps

With a solid understanding of AI-robot brain implementation, continue to [Module 3 Troubleshooting](./troubleshooting.md) to learn how to diagnose and fix common issues that arise when implementing AI systems in humanoid robotics applications.