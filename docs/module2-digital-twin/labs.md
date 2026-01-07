---
title: Module 2 Labs
sidebar_position: 8
---

# Module 2 Labs: Digital Twin and Simulation for Humanoid Robotics

This lab section provides hands-on exercises to reinforce the simulation concepts you've learned in Module 2. Each lab builds on the previous one to give you practical experience with Gazebo, Unity, and other simulation tools in the context of humanoid robotics.

## Lab 2A: Setting Up Gazebo Simulation Environment

### Objective
Set up a basic Gazebo simulation environment and run your first simulation with a simple robot model.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Gazebo Garden installed
- Basic command line skills
- Understanding of URDF/SDF concepts

### Steps

1. **Verify Gazebo Installation**:
   ```bash
   # Check if Gazebo is installed
   gz --version

   # Check if Gazebo ROS packages are installed
   ros2 pkg list | grep gazebo
   ```

2. **Launch Basic Gazebo Environment**:
   ```bash
   # Launch Gazebo with empty world
   gz sim -v 4

   # Or launch with GUI
   gz sim -g -v 4
   ```

3. **Create a Simple World File**:
   Create `~/ros2_ws/src/my_robot_gazebo/worlds/simple_world.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="simple_world">
       <!-- Physics -->
       <physics name="1ms" type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
         <gravity>0 0 -9.8</gravity>
       </physics>

       <!-- Sun light -->
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <direction>-0.4 0.2 -0.9</direction>
       </light>

       <!-- Ground plane -->
       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
             <material>
               <ambient>0.7 0.7 0.7 1</ambient>
               <diffuse>0.7 0.7 0.7 1</diffuse>
               <specular>0.0 0.0 0.0 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <!-- Simple box obstacle -->
       <model name="box_obstacle">
         <pose>2 0 0.5 0 0 0</pose>
         <link name="box_link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>1 1 1</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>1 1 1</size>
             </box>
             </geometry>
             <material>
               <ambient>0.8 0.3 0.3 1</ambient>
               <diffuse>0.8 0.3 0.3 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>1.0</mass>
             <inertia>
               <ixx>0.1</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.1</iyy>
               <iyz>0</iyz>
               <izz>0.1</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

4. **Launch Gazebo with Your World**:
   ```bash
   # Launch Gazebo with your custom world
   gz sim -r -v 4 ~/ros2_ws/src/my_robot_gazebo/worlds/simple_world.sdf
   ```

5. **Interact with the Simulation**:
   - Use the GUI controls to navigate the 3D view
   - Add models from the model database
   - Test physics by adding objects

### Expected Output
- Gazebo GUI launches successfully
- Custom world with ground plane and box obstacle loads
- Physics simulation runs smoothly
- Objects respond to physics laws

### Troubleshooting
- If Gazebo fails to launch, check installation and graphics drivers
- If world doesn't load, verify SDF file syntax
- If physics are unstable, adjust time step parameters

## Lab 2B: Creating a Simple Robot Model in Gazebo

### Objective
Create a simple robot model with joints and visualize it in Gazebo simulation.

### Steps

1. **Create a Simple Robot URDF**:
   Create `~/ros2_ws/src/my_robot_description/urdf/simple_robot.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Upper body -->
     <link name="upper_body">
       <visual>
         <geometry>
           <cylinder radius="0.15" length="0.3"/>
         </geometry>
         <material name="gray">
           <color rgba="0.5 0.5 0.5 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.15" length="0.3"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.5"/>
         <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
       </inertial>
     </link>

     <!-- Joint connecting base to upper body -->
     <joint name="base_to_upper" type="revolute">
       <parent link="base_link"/>
       <child link="upper_body"/>
       <origin xyz="0 0 0.25" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
     </joint>

     <!-- Simple wheel -->
     <link name="wheel">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.2"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <!-- Joint connecting wheel to base -->
     <joint name="wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="wheel"/>
       <origin xyz="0.2 0 -0.15" rpy="1.57 0 0"/>
       <axis xyz="0 0 1"/>
     </joint>
   </robot>
   ```

2. **Convert URDF to SDF**:
   ```bash
   # Convert URDF to SDF
   gz sdf -p ~/ros2_ws/src/my_robot_description/urdf/simple_robot.urdf > ~/ros2_ws/src/my_robot_gazebo/models/simple_robot/model.sdf
   ```

3. **Create Model Configuration**:
   Create directory and file `~/ros2_ws/src/my_robot_gazebo/models/simple_robot/model.config`:
   ```xml
   <?xml version="1.0"?>
   <model>
     <name>Simple Robot</name>
     <version>1.0</version>
     <sdf version="1.7">model.sdf</sdf>

     <author>
       <name>Your Name</name>
       <email>your.email@example.com</email>
     </author>

     <description>
       A simple robot model for simulation.
     </description>
   </model>
   ```

4. **Create a World with Your Robot**:
   Create `~/ros2_ws/src/my_robot_gazebo/worlds/robot_world.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="robot_world">
       <!-- Physics -->
       <physics name="1ms" type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
         <gravity>0 0 -9.8</gravity>
       </physics>

       <!-- Lighting -->
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <direction>-0.4 0.2 -0.9</direction>
       </light>

       <!-- Ground plane -->
       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>100 100</size>
               </plane>
             </geometry>
             <material>
               <ambient>0.7 0.7 0.7 1</ambient>
               <diffuse>0.7 0.7 0.7 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <!-- Include your robot -->
       <include>
         <uri>model://simple_robot</uri>
         <pose>0 0 1 0 0 0</pose>
       </include>
     </world>
   </sdf>
   ```

5. **Launch and Test**:
   ```bash
   # Launch with your robot world
   gz sim -r -v 4 ~/ros2_ws/src/my_robot_gazebo/worlds/robot_world.sdf
   ```

### Expected Output
- Robot model appears in the simulation
- Joints function correctly
- Physics simulation works properly
- Robot responds to gravity and collisions

## Lab 2C: Adding Sensors to Your Robot

### Objective
Add various sensors to your robot model and verify they publish data correctly.

### Steps

1. **Add Sensors to Your Robot URDF**:
   Update your robot URDF file to include sensors:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot_with_sensors">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder radius="0.2" length="0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.2" length="0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Camera sensor -->
     <link name="camera_link">
       <visual>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.01"/>
         <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
       </inertial>
     </link>

     <!-- Joint to attach camera -->
     <joint name="camera_joint" type="fixed">
       <parent link="base_link"/>
       <child link="camera_link"/>
       <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
     </joint>

     <!-- Add Gazebo-specific sensor definition -->
     <gazebo reference="camera_link">
       <sensor name="camera" type="camera">
         <pose>0 0 0 0 0 0</pose>
         <camera name="head_camera">
           <horizontal_fov>1.047</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>10</far>
           </clip>
           <noise>
             <type>gaussian</type>
             <mean>0.0</mean>
             <stddev>0.007</stddev>
           </noise>
         </camera>
         <always_on>1</always_on>
         <update_rate>30</update_rate>
         <visualize>true</visualize>
       </sensor>
     </gazebo>

     <!-- IMU sensor -->
     <link name="imu_link">
       <inertial>
         <mass value="0.01"/>
         <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
       </inertial>
     </link>

     <joint name="imu_joint" type="fixed">
       <parent link="base_link"/>
       <child link="imu_link"/>
       <origin xyz="0 0 0.1" rpy="0 0 0"/>
     </joint>

     <gazebo reference="imu_link">
       <sensor name="imu" type="imu">
         <always_on>true</always_on>
         <update_rate>100</update_rate>
         <imu>
           <angular_velocity>
             <x>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e-4</stddev>
                 <bias_mean>0.0000075</bias_mean>
                 <bias_stddev>0.0000008</bias_stddev>
               </noise>
             </x>
             <y>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e-4</stddev>
                 <bias_mean>0.0000075</bias_mean>
                 <bias_stddev>0.0000008</bias_stddev>
               </noise>
             </y>
             <z>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>2e-4</stddev>
                 <bias_mean>0.0000075</bias_mean>
                 <bias_stddev>0.0000008</bias_stddev>
               </noise>
             </z>
           </angular_velocity>
           <linear_acceleration>
             <x>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e-2</stddev>
                 <bias_mean>0.1</bias_mean>
                 <bias_stddev>0.001</bias_stddev>
               </noise>
             </x>
             <y>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e-2</stddev>
                 <bias_mean>0.1</bias_mean>
                 <bias_stddev>0.001</bias_stddev>
               </noise>
             </y>
             <z>
               <noise type="gaussian">
                 <mean>0.0</mean>
                 <stddev>1.7e-2</stddev>
                 <bias_mean>0.1</bias_mean>
                 <bias_stddev>0.001</bias_stddev>
               </noise>
             </z>
           </linear_acceleration>
         </imu>
       </sensor>
     </gazebo>

     <!-- LiDAR sensor -->
     <link name="lidar_link">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0002"/>
       </inertial>
     </link>

     <joint name="lidar_joint" type="fixed">
       <parent link="base_link"/>
       <child link="lidar_link"/>
       <origin xyz="0.15 0 0.15" rpy="0 0 0"/>
     </joint>

     <gazebo reference="lidar_link">
       <sensor name="lidar" type="ray">
         <pose>0 0 0 0 0 0</pose>
         <ray>
           <scan>
             <horizontal>
               <samples>720</samples>
               <resolution>1</resolution>
               <min_angle>-3.14159</min_angle>
               <max_angle>3.14159</max_angle>
             </horizontal>
           </scan>
           <range>
             <min>0.1</min>
             <max>30.0</max>
             <resolution>0.01</resolution>
           </range>
         </ray>
         <always_on>1</always_on>
         <update_rate>10</update_rate>
         <visualize>true</visualize>
       </sensor>
     </gazebo>
   </robot>
   ```

2. **Create a Launch File for ROS 2 Integration**:
   Create `~/ros2_ws/src/my_robot_bringup/launch/sim_robot.launch.py`:
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
   from launch.conditions import IfCondition
   from launch.launch_description_sources import PythonLaunchDescriptionSource
   from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
   from launch_ros.actions import Node
   from launch_ros.substitutions import FindPackageShare


   def generate_launch_description():
       # Launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')
       use_rviz = LaunchConfiguration('use_rviz')
       robot_name = LaunchConfiguration('robot_name')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='true',
               description='Use simulation (Gazebo) clock if true'
           ),
           DeclareLaunchArgument(
               'use_rviz',
               default_value='true',
               description='Whether to launch RViz'
           ),
           DeclareLaunchArgument(
               'robot_name',
               default_value='simple_robot',
               description='Name of the robot to spawn'
           ),

           # Launch Gazebo
           IncludeLaunchDescription(
               PythonLaunchDescriptionSource([
                   PathJoinSubstitution([
                       FindPackageShare('gazebo_ros'),
                       'launch',
                       'gazebo.launch.py'
                   ])
               ]),
           ),

           # Robot State Publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               output='screen',
               parameters=[
                   {'use_sim_time': use_sim_time},
                   {'robot_description': open(PathJoinSubstitution([
                       FindPackageShare('my_robot_description'),
                       'urdf',
                       'simple_robot_with_sensors.urdf'
                   ]).perform(None)).read()}
               ]
           ),

           # Spawn robot in Gazebo
           Node(
               package='gazebo_ros',
               executable='spawn_entity.py',
               arguments=[
                   '-topic', 'robot_description',
                   '-entity', robot_name,
                   '-x', '0',
                   '-y', '0',
                   '-z', '1'
               ],
               output='screen'
           ),

           # Launch RViz if requested
           Node(
               package='rviz2',
               executable='rviz2',
               name='rviz2',
               output='screen',
               condition=IfCondition(use_rviz),
               parameters=[{'use_sim_time': use_sim_time}]
           )
       ])
   ```

3. **Build and Launch**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_bringup my_robot_description
   source install/setup.bash

   # Launch the simulation
   ros2 launch my_robot_bringup sim_robot.launch.py
   ```

4. **Verify Sensor Data**:
   ```bash
   # Check if sensor topics are being published
   ros2 topic list | grep sensor

   # Echo camera data
   ros2 topic echo /camera/image_raw

   # Echo IMU data
   ros2 topic echo /imu/data

   # Echo LiDAR data
   ros2 topic echo /scan
   ```

### Expected Output
- Robot with sensors appears in Gazebo
- Sensor topics are published in ROS 2
- Data streams from camera, IMU, and LiDAR
- RViz shows sensor data visualization

## Lab 2D: Creating a Humanoid-Specific Simulation

### Objective
Create a more complex humanoid robot simulation with multiple joints and appropriate sensors for humanoid applications.

### Steps

1. **Create a Humanoid Robot URDF**:
   Create `~/ros2_ws/src/my_robot_description/urdf/humanoid_robot.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="humanoid_robot">
     <!-- Torso -->
     <link name="torso">
       <visual>
         <geometry>
           <cylinder radius="0.1" length="0.5"/>
         </geometry>
         <material name="gray">
           <color rgba="0.5 0.5 0.5 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.1" length="0.5"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Head -->
     <link name="head">
       <visual>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
         <material name="white">
           <color rgba="1 1 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <sphere radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <!-- Neck joint -->
     <joint name="neck_joint" type="revolute">
       <parent link="torso"/>
       <child link="head"/>
       <origin xyz="0 0 0.3" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
     </joint>

     <!-- Left Arm -->
     <link name="left_upper_arm">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="left_shoulder" type="revolute">
       <parent link="torso"/>
       <child link="left_upper_arm"/>
       <origin xyz="0.15 0.1 0.2" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
     </joint>

     <link name="left_forearm">
       <visual>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.8"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="left_elbow" type="revolute">
       <parent link="left_upper_arm"/>
       <child link="left_forearm"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="0" effort="50" velocity="2"/>
     </joint>

     <!-- Right Arm (mirrored) -->
     <link name="right_upper_arm">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.3"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.0"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="right_shoulder" type="revolute">
       <parent link="torso"/>
       <child link="right_upper_arm"/>
       <origin xyz="0.15 -0.1 0.2" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
     </joint>

     <link name="right_forearm">
       <visual>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.04" length="0.3"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.8"/>
         <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="right_elbow" type="revolute">
       <parent link="right_upper_arm"/>
       <child link="right_forearm"/>
       <origin xyz="0 0 -0.3" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.57" upper="0" effort="50" velocity="2"/>
     </joint>

     <!-- Left Leg -->
     <link name="left_thigh">
       <visual>
         <geometry>
           <cylinder radius="0.06" length="0.5"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.06" length="0.5"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="2.0"/>
         <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <joint name="left_hip" type="revolute">
       <parent link="torso"/>
       <child link="left_thigh"/>
       <origin xyz="0 -0.05 -0.25" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
     </joint>

     <link name="left_shin">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.5"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.5"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.5"/>
         <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="left_knee" type="revolute">
       <parent link="left_thigh"/>
       <child link="left_shin"/>
       <origin xyz="0 0 -0.5" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="0" upper="2.3" effort="100" velocity="2"/>
     </joint>

     <link name="left_foot">
       <visual>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.5"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <joint name="left_ankle" type="revolute">
       <parent link="left_shin"/>
       <child link="left_foot"/>
       <origin xyz="0 0 -0.5" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
     </joint>

     <!-- Right Leg (mirrored) -->
     <link name="right_thigh">
       <visual>
         <geometry>
           <cylinder radius="0.06" length="0.5"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.06" length="0.5"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="2.0"/>
         <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <joint name="right_hip" type="revolute">
       <parent link="torso"/>
       <child link="right_thigh"/>
       <origin xyz="0 0.05 -0.25" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
     </joint>

     <link name="right_shin">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.5"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.5"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.5"/>
         <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="right_knee" type="revolute">
       <parent link="right_thigh"/>
       <child link="right_shin"/>
       <origin xyz="0 0 -0.5" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="0" upper="2.3" effort="100" velocity="2"/>
     </joint>

     <link name="right_foot">
       <visual>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.2 0.1 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.5"/>
         <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
       </inertial>
     </link>

     <joint name="right_ankle" type="revolute">
       <parent link="right_shin"/>
       <child link="right_foot"/>
       <origin xyz="0 0 -0.5" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
     </joint>

     <!-- Sensors -->
     <!-- IMU in torso for balance -->
     <link name="torso_imu">
       <inertial>
         <mass value="0.01"/>
         <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
       </inertial>
     </link>

     <joint name="torso_imu_joint" type="fixed">
       <parent link="torso"/>
       <child link="torso_imu"/>
       <origin xyz="0 0 0.1" rpy="0 0 0"/>
     </joint>

     <gazebo reference="torso_imu">
       <sensor name="torso_imu" type="imu">
         <always_on>true</always_on>
         <update_rate>100</update_rate>
       </sensor>
     </gazebo>

     <!-- Camera in head for vision -->
     <link name="head_camera">
       <visual>
         <geometry>
           <box size="0.02 0.02 0.02"/>
         </geometry>
       </visual>
       <inertial>
         <mass value="0.01"/>
         <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
       </inertial>
     </link>

     <joint name="head_camera_joint" type="fixed">
       <parent link="head"/>
       <child link="head_camera"/>
       <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
     </joint>

     <gazebo reference="head_camera">
       <sensor name="head_camera" type="camera">
         <camera name="head_cam">
           <horizontal_fov>1.047</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>10</far>
           </clip>
         </camera>
         <always_on>1</always_on>
         <update_rate>30</update_rate>
         <visualize>true</visualize>
       </sensor>
     </gazebo>
   </robot>
   ```

2. **Create a Humanoid World**:
   Create `~/ros2_ws/src/my_robot_gazebo/worlds/humanoid_world.sdf`:
   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="humanoid_world">
       <!-- Physics optimized for humanoid -->
       <physics name="humanoid_physics" type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
         <gravity>0 0 -9.8</gravity>
         <ode>
           <solver>
             <type>quick</type>
             <iters>100</iters>
             <sor>1.3</sor>
           </solver>
           <constraints>
             <cfm>1e-5</cfm>
             <erp>0.2</erp>
           </constraints>
         </ode>
       </physics>

       <!-- Lighting -->
       <light name="sun" type="directional">
         <cast_shadows>true</cast_shadows>
         <pose>0 0 10 0 0 0</pose>
         <diffuse>0.8 0.8 0.8 1</diffuse>
         <specular>0.2 0.2 0.2 1</specular>
         <direction>-0.4 0.2 -0.9</direction>
       </light>

       <!-- Ground plane with appropriate friction for walking -->
       <model name="ground_plane">
         <static>true</static>
         <link name="link">
           <collision name="collision">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>20 20</size>
               </plane>
             </geometry>
             <surface>
               <friction>
                 <ode>
                   <mu>0.8</mu>
                   <mu2>0.8</mu2>
                 </ode>
               </friction>
             </surface>
           </collision>
           <visual name="visual">
             <geometry>
               <plane>
                 <normal>0 0 1</normal>
                 <size>20 20</size>
               </plane>
             </geometry>
             <material>
               <ambient>0.7 0.7 0.7 1</ambient>
               <diffuse>0.7 0.7 0.7 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <!-- Add some obstacles for navigation -->
       <model name="obstacle_1">
         <pose>2 0 0.5 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.5 0.5 1</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.5 0.5 1</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.3 0.3 1</ambient>
               <diffuse>0.8 0.3 0.3 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>10.0</mass>
             <inertia>
               <ixx>1</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>1</iyy>
               <iyz>0</iyz>
               <izz>1</izz>
             </inertia>
           </inertial>
         </link>
       </model>

       <model name="obstacle_2">
         <pose>-2 1 0.3 0 0 0</pose>
         <link name="link">
           <collision name="collision">
             <geometry>
               <cylinder>
                 <radius>0.3</radius>
                 <length>0.6</length>
               </cylinder>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <cylinder>
                 <radius>0.3</radius>
                 <length>0.6</length>
               </cylinder>
             </geometry>
             <material>
               <ambient>0.3 0.8 0.3 1</ambient>
               <diffuse>0.3 0.8 0.3 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>5.0</mass>
             <inertia>
               <ixx>0.5</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.5</iyy>
               <iyz>0</iyz>
               <izz>0.5</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

3. **Build and Test the Humanoid Simulation**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select my_robot_description my_robot_gazebo my_robot_bringup
   source install/setup.bash

   # Test the humanoid model in isolation first
   gz sim -r -v 4 ~/ros2_ws/src/my_robot_gazebo/worlds/humanoid_world.sdf
   ```

### Expected Output
- Complex humanoid robot model loads in Gazebo
- All joints function correctly
- Physics simulation handles the complex model
- Robot can be manipulated in the simulation environment

## Lab 2E: Unity Simulation Environment (Optional)

### Objective
Set up a basic Unity simulation environment for humanoid robotics (if Unity is available).

### Steps

1. **Install Unity Robotics Package** (if Unity is available):
   - Open Unity Hub
   - Install Unity Editor (2021.3 LTS or newer)
   - Create a new 3D project
   - Add Unity Robotics Package via Package Manager

2. **Create Basic Scene**:
   ```csharp
   // Create a simple script to demonstrate Unity robotics concepts
   using UnityEngine;
   using Unity.Robotics.ROSTCPConnector;
   using RosMessageTypes.Sensor;

   public class UnityRobotSimulation : MonoBehaviour
   {
       [SerializeField] private string cameraTopic = "/unity_camera/image_raw";
       [SerializeField] private float updateRate = 30.0f;

       private RosConnection ros;
       private Camera unityCamera;
       private float updateInterval;
       private float lastUpdateTime;

       void Start()
       {
           ros = RosConnection.GetOrCreateInstance();
           unityCamera = GetComponent<Camera>();
           updateInterval = 1.0f / updateRate;
       }

       void Update()
       {
           if (Time.time - lastUpdateTime >= updateInterval)
           {
               PublishCameraImage();
               lastUpdateTime = Time.time;
           }
       }

       void PublishCameraImage()
       {
           // This is a simplified example
           // In a real implementation, you would capture the camera image
           // and convert it to a ROS sensor_msgs/Image message
           Debug.Log("Camera image would be published here");
       }
   }
   ```

### Expected Output
- Unity project created with robotics package
- Basic camera simulation implemented
- ROS connection established (if ROS bridge is running)

## Lab Summary

In these labs, you've learned to:
1. Set up and configure Gazebo simulation environments
2. Create robot models with proper URDF/SDF definitions
3. Add various sensors to robot models
4. Integrate with ROS 2 for complete simulation workflows
5. Create complex humanoid robot models
6. (Optional) Explore Unity simulation capabilities

These skills form the foundation for simulation-based robotics development. Continue to practice these concepts and explore how they integrate with the other modules in this course.

## Next Steps

After completing these labs, you should be comfortable with:
- Creating and configuring simulation environments
- Building robot models for simulation
- Adding sensors and validating their outputs
- Integrating simulation with ROS 2

Continue to [Module 2 Troubleshooting](./troubleshooting.md) to learn how to diagnose and fix common issues you might encounter when working with simulation environments in humanoid robotics applications.