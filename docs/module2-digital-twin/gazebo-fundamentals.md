---
title: Gazebo Fundamentals
sidebar_position: 3
---

# Gazebo Fundamentals

Gazebo is a powerful 3D simulation environment that plays a crucial role in robotics development. This lesson covers the fundamental concepts of Gazebo and how to use it effectively for humanoid robotics simulation.

## Introduction to Gazebo

Gazebo is a physics-based simulation environment that provides:
- **Realistic physics simulation** using Open Dynamics Engine (ODE), Bullet, or DART
- **High-quality 3D rendering** using OGRE
- **Sensors simulation** including cameras, LiDAR, IMU, GPS, and more
- **Robust plugin system** for extending functionality
- **ROS/ROS 2 integration** for seamless development workflows

### Key Features

1. **Physics Simulation**: Accurate modeling of rigid body dynamics, collisions, and contacts
2. **Sensor Simulation**: Realistic simulation of various robot sensors
3. **Environment Modeling**: Creation of complex indoor and outdoor environments
4. **Real-time Visualization**: Interactive 3D visualization of the simulation
5. **Plugin Architecture**: Extensible system for custom functionality
6. **ROS Integration**: Native support for ROS/ROS 2 communication

## Gazebo Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Client    │    │  Gazebo Server  │    │  Physics Engine │
│                 │    │                 │    │                 │
│ • Visualization │    │ • World Loading │    │ • Collision     │
│ • Controls      │    │ • Physics Step  │    │ • Dynamics      │
│ • Scene Editing │    │ • Sensor Update │    │ • Constraints   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │     Plugin System         │
                    │                           │
                    │ • Sensor Plugins          │
                    │ • Controller Plugins      │
                    │ • GUI Plugins             │
                    └───────────────────────────┘
```

### Simulation Loop

Gazebo operates on a fixed time step simulation loop:

1. **Physics Update**: Update positions and velocities based on forces
2. **Collision Detection**: Detect and resolve collisions
3. **Sensor Update**: Update sensor readings based on world state
4. **Plugin Update**: Execute custom plugin logic
5. **Visualization Update**: Update the 3D visualization

## Installing and Running Gazebo

### Installation

For ROS 2 Humble with Gazebo Garden:

```bash
# Install Gazebo Garden
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-dev

# Or install standalone Gazebo Garden
sudo apt install gz-garden
```

### Running Gazebo

```bash
# Launch Gazebo with GUI
gz sim -v 4

# Launch without GUI (headless)
gz sim -s -r

# Launch with a specific world
gz sim -r -v 4 empty.sdf
```

## World Files

Gazebo uses SDF (Simulation Description Format) to define simulation worlds. SDF is an XML-based format that describes:

- **World**: The complete simulation environment
- **Models**: Robots, objects, and static entities
- **Lights**: Lighting conditions
- **Physics**: Physics engine parameters
- **GUI**: Visualization settings

### Basic World File Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Environment lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
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
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Example robot model -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

## Models in Gazebo

### Model Structure

Gazebo models are defined in SDF format and typically include:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <!-- Model pose -->
    <pose>0 0 0.5 0 0 0</pose>

    <!-- Links define rigid bodies -->
    <link name="base_link">
      <!-- Inertial properties -->
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

      <!-- Visual properties -->
      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>

      <!-- Collision properties -->
      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.5</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Joints connect links -->
    <joint name="joint1" type="revolute">
      <parent>base_link</parent>
      <child>upper_link</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>

    <link name="upper_link">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.05</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.05</iyy>
          <iyz>0</iyz>
          <izz>0.05</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

### Model Directory Structure

```
~/.gazebo/models/my_robot/
├── model.config          # Model metadata
├── model.sdf            # Model definition
├── meshes/              # 3D mesh files
│   ├── link1.dae
│   └── link2.stl
└── materials/
    └── textures/
        └── texture.png
```

### model.config Example

```xml
<?xml version="1.0"?>
<model>
  <name>My Robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>

  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>

  <description>
    A sample robot model for simulation.
  </description>
</model>
```

## Sensors in Gazebo

Gazebo provides realistic simulation of various sensors commonly used in robotics:

### Camera Sensors

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LiDAR Sensors

```xml
<sensor name="lidar" type="ray">
  <pose>0.15 0 0.1 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>
        <max_angle>1.570796</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensors

```xml
<sensor name="imu" type="imu">
  <pose>0 0 0.1 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Gazebo with ROS 2

Gazebo integrates seamlessly with ROS 2 through the `gazebo_ros_pkgs`:

### Launching Gazebo with ROS 2

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('my_robot_gazebo'),
                    'worlds',
                    'my_world.sdf'
                ])
            }.items()
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': True}
            ]
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'my_robot',
                '-x', '0',
                '-y', '0',
                '-z', '1'
            ],
            output='screen'
        )
    ])
```

### Common Gazebo-ROS Interfaces

Gazebo uses standard ROS 2 message types:

- **Joint States**: `/joint_states` (sensor_msgs/JointState)
- **Robot Commands**: `/joint_commands` (trajectory_msgs/JointTrajectory)
- **Camera Data**: `/camera/image_raw` (sensor_msgs/Image)
- **LiDAR Data**: `/scan` (sensor_msgs/LaserScan)
- **IMU Data**: `/imu` (sensor_msgs/Imu)
- **Odometry**: `/odom` (nav_msgs/Odometry)

## Physics Configuration

### Physics Engine Options

Gazebo supports multiple physics engines:

1. **ODE (Open Dynamics Engine)**: Default, good balance of speed and accuracy
2. **Bullet**: Good for complex collisions
3. **DART**: Advanced physics simulation

### Physics Parameters

```xml
<physics type="ode">
  <!-- Time step configuration -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Solver parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Plugins System

Gazebo's plugin system allows extending functionality:

### Common Plugin Types

1. **Model Plugins**: Attach to models to control behavior
2. **Sensor Plugins**: Process sensor data
3. **World Plugins**: Control world behavior
4. **GUI Plugins**: Extend visualization interface

### Example Model Plugin

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class JointControlPlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&JointControlPlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply a small linear velocity to the model
      this->model->SetLinearVel(math::Vector3(0.01, 0, 0));
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(JointControlPlugin)
}
```

## Humanoid Robotics Considerations

### Balance Simulation

For humanoid robots, physics accuracy is crucial:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Small step for stability -->
  <real_time_factor>1</real_time_factor>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>  <!-- More iterations for stability -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>  <!-- Constraint force mixing -->
      <erp>0.2</erp>       <!-- Error reduction parameter -->
    </constraints>
  </ode>
</physics>
```

### Complex Joint Simulation

Humanoid robots require complex joint configurations:

```xml
<!-- 6-DOF joint for pelvis (using multiple single DOF joints) -->
<joint name="pelvis_tx" type="prismatic">
  <parent>world</parent>
  <child>pelvis</child>
  <axis>
    <xyz>1 0 0</xyz>
  </axis>
</joint>

<joint name="pelvis_tz" type="prismatic">
  <parent>pelvis</parent>
  <child>pelvis_z</child>
  <axis>
    <xyz>0 0 1</xyz>
  </axis>
</joint>

<joint name="pelvis_ry" type="revolute">
  <parent>pelvis_z</parent>
  <child>pelvis_ry_link</child>
  <axis>
    <xyz>0 1 0</xyz>
  </axis>
</joint>
```

## Performance Optimization

### Rendering Optimization

```bash
# Reduce rendering quality for better performance
gz sim -v 0  # Minimal rendering
gz sim -v 4  # Full rendering
```

### Physics Optimization

- **Reduce update rate**: Lower `real_time_update_rate` for less CPU usage
- **Increase step size**: Larger `max_step_size` (less accurate but faster)
- **Simplify models**: Use simpler collision geometry
- **Limit simulation steps**: Use fixed step count for testing

## Best Practices

### 1. Model Design

- Use appropriate collision geometry (simpler than visual)
- Set realistic inertial properties
- Use proper joint limits
- Include sensor noise models

### 2. World Design

- Start simple and add complexity gradually
- Use appropriate physics parameters
- Include realistic lighting and textures
- Consider computational requirements

### 3. Integration with ROS 2

- Use standard message types
- Implement proper error handling
- Consider real-time requirements
- Test both simulation and real hardware code paths

### 4. Validation

- Compare simulation results with real robot data
- Validate sensor models against real sensors
- Test control algorithms in both environments
- Document simulation assumptions and limitations

## Troubleshooting Common Issues

### Physics Issues

- **Robot falls through ground**: Check collision geometry and physics parameters
- **Unstable simulation**: Reduce time step or increase solver iterations
- **Joints not moving properly**: Check joint limits and actuator parameters

### Rendering Issues

- **Slow rendering**: Reduce visual complexity or rendering quality
- **Missing textures**: Check model file paths and permissions
- **Lighting problems**: Adjust light properties in world file

### ROS Integration Issues

- **No communication**: Verify ROS domain settings and network configuration
- **Topic mismatches**: Check topic names and message types
- **Timing issues**: Use `use_sim_time` parameter consistently

## Next Steps

With a solid understanding of Gazebo fundamentals, continue to [Worlds & Physics Settings](./worlds-physics.md) to learn how to create and configure simulation environments specifically for humanoid robotics applications.