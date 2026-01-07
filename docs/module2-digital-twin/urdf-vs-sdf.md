---
title: URDF vs SDF for Simulation
sidebar_position: 5
---

# URDF vs SDF for Simulation

Understanding the differences between URDF (Unified Robot Description Format) and SDF (Simulation Description Format) is crucial for effective robotics simulation. This lesson covers when and how to use each format, particularly in the context of humanoid robotics simulation.

## Introduction to URDF and SDF

### URDF (Unified Robot Description Format)
- **Purpose**: Describes robot structure and kinematics
- **Primary Use**: ROS-based robot description
- **Scope**: Robot model, joints, links, and basic properties
- **Integration**: Native ROS integration

### SDF (Simulation Description Format)
- **Purpose**: Describes simulation environment and robot for physics engines
- **Primary Use**: Gazebo and other simulation environments
- **Scope**: Full simulation model including physics, sensors, and environment
- **Integration**: Simulation engine integration

## Key Differences

### URDF Characteristics
```xml
<!-- URDF focuses on robot description -->
<?xml version="1.0"?>
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

### SDF Characteristics
```xml
<!-- SDF focuses on simulation aspects -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_robot">
    <pose>0 0 0.5 0 0 0</pose>

    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>

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

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 1 1</ambient>
          <diffuse>0 0 1 1</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.5 0.2</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
    </link>

    <joint name="joint1" type="revolute">
      <parent>base_link</parent>
      <child>link1</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

## When to Use URDF

### Appropriate Use Cases

#### 1. ROS-Based Robot Development
- When building robots for ROS/ROS 2 ecosystems
- For kinematic and dynamic analysis
- For robot calibration and configuration

#### 2. Robot-Agnostic Descriptions
- When describing robot structure independently of simulation
- For manufacturing and design purposes
- For multi-simulation platform compatibility

#### 3. Control System Development
- When developing controllers that don't require simulation-specific features
- For trajectory planning and inverse kinematics

### URDF Advantages
- **Simplicity**: Clean, focused syntax for robot description
- **ROS Integration**: Native support in ROS ecosystem
- **Tool Support**: Extensive tooling for URDF (RViz, MoveIt, etc.)
- **Community**: Large community and examples

### URDF Limitations
- **Simulation Features**: Lacks simulation-specific features
- **Physics Details**: Limited physics parameter control
- **Sensors**: No sensor definitions
- **Environment**: Cannot describe simulation environments

## When to Use SDF

### Appropriate Use Cases

#### 1. Simulation-First Development
- When primary goal is simulation
- For physics-based validation
- For sensor simulation

#### 2. Gazebo-Specific Features
- When using Gazebo plugins
- For complex sensor configurations
- For environment modeling

#### 3. Multi-Robot Simulation
- When simulating multiple robots in shared environments
- For complex world scenarios

### SDF Advantages
- **Simulation Features**: Rich support for simulation aspects
- **Physics Control**: Detailed physics parameter configuration
- **Sensors**: Native sensor definitions and properties
- **Environment**: Full world description capabilities
- **Plugins**: Support for custom simulation plugins

### SDF Limitations
- **Complexity**: More verbose and complex syntax
- **ROS Integration**: Requires additional tools for ROS integration
- **Tool Support**: Fewer tools compared to URDF
- **Learning Curve**: Steeper learning curve

## Converting Between Formats

### URDF to SDF Conversion

#### Using Command Line Tools
```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Convert with specific SDF version
gz sdf -p --version 1.7 robot.urdf > robot.sdf
```

#### Programmatic Conversion
```python
import xml.etree.ElementTree as ET

def urdf_to_sdf_basic(urdf_content):
    """Basic example of converting URDF to SDF structure"""
    urdf_root = ET.fromstring(urdf_content)

    # Create SDF root
    sdf_root = ET.Element("sdf", version="1.7")
    world = ET.SubElement(sdf_root, "model", name=urdf_root.get("name"))

    # Convert links
    for link in urdf_root.findall("link"):
        sdf_link = ET.SubElement(world, "link", name=link.get("name"))

        # Convert visual
        for visual in link.findall("visual"):
            sdf_visual = ET.SubElement(sdf_link, "visual", name="visual")
            # ... convert visual properties

        # Convert collision
        for collision in link.findall("collision"):
            sdf_collision = ET.SubElement(sdf_link, "collision", name="collision")
            # ... convert collision properties

        # Convert inertial
        for inertial in link.findall("inertial"):
            sdf_inertial = ET.SubElement(sdf_link, "inertial")
            # ... convert inertial properties

    return ET.tostring(sdf_root, encoding='unicode')
```

### Adding Simulation-Specific Features

When converting from URDF to SDF, you often need to add simulation-specific elements:

```xml
<!-- Original URDF link -->
<link name="link_with_simulation">
  <!-- Original URDF elements -->
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>

  <!-- SDF additions for simulation -->
  <surface>
    <friction>
      <ode>
        <mu>0.5</mu>
        <mu2>0.5</mu2>
      </ode>
    </friction>
    <contact>
      <ode>
        <kp>1e6</kp>
        <kd>100</kd>
      </ode>
    </contact>
  </surface>
</link>
```

## Practical Examples for Humanoid Robotics

### URDF for Humanoid Robot
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
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
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
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
    <origin xyz="0.15 0.1 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>
</robot>
```

### Enhanced SDF for Simulation
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="humanoid_robot">
    <static>false</static>
    <self_collide>false</self_collide>
    <enable_wind>false</enable_wind>
    <pose>0 0 1 0 0 0</pose>

    <!-- Torso -->
    <link name="torso">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <visual name="torso_visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>

      <collision name="torso_collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.8</mu>
              <mu2>0.8</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>1e6</kp>
              <kd>100</kd>
            </ode>
          </contact>
        </surface>
      </collision>
    </link>

    <!-- Head with sensor -->
    <link name="head">
      <pose>0 0 0.3 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.002</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.002</iyy>
          <iyz>0</iyz>
          <izz>0.002</izz>
        </inertia>
      </inertial>

      <!-- Camera sensor in head -->
      <sensor name="head_camera" type="camera">
        <pose>0.05 0 0.05 0 0 0</pose>
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
    </link>

    <!-- Neck joint -->
    <joint name="neck_joint" type="revolute">
      <parent>torso</parent>
      <child>head</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>10</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

## Integration with ROS 2

### Using URDF in Gazebo

The `robot_state_publisher` and `gazebo_ros` packages allow using URDF with Gazebo:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
            ])
        ),

        # Robot State Publisher (loads URDF)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': True},
                {'robot_description': open('urdf/humanoid.urdf').read()}
            ]
        ),

        # Spawn robot from URDF
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'humanoid_robot',
                '-x', '0',
                '-y', '0',
                '-z', '1'
            ],
            output='screen'
        )
    ])
```

### Xacro for Both Formats

Xacro can be used to generate both URDF and SDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.14159265359"/>
  <xacro:property name="torso_mass" value="5.0"/>
  <xacro:property name="arm_mass" value="1.0"/>

  <!-- Macro for humanoid link -->
  <xacro:macro name="humanoid_link" params="name mass *geometry *inertial_values">
    <link name="${name}">
      <visual>
        <xacro:insert_block name="geometry"/>
        <material name="gray">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
      <collision>
        <xacro:insert_block name="geometry"/>
      </collision>
      <xacro:insert_block name="inertial_values"/>
    </link>
  </xacro:macro>

  <!-- Torso -->
  <xacro:humanoid_link name="torso" mass="${torso_mass}">
    <geometry>
      <cylinder radius="0.1" length="0.5"/>
    </geometry>
    <inertial_values>
      <inertial>
        <mass value="${torso_mass}"/>
        <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
      </inertial>
    </inertial_values>
  </xacro:humanoid_link>

  <!-- Simulation-specific additions -->
  <gazebo reference="torso">
    <material>Gazebo/Gray</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>
</robot>
```

## Best Practices for Humanoid Robotics

### 1. Use URDF for Primary Description
- Keep your main robot description in URDF
- Use Xacro for complex humanoid models
- Focus on kinematic and dynamic properties

### 2. Extend with Gazebo Tags in URDF
- Add simulation-specific properties using `<gazebo>` tags
- Keep everything in one file for easier management
- Use conditional blocks for different simulation scenarios

```xml
<!-- Add simulation properties to URDF -->
<gazebo reference="left_foot">
  <mu1>1.0</mu1>
  <mu2>1.0</mu2>
  <fdir1>1 0 0</fdir1>
  <kp>10000000.0</kp>
  <kd>1000000.0</kd>
</gazebo>

<!-- Add sensors to URDF -->
<gazebo reference="head">
  <sensor name="camera" type="camera">
    <pose>0.05 0 0.05 0 0 0</pose>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
      </image>
    </camera>
    <always_on>1</always_on>
    <update_rate>30</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### 3. Separate Complex Simulation Models
- Use SDF for complex multi-robot scenarios
- Use SDF for custom world models
- Use SDF when you need advanced simulation features not available through Gazebo tags

### 4. Maintain Consistency
- Keep inertial properties consistent between formats
- Use the same joint limits and ranges
- Ensure visual and collision models match

## Migration Strategies

### From URDF to SDF-Only
```bash
# Convert and manually enhance
gz sdf -p robot.urdf > robot.sdf

# Then manually add simulation-specific features
# - Physics parameters
# - Sensors
# - Plugins
# - Surface properties
```

### From SDF to URDF-Only
This is generally not recommended as SDF contains more information than URDF can represent.

## Tools and Utilities

### Command Line Tools
```bash
# Validate URDF
check_urdf robot.urdf

# Convert URDF to SDF
gz sdf -p robot.urdf

# Validate SDF
gz sdf -k robot.sdf

# Visualize SDF
gz sdf -g robot.sdf
```

### ROS Tools
```bash
# Check robot description
ros2 param get /robot_state_publisher robot_description

# Visualize in RViz
ros2 run rviz2 rviz2

# Check transforms
ros2 run tf2_tools view_frames
```

## Common Pitfalls

### 1. Inertial Property Mismatches
- Ensure inertial properties are realistic
- Verify mass and inertia values
- Use consistent units across formats

### 2. Collision vs Visual Geometry
- Use simplified collision geometry for performance
- Use detailed visual geometry for appearance
- Ensure both represent the same physical object

### 3. Joint Limit Discrepancies
- Keep joint limits consistent between formats
- Consider real hardware limits
- Account for safety margins

### 4. Sensor Placement Issues
- Verify sensor positions in both formats
- Check coordinate frame orientations
- Validate sensor parameters

## Performance Considerations

### URDF Advantages
- Faster parsing and loading
- Simpler for basic robot description
- Better tool support

### SDF Advantages
- More detailed physics control
- Better simulation performance with optimized settings
- Richer feature set for complex scenarios

## Troubleshooting

### Common Issues

**Problem**: Robot falls through floor in simulation
- **Cause**: Collision geometry issues or physics parameters
- **Solution**: Check collision models and contact parameters

**Problem**: URDF doesn't work in Gazebo
- **Cause**: Missing gazebo tags or incorrect references
- **Solution**: Add proper gazebo blocks to URDF

**Problem**: Different behavior in URDF vs SDF
- **Cause**: Inconsistent parameters between formats
- **Solution**: Ensure parameter consistency

## Next Steps

With a solid understanding of URDF vs SDF, continue to [Sensor Simulation](./sensor-simulation.md) to learn how to simulate various sensor types in Gazebo and how to use them effectively in humanoid robotics applications.