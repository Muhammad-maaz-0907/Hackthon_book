---
title: URDF Primer for Humanoids
sidebar_position: 7
---

# URDF Primer for Humanoids

URDF (Unified Robot Description Format) is essential for describing humanoid robots in ROS 2. This lesson covers the fundamentals of URDF with a focus on humanoid-specific applications, including how to structure complex humanoid models and connect them to ROS 2 systems.

## Understanding URDF

URDF is an XML format that describes robots in terms of their links, joints, and other components. For humanoid robots, URDF becomes particularly important due to the complex kinematic structure and multiple degrees of freedom.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Links define rigid bodies -->
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
</robot>
```

## URDF Components for Humanoid Robots

### Links: The Building Blocks

Links represent rigid bodies in the robot. For humanoid robots, these include:

```xml
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
```

### Joints: Connecting the Links

Joints define how links move relative to each other. Humanoid robots require many joints to achieve human-like movement:

```xml
<!-- Joint connecting torso to head -->
<joint name="neck_joint" type="revolute">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Example leg joints -->
<joint name="left_hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0 -0.1 -0.25" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="200" velocity="2.0"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="2.3" effort="200" velocity="2.0"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<joint name="left_ankle_joint" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Humanoid-Specific Joint Types

### 1. Spherical Joints for Shoulders and Hips

For joints that need to move in multiple directions:

```xml
<!-- Spherical joint for shoulder (using 3 revolute joints) -->
<joint name="left_shoulder_yaw" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.1 0.1 0.2" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="150" velocity="1.5"/>
</joint>

<joint name="left_shoulder_pitch" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_forearm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.0" upper="1.0" effort="150" velocity="1.5"/>
</joint>
```

### 2. Fixed Joints for Non-Moving Parts

```xml
<!-- Fixed joint for sensors that don't move relative to their mounting link -->
<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<link name="imu_link"/>
```

## Complete Humanoid URDF Example

Here's a simplified but complete humanoid URDF that demonstrates the key concepts:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.5"/>
      </geometry>
      <material name="grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.3 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.25"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_forearm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_forearm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm (similar to left, mirrored) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.15 -0.1 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_forearm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_forearm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0 -0.05 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.3" effort="100" velocity="2.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.025"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right Leg (similar to left) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_thigh"/>
    <origin xyz="0 0.05 -0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="2.3" effort="100" velocity="2.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="1.57 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.03" ixy="0" ixz="0" iyy="0.03" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0.025"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Sensors -->
  <joint name="imu_joint" type="fixed">
    <parent link="torso"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="imu_link"/>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="camera_link"/>
</robot>
```

## Xacro for Complex Humanoid Models

Xacro is a macro language that extends URDF, making it easier to create complex humanoid models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_length" value="0.5" />
  <xacro:property name="upper_arm_length" value="0.3" />
  <xacro:property name="lower_arm_length" value="0.3" />
  <xacro:property name="upper_leg_length" value="0.5" />
  <xacro:property name="lower_leg_length" value="0.5" />

  <!-- Materials -->
  <xacro:macro name="default_material" params="name color">
    <material name="${name}">
      <color rgba="${color}"/>
    </material>
  </xacro:macro>

  <!-- Link macro -->
  <xacro:macro name="simple_link" params="name mass visual_color *geometry *inertial_values">
    <link name="${name}">
      <visual>
        <xacro:insert_block name="geometry"/>
        <xacro:default_material name="link_material" color="${visual_color}"/>
      </visual>
      <collision>
        <xacro:insert_block name="geometry"/>
      </collision>
      <xacro:insert_block name="inertial_values"/>
    </link>
  </xacro:macro>

  <!-- Define a humanoid limb -->
  <xacro:macro name="humanoid_arm" params="side parent_link parent_joint_origin">
    <!-- Upper arm -->
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${parent_joint_origin}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
    </joint>

    <xacro:simple_link name="${side}_upper_arm" mass="1.0" visual_color="0.0 0.0 0.8 1.0">
      <geometry>
        <cylinder radius="0.05" length="${upper_arm_length}"/>
      </geometry>
      <inertial_values>
        <inertial>
          <mass value="1.0"/>
          <origin xyz="0 0 -${upper_arm_length/2}"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
        </inertial>
      </inertial_values>
    </xacro:simple_link>

    <!-- Elbow joint -->
    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_forearm"/>
      <origin xyz="0 0 -${upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
    </joint>

    <xacro:simple_link name="${side}_forearm" mass="0.8" visual_color="0.0 0.0 0.8 1.0">
      <geometry>
        <cylinder radius="0.04" length="${lower_arm_length}"/>
      </geometry>
      <inertial_values>
        <inertial>
          <mass value="0.8"/>
          <origin xyz="0 0 -${lower_arm_length/2}"/>
          <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
        </inertial>
      </inertial_values>
    </xacro:simple_link>
  </xacro:macro>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.3 ${torso_length}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.3 ${torso_length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 ${torso_length/2}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Use the macro to create both arms -->
  <xacro:humanoid_arm side="left" parent_link="torso" parent_joint_origin="0.15 0.1 0.3"/>
  <xacro:humanoid_arm side="right" parent_link="torso" parent_joint_origin="0.15 -0.1 0.3"/>
</robot>
```

## URDF and ROS 2 Integration

### Robot State Publisher

The robot state publisher node uses your URDF to publish TF transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math


class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Robot description parameter
        self.declare_parameter('robot_description', '')

    def joint_state_callback(self, msg):
        # Process joint states and broadcast transforms
        for i, name in enumerate(msg.name):
            if name in ['left_hip_joint', 'right_knee_joint']:  # Example joints
                # Create transform for this joint
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = f'{name}_parent'
                t.child_frame_id = f'{name}_child'

                # Calculate transform based on joint position
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                # Convert joint position to rotation
                t.transform.rotation = self.euler_to_quaternion(0, 0, msg.position[i])

                self.tf_broadcaster.sendTransform(t)
```

### Loading URDF in Launch Files

```python
from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Path to URDF file
    urdf_path = PathJoinSubstitution([
        FindPackageShare('my_robot_description'),
        'urdf',
        'humanoid.urdf.xacro'
    ])

    # Robot description command
    robot_description = Command(['xacro ', urdf_path])

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description': robot_description}
            ],
            output='screen'
        )
    ])
```

## Humanoid-Specific Considerations

### 1. Center of Mass and Stability

For humanoid robots, the center of mass is critical for stability:

```xml
<!-- Ensure torso link has proper inertial properties for balance control -->
<link name="torso">
  <visual>
    <geometry>
      <box size="0.2 0.3 0.5"/>
    </geometry>
    <material name="grey">
      <color rgba="0.5 0.5 0.5 1"/>
    </material>
  </visual>
  <inertial>
    <mass value="10.0"/>
    <origin xyz="0 0 0.25"/>  <!-- Lower CoM for stability -->
    <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.2"/>
  </inertial>
</link>
```

### 2. Degrees of Freedom for Human-like Movement

Humanoid robots need sufficient DOF to perform human-like tasks:

```xml
<!-- Example of a more complex hand with multiple DOF -->
<joint name="left_hand_yaw" type="revolute">
  <parent link="left_forearm"/>
  <child link="left_hand"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="20" velocity="1.0"/>
</joint>

<!-- Individual finger joints -->
<joint name="left_thumb_joint" type="revolute">
  <parent link="left_hand"/>
  <child link="left_thumb"/>
  <origin xyz="0.05 0.02 -0.05" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="1.0" effort="5" velocity="0.5"/>
</joint>
```

### 3. Sensor Integration

Humanoid robots require many sensors integrated into the URDF:

```xml
<!-- IMU mounted on torso -->
<joint name="torso_imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="torso_imu_frame"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<link name="torso_imu_frame"/>

<!-- Multiple cameras for perception -->
<joint name="left_camera_joint" type="fixed">
  <parent link="head"/>
  <child link="left_camera_frame"/>
  <origin xyz="0.05 0.05 0.05" rpy="0 0 0"/>
</joint>

<link name="left_camera_frame"/>

<joint name="right_camera_joint" type="fixed">
  <parent link="head"/>
  <child link="right_camera_frame"/>
  <origin xyz="0.05 -0.05 0.05" rpy="0 0 0"/>
</joint>

<link name="right_camera_frame"/>

<!-- Force/torque sensors in feet -->
<joint name="left_foot_ft_sensor_joint" type="fixed">
  <parent link="left_foot"/>
  <child link="left_foot_center"/>
  <origin xyz="0 0 0.025" rpy="0 0 0"/>  <!-- Move to center of foot -->
</joint>

<link name="left_foot_center"/>
```

## Validation and Testing URDF

### URDF Validation

Always validate your URDF files:

```bash
# Check URDF syntax
check_urdf /path/to/your/robot.urdf

# Visualize the robot in RViz
ros2 run rviz2 rviz2
```

### Kinematic Chain Verification

Verify that your kinematic chains are properly formed:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState


class URDFValidator(Node):
    def __init__(self):
        super().__init__('urdf_validator')

        # Create TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.validate_kinematics, 10)

    def validate_kinematics(self, msg):
        # Check if transforms exist between connected links
        try:
            # Example: Check transform from torso to head
            transform = self.tf_buffer.lookup_transform(
                'torso', 'head', rclpy.time.Time())
            self.get_logger().info('Valid kinematic chain from torso to head')
        except Exception as e:
            self.get_logger().error(f'Kinematic chain error: {e}')
```

## Best Practices for Humanoid URDF

### 1. Modular Design

Break complex humanoid models into modules:

```xml
<!-- Include different body parts as separate xacro files -->
<xacro:include filename="$(find my_robot_description)/urdf/torso.urdf.xacro"/>
<xacro:include filename="$(find my_robot_description)/urdf/arms.urdf.xacro"/>
<xacro:include filename="$(find my_robot_description)/urdf/legs.urdf.xacro"/>
```

### 2. Proper Scaling

Ensure your model is properly scaled:

```xml
<!-- Use realistic dimensions based on actual robot or human proportions -->
<xacro:property name="human_height" value="1.7" />
<xacro:property name="torso_height" value="${human_height * 0.3}" />
<xacro:property name="leg_length" value="${human_height * 0.45}" />
```

### 3. Mass Distribution

Set realistic masses and inertias:

```xml
<!-- Use realistic mass values based on actual hardware -->
<link name="left_thigh">
  <inertial>
    <mass value="3.5"/>  <!-- Based on actual actuator + structure mass -->
    <origin xyz="0 0 -0.25"/>  <!-- CoM location -->
    <inertia ixx="0.08" ixy="0" ixz="0" iyy="0.08" iyz="0" izz="0.01"/>
  </inertial>
</link>
```

## Next Steps

Now that you understand URDF for humanoid robots, continue to [Module 1 Labs](./labs.md) to practice implementing these concepts in hands-on exercises that connect URDF to actual ROS 2 nodes and systems.