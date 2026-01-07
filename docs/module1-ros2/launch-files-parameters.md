---
title: Launch Files & Parameters
sidebar_position: 5
---

# Launch Files & Parameters

This lesson covers how to use launch files to coordinate multiple nodes and how to configure your robotic systems using parameters in ROS 2. These tools are essential for managing complex humanoid robotics systems.

## Launch Files: Coordinating Complex Systems

Launch files allow you to start multiple nodes with a single command, making it easier to manage complex robotic systems. This is particularly important in humanoid robotics where multiple sensors, controllers, and perception systems need to work together.

### Basic Launch File Structure

A launch file is a Python file that defines which nodes to run and how to configure them:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            output='screen'
        )
    ])
```

### Launch File with Parameters

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_robot_name_cmd = DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    )

    # Create nodes
    robot_node = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name}
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_robot_name_cmd,
        robot_node
    ])
```

### Launch File with Conditional Logic

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_vision = LaunchConfiguration('enable_vision')

    # Conditional nodes
    vision_node = Node(
        package='my_robot_package',
        executable='vision_node',
        name='vision_node',
        condition=IfCondition(enable_vision),
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'enable_vision',
            default_value='true',
            description='Enable vision processing nodes'
        ),
        vision_node
    ])
```

## Advanced Launch File Patterns

### Including Other Launch Files

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include another launch file
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_sensors'),
                'launch',
                'sensors_launch.py'
            ])
        ])
    )

    return LaunchDescription([
        sensors_launch,
        # Additional nodes can be added here
    ])
```

### Launch File with Remappings

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor',
            remappings=[
                ('/input/laser_scan', '/laser_scan'),
                ('/output/processed_data', '/processed_laser_data')
            ],
            output='screen'
        )
    ])
```

### Launch File with Timed Actions

```python
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Start the main controller immediately
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            output='screen'
        ),
        # Start the UI after 5 seconds
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='my_robot_package',
                    executable='robot_ui',
                    name='robot_ui',
                    output='screen'
                )
            ]
        )
    ])
```

## Parameter Management

Parameters in ROS 2 allow you to configure nodes at runtime, making them more flexible and reusable.

### Declaring and Using Parameters

```python
import rclpy
from rclpy.node import Node


class ParameterExampleNode(Node):

    def __init__(self):
        super().__init__('parameter_example_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_enabled', True)
        self.declare_parameter('joint_limits', [1.57, 1.57, 3.14])  # Example list

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_enabled = self.get_parameter('safety_enabled').value
        self.joint_limits = self.get_parameter('joint_limits').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety enabled: {self.safety_enabled}')
        self.get_logger().info(f'Joint limits: {self.joint_limits}')

        # Create a timer to periodically check parameter changes
        self.timer = self.create_timer(1.0, self.check_parameters)

    def check_parameters(self):
        # This allows parameters to be changed at runtime
        self.max_velocity = self.get_parameter('max_velocity').value
        self.get_logger().info(f'Current max velocity: {self.max_velocity}')
```

### Parameter Descriptions and Validation

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter


class ParameterValidationNode(Node):

    def __init__(self):
        super().__init__('parameter_validation_node')

        # Declare parameter with description
        self.declare_parameter(
            'control_frequency',
            100,  # default value
            ParameterDescriptor(
                description='Control loop frequency in Hz',
                integer_range=[ParameterIntegerRange(from_value=10, to_value=1000, step=1)]
            )
        )

        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameters_callback)

    def parameters_callback(self, parameters):
        for param in parameters:
            if param.name == 'control_frequency' and param.type_ == Parameter.Type.INTEGER:
                if param.value < 10 or param.value > 1000:
                    return SetParametersResult(successful=False, reason='Control frequency must be between 10 and 1000 Hz')
        return SetParametersResult(successful=True)
```

### YAML Parameter Files

Create a parameter file `config/robot_config.yaml`:

```yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    log_level: 'info'

robot_controller:
  ros__parameters:
    max_velocity: 2.0
    acceleration_limit: 1.0
    safety_enabled: true
    joint_limits:
      - 1.57
      - 1.57
      - 3.14

vision_node:
  ros__parameters:
    image_width: 640
    image_height: 480
    detection_threshold: 0.7
```

### Loading Parameters from YAML

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Path to the parameter file
    params_file = PathJoinSubstitution([
        FindPackageShare('my_robot_package'),
        'config',
        'robot_config.yaml'
    ])

    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[params_file],
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='vision_node',
            name='vision_node',
            parameters=[params_file],
            output='screen'
        )
    ])
```

## Humanoid Robotics Launch Patterns

### Full Humanoid Robot Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),

        # Robot state publisher (publishes TF transforms)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'robot_description': open('urdf/humanoid.urdf').read()}
            ],
            output='screen'
        ),

        # Joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Perception nodes
        Node(
            package='my_robot_perception',
            executable='vision_node',
            name='vision_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Control nodes
        Node(
            package='my_robot_control',
            executable='humanoid_controller',
            name='humanoid_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Navigation nodes
        Node(
            package='nav2_bringup',
            executable='nav2_launch',
            name='nav2_bringup',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

### Modular Launch Architecture

For complex humanoid systems, it's helpful to break launch files into modules:

```python
# launch/humanoid_base.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Include sensor launch
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_sensors'),
                'launch',
                'sensors.launch.py'
            ])
        ])
    )

    # Include control launch
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_control'),
                'launch',
                'control.launch.py'
            ])
        ])
    )

    # Include perception launch
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('my_robot_perception'),
                'launch',
                'perception.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        sensors_launch,
        control_launch,
        perception_launch
    ])
```

## Parameter Best Practices for Humanoid Robotics

### 1. Organize Parameters by Function

```yaml
# config/humanoid_params.yaml
humanoid_controller:
  ros__parameters:
    # Joint control parameters
    joint_control:
      position_tolerance: 0.01
      velocity_tolerance: 0.1
      effort_tolerance: 0.5

    # Balance control parameters
    balance_control:
      com_height: 0.8
      com_tolerance: 0.05
      zmp_tolerance: 0.02

    # Walking parameters
    walking:
      step_height: 0.05
      step_length: 0.3
      step_duration: 1.0
```

### 2. Use Parameter Namespaces

```python
# In your node
class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Use namespaces for organized parameter access
        self.declare_parameter('walking.step_length', 0.3)
        self.declare_parameter('walking.step_height', 0.05)
        self.declare_parameter('balance.com_height', 0.8)

        # Access with namespace
        self.step_length = self.get_parameter('walking.step_length').value
        self.com_height = self.get_parameter('balance.com_height').value
```

### 3. Runtime Parameter Reconfiguration

```python
# Add dynamic parameter support
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

class ConfigurableHumanoidNode(Node):
    def __init__(self):
        super().__init__('configurable_humanoid_node')

        # Declare parameters with proper types
        self.declare_parameter(
            'gait_type',
            'walk',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )

        # Add callback to handle parameter changes
        self.add_on_set_parameters_callback(self.param_change_callback)

    def param_change_callback(self, params):
        for param in params:
            if param.name == 'gait_type':
                self.change_gait(param.value)
        return SetParametersResult(successful=True)

    def change_gait(self, gait_type):
        # Implement gait change logic
        self.get_logger().info(f'Changing gait to: {gait_type}')
```

## Launch File Best Practices

### 1. Use Descriptive Names

```python
# Good: Descriptive launch file names
# launch/humanoid_walking_demo.launch.py
# launch/humanoid_manipulation_demo.launch.py
# launch/humanoid_full_system.launch.py
```

### 2. Document Launch Arguments

```python
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true. Set to true when running in Gazebo simulation.'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='atlas',
            choices=['atlas', 'valkyrie', 'darwin'],
            description='Type of humanoid robot model to load'
        ),
        # ... rest of launch description
    ])
```

### 3. Error Handling in Launch Files

```python
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    controller_node = Node(
        package='my_robot_control',
        executable='humanoid_controller',
        name='humanoid_controller'
    )

    # Handle node crashes
    controller_error_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_node,
            on_exit=[],
            # Add logic for restart or shutdown here
        )
    )

    return LaunchDescription([
        controller_node,
        controller_error_handler
    ])
```

## Running Launch Files

### Basic Launch Command

```bash
# Run a launch file
ros2 launch my_robot_package robot_launch.py

# Run with arguments
ros2 launch my_robot_package robot_launch.py use_sim_time:=true robot_name:=atlas

# Run with parameter file
ros2 launch my_robot_package robot_launch.py --params-file config/robot_config.yaml
```

### Checking Launch Status

```bash
# List all active nodes
ros2 node list

# Check the launch file structure
ros2 launch --show-args my_robot_package robot_launch.py

# Monitor parameter changes
ros2 param list
ros2 param get /robot_controller robot_name
```

## Troubleshooting Launch Files

### Common Issues and Solutions

1. **Node not found**: Check that the package is built and sourced
2. **Parameter not loaded**: Verify parameter file path and format
3. **Node crashes on startup**: Check parameter values and dependencies
4. **Timing issues**: Use TimerAction to control startup order

### Debugging Launch Files

```bash
# Enable verbose output
ros2 launch my_robot_package robot_launch.py --log-level debug

# Dry run to check syntax
ros2 launch --dry-run my_robot_package robot_launch.py
```

## Next Steps

Now that you understand launch files and parameters, continue to [Humanoid Context Applications](./humanoid-context.md) to learn how these tools apply specifically to humanoid robotics systems and how to create effective architectures for complex humanoid robots.