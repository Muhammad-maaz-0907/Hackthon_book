---
title: Practical ROS 2 Development
sidebar_position: 4
---

# Practical ROS 2 Development

This lesson focuses on practical ROS 2 development using Python and rclpy. You'll learn to create real ROS 2 packages, nodes, and implement the communication patterns you learned about in the previous lesson.

## Setting Up a ROS 2 Workspace

Before creating your first package, you need to set up a ROS 2 workspace:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Build the workspace (even though it's empty)
colcon build
source install/setup.bash
```

## Creating Your First ROS 2 Package

### Using ros2 pkg create

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package
```

This creates a basic Python package structure:

```
my_robot_package/
├── my_robot_package/
│   ├── __init__.py
│   └── my_node.py
├── setup.py
├── setup.cfg
├── package.xml
└── resource/my_robot_package
```

### Package.xml Configuration

Edit the `package.xml` file to define dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example package for humanoid robotics</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### Setup.py Configuration

Update the `setup.py` file:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Example package for humanoid robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.my_node:main',
            'listener = my_robot_package.my_node:main',
        ],
    },
)
```

## Creating a Basic Publisher Node

Create a publisher node in `my_robot_package/my_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Creating a Basic Subscriber Node

Create a subscriber node in the same file:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def subscriber_main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


# Add to the main function to select which node to run
def main(args=None):
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'subscriber':
        subscriber_main(args)
    else:
        publisher_main(args)


def publisher_main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()
```

## Creating a Service Server and Client

### Service Server Implementation

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response


def service_main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()
```

### Service Client Implementation

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        return self.future


def client_main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    future = minimal_client.send_request(1, 2)

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                minimal_client.get_logger().info('Service call failed %r' % (e,))
            else:
                minimal_client.get_logger().info('Result of add_two_ints: %d' % (response.sum,))
            break

    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Humanoid Robotics Example: Joint State Publisher

Let's create a practical example for humanoid robotics - a joint state publisher:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time


class HumanoidJointStatePublisher(Node):

    def __init__(self):
        super().__init__('humanoid_joint_state_publisher')

        # Create publisher for joint states
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

        # Timer for publishing joint states
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Joint names for a simple humanoid (2 legs, 2 arms)
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        self.time_prev = time.time()

    def timer_callback(self):
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Generate some example joint positions (oscillating for demo)
        current_time = time.time()
        time_diff = current_time - self.time_prev

        positions = []
        for i, name in enumerate(self.joint_names):
            # Create different oscillation patterns for each joint
            position = math.sin(current_time + i) * 0.5
            positions.append(position)

        msg.position = positions
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        self.publisher_.publish(msg)
        self.time_prev = current_time


def joint_state_publisher_main(args=None):
    rclpy.init(args=args)
    node = HumanoidJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Launch Files for Coordinating Multiple Nodes

Launch files allow you to start multiple nodes with a single command.

### Creating a Launch File

Create `my_robot_package/launch/demo_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='talker',
            name='publisher_node',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='listener',
            name='subscriber_node',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            output='screen'
        )
    ])
```

### Launch File with Parameters

Create `my_robot_package/launch/robot_with_params_launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # Create nodes
    robot_node = Node(
        package='my_robot_package',
        executable='joint_state_publisher',
        name='humanoid_controller',
        parameters=[
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        robot_node
    ])
```

## Parameters in ROS 2

Parameters allow you to configure nodes at runtime:

```python
import rclpy
from rclpy.node import Node


class ParameterNode(Node):

    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_enabled', True)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_enabled = self.get_parameter('safety_enabled').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety enabled: {self.safety_enabled}')

        # Create a timer to periodically check parameter changes
        self.timer = self.create_timer(1.0, self.check_parameters)

    def check_parameters(self):
        # This allows parameters to be changed at runtime
        self.max_velocity = self.get_parameter('max_velocity').value
        self.get_logger().info(f'Current max velocity: {self.max_velocity}')


def parameter_main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Testing Your Nodes

### Writing Basic Tests

Create `test/test_my_node.py`:

```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_package.my_node import MinimalPublisher, MinimalSubscriber


class TestMinimalNodes(unittest.TestCase):

    def setUp(self):
        rclpy.init()

    def tearDown(self):
        rclpy.shutdown()

    def test_publisher_subscriber(self):
        publisher_node = MinimalPublisher()
        subscriber_node = MinimalSubscriber()

        executor = SingleThreadedExecutor()
        executor.add_node(publisher_node)
        executor.add_node(subscriber_node)

        # Run for a short time to allow communication
        executor.spin_once(timeout_sec=0.1)

        # Publish a message
        publisher_node.timer_callback()

        # Check that the subscriber received the message
        # (This is a basic test - you'd want more sophisticated checks)
        self.assertTrue(publisher_node.i > 0)


if __name__ == '__main__':
    unittest.main()
```

## Building and Running Your Package

### Building the Package

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

### Running Nodes

```bash
# Run the publisher
ros2 run my_robot_package talker

# Run the subscriber in another terminal
ros2 run my_robot_package listener

# Run with launch file
ros2 launch my_robot_package demo_launch.py
```

### Using Parameters

```bash
# Run with custom parameters
ros2 run my_robot_package parameter_node --ros-args -p robot_name:=my_humanoid -p max_velocity:=2.0
```

## Common Development Patterns

### 1. Publisher-Subscriber Pattern

```python
class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        self.subscriber = self.create_subscription(
            SensorMsg, 'input_topic', self.process_data, 10)
        self.publisher = self.create_publisher(
            ProcessedMsg, 'output_topic', 10)

    def process_data(self, msg):
        # Process the input data
        processed_msg = self.process(msg)
        # Publish the result
        self.publisher.publish(processed_msg)
```

### 2. Service-Based Configuration

```python
class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')
        self.config_value = 1.0
        self.service = self.create_service(
            SetConfiguration, 'set_config', self.set_config_callback)

    def set_config_callback(self, request, response):
        self.config_value = request.value
        response.success = True
        response.message = f'Config set to {self.config_value}'
        return response
```

### 3. State Machine Pattern

```python
from enum import Enum

class RobotState(Enum):
    IDLE = 1
    MOVING = 2
    MANIPULATING = 3
    ERROR = 4

class StateMachineNode(Node):
    def __init__(self):
        super().__init__('state_machine_node')
        self.current_state = RobotState.IDLE
        self.timer = self.create_timer(0.1, self.state_machine_loop)

    def state_machine_loop(self):
        if self.current_state == RobotState.IDLE:
            # Check for commands to start moving
            pass
        elif self.current_state == RobotState.MOVING:
            # Execute movement
            pass
        # ... handle other states
```

## Best Practices for Humanoid Robotics

### 1. Error Handling and Safety

```python
class SafeHumanoidController(Node):
    def __init__(self):
        super().__init__('safe_humanoid_controller')
        self.safety_enabled = True
        self.emergency_stop = False

    def check_safety(self, joint_commands):
        # Check if commands are within safe limits
        for cmd in joint_commands:
            if abs(cmd) > self.max_safe_value:
                self.get_logger().error('Safety limit exceeded!')
                self.emergency_stop = True
                return False
        return True
```

### 2. Real-time Considerations

```python
class RealTimeController(Node):
    def __init__(self):
        super().__init__('real_time_controller')
        # Use high-frequency timer for real-time control
        self.control_timer = self.create_timer(
            0.001,  # 1kHz control loop
            self.control_callback,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME)
        )
```

## Next Steps

Now that you've learned practical ROS 2 development, continue to [Launch Files & Parameters](./launch-files-parameters.md) to learn how to coordinate complex multi-node systems and configure them effectively for humanoid robotics applications.