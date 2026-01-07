---
title: Humanoid Context Applications
sidebar_position: 6
---

# Humanoid Context Applications

This lesson explores how ROS 2 concepts apply specifically to humanoid robotics systems. You'll learn how to design ROS 2 architectures that effectively support the complex, multi-domain nature of humanoid robots.

## Understanding Humanoid Robotics Architecture

Humanoid robots present unique challenges compared to simpler robotic systems. They must coordinate multiple subsystems including perception, planning, control, and interaction in real-time to achieve human-like behavior.

### The Humanoid System Architecture

A typical humanoid robot system includes these interconnected subsystems:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │    Planning     │    │     Control     │
│                 │    │                 │    │                 │
│ • Vision        │    │ • Motion        │    │ • Joint         │
│ • Audio         │◄──►│   Planning      │◄──►│   Control       │
│ • Tactile       │    │ • Path          │    │ • Balance       │
│ • IMU/Sensors   │    │   Planning      │    │ • Walking       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │    Humanoid Framework     │
                    │                           │
                    │ • State Management        │
                    │ • Coordination Layer      │
                    │ • Safety & Monitoring   │
                    └───────────────────────────┘
```

## ROS 2 Communication Patterns in Humanoid Systems

### Topics: Sensor Data Streaming

Humanoid robots generate vast amounts of sensor data that needs to be processed by multiple systems simultaneously.

#### Example: Sensor Data Topics

```python
# Sensor data topics in a humanoid system
class HumanoidSensors(Node):
    def __init__(self):
        super().__init__('humanoid_sensors')

        # Joint state publisher (multiple subscribers)
        self.joint_state_pub = self.create_publisher(
            JointState, 'joint_states', 10)

        # IMU data (balance, perception, logging)
        self.imu_pub = self.create_publisher(
            Imu, 'imu/data', 10)

        # Camera feeds (multiple perception nodes)
        self.camera_pub = self.create_publisher(
            Image, 'camera/image_raw', 10)

        # Force/torque sensors (control, safety)
        self.ft_pub = self.create_publisher(
            WrenchStamped, 'l_foot/force_torque', 10)
```

**Use Cases in Humanoid Systems:**
- Joint position/velocity/effort streaming
- IMU data for balance control
- Camera feeds for perception
- LIDAR scans for navigation
- Force/torque sensor data
- Microphone audio streams

### Services: Configuration and Calibration

Services are used for operations that require immediate responses, such as calibration or configuration changes.

#### Example: Humanoid Service Interface

```python
# Calibration service
class CalibrationService(Node):
    def __init__(self):
        super().__init__('calibration_service')
        self.calibration_srv = self.create_service(
            CalibrateSensors, 'calibrate_sensors', self.calibrate_callback)

    def calibrate_callback(self, request, response):
        # Perform calibration based on request parameters
        if request.sensor_type == 'imu':
            self.calibrate_imu()
        elif request.sensor_type == 'camera':
            self.calibrate_camera()

        response.success = True
        response.message = f'Calibrated {request.sensor_type}'
        return response
```

**Use Cases in Humanoid Systems:**
- Sensor calibration
- Robot configuration changes
- Emergency stop activation
- Mode switching
- Parameter updates

### Actions: Complex Humanoid Behaviors

Actions are perfect for long-running humanoid behaviors that need feedback and can be interrupted.

#### Example: Walking Action Server

```python
from rclpy.action import ActionServer
from my_robot_msgs.action import WalkToGoal

class WalkingActionServer(Node):
    def __init__(self):
        super().__init__('walking_action_server')
        self._action_server = ActionServer(
            self,
            WalkToGoal,
            'walk_to_goal',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walking goal...')

        # Walking with feedback
        feedback_msg = WalkToGoal.Feedback()

        for step in self.walk_to_position(goal_handle.request.target_pose):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return WalkToGoal.Result()

            # Provide feedback on walking progress
            feedback_msg.current_pose = step.current_pose
            feedback_msg.progress = step.progress
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = WalkToGoal.Result()
        result.success = True
        result.final_pose = feedback_msg.current_pose
        return result
```

**Use Cases in Humanoid Systems:**
- Navigation and walking
- Manipulation tasks
- Grasping and releasing
- Speech synthesis with feedback
- Complex motion sequences
- Human interaction behaviors

## Humanoid-Specific Design Patterns

### 1. The State Machine Pattern for Humanoid Behaviors

Humanoid robots often need to maintain complex internal states that coordinate multiple subsystems.

```python
from enum import Enum

class HumanoidState(Enum):
    IDLE = 1
    WALKING = 2
    MANIPULATING = 3
    BALANCING = 4
    EMERGENCY_STOP = 5

class HumanoidStateMachine(Node):
    def __init__(self):
        super().__init__('humanoid_state_machine')
        self.current_state = HumanoidState.IDLE

        # Publishers for different subsystems
        self.command_pub = self.create_publisher(
            HumanoidCommand, 'humanoid_commands', 10)

        # State transition timer
        self.state_timer = self.create_timer(0.1, self.state_machine_loop)

    def state_machine_loop(self):
        if self.current_state == HumanoidState.IDLE:
            self.handle_idle_state()
        elif self.current_state == HumanoidState.WALKING:
            self.handle_walking_state()
        elif self.current_state == HumanoidState.BALANCING:
            self.handle_balancing_state()
        # ... other states
```

### 2. The Sensor Fusion Pattern

Humanoid robots must combine data from multiple sensors to create a coherent understanding of their environment.

```python
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Multiple sensor subscribers
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Fused state publisher
        self.fused_state_pub = self.create_publisher(
            HumanoidState, 'fused_state', 10)

        self.fused_data = {}

    def imu_callback(self, msg):
        self.fused_data['imu'] = msg
        self.publish_fused_state()

    def odom_callback(self, msg):
        self.fused_data['odom'] = msg
        self.publish_fused_state()

    def joint_callback(self, msg):
        self.fused_data['joints'] = msg
        self.publish_fused_state()

    def publish_fused_state(self):
        # Combine sensor data into a coherent state
        fused_msg = self.combine_sensor_data()
        self.fused_state_pub.publish(fused_msg)
```

### 3. The Safety Monitor Pattern

Safety is critical in humanoid robotics, requiring constant monitoring of multiple systems.

```python
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Subscribe to critical topics
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        self.command_pub = self.create_publisher(
            Bool, 'emergency_stop', 10)

        self.safety_timer = self.create_timer(0.01, self.safety_check)  # 100Hz

        # Safety parameters
        self.declare_parameter('max_joint_effort', 100.0)
        self.declare_parameter('max_angular_velocity', 3.14)
        self.declare_parameter('fall_threshold', 0.5)

    def safety_check(self):
        # Check for safety violations
        if self.check_joint_limits() or self.check_balance():
            self.trigger_emergency_stop()

    def check_joint_limits(self):
        # Check if joint efforts are within safe limits
        if hasattr(self, 'last_joint_state'):
            max_effort = self.get_parameter('max_joint_effort').value
            for effort in self.last_joint_state.effort:
                if abs(effort) > max_effort:
                    return True
        return False

    def check_balance(self):
        # Check if robot is tilting too much
        if hasattr(self, 'last_imu'):
            threshold = self.get_parameter('fall_threshold').value
            roll, pitch = self.get_orientation_angles(self.last_imu)
            return abs(roll) > threshold or abs(pitch) > threshold
        return False
```

## Humanoid-Specific Parameters

Humanoid robots require many parameters that are specific to their complex nature:

### Balance Control Parameters

```yaml
balance_controller:
  ros__parameters:
    # Center of mass control
    com_height: 0.85  # meters
    com_tolerance: 0.02  # meters
    zmp_tolerance: 0.015  # meters

    # Balance gains
    balance_p_gain: 100.0
    balance_d_gain: 10.0

    # Safety limits
    max_tilt_angle: 0.3  # radians
    fall_threshold: 0.4  # radians
```

### Walking Pattern Parameters

```yaml
walking_controller:
  ros__parameters:
    # Step parameters
    step_length: 0.3  # meters
    step_width: 0.2   # meters
    step_height: 0.05 # meters
    step_duration: 1.0 # seconds

    # Walking gait
    walking_speed: 0.5  # m/s
    max_turn_rate: 0.5  # rad/s

    # Support polygon
    support_margin: 0.05 # meters
```

### Joint Control Parameters

```yaml
joint_controller:
  ros__parameters:
    # Joint limits (per joint)
    joint_limits:
      left_hip:
        position_min: -1.57
        position_max: 1.57
        velocity_max: 2.0
        effort_max: 100.0
      right_knee:
        position_min: 0.0
        position_max: 2.3
        velocity_max: 2.0
        effort_max: 100.0
    # ... more joints
```

## Launch Files for Humanoid Systems

### Full Humanoid System Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    config_file = LaunchConfiguration('config_file')

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
        DeclareLaunchArgument(
            'config_file',
            default_value='humanoid_config.yaml',
            description='Configuration file to load'
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

        # Start sensors after other nodes are up
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='my_robot_sensors',
                    executable='sensor_manager',
                    name='sensor_manager',
                    parameters=[{'use_sim_time': use_sim_time}],
                    output='screen'
                )
            ]
        ),

        # Core controllers
        Node(
            package='my_robot_control',
            executable='balance_controller',
            name='balance_controller',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),

        Node(
            package='my_robot_control',
            executable='walking_controller',
            name='walking_controller',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Perception system
        Node(
            package='my_robot_perception',
            executable='perception_manager',
            name='perception_manager',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Humanoid state machine
        Node(
            package='my_robot_behavior',
            executable='state_machine',
            name='state_machine',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Safety monitor
        Node(
            package='my_robot_safety',
            executable='safety_monitor',
            name='safety_monitor',
            parameters=[config_file, {'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## Quality of Service Considerations for Humanoid Systems

Humanoid robots have real-time requirements that affect QoS settings:

### Critical Control Topics

```python
# For critical control commands
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

critical_qos = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

self.joint_command_pub = self.create_publisher(
    JointCommand, 'joint_commands', critical_qos)
```

### Sensor Data Topics

```python
# For high-frequency sensor data
sensor_qos = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST
)

self.camera_pub = self.create_publisher(
    Image, 'camera/image_raw', sensor_qos)
```

## Humanoid Robotics Best Practices

### 1. Real-time Considerations

```python
class RealTimeHumanoidController(Node):
    def __init__(self):
        super().__init__('real_time_controller')

        # High-frequency control loop (1kHz)
        self.control_timer = self.create_timer(
            0.001,  # 1ms period
            self.control_loop,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME)
        )

        # Use real-time safe operations in control loop
```

### 2. Fault Tolerance

```python
class FaultTolerantHumanoid(Node):
    def __init__(self):
        super().__init__('fault_tolerant_humanoid')

        # Monitor critical nodes
        self.node_monitor_timer = self.create_timer(1.0, self.monitor_nodes)

    def monitor_nodes(self):
        # Check if critical nodes are alive
        active_nodes = self.get_node_names()

        critical_nodes = ['balance_controller', 'sensor_manager', 'safety_monitor']
        for node in critical_nodes:
            if node not in active_nodes:
                self.get_logger().error(f'Critical node {node} is not running!')
                self.activate_safety_mode()
```

### 3. Graceful Degradation

```python
class DegradableHumanoid(Node):
    def __init__(self):
        super().__init__('degradable_humanoid')

        # Handle sensor failures gracefully
        self.vision_available = True
        self.torso_imu_available = True

    def check_sensor_availability(self):
        # Check which sensors are available
        if not self.check_vision_system():
            self.vision_available = False
            self.get_logger().warn('Vision system unavailable, degrading functionality')

        if not self.check_imu_system():
            self.torso_imu_available = False
            self.get_logger().warn('Torso IMU unavailable, using joint-based estimation')

    def adjust_behavior_for_sensor_loss(self):
        # Adjust behavior based on available sensors
        if not self.vision_available:
            # Use alternative perception methods
            pass
        if not self.torso_imu_available:
            # Use joint-based balance estimation
            pass
```

## Integration with Other Modules

The ROS 2 foundation you're building here integrates with:
- **Module 2**: Sensor data from simulation
- **Module 3**: Perception results for planning
- **Module 4**: Voice commands for behavior control

### Example Integration Points

```python
# Integration with perception (Module 3)
class PerceptionIntegrator(Node):
    def __init__(self):
        super().__init__('perception_integrator')

        # Subscribe to perception results
        self.detection_sub = self.create_subscription(
            ObjectDetection, 'object_detections',
            self.detections_callback, 10)

    def detections_callback(self, msg):
        # Use perception results for navigation planning
        if self.current_behavior == 'navigation':
            self.update_navigation_goals(msg.objects)

# Integration with VLA (Module 4)
class VoiceCommandIntegrator(Node):
    def __init__(self):
        super().__init__('voice_command_integrator')

        # Subscribe to voice commands
        self.voice_sub = self.create_subscription(
            VoiceCommand, 'voice_commands',
            self.voice_command_callback, 10)

    def voice_command_callback(self, msg):
        # Convert voice commands to robot behaviors
        if msg.command_type == 'WALK':
            self.initiate_walking(msg.destination)
        elif msg.command_type == 'GRASP':
            self.initiate_grasping(msg.object_name)
```

## Next Steps

With a solid understanding of how ROS 2 applies to humanoid systems, continue to [URDF Primer for Humanoids](./urdf-primer.md) to learn how to represent humanoid robots in ROS 2 and connect your communication architecture to robot description.