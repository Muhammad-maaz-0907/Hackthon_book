---
title: Module 1 Labs
sidebar_position: 8
---

# Module 1 Labs: ROS 2 Fundamentals for Humanoid Robotics

This lab section provides hands-on exercises to reinforce the ROS 2 concepts you've learned in Module 1. Each lab builds on the previous one to give you practical experience with ROS 2 in the context of humanoid robotics.

## Lab 1A: Setting Up Your ROS 2 Workspace

### Objective
Set up a ROS 2 development environment and create your first ROS 2 package.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Basic command line skills
- Understanding of ROS 2 workspace concepts

### Steps

1. **Create a workspace directory:**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

2. **Source the ROS 2 environment:**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. **Create a new package:**
   ```bash
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python humanoid_robot_tutorial --dependencies rclpy std_msgs sensor_msgs geometry_msgs
   ```

4. **Build the workspace:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

### Expected Output
- A new ROS 2 package named `humanoid_robot_tutorial`
- Successful compilation without errors
- Ability to run `ros2 run humanoid_robot_tutorial` without errors

### Troubleshooting
- If you get "command not found" errors, ensure ROS 2 is properly sourced
- If build fails, check that all dependencies are installed
- Make sure you're using Python 3.8 or higher

## Lab 1B: Creating a Simple Publisher and Subscriber

### Objective
Create a publisher node that sends messages and a subscriber node that receives them.

### Steps

1. **Create the publisher node:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/joint_publisher.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState
   import math
   import time


   class JointStatePublisher(Node):

       def __init__(self):
           super().__init__('joint_state_publisher')
           self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
           timer_period = 0.1  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0
           self.time_prev = time.time()

       def timer_callback(self):
           msg = JointState()
           msg.name = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint']

           # Create oscillating joint positions
           current_time = time.time()
           msg.position = [
               math.sin(current_time) * 0.5,      # left hip
               math.sin(current_time + 0.5) * 0.3, # left knee
               math.sin(current_time + 1.0) * 0.5, # right hip
               math.sin(current_time + 1.5) * 0.3  # right knee
           ]

           msg.velocity = [0.0] * len(msg.position)
           msg.effort = [0.0] * len(msg.position)

           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing joint states: {msg.position}')


   def main(args=None):
       rclpy.init(args=args)
       joint_state_publisher = JointStatePublisher()
       rclpy.spin(joint_state_publisher)
       joint_state_publisher.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. **Create the subscriber node:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/joint_subscriber.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState


   class JointStateSubscriber(Node):

       def __init__(self):
           super().__init__('joint_state_subscriber')
           self.subscription = self.create_subscription(
               JointState,
               'joint_states',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'Received joint states:')
           for i, name in enumerate(msg.name):
               self.get_logger().info(f'  {name}: pos={msg.position[i]:.2f}, vel={msg.velocity[i]:.2f}')


   def main(args=None):
       rclpy.init(args=args)
       joint_state_subscriber = JointStateSubscriber()
       rclpy.spin(joint_state_subscriber)
       joint_state_subscriber.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Update setup.py to add executables:**
   Add to the `entry_points` section in `setup.py`:
   ```python
   entry_points={
       'console_scripts': [
           'joint_publisher = humanoid_robot_tutorial.joint_publisher:main',
           'joint_subscriber = humanoid_robot_tutorial.joint_subscriber:main',
       ],
   },
   ```

4. **Build and test:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

5. **Run the publisher in one terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial joint_publisher
   ```

6. **Run the subscriber in another terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial joint_subscriber
   ```

### Expected Output
- Publisher terminal shows joint positions being published
- Subscriber terminal shows received joint states with positions and velocities
- Both nodes running without errors

### Troubleshooting
- Ensure both terminals have sourced the ROS 2 environment
- Check that the topic names match exactly
- Verify that the message types are compatible

## Lab 1C: Creating a Service Server and Client

### Objective
Implement a service-based interface for humanoid robot control.

### Steps

1. **Create a custom service definition:**
   Create directory and file `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/srv/SetJointPosition.srv`:
   ```
   string joint_name
   float64 position
   ---
   bool success
   string message
   ```

2. **Create the service server:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/joint_service_server.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import SetBool
   # For this example, we'll use a standard service since custom services need special setup


   class JointServiceServer(Node):

       def __init__(self):
           super().__init__('joint_service_server')
           self.srv = self.create_service(
               SetBool,
               'set_balance_mode',
               self.set_balance_mode_callback)
           self.balance_mode = False

       def set_balance_mode_callback(self, request, response):
           self.balance_mode = request.data
           response.success = True
           if request.data:
               response.message = 'Balance mode enabled'
               self.get_logger().info('Balance mode enabled')
           else:
               response.message = 'Balance mode disabled'
               self.get_logger().info('Balance mode disabled')
           return response


   def main(args=None):
       rclpy.init(args=args)
       joint_service_server = JointServiceServer()
       rclpy.spin(joint_service_server)
       joint_service_server.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

3. **Create the service client:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/joint_service_client.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from example_interfaces.srv import SetBool


   class JointServiceClient(Node):

       def __init__(self):
           super().__init__('joint_service_client')
           self.cli = self.create_client(SetBool, 'set_balance_mode')
           while not self.cli.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Service not available, waiting again...')
           self.req = SetBool.Request()

       def send_request(self, enable_balance):
           self.req.data = enable_balance
           self.future = self.cli.call_async(self.req)
           return self.future


   def main(args=None):
       rclpy.init(args=args)
       client = JointServiceClient()

       # Enable balance mode
       future = client.send_request(True)
       rclpy.spin_until_future_complete(client, future)
       response = future.result()
       client.get_logger().info(f'Response: success={response.success}, message={response.message}')

       # Disable balance mode after a delay
       import time
       time.sleep(2)
       future = client.send_request(False)
       rclpy.spin_until_future_complete(client, future)
       response = future.result()
       client.get_logger().info(f'Response: success={response.success}, message={response.message}')

       client.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

4. **Update setup.py:**
   Add to `entry_points`:
   ```python
   'console_scripts': [
       'joint_publisher = humanoid_robot_tutorial.joint_publisher:main',
       'joint_subscriber = humanoid_robot_tutorial.joint_subscriber:main',
       'joint_service_server = humanoid_robot_tutorial.joint_service_server:main',
       'joint_service_client = humanoid_robot_tutorial.joint_service_client:main',
   ],
   ```

5. **Build and test:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

6. **Run the server in one terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial joint_service_server
   ```

7. **Run the client in another terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial joint_service_client
   ```

### Expected Output
- Service server receives requests and responds appropriately
- Client receives responses from the server
- Balance mode state changes as requested

## Lab 1D: Creating an Action Server and Client

### Objective
Implement an action-based interface for humanoid robot walking behavior.

### Steps

1. **Create the action server:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/walking_action_server.py`:
   ```python
   import time
   import rclpy
   from rclpy.action import ActionServer, CancelResponse, GoalResponse
   from rclpy.node import Node
   from example_interfaces.action import Fibonacci  # Using standard action for simplicity


   class WalkingActionServer(Node):

       def __init__(self):
           super().__init__('walking_action_server')
           # Using Fibonacci as an example - in real humanoid robot you'd define custom action
           self._action_server = ActionServer(
               self,
               Fibonacci,
               'walk_to_goal',
               execute_callback=self.execute_callback,
               goal_callback=self.goal_callback,
               cancel_callback=self.cancel_callback)

       def goal_callback(self, goal_request):
           self.get_logger().info('Received walking goal request')
           return GoalResponse.ACCEPT

       def cancel_callback(self, goal_handle):
           self.get_logger().info('Received request to cancel walking goal')
           return CancelResponse.ACCEPT

       def execute_callback(self, goal_handle):
           self.get_logger().info('Executing walking goal...')

           feedback_msg = Fibonacci.Feedback()
           feedback_msg.sequence = [0, 1]

           for i in range(1, goal_handle.request.order):
               if goal_handle.is_cancel_requested:
                   goal_handle.canceled()
                   self.get_logger().info('Walking goal canceled')
                   return Fibonacci.Result()

               feedback_msg.sequence.append(
                   feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

               self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')
               goal_handle.publish_feedback(feedback_msg)

               time.sleep(0.5)  # Simulate walking time

           goal_handle.succeed()
           result = Fibonacci.Result()
           result.sequence = feedback_msg.sequence
           self.get_logger().info(f'Walking goal succeeded with result: {result.sequence}')

           return result


   def main(args=None):
       rclpy.init(args=args)
       walking_action_server = WalkingActionServer()
       rclpy.spin(walking_action_server)
       walking_action_server.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. **Create the action client:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/walking_action_client.py`:
   ```python
   import time
   import rclpy
   from rclpy.action import ActionClient
   from rclpy.node import Node
   from example_interfaces.action import Fibonacci


   class WalkingActionClient(Node):

       def __init__(self):
           super().__init__('walking_action_client')
           self._action_client = ActionClient(self, Fibonacci, 'walk_to_goal')

       def send_goal(self, order=5):
           goal_msg = Fibonacci.Goal()
           goal_msg.order = order

           self._action_client.wait_for_server()
           self._send_goal_future = self._action_client.send_goal_async(
               goal_msg,
               feedback_callback=self.feedback_callback)

           self._send_goal_future.add_done_callback(self.goal_response_callback)

       def goal_response_callback(self, future):
           goal_handle = future.result()
           if not goal_handle.accepted:
               self.get_logger().info('Goal rejected')
               return

           self.get_logger().info('Goal accepted')
           self._get_result_future = goal_handle.get_result_async()
           self._get_result_future.add_done_callback(self.get_result_callback)

       def feedback_callback(self, feedback_msg):
           self.get_logger().info(f'Received feedback: {feedback_msg.feedback.sequence}')

       def get_result_callback(self, future):
           result = future.result().result
           self.get_logger().info(f'Result: {result.sequence}')
           rclpy.shutdown()


   def main(args=None):
       rclpy.init(args=args)
       action_client = WalkingActionClient()
       action_client.send_goal(order=5)

       while rclpy.ok():
           rclpy.spin_once(action_client)

       action_client.destroy_node()


   if __name__ == '__main__':
       main()
   ```

3. **Update setup.py:**
   Add to `entry_points`:
   ```python
   'console_scripts': [
       'joint_publisher = humanoid_robot_tutorial.joint_publisher:main',
       'joint_subscriber = humanoid_robot_tutorial.joint_subscriber:main',
       'joint_service_server = humanoid_robot_tutorial.joint_service_server:main',
       'joint_service_client = humanoid_robot_tutorial.joint_service_client:main',
       'walking_action_server = humanoid_robot_tutorial.walking_action_server:main',
       'walking_action_client = humanoid_robot_tutorial.walking_action_client:main',
   ],
   ```

4. **Build and test:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

5. **Run the action server in one terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial walking_action_server
   ```

6. **Run the action client in another terminal:**
   ```bash
   ros2 run humanoid_robot_tutorial walking_action_client
   ```

### Expected Output
- Action server receives goal and provides feedback during execution
- Action client receives feedback and final result
- Goal can be canceled if needed

## Lab 1E: Creating Launch Files for Humanoid Systems

### Objective
Create launch files to coordinate multiple ROS 2 nodes for humanoid robot systems.

### Steps

1. **Create launch directory:**
   ```bash
   mkdir -p ~/ros2_ws/src/humanoid_robot_tutorial/launch
   ```

2. **Create a basic launch file:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/launch/humanoid_demo_launch.py`:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration


   def generate_launch_description():
       # Declare launch arguments
       use_sim_time = LaunchConfiguration('use_sim_time')

       return LaunchDescription([
           DeclareLaunchArgument(
               'use_sim_time',
               default_value='false',
               description='Use simulation (Gazebo) clock if true'
           ),

           # Joint state publisher
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_publisher',
               name='joint_state_publisher',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),

           # Joint state subscriber
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_subscriber',
               name='joint_state_subscriber',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),

           # Service server
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_service_server',
               name='balance_service_server',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),
       ])
   ```

3. **Create a more complex launch file with parameters:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/launch/humanoid_full_system_launch.py`:
   ```python
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument, TimerAction
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node


   def generate_launch_description():
       # Declare launch arguments
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
               default_value='$(find humanoid_robot_tutorial)/config/robot_config.yaml',
               description='Configuration file to load'
           ),

           # Robot state publisher
           Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               parameters=[
                   {'use_sim_time': use_sim_time},
                   # Note: You would load a URDF here in a real system
               ],
               output='screen'
           ),

           # Joint state publisher (simulated)
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_publisher',
               name='joint_state_publisher',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),

           # Main controller (simulated)
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_subscriber',
               name='controller',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),

           # Service server for balance control
           Node(
               package='humanoid_robot_tutorial',
               executable='joint_service_server',
               name='balance_service',
               parameters=[
                   {'use_sim_time': use_sim_time}
               ],
               output='screen'
           ),

           # Action server for walking (delayed start)
           TimerAction(
               period=2.0,
               actions=[
                   Node(
                       package='humanoid_robot_tutorial',
                       executable='walking_action_server',
                       name='walking_action_server',
                       parameters=[
                           {'use_sim_time': use_sim_time}
                       ],
                       output='screen'
                   )
               ]
           )
       ])
   ```

4. **Create a parameter file:**
   Create directory and file:
   ```bash
   mkdir -p ~/ros2_ws/src/humanoid_robot_tutorial/config
   ```

   Create `~/ros2_ws/src/humanoid_robot_tutorial/config/robot_config.yaml`:
   ```yaml
   /**:  # Applies to all nodes
     ros__parameters:
       use_sim_time: false
       log_level: 'info'

   joint_state_publisher:
     ros__parameters:
       publish_rate: 50.0
       joint_state_topic: 'joint_states'

   balance_service_server:
     ros__parameters:
       default_balance_mode: false
       balance_sensitivity: 0.1
   ```

5. **Build and test the launch files:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

6. **Run the basic launch file:**
   ```bash
   ros2 launch humanoid_robot_tutorial humanoid_demo_launch.py
   ```

7. **Run the full system launch file:**
   ```bash
   ros2 launch humanoid_robot_tutorial humanoid_full_system_launch.py
   ```

### Expected Output
- Launch files start multiple nodes simultaneously
- All nodes run with proper parameter configurations
- Nodes communicate with each other as designed
- No conflicts between node names or topics

## Lab 1F: Parameter Management for Humanoid Systems

### Objective
Learn to use parameters to configure humanoid robot behavior at runtime.

### Steps

1. **Create a parameter-based node:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/parameter_demo_node.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from rclpy.parameter import Parameter
   from rcl_interfaces.msg import ParameterDescriptor, ParameterType


   class ParameterDemoNode(Node):

       def __init__(self):
           super().__init__('parameter_demo_node')

           # Declare parameters with descriptions and constraints
           self.declare_parameter(
               'robot_name',
               'humanoid_robot',
               ParameterDescriptor(
                   description='Name of the humanoid robot',
                   type=ParameterType.PARAMETER_STRING
               )
           )

           self.declare_parameter(
               'walking_speed',
               0.5,
               ParameterDescriptor(
                   description='Walking speed in m/s',
                   type=ParameterType.PARAMETER_DOUBLE
               )
           )

           self.declare_parameter(
               'max_joint_effort',
               100.0,
               ParameterDescriptor(
                   description='Maximum joint effort in Nm',
                   type=ParameterType.PARAMETER_DOUBLE
               )
           )

           self.declare_parameter(
               'joints_to_control',
               ['left_hip', 'right_hip', 'left_knee', 'right_knee'],
               ParameterDescriptor(
                   description='List of joints to control',
                   type=ParameterType.PARAMETER_STRING_ARRAY
               )
           )

           # Get initial parameter values
           self.update_parameters()

           # Create a timer to periodically check for parameter changes
           self.timer = self.create_timer(1.0, self.check_parameters)

           # Add callback for parameter changes
           self.add_on_set_parameters_callback(self.parameters_callback)

       def update_parameters(self):
           self.robot_name = self.get_parameter('robot_name').value
           self.walking_speed = self.get_parameter('walking_speed').value
           self.max_joint_effort = self.get_parameter('max_joint_effort').value
           self.joints_to_control = self.get_parameter('joints_to_control').value

           self.get_logger().info(f'Updated parameters:')
           self.get_logger().info(f'  Robot name: {self.robot_name}')
           self.get_logger().info(f'  Walking speed: {self.walking_speed} m/s')
           self.get_logger().info(f'  Max joint effort: {self.max_joint_effort} Nm')
           self.get_logger().info(f'  Joints to control: {self.joints_to_control}')

       def check_parameters(self):
           # This method is called periodically to check for parameter changes
           current_walking_speed = self.get_parameter('walking_speed').value
           if current_walking_speed != self.walking_speed:
               self.get_logger().info(f'Walking speed changed from {self.walking_speed} to {current_walking_speed}')
               self.walking_speed = current_walking_speed

       def parameters_callback(self, parameters):
           for param in parameters:
               if param.name == 'walking_speed':
                   if param.value <= 0 or param.value > 2.0:
                       return rclpy.node.SetParametersResult(
                           successful=False,
                           reason='Walking speed must be between 0 and 2.0 m/s'
                       )
               elif param.name == 'max_joint_effort':
                   if param.value <= 0 or param.value > 500.0:
                       return rclpy.node.SetParametersResult(
                           successful=False,
                           reason='Max joint effort must be between 0 and 500.0 Nm'
                       )

           return rclpy.node.SetParametersResult(successful=True)


   def main(args=None):
       rclpy.init(args=args)
       node = ParameterDemoNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. **Update setup.py:**
   Add to `entry_points`:
   ```python
   'console_scripts': [
       'joint_publisher = humanoid_robot_tutorial.joint_publisher:main',
       'joint_subscriber = humanoid_robot_tutorial.joint_subscriber:main',
       'joint_service_server = humanoid_robot_tutorial.joint_service_server:main',
       'joint_service_client = humanoid_robot_tutorial.joint_service_client:main',
       'walking_action_server = humanoid_robot_tutorial.walking_action_server:main',
       'walking_action_client = humanoid_robot_tutorial.walking_action_client:main',
       'parameter_demo = humanoid_robot_tutorial.parameter_demo_node:main',
   ],
   ```

3. **Build and test:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

4. **Run the parameter demo node:**
   ```bash
   ros2 run humanoid_robot_tutorial parameter_demo
   ```

5. **Test parameter changes from another terminal:**
   ```bash
   # Change a parameter
   ros2 param set /parameter_demo walking_speed 1.0

   # List all parameters
   ros2 param list

   # Get a specific parameter
   ros2 param get /parameter_demo robot_name
   ```

### Expected Output
- Node starts with default parameter values
- Parameter changes are detected and logged
- Parameter validation prevents invalid values
- Parameters can be changed at runtime using ros2 param commands

## Lab 1G: Humanoid Robot State Machine

### Objective
Implement a state machine for humanoid robot behaviors using ROS 2.

### Steps

1. **Create a state machine node:**
   Create `~/ros2_ws/src/humanoid_robot_tutorial/humanoid_robot_tutorial/humanoid_state_machine.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from enum import Enum


   class HumanoidState(Enum):
       IDLE = "idle"
       WALKING = "walking"
       STANDING = "standing"
       SITTING = "sitting"
       EMERGENCY_STOP = "emergency_stop"


   class HumanoidStateMachine(Node):

       def __init__(self):
           super().__init__('humanoid_state_machine')

           # Initialize state
           self.current_state = HumanoidState.IDLE
           self.get_logger().info(f'Starting in state: {self.current_state.value}')

           # Publisher for state changes
           self.state_publisher = self.create_publisher(String, 'robot_state', 10)

           # Subscriber for commands
           self.command_subscriber = self.create_subscription(
               String,
               'robot_commands',
               self.command_callback,
               10
           )

           # Timer for state machine loop
           self.state_timer = self.create_timer(0.1, self.state_machine_loop)

           # Publish initial state
           self.publish_state()

       def command_callback(self, msg):
           command = msg.data.lower()
           self.get_logger().info(f'Received command: {command}')

           # Handle commands to change state
           if command == 'walk' and self.current_state == HumanoidState.STANDING:
               self.current_state = HumanoidState.WALKING
               self.get_logger().info('Transitioning to WALKING state')
           elif command == 'stop' and self.current_state == HumanoidState.WALKING:
               self.current_state = HumanoidState.STANDING
               self.get_logger().info('Transitioning to STANDING state')
           elif command == 'sit' and self.current_state == HumanoidState.STANDING:
               self.current_state = HumanoidState.SITTING
               self.get_logger().info('Transitioning to SITTING state')
           elif command == 'stand' and self.current_state == HumanoidState.SITTING:
               self.current_state = HumanoidState.STANDING
               self.get_logger().info('Transitioning to STANDING state')
           elif command == 'idle':
               self.current_state = HumanoidState.IDLE
               self.get_logger().info('Transitioning to IDLE state')
           elif command == 'emergency_stop':
               self.current_state = HumanoidState.EMERGENCY_STOP
               self.get_logger().info('Transitioning to EMERGENCY_STOP state')
           else:
               self.get_logger().warn(f'Invalid command "{command}" in state "{self.current_state.value}"')

           self.publish_state()

       def state_machine_loop(self):
           # Perform state-specific actions
           if self.current_state == HumanoidState.WALKING:
               # Walking-specific logic
               self.get_logger().debug('Walking state - monitoring progress')
           elif self.current_state == HumanoidState.STANDING:
               # Standing-specific logic
               self.get_logger().debug('Standing state - maintaining balance')
           elif self.current_state == HumanoidState.SITTING:
               # Sitting-specific logic
               self.get_logger().debug('Sitting state - monitoring stability')
           elif self.current_state == HumanoidState.EMERGENCY_STOP:
               # Emergency stop logic
               self.get_logger().warn('EMERGENCY STOP - all motion disabled')
           elif self.current_state == HumanoidState.IDLE:
               # Idle-specific logic
               self.get_logger().debug('Idle state - ready for commands')

           # Check for safety conditions that might force state changes
           self.check_safety_conditions()

       def check_safety_conditions(self):
           # In a real system, this would check sensor data
           # For this example, we'll just log the check
           pass

       def publish_state(self):
           msg = String()
           msg.data = self.current_state.value
           self.state_publisher.publish(msg)


   def main(args=None):
       rclpy.init(args=args)
       state_machine = HumanoidStateMachine()

       # Example: Send some commands
       import threading
       import time

       def send_test_commands():
           time.sleep(2)
           # Create a temporary node to send commands
           temp_node = rclpy.create_node('temp_commander')
           cmd_publisher = temp_node.create_publisher(String, 'robot_commands', 10)

           commands = ['stand', 'walk', 'stop', 'sit', 'stand', 'idle']
           for cmd in commands:
               msg = String()
               msg.data = cmd
               cmd_publisher.publish(msg)
               state_machine.get_logger().info(f'Sent command: {cmd}')
               time.sleep(3)

           temp_node.destroy_node()

       # Start command thread
       cmd_thread = threading.Thread(target=send_test_commands)
       cmd_thread.start()

       rclpy.spin(state_machine)
       cmd_thread.join()

       state_machine.destroy_node()
       rclpy.shutdown()


   if __name__ == '__main__':
       main()
   ```

2. **Update setup.py:**
   Add to `entry_points`:
   ```python
   'console_scripts': [
       'joint_publisher = humanoid_robot_tutorial.joint_publisher:main',
       'joint_subscriber = humanoid_robot_tutorial.joint_subscriber:main',
       'joint_service_server = humanoid_robot_tutorial.joint_service_server:main',
       'joint_service_client = humanoid_robot_tutorial.joint_service_client:main',
       'walking_action_server = humanoid_robot_tutorial.walking_action_server:main',
       'walking_action_client = humanoid_robot_tutorial.walking_action_client:main',
       'parameter_demo = humanoid_robot_tutorial.parameter_demo_node:main',
       'state_machine = humanoid_robot_tutorial.humanoid_state_machine:main',
   ],
   ```

3. **Build and test:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select humanoid_robot_tutorial
   source install/setup.bash
   ```

4. **Run the state machine:**
   ```bash
   ros2 run humanoid_robot_tutorial state_machine
   ```

5. **Send commands from another terminal:**
   ```bash
   # Send a command to make the robot stand
   ros2 topic pub /robot_commands std_msgs/String "data: 'stand'" --once

   # Send a command to make the robot walk
   ros2 topic pub /robot_commands std_msgs/String "data: 'walk'" --once
   ```

### Expected Output
- State machine starts in IDLE state
- Commands successfully change the robot state
- State transitions are logged
- Current state is published to the robot_state topic

## Lab Summary

In these labs, you've learned to:
1. Set up a ROS 2 workspace and create packages
2. Implement publisher/subscriber communication patterns
3. Create service-based interfaces for synchronous communication
4. Build action-based interfaces for goal-oriented behaviors
5. Coordinate multiple nodes using launch files
6. Manage configuration using parameters
7. Implement state machines for robot behavior control

These skills form the foundation for building complex humanoid robotics systems. Continue to practice these concepts and explore how they integrate with the other modules in this course.

## Next Steps

After completing these labs, you should be comfortable with:
- Creating and running ROS 2 nodes
- Implementing different communication patterns
- Using launch files and parameters
- Building robot control systems

Continue to [Module 1 Troubleshooting](./troubleshooting.md) to learn how to diagnose and fix common issues you might encounter when working with ROS 2 in humanoid robotics applications.