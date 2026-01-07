---
title: Module 1 Troubleshooting
sidebar_position: 9
---

# Module 1 Troubleshooting: ROS 2 for Humanoid Robotics

This troubleshooting guide covers common issues you may encounter when working with ROS 2 in the context of humanoid robotics. Each section provides diagnostic steps and solutions for specific problems.

## General ROS 2 Issues

### Environment Not Sourced

**Problem**: Getting "command not found" errors for ROS 2 commands.

**Symptoms**:
- `ros2` command not found
- `colcon` command not found
- `ament` command not found

**Diagnosis**:
```bash
# Check if ROS 2 environment is sourced
env | grep -i ros
```

**Solutions**:
1. **Source ROS 2 environment**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Add to your shell profile** (for permanent solution):
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify installation**:
   ```bash
   # Check if ROS 2 is installed
   ls /opt/ros/humble/

   # Check if setup.bash exists
   ls /opt/ros/humble/setup.bash
   ```

### Workspace Not Built

**Problem**: Nodes fail to run with "executable not found" errors.

**Symptoms**:
- `ros2 run package_name node_name` fails
- Executable not found errors
- Package not found errors

**Solutions**:
1. **Build the workspace**:
   ```bash
   cd ~/ros2_ws
   colcon build
   ```

2. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

3. **Build specific package only**:
   ```bash
   colcon build --packages-select package_name
   source install/setup.bash
   ```

4. **Clean build if needed**:
   ```bash
   rm -rf build/ install/ log/
   colcon build
   source install/setup.bash
   ```

## Communication Issues

### Topic Not Found

**Problem**: Publisher and subscriber nodes cannot communicate.

**Symptoms**:
- Publisher shows "Publishing" but subscriber shows nothing
- Topic not listed in `ros2 topic list`
- No data exchange between nodes

**Diagnosis**:
```bash
# Check topic list
ros2 topic list

# Check specific topic info
ros2 topic info /topic_name

# Check for matching topic names
ros2 topic list | grep topic_name
```

**Solutions**:
1. **Verify topic names match exactly**:
   - Check spelling and case sensitivity
   - Ensure no extra spaces or characters
   - Verify namespaces if used

2. **Check Quality of Service (QoS) settings**:
   ```python
   # Ensure publisher and subscriber have compatible QoS
   from rclpy.qos import QoSProfile

   # Both should use compatible settings
   qos = QoSProfile(depth=10)
   publisher = node.create_publisher(MsgType, 'topic_name', qos)
   subscriber = node.create_subscription(MsgType, 'topic_name', callback, qos)
   ```

3. **Wait for connections**:
   ```python
   # In publisher, wait for subscriber
   while publisher.get_subscription_count() == 0:
       print("Waiting for subscriber...")
       time.sleep(0.1)
   ```

### Service Not Available

**Problem**: Service client cannot connect to service server.

**Symptoms**:
- Client waits indefinitely for service
- "Service not available" messages
- `ros2 service list` doesn't show the service

**Solutions**:
1. **Ensure service server is running**:
   ```bash
   # Check if service exists
   ros2 service list

   # Check if your service is listed
   ros2 service list | grep service_name
   ```

2. **Add proper service waiting in client**:
   ```python
   # In your client node
   while not client.wait_for_service(timeout_sec=1.0):
       node.get_logger().info('Service not available, waiting again...')
   ```

3. **Verify service type matches**:
   - Check that client and server use the same service type
   - Verify service name is identical

## Node Issues

### Node Not Starting

**Problem**: Node fails to start or crashes immediately.

**Symptoms**:
- Node exits with error code
- Import errors in Python
- Missing dependencies

**Diagnosis**:
```bash
# Run with verbose output
ros2 run package_name node_name --ros-args -r __log_level:=DEBUG

# Check for import errors
python3 -c "import rclpy"  # Should not produce errors
```

**Solutions**:
1. **Check Python imports**:
   ```python
   # Ensure all imports are correct
   import rclpy
   from rclpy.node import Node
   # Add other necessary imports
   ```

2. **Verify package.xml dependencies**:
   ```xml
   <depend>rclpy</depend>
   <depend>std_msgs</depend>
   <depend>sensor_msgs</depend>
   ```

3. **Check setup.py console scripts**:
   ```python
   entry_points={
       'console_scripts': [
           'node_name = package_name.file_name:main',
       ],
   },
   ```

### Node Crashes During Execution

**Problem**: Node runs initially but crashes during operation.

**Symptoms**:
- Node stops unexpectedly
- Segmentation faults
- Memory errors
- Unhandled exceptions

**Solutions**:
1. **Add proper error handling**:
   ```python
   def callback(self, msg):
       try:
           # Process message
           self.process_data(msg)
       except Exception as e:
           self.get_logger().error(f'Error processing message: {e}')
   ```

2. **Check for memory issues**:
   ```python
   # Limit message queue size
   subscriber = self.create_subscription(MsgType, 'topic', callback, 10)
   ```

3. **Monitor resource usage**:
   ```bash
   # Check memory usage
   htop

   # Check for memory leaks
   valgrind --tool=memcheck ros2 run package_name node_name
   ```

## Launch File Issues

### Launch File Not Working

**Problem**: Launch file fails to start or nodes don't run as expected.

**Symptoms**:
- Launch file exits with errors
- Some nodes don't start
- Parameter loading failures

**Diagnosis**:
```bash
# Run launch file with verbose output
ros2 launch package_name launch_file.py --log-level debug

# Check launch file syntax
ros2 launch --dry-run package_name launch_file.py
```

**Solutions**:
1. **Verify Python syntax in launch file**:
   ```bash
   python3 -m py_compile launch_file.py
   ```

2. **Check for missing dependencies**:
   - Ensure all packages in launch file are installed
   - Verify executable names match setup.py

3. **Add proper error handling in launch file**:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration

   def generate_launch_description():
       return LaunchDescription([
           # Your nodes here
       ])
   ```

### Parameter Issues in Launch Files

**Problem**: Parameters not loading correctly from launch files.

**Symptoms**:
- Default parameter values used instead of launch file values
- Parameter validation errors
- Nodes not responding to parameter changes

**Solutions**:
1. **Verify parameter file path**:
   ```python
   # Use proper path joining
   from launch.substitutions import PathJoinSubstitution
   from launch_ros.substitutions import FindPackageShare

   params_file = PathJoinSubstitution([
       FindPackageShare('package_name'),
       'config',
       'params.yaml'
   ])
   ```

2. **Check YAML parameter file format**:
   ```yaml
   node_name:
     ros__parameters:
       param_name: param_value
   ```

3. **Use proper parameter declaration in nodes**:
   ```python
   def __init__(self):
       super().__init__('node_name')
       self.declare_parameter('param_name', default_value)
   ```

## Humanoid-Specific Issues

### Joint State Issues

**Problem**: Joint states not publishing or updating correctly.

**Symptoms**:
- Joint states topic shows no data
- RViz not showing robot model moving
- Joint state publisher not working

**Solutions**:
1. **Verify joint names match URDF**:
   ```python
   # Joint names in message must match URDF
   msg.name = ['joint1', 'joint2', 'joint3']  # Must match URDF exactly
   ```

2. **Check joint state message structure**:
   ```python
   from sensor_msgs.msg import JointState
   import math

   msg = JointState()
   msg.name = joint_names
   msg.position = joint_positions  # Length must match names
   msg.velocity = joint_velocities  # Optional, but if provided, length must match
   msg.effort = joint_efforts      # Optional, but if provided, length must match
   ```

3. **Verify robot state publisher**:
   ```bash
   # Check if robot description is loaded
   ros2 param list | grep robot_description

   # Check TF frames
   ros2 run tf2_tools view_frames
   ```

### URDF Loading Issues

**Problem**: Robot model not displaying correctly in RViz.

**Symptoms**:
- Robot model not showing in RViz
- TF tree incomplete
- Joint state visualization problems

**Solutions**:
1. **Validate URDF file**:
   ```bash
   # Check URDF syntax
   check_urdf /path/to/robot.urdf

   # Visualize URDF
   urdf_to_graphiz /path/to/robot.urdf
   ```

2. **Verify robot state publisher setup**:
   ```bash
   # Run robot state publisher
   ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat robot.urdf)'
   ```

3. **Check joint state publisher**:
   ```bash
   # Ensure joint states are being published
   ros2 topic echo /joint_states
   ```

## Performance Issues

### High CPU Usage

**Problem**: ROS 2 nodes consuming excessive CPU resources.

**Symptoms**:
- High CPU usage shown in system monitor
- Nodes running slower than expected
- System becoming unresponsive

**Solutions**:
1. **Optimize timer rates**:
   ```python
   # Don't use very high frequency timers unnecessarily
   # For visualization: 30Hz is usually sufficient
   # For control: Match hardware capabilities
   self.timer = self.create_timer(0.033, callback)  # ~30Hz
   ```

2. **Reduce message publishing rate**:
   ```python
   # Don't publish more than needed
   # Use latching for static data
   # Use compression for large messages
   ```

3. **Profile node performance**:
   ```bash
   # Use ROS 2 tools to profile
   ros2 run plotjuggler plotjuggler

   # Or use system tools
   htop
   ```

### Memory Leaks

**Problem**: Memory usage growing over time.

**Symptoms**:
- Increasing memory usage over time
- System running out of memory
- Nodes becoming slower

**Solutions**:
1. **Check for message accumulation**:
   ```python
   # Limit queue sizes
   subscriber = self.create_subscription(MsgType, 'topic', callback, 10)

   # Clear old data structures periodically
   if len(self.data_buffer) > MAX_SIZE:
       self.data_buffer = self.data_buffer[-MAX_SIZE:]
   ```

2. **Properly destroy nodes and publishers**:
   ```python
   def destroy_node(self):
       # Clean up resources
       if self.publisher:
           self.publisher.destroy()
       if self.subscriber:
           self.subscriber.destroy()
       super().destroy_node()
   ```

## Network and Multi-Machine Issues

### Multi-Machine Communication

**Problem**: Nodes on different machines cannot communicate.

**Symptoms**:
- Nodes on different machines cannot see each other
- Topics/services not shared across machines
- Network timeouts

**Solutions**:
1. **Set proper ROS_DOMAIN_ID**:
   ```bash
   # Ensure all machines use the same domain ID
   export ROS_DOMAIN_ID=0
   ```

2. **Configure network settings**:
   ```bash
   # Set ROS IP if needed
   export ROS_IP=192.168.1.100  # Your machine's IP
   export ROS_HOSTNAME=your-hostname
   ```

3. **Check firewall settings**:
   ```bash
   # Ensure DDS ports are open (usually 11811 and others)
   sudo ufw allow 11811
   ```

### Discovery Issues

**Problem**: Nodes cannot discover each other on the same machine.

**Symptoms**:
- Nodes running on same machine cannot communicate
- `ros2 node list` doesn't show all nodes
- Topic/service discovery fails

**Solutions**:
1. **Check for multiple ROS distributions**:
   ```bash
   # Ensure only one ROS distribution is sourced
   env | grep -i ros
   ```

2. **Verify DDS configuration**:
   ```bash
   # Check which DDS implementation is being used
   printenv | grep -i dds
   ```

3. **Restart daemon if needed**:
   ```bash
   # Kill and restart ROS 2 daemon
   killall -9 ros2
   ros2 daemon stop
   ros2 daemon start
   ```

## Debugging Tools and Techniques

### ROS 2 Command Line Tools

**Useful debugging commands**:
```bash
# List all nodes
ros2 node list

# Get info about a specific node
ros2 node info node_name

# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo /topic_name

# List all services
ros2 service list

# Call a service
ros2 service call /service_name service_type "{request: data}"

# List all parameters
ros2 param list

# Get a parameter value
ros2 param get /node_name param_name
```

### Logging and Debugging

**Effective logging in nodes**:
```python
def __init__(self):
    super().__init__('node_name')
    self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

def some_method(self):
    self.get_logger().debug('Debug message')
    self.get_logger().info('Info message')
    self.get_logger().warn('Warning message')
    self.get_logger().error('Error message')
    self.get_logger().fatal('Fatal message')
```

### Using rqt and RViz for Debugging

**Common debugging setups**:
```bash
# Launch rqt with useful plugins
rqt

# Launch RViz for visualization
rviz2

# Use plotjuggler for plotting
ros2 run plotjuggler plotjuggler
```

## Common Humanoid Robotics Patterns

### Safety System Issues

**Problem**: Safety systems triggering unexpectedly or not working.

**Solutions**:
1. **Implement proper safety monitoring**:
   ```python
   class SafetyMonitor(Node):
       def __init__(self):
           super().__init__('safety_monitor')
           self.safety_timer = self.create_timer(0.01, self.check_safety)  # 100Hz

       def check_safety(self):
           # Check joint limits, velocities, efforts
           # Check balance stability
           # Check for hardware faults
           pass
   ```

2. **Ensure proper emergency stop handling**:
   ```python
   def emergency_stop(self):
       # Stop all motion
       # Set joints to safe positions
       # Log the event
       pass
   ```

### Real-time Performance Issues

**Problem**: Control loops not meeting timing requirements.

**Solutions**:
1. **Optimize control loop timing**:
   ```python
   # Use appropriate control frequencies
   # 1kHz for low-level control
   # 100Hz for high-level control
   # 10Hz for planning
   self.control_timer = self.create_timer(0.001, self.control_callback)  # 1kHz
   ```

2. **Minimize computation in control loops**:
   ```python
   # Pre-compute values outside the loop
   # Use efficient algorithms
   # Avoid memory allocation in loops
   ```

## Preventive Measures

### Code Quality Practices

1. **Use proper error handling**:
   ```python
   try:
       result = self.process_data(data)
   except ValueError as e:
       self.get_logger().error(f'Invalid data: {e}')
       return
   except Exception as e:
       self.get_logger().error(f'Unexpected error: {e}')
       return
   ```

2. **Implement proper node lifecycle**:
   ```python
   def destroy_node(self):
       # Clean up resources
       if hasattr(self, 'publisher'):
           self.publisher.destroy()
       if hasattr(self, 'timer'):
           self.timer.destroy()
       super().destroy_node()
   ```

3. **Validate inputs**:
   ```python
   def set_joint_positions(self, positions):
       if len(positions) != self.num_joints:
           raise ValueError(f'Expected {self.num_joints} positions, got {len(positions)}')
       # Validate joint limits
       for pos in positions:
           if pos < self.min_limit or pos > self.max_limit:
               raise ValueError(f'Joint position {pos} out of limits')
   ```

### Testing and Validation

1. **Test nodes individually**:
   ```bash
   # Test each node separately before integration
   ros2 run package_name test_node
   ```

2. **Use simulation for testing**:
   - Test control algorithms in simulation first
   - Verify safety systems in simulation
   - Validate parameter configurations

3. **Monitor system health**:
   ```bash
   # Monitor CPU usage
   htop

   # Monitor memory usage
   free -h

   # Monitor network
   iftop
   ```

## Getting Help

### When to Seek Help

- Issues persist after trying troubleshooting steps
- Problems with ROS 2 installation or setup
- Complex integration issues
- Performance problems affecting safety

### Resources

- **ROS 2 Documentation**: https://docs.ros.org/
- **ROS Answers**: https://answers.ros.org/
- **ROS Discourse**: https://discourse.ros.org/
- **GitHub Issues**: For specific packages
- **Community Forums**: Local ROS user groups

### Creating Good Bug Reports

When reporting issues, include:
1. ROS 2 distribution and version
2. Ubuntu version
3. Steps to reproduce
4. Expected vs. actual behavior
5. Error messages and logs
6. Relevant code snippets

## Next Steps

After troubleshooting issues, continue to:
- [Module 2: The Digital Twin (Gazebo & Unity)](/docs/module2-digital-twin/index) to learn about simulation
- [Module 3: The AI-Robot Brain (NVIDIA Isaac)](/docs/module3-ai-brain/index) for perception and AI
- [Module 4: Vision-Language-Action (VLA) & Conversational Robotics](/docs/module4-vla/index) for advanced interaction

Remember that troubleshooting is a normal part of ROS 2 development. The key is to systematically identify, diagnose, and resolve issues while maintaining safety in humanoid robotics applications.