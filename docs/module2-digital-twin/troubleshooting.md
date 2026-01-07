---
title: Module 2 Troubleshooting
sidebar_position: 9
---

# Module 2 Troubleshooting: Digital Twin and Simulation

This troubleshooting guide covers common issues you may encounter when working with Gazebo, Unity, and other simulation environments in the context of humanoid robotics. Each section provides diagnostic steps and solutions for specific problems.

## General Simulation Issues

### Gazebo Installation and Setup

**Problem**: Gazebo fails to launch or crashes immediately.

**Symptoms**:
- `gz sim` command fails
- Gazebo GUI crashes on startup
- No graphics rendering

**Diagnosis**:
```bash
# Check Gazebo installation
gz --version

# Check graphics drivers
glxinfo | grep -i opengl

# Check for missing dependencies
ldd $(which gz)
```

**Solutions**:
1. **Install missing dependencies**:
   ```bash
   sudo apt update
   sudo apt install nvidia-prime nvidia-driver-470 # For NVIDIA GPUs
   sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
   ```

2. **Fix OpenGL issues**:
   ```bash
   # Try software rendering if hardware acceleration fails
   export LIBGL_ALWAYS_SOFTWARE=1
   gz sim
   ```

3. **Check GPU drivers**:
   ```bash
   # Verify GPU is detected
   lspci | grep -i vga

   # Check if GPU driver is loaded
   nvidia-smi  # For NVIDIA
   glxinfo | grep -i renderer
   ```

### ROS 2 and Gazebo Integration

**Problem**: ROS 2 nodes can't communicate with Gazebo simulation.

**Symptoms**:
- No sensor data from simulation
- Robot commands don't affect simulation
- Topics not published by Gazebo plugins

**Diagnosis**:
```bash
# Check if Gazebo ROS bridge is running
ps aux | grep gz

# Check ROS 2 topics
ros2 topic list

# Verify network configuration
echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
```

**Solutions**:
1. **Verify Gazebo ROS plugins are loaded**:
   ```bash
   # Check if gazebo_ros_pkgs is installed
   ros2 pkg list | grep gazebo
   ```

2. **Check plugin paths**:
   ```bash
   # Verify plugin paths are set
   echo $GAZEBO_PLUGIN_PATH
   echo $GZ_SIM_SYSTEM_PLUGIN_PATH

   # Add to your .bashrc if missing
   export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH
   export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH
   ```

3. **Test basic communication**:
   ```bash
   # Publish a test message to verify ROS 2 is working
   ros2 topic pub /test std_msgs/String "data: 'test'"
   ```

## Physics Simulation Issues

### Robot Falling Through Floor

**Problem**: Robot falls through the ground plane in simulation.

**Symptoms**:
- Robot disappears below the ground
- Physics collision not working properly
- Robot doesn't stay on surface

**Solutions**:
1. **Check collision geometry**:
   ```xml
   <!-- Ensure collision geometry is properly defined -->
   <collision name="collision">
     <geometry>
       <cylinder>
         <radius>0.1</radius>
         <length>0.2</length>
       </cylinder>
     </geometry>
   </collision>
   ```

2. **Verify ground plane definition**:
   ```xml
   <!-- Ground plane should be static -->
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
     </link>
   </model>
   ```

3. **Adjust physics parameters**:
   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_factor>1</real_time_factor>
     <ode>
       <solver>
         <iters>100</iters>  <!-- Increase solver iterations -->
       </solver>
       <constraints>
         <cfm>1e-5</cfm>      <!-- Reduce constraint force mixing -->
         <erp>0.2</erp>       <!-- Increase error reduction parameter -->
       </constraints>
     </ode>
   </physics>
   ```

### Robot Instability and Jittering

**Problem**: Robot joints are unstable or jitter during simulation.

**Symptoms**:
- Robot wobbles uncontrollably
- Joint positions oscillate
- Unstable walking behavior

**Solutions**:
1. **Reduce time step**:
   ```xml
   <physics type="ode">
     <max_step_size>0.0005</max_step_size>  <!-- Smaller time step -->
   </physics>
   ```

2. **Increase solver iterations**:
   ```xml
   <physics type="ode">
     <ode>
       <solver>
         <iters>200</iters>  <!-- More iterations for stability -->
       </solver>
     </ode>
   </physics>
   ```

3. **Adjust joint parameters**:
   ```xml
   <joint name="joint_name" type="revolute">
     <parent link="parent_link"/>
     <child link="child_link"/>
     <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     <dynamics damping="0.5" friction="0.1"/>  <!-- Add damping -->
   </joint>
   ```

### Joint Limit Violations

**Problem**: Robot joints exceed their defined limits.

**Symptoms**:
- Joints move beyond specified limits
- Robot behaves unexpectedly
- Joint positions outside expected range

**Solutions**:
1. **Verify joint limits in URDF/SDF**:
   ```xml
   <joint name="limited_joint" type="revolute">
     <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
   </joint>
   ```

2. **Add soft limits**:
   ```xml
   <joint name="soft_limited_joint" type="revolute">
     <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
     <safety_controller k_position="10" k_velocity="10" soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
   </joint>
   ```

## Sensor Simulation Issues

### Camera Not Publishing Data

**Problem**: Camera sensor in simulation doesn't publish images.

**Symptoms**:
- No images on `/camera/image_raw` topic
- Camera topic doesn't appear in `ros2 topic list`
- RViz shows no camera feed

**Diagnosis**:
```bash
# Check if camera topic exists
ros2 topic list | grep camera

# Check topic info
ros2 topic info /camera/image_raw

# Monitor topic for activity
ros2 topic echo /camera/image_raw --field header.stamp
```

**Solutions**:
1. **Verify sensor definition**:
   ```xml
   <sensor name="camera" type="camera">
     <camera name="head_camera">
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
   ```

2. **Check Gazebo plugins**:
   ```bash
   # Verify sensor plugin is loaded
   gz plugin list | grep camera
   ```

3. **Test sensor in isolation**:
   ```bash
   # Launch simple world with just camera
   gz sim -r simple_camera.sdf
   ```

### LiDAR Range Issues

**Problem**: LiDAR sensor returns unexpected range values.

**Symptoms**:
- All ranges show maximum value
- No obstacles detected
- Unexpected minimum/maximum ranges

**Solutions**:
1. **Verify range parameters**:
   ```xml
   <sensor name="lidar" type="ray">
     <ray>
       <range>
         <min>0.1</min>
         <max>30.0</max>
         <resolution>0.01</resolution>
       </range>
     </ray>
   </sensor>
   ```

2. **Check for ray intersection issues**:
   - Ensure LiDAR is positioned correctly
   - Verify no obstructions in sensor view
   - Check collision geometry of objects in environment

3. **Validate point cloud data**:
   ```bash
   # Monitor LiDAR data
   ros2 topic echo /scan --field ranges --field angle_min --field angle_max
   ```

### IMU Data Problems

**Problem**: IMU sensor produces incorrect or noisy data.

**Symptoms**:
- Acceleration values don't match expected gravity
- Angular velocity constantly drifting
- Orientation quaternion invalid

**Solutions**:
1. **Check IMU placement**:
   ```xml
   <!-- IMU should be placed on the body whose orientation you want to measure -->
   <sensor name="imu" type="imu">
     <pose>0 0 0 0 0 0</pose>  <!-- Center of the link -->
   </sensor>
   ```

2. **Verify noise parameters**:
   ```xml
   <sensor name="realistic_imu" type="imu">
     <imu>
       <angular_velocity>
         <x>
           <noise type="gaussian">
             <stddev>2e-4</stddev>
           </noise>
         </x>
       </angular_velocity>
       <linear_acceleration>
         <z>
           <noise type="gaussian">
             <stddev>1.7e-2</stddev>
           </noise>
         </z>
       </linear_acceleration>
     </imu>
   </sensor>
   ```

## Performance Issues

### Slow Simulation Speed

**Problem**: Simulation runs significantly slower than real-time.

**Symptoms**:
- Real-time factor much less than 1.0
- High CPU usage
- Choppiness in visualization

**Diagnosis**:
```bash
# Check real-time factor in Gazebo
# Look at the bottom of the Gazebo GUI or check the console output

# Monitor system resources
htop
nvidia-smi  # For GPU monitoring
```

**Solutions**:
1. **Reduce visual complexity**:
   ```bash
   # Launch with reduced rendering
   gz sim -v 0  # Minimal visualization
   gz sim -v 2  # Reduced quality
   ```

2. **Optimize physics parameters**:
   ```xml
   <physics type="ode">
     <max_step_size>0.002</max_step_size>  <!-- Larger time step -->
     <ode>
       <solver>
         <iters>50</iters>  <!-- Fewer iterations -->
       </solver>
     </ode>
   </physics>
   ```

3. **Simplify robot models**:
   - Use simpler collision geometry
   - Reduce number of joints if possible
   - Lower sensor update rates

### High Memory Usage

**Problem**: Simulation consumes excessive memory.

**Symptoms**:
- Memory usage grows over time
- System becomes sluggish
- Out of memory errors

**Solutions**:
1. **Reduce sensor update rates**:
   ```xml
   <sensor name="slow_camera" type="camera">
     <update_rate>10</update_rate>  <!-- Lower rate -->
   </sensor>
   ```

2. **Limit sensor resolution**:
   ```xml
   <sensor name="low_res_lidar" type="ray">
     <ray>
       <scan>
         <horizontal>
           <samples>360</samples>  <!-- Half the samples -->
         </horizontal>
       </scan>
     </ray>
   </sensor>
   ```

3. **Monitor and clean up resources**:
   ```bash
   # Check memory usage
   free -h

   # Restart simulation if needed
   killall -9 gz
   ```

## Humanoid-Specific Issues

### Balance and Walking Problems

**Problem**: Humanoid robot cannot maintain balance in simulation.

**Symptoms**:
- Robot falls over immediately
- Unstable walking pattern
- Balance control fails

**Solutions**:
1. **Verify center of mass**:
   ```xml
   <link name="torso">
     <inertial>
       <mass value="5.0"/>
       <origin xyz="0 0 0.25"/>  <!-- CoM at 25cm height -->
       <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
     </inertial>
   </link>
   ```

2. **Check physics parameters for stability**:
   ```xml
   <physics type="ode">
     <max_step_size>0.0005</max_step_size>  <!-- Very small for stability -->
     <ode>
       <solver>
         <iters>150</iters>  <!-- More iterations -->
       </solver>
       <constraints>
         <cfm>1e-6</cfm>      <!-- Very low constraint mixing -->
         <erp>0.1</erp>       <!-- Lower error reduction -->
       </constraints>
     </ode>
   </physics>
   ```

3. **Verify foot contact properties**:
   ```xml
   <link name="left_foot">
     <collision name="foot_collision">
       <geometry>
         <box>
           <size>0.2 0.1 0.05</size>
         </box>
       </geometry>
       <surface>
         <friction>
           <ode>
             <mu>1.0</mu>    <!-- High friction for stability -->
             <mu2>1.0</mu2>
           </ode>
         </friction>
         <contact>
           <ode>
             <kp>1e6</kp>    <!-- High contact stiffness -->
             <kd>100</kd>    <!-- Appropriate damping -->
           </ode>
         </contact>
       </surface>
     </collision>
   </link>
   ```

### Joint Control Issues

**Problem**: Robot joints don't respond to control commands.

**Symptoms**:
- Joint states don't update
- Control commands ignored
- Robot stays in initial position

**Solutions**:
1. **Verify joint type matches control method**:
   - Use `revolute` for position/velocity control
   - Use `continuous` for wheels
   - Use `prismatic` for linear joints

2. **Check joint limits and effort**:
   ```xml
   <joint name="controlled_joint" type="revolute">
     <limit lower="-1.57" upper="1.57" effort="200" velocity="2"/>  <!-- Adequate effort -->
   </joint>
   ```

3. **Verify controller configuration**:
   ```yaml
   # controller_manager.yaml
   controller_manager:
     ros__parameters:
       update_rate: 1000  # Match Gazebo physics rate

   position_trajectory_controller:
     ros__parameters:
       joints:
         - joint1
         - joint2
   ```

## Unity Simulation Issues (if applicable)

### Unity Robotics Package Problems

**Problem**: Unity fails to connect to ROS 2.

**Symptoms**:
- No communication between Unity and ROS 2
- Sensor data not published from Unity
- Robot commands not received by Unity

**Solutions**:
1. **Check Unity ROS TCP connector**:
   - Verify ROS TCP Connector package is imported
   - Check connection settings in Unity scene
   - Ensure ROS master is running

2. **Network configuration**:
   ```bash
   # Check network settings
   echo $ROS_IP
   echo $ROS_HOSTNAME

   # Set if needed
   export ROS_IP=127.0.0.1
   export ROS_HOSTNAME=localhost
   ```

3. **Firewall settings**:
   ```bash
   # Allow Unity ROS communication
   sudo ufw allow 10000:65535/tcp
   ```

## Common Error Messages and Solutions

### "Failed to load Gazebo ROS plugin"

**Error**: `Failed to load Gazebo ROS plugin`

**Cause**: Plugin path not set or plugin missing

**Solution**:
```bash
# Set plugin paths
export GAZEBO_PLUGIN_PATH=/opt/ros/humble/lib:$GAZEBO_PLUGIN_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH

# Verify plugin exists
ls /opt/ros/humble/lib/libgazebo_ros_init.so
```

### "No matching device found" for sensors

**Error**: Sensor reports "No matching device found"

**Cause**: Sensor configuration mismatch

**Solution**: Check sensor definition matches expected ROS message type and frame names.

### "URDF is not valid" error

**Error**: `URDF is not valid` when loading robot model

**Solution**: Validate URDF with:
```bash
check_urdf /path/to/robot.urdf
```

## Diagnostic Tools and Techniques

### Gazebo Tools

**Useful Gazebo commands**:
```bash
# List all models in simulation
gz model -m

# Get model info
gz model -m model_name -i

# List all topics
gz topic -l

# Echo a topic
gz topic -e -t /topic_name
```

### ROS 2 Debugging

**Useful ROS 2 commands**:
```bash
# List all nodes
ros2 node list

# Check node info
ros2 node info node_name

# List topics
ros2 topic list

# Echo topic data
ros2 topic echo /topic_name

# Check service list
ros2 service list

# Check parameters
ros2 param list
```

### Physics Debugging

**Enable physics debugging**:
```xml
<!-- Add to your world file -->
<physics type="ode">
  <debug>true</debug>  <!-- Enable physics debugging -->
</physics>
```

## Performance Monitoring

### System Resources
```bash
# Monitor CPU and memory
htop

# Monitor GPU (if applicable)
nvidia-smi

# Monitor network
iftop
```

### Simulation Performance
```bash
# Monitor real-time factor in Gazebo GUI
# Check physics update statistics
gz stats
```

## Best Practices for Simulation Stability

### 1. Physics Parameter Tuning
- Start with conservative parameters
- Gradually optimize for performance
- Balance accuracy with speed

### 2. Model Complexity Management
- Use simplified collision geometry
- Reduce visual complexity for performance
- Optimize joint configurations

### 3. Sensor Configuration
- Use appropriate update rates
- Validate sensor noise models
- Test sensor placement early

### 4. Testing and Validation
- Test simple models before complex ones
- Validate physics behavior against expectations
- Compare simulation vs real robot data when possible

## Common Humanoid Robotics Simulation Patterns

### Balance Control Validation
```python
def validate_balance_control(sim_robot):
    """Validate balance control in simulation"""
    # Check if robot maintains upright position
    # Verify CoM stays within support polygon
    # Test response to disturbances
    pass
```

### Walking Pattern Testing
```python
def test_walking_pattern(sim_robot):
    """Test walking pattern in simulation"""
    # Verify step timing and coordination
    # Check for stable gait
    # Validate foot placement
    pass
```

## Getting Help

### When to Seek Help

- Issues persist after trying troubleshooting steps
- Complex physics problems affecting safety
- Performance issues preventing development
- Integration problems between simulation and real systems

### Resources

- **Gazebo Documentation**: https://gazebosim.org/
- **ROS 2 Documentation**: https://docs.ros.org/
- **Unity Robotics**: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- **Gazebo Answers**: https://answers.gazebosim.org/
- **ROS Answers**: https://answers.ros.org/

### Creating Good Bug Reports

When reporting simulation issues:

1. **Environment**: ROS 2 distribution, Gazebo version, Ubuntu version
2. **Steps to Reproduce**: Clear sequence of actions
3. **Expected vs Actual**: What should happen vs what happens
4. **Error Messages**: Complete error output
5. **Configuration Files**: Relevant URDF/SDF/launch files
6. **Logs**: Console output and log files

## Next Steps

After troubleshooting simulation issues, continue to:
- [Module 3: The AI-Robot Brain (NVIDIA Isaac)](/docs/module3-ai-brain/index) to learn about perception and navigation
- [Module 4: Vision-Language-Action (VLA) & Conversational Robotics](/docs/module4-vla/index) for advanced interaction
- [Capstone: Autonomous Humanoid](/docs/capstone/index) to integrate all concepts

Remember that simulation is a tool for development, and real-world validation is always necessary for humanoid robotics applications. The goal is to minimize the sim-to-real gap while maintaining safety during development.