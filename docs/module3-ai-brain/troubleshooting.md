---
title: Module 3 Troubleshooting
sidebar_position: 9
---

# Module 3 Troubleshooting: AI-Robot Brain Implementation

This troubleshooting guide addresses common issues encountered when implementing AI-robot brain systems using Isaac Sim, Isaac ROS, VSLAM, and navigation stacks for humanoid robotics.

## Isaac Sim Troubleshooting

### Installation Issues

**Problem**: Isaac Sim fails to launch or crashes immediately

**Symptoms**:
- Isaac Sim exits with error code
- Black screen or no rendering
- CUDA/OpenGL errors

**Diagnosis**:
```bash
# Check system requirements
nvidia-smi
glxinfo | grep -i nvidia
free -h
df -h

# Check Isaac Sim logs
cat ~/.nvidia-isaac/logs/*
```

**Solutions**:
1. **Verify GPU Compatibility**:
   ```bash
   # Check if GPU supports required compute capability
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

   # Verify CUDA installation
   nvcc --version
   ```

2. **Driver Issues**:
   ```bash
   # Update NVIDIA drivers
   sudo apt update
   sudo ubuntu-drivers autoinstall

   # Or install specific driver
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

3. **Permissions and Dependencies**:
   ```bash
   # Install required dependencies
   sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

   # Check for missing libraries
   ldd IsaacSim/Isaac-Sim-Windows-x86_64/IsaacSim.sh
   ```

### Rendering Issues

**Problem**: Poor performance or visual artifacts in Isaac Sim

**Symptoms**:
- Low frame rate (< 30 FPS)
- Visual glitches or missing textures
- Lighting issues

**Solutions**:
1. **Graphics Settings**:
   ```bash
   # Launch with reduced graphics quality
   ./isaac-sim-gui.sh -- --graphics=low

   # Or disable rendering for headless operation
   ./isaac-sim-headless.sh
   ```

2. **VRAM Issues**:
   ```bash
   # Monitor GPU memory usage
   watch -n 1 nvidia-smi

   # Reduce scene complexity in simulation
   # Lower texture resolutions
   # Reduce number of active lights
   ```

3. **Display Configuration**:
   ```bash
   # For remote desktop/SSH connections
   export __GLX_VENDOR_LIBRARY_NAME=nvidia
   export MESA_GL_VERSION_OVERRIDE=4.6
   ```

### Physics Simulation Issues

**Problem**: Robot behaves unrealistically in simulation

**Symptoms**:
- Robot falls through floors
- Unstable joint behavior
- Incorrect mass/inertia properties

**Solutions**:
1. **Check Physics Parameters**:
   ```python
   # Verify mass and inertia in URDF/USD
   import omni
   from pxr import Gf, UsdPhysics, PhysxSchema

   # Example: Set proper mass and inertia
   def configure_rigid_body(prim_path, mass, principal_axes, moments_of_inertia):
       rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(prim_path))
       physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(prim_path))

       # Set mass
       mass_api = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(prim_path))
       mass_api.massAttr = mass

       # Set inertia
       mass_api.centerOfMassAttr = Gf.Vec3f(0, 0, 0)  # Center of mass
       mass_api.diagonalInertiaAttr = Gf.Vec3f(moments_of_inertia[0],
                                               moments_of_inertia[1],
                                               moments_of_inertia[2])
   ```

2. **Friction and Contact Parameters**:
   ```python
   # Configure friction and contact properties
   def configure_friction(prim_path, static_friction, dynamic_friction):
       friction_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(prim_path))
       friction_api.staticFrictionAttr = static_friction
       friction_api.dynamicFrictionAttr = dynamic_friction
   ```

## Isaac ROS Integration Issues

### ROS Bridge Connection Problems

**Problem**: Isaac Sim and ROS 2 cannot communicate

**Symptoms**:
- No topics published from Isaac Sim
- ROS nodes cannot control simulation
- Clock synchronization issues

**Diagnosis**:
```bash
# Check ROS 2 setup
source /opt/ros/humble/setup.bash
echo $ROS_DISTRO
echo $ROS_DOMAIN_ID

# Verify topics
ros2 topic list
ros2 node list
```

**Solutions**:
1. **Verify Isaac ROS Installation**:
   ```bash
   # Check if Isaac ROS packages are installed
   apt list --installed | grep isaac-ros

   # Install missing packages
   sudo apt update
   sudo apt install ros-humble-isaac-ros-common
   sudo apt install ros-humble-isaac-ros-perception
   sudo apt install ros-humble-isaac-ros-navigation
   ```

2. **Network Configuration**:
   ```bash
   # Check network settings
   hostname
   echo $ROS_HOSTNAME
   echo $ROS_IP

   # For localhost communication
   export ROS_HOSTNAME=localhost
   export ROS_IP=127.0.0.1
   export ROS_DOMAIN_ID=0
   ```

3. **Isaac Sim Extension Activation**:
   ```bash
   # In Isaac Sim UI, ensure extensions are enabled:
   # Window > Extensions > Isaac ROS Bridge > Enable
   ```

### Sensor Data Issues

**Problem**: Sensor data from Isaac Sim is incorrect or missing

**Symptoms**:
- Empty sensor topics
- Incorrect sensor readings
- High latency in sensor data

**Solutions**:
1. **Check Sensor Configuration**:
   ```python
   # Verify sensor configuration in USD
   import omni
   from pxr import UsdLux, UsdGeom

   # Example: Verify camera configuration
   def verify_camera_configuration(camera_prim_path):
       camera_prim = stage.GetPrimAtPath(camera_prim_path)
       if not camera_prim.IsValid():
           print(f"Camera prim {camera_prim_path} not found!")
           return False

       # Check camera properties
       camera = UsdGeom.Camera(camera_prim)
       if camera.GetHorizontalApertureAttr().Get() == 0:
           print("Camera aperture not configured!")
           return False

       return True
   ```

2. **Sensor Update Rates**:
   ```yaml
   # In sensor configuration, verify update rates
   camera_config:
     update_rate: 30.0  # Hz
     resolution: [640, 480]
     fov: 1.047  # radians (60 degrees)

   lidar_config:
     update_rate: 10.0  # Hz (typical for LiDAR)
     samples: 720  # Horizontal samples
     range: [0.1, 25.0]  # Min/max range in meters
   ```

3. **TF Frame Issues**:
   ```bash
   # Check TF tree
   ros2 run tf2_tools view_frames

   # Echo TF transforms
   ros2 run tf2_ros tf2_echo map base_link
   ```

## VSLAM Troubleshooting

### Tracking Failures

**Problem**: VSLAM system loses tracking frequently

**Symptoms**:
- Frequent relocalization
- Drifting pose estimates
- Featureless environments causing failures

**Diagnosis**:
```bash
# Monitor VSLAM performance
ros2 topic echo /visual_slam/tracking_accuracy
ros2 topic echo /visual_slam/feature_count

# Check image quality
ros2 run image_view image_view _image:=/camera/image_raw
```

**Solutions**:
1. **Feature Detection Tuning**:
   ```yaml
   # VSLAM configuration for better feature detection
   visual_slam:
     ros__parameters:
       # Increase feature count for better tracking
       num_features: 2000  # Default is often 1000

       # Adjust feature detection thresholds
       min_feature_threshold: 20  # Lower for feature-poor environments
       max_track_length: 100  # Increase for longer feature tracks

       # Enable dynamic feature management
       enable_dynamic_feature_management: true
       feature_retention_strategy: "keep_best"
   ```

2. **Motion Model Issues**:
   ```python
   # Ensure proper motion model for humanoid movement
   class HumanoidMotionModel:
       def __init__(self):
           # Humanoid-specific motion constraints
           self.max_linear_velocity = 0.5  # m/s (walking speed)
           self.max_angular_velocity = 0.6  # rad/s (turning speed)
           self.motion_uncertainty = 0.1    # Higher for dynamic movement

       def predict_motion(self, dt):
           # Humanoid-specific motion prediction
           # Consider bipedal dynamics and balance constraints
           pass
   ```

3. **Illumination Sensitivity**:
   ```python
   # Improve illumination robustness
   def enhance_image_contrast(image):
       # Apply histogram equalization or CLAHE
       import cv2
       lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
       l_channel, a, b = cv2.split(lab)

       # Apply CLAHE to L-channel
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
       cl = clahe.apply(l_channel)

       # Merge channels and convert back to BGR
       limg = cv2.merge((cl,a,b))
       enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

       return enhanced_image
   ```

### Mapping Issues

**Problem**: Poor map quality or inconsistent mapping

**Symptoms**:
- Inaccurate maps
- Duplicate structures
- Scale drift over time

**Solutions**:
1. **Loop Closure Configuration**:
   ```yaml
   # Loop closure settings for better map consistency
   visual_slam:
     ros__parameters:
       # Loop closure parameters
       enable_loop_closure: true
       loop_closure_minimum_similarity: 0.7  # Similarity threshold
       loop_closure_inlier_threshold: 0.1    # Geometric consistency
       loop_closure_maximum_distance: 10.0   # Max distance for loop closure

       # Bundle adjustment settings
       enable_bundle_adjustment: true
       bundle_adjustment_window_size: 50     # Frames in optimization window
       bundle_adjustment_max_iterations: 100 # Max optimization iterations
   ```

2. **Scale Estimation**:
   ```python
   # Ensure proper scale estimation for humanoid navigation
   class ScaleEstimator:
       def __init__(self):
           self.scale_factor = 1.0
           self.reference_height = 1.7  # Average humanoid height

       def estimate_scale_from_humanoid(self, detected_person):
           """Estimate scale based on known humanoid dimensions"""
           if detected_person.height_pixels > 0:
               # Calculate scale from known height vs detected height
               scale = self.reference_height / detected_person.real_height_meters
               self.update_scale_factor(scale)
   ```

## Navigation Stack Issues

### Path Planning Problems

**Problem**: Nav2 fails to find valid paths or plans unrealistic paths

**Symptoms**:
- "No valid path found" errors
- Robot taking very long routes
- Path planning fails in known navigable areas

**Diagnosis**:
```bash
# Check costmaps
ros2 run rviz2 rviz2  # Visualize costmaps
ros2 topic echo /global_costmap/costmap
ros2 topic echo /local_costmap/costmap

# Check planner status
ros2 service list | grep plan
ros2 action list | grep compute_path
```

**Solutions**:
1. **Costmap Configuration**:
   ```yaml
   # Costmap configuration for humanoid robots
   global_costmap:
     ros__parameters:
       # Increase robot footprint for humanoid
       footprint: [[-0.3, -0.3], [-0.3, 0.3], [0.3, 0.3], [0.3, -0.3]]  # 60cm square (larger for humanoid)
       footprint_padding: 0.1  # Extra safety margin

       # Increase inflation radius for humanoid safety
       inflation_layer:
         inflation_radius: 0.8  # Larger than default 0.55
         cost_scaling_factor: 3.0  # Higher scaling for humanoid safety

   local_costmap:
     ros__parameters:
       # Local costmap for humanoid navigation
       footprint: [[-0.25, -0.25], [-0.25, 0.25], [0.4, 0.25], [0.4, -0.25]]  # Rectangular footprint for forward movement

       # Increase local costmap size for humanoid turning radius
       width: 8.0  # Larger than default 10.0
       height: 8.0  # Larger than default 10.0
       resolution: 0.05  # Keep high resolution for precision
   ```

2. **Planner Configuration**:
   ```yaml
   # Planner configuration for humanoid mobility
   planner_server:
     ros__parameters:
       planner_plugins: ["GridBased"]
       GridBased:
         plugin: "nav2_navfn_planner/NavfnPlanner"
         # Increase tolerance for humanoid navigation
         tolerance: 0.5  # Larger tolerance for humanoid (was 0.5)
         use_astar: true  # Use A* for better path quality
         allow_unknown: true  # Allow planning through unknown areas with penalty
   ```

3. **Controller Configuration**:
   ```yaml
   # Controller configuration for humanoid stability
   controller_server:
     ros__parameters:
       controller_frequency: 20.0  # Higher frequency for stability
       min_x_velocity_threshold: 0.001
       min_y_velocity_threshold: 0.001
       min_theta_velocity_threshold: 0.001

       controller_plugins: ["FollowPath"]

       FollowPath:
         plugin: "nav2_mppi_controller::MPPICtrl"
         # Humanoid-specific parameters
         max_vel_x: 0.3  # Reduced for stability (was 0.5)
         max_vel_theta: 0.4  # Slower turning for balance (was 0.6)

         # Increase safety margins
         xy_goal_tolerance: 0.3  # Larger for humanoid (was 0.25)
         yaw_goal_tolerance: 0.3  # Larger for humanoid (was 0.25)

         # Humanoid-specific control parameters
         sim_time: 2.0  # Longer prediction horizon for stability
         vx_samples: 15  # More samples for smoother control
         vtheta_samples: 20  # More angular samples
   ```

### Localization Issues

**Problem**: Robot loses localization or has inaccurate pose estimates

**Symptoms**:
- Robot pose jumps randomly
- High covariance in pose estimates
- AMCL fails to converge

**Solutions**:
1. **AMCL Configuration**:
   ```yaml
   # AMCL configuration for humanoid environments
   amcl:
     ros__parameters:
       # Increase particle count for complex humanoid environments
       min_particles: 1000  # Increase from default 500
       max_particles: 3000  # Increase from default 2000

       # Adjust for humanoid sensor configuration
       alpha1: 0.1  # Lower for more stable motion model (was 0.2)
       alpha2: 0.1  # Lower for more stable motion model (was 0.2)
       alpha3: 0.1  # Lower for more stable motion model (was 0.2)
       alpha4: 0.1  # Lower for more stable motion model (was 0.2)
       alpha5: 0.1  # Additional parameter for 3D motion

       # Increase update frequency for humanoid dynamics
       update_min_d: 0.1  # Distance threshold for updates (was 0.25)
       update_min_a: 0.1  # Angle threshold for updates (was 0.2)

       # Adjust for humanoid height and perspective
       laser_z_hit: 0.4  # Higher weight for valid readings (was 0.5)
       laser_z_short: 0.1  # Weight for short readings (was 0.05)
       laser_z_max: 0.05  # Weight for max range readings (was 0.05)
       laser_z_rand: 0.5  # Weight for random readings (was 0.5)
   ```

2. **Initial Pose Estimation**:
   ```python
   # Better initial pose estimation for humanoid robots
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseWithCovarianceStamped
   from std_msgs.msg import String
   import numpy as np

   class EnhancedInitialPoseEstimator(Node):
       def __init__(self):
           super().__init__('enhanced_initial_pose_estimator')

           self.initial_pose_pub = self.create_publisher(
               PoseWithCovarianceStamped, 'initialpose', 10
           )

           self.environment_analyzer = self.create_subscription(
               String, 'environment_analysis', self.analyze_environment, 10
           )

       def analyze_environment(self, msg):
           """Analyze environment to provide better initial pose estimate"""
           environment_data = eval(msg.data)  # In practice, use proper serialization

           # Estimate position based on environment features
           estimated_pose = self.estimate_pose_from_features(environment_data)

           # Publish with reduced covariance based on feature confidence
           self.publish_refined_initial_pose(estimated_pose)

       def estimate_pose_from_features(self, env_data):
           """Estimate pose based on environmental features"""
           # Use known landmarks, hallway geometry, door positions, etc.
           # to estimate more accurate initial pose
           pass
   ```

## GPU and Performance Issues

### CUDA/OpenGL Errors

**Problem**: GPU acceleration fails or causes crashes

**Symptoms**:
- CUDA errors during execution
- OpenGL context errors
- GPU memory allocation failures

**Diagnosis**:
```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version
nvidia-ml-py3 --version

# Monitor GPU memory usage
watch -n 0.5 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
```

**Solutions**:
1. **CUDA Environment Setup**:
   ```bash
   # Ensure proper CUDA environment
   export CUDA_HOME=/usr/local/cuda
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
   export PATH=$PATH:$CUDA_HOME/bin

   # Verify CUDA works
   nvidia-smi
   nvcc -V
   ```

2. **Memory Management**:
   ```python
   # Proper GPU memory management
   import torch
   import gc

   def optimize_gpu_memory():
       """Optimize GPU memory usage"""
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           gc.collect()

           # Set memory fraction if needed
           torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

       # Monitor memory usage
       if torch.cuda.is_available():
           print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
           print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
   ```

3. **Isaac Sim GPU Configuration**:
   ```bash
   # Launch Isaac Sim with specific GPU settings
   export ISAAC_SIM_DISABLE_GPU_FRUSTUM_CULLING=1
   export OMNI_DEBUG_USE_GPU=1
   export CUDA_VISIBLE_DEVICES=0

   ./isaac-sim-gui.sh
   ```

### Performance Optimization

**Problem**: AI-robot brain runs too slowly

**Symptoms**:
- Low frame rates
- High latency in responses
- CPU/GPU bottlenecks

**Solutions**:
1. **Profiling and Optimization**:
   ```python
   # Performance profiling for AI-robot brain
   import cProfile
   import pstats
   import time
   from functools import wraps

   def profile_function(func):
       """Decorator to profile function performance"""
       @wraps(func)
       def wrapper(*args, **kwargs):
           pr = cProfile.Profile()
           pr.enable()
           result = func(*args, **kwargs)
           pr.disable()

           # Sort and print stats
           stats = pstats.Stats(pr)
           stats.sort_stats('cumulative')
           stats.print_stats(10)  # Print top 10 functions

           return result
       return wrapper

   class PerformanceMonitor:
       def __init__(self):
           self.timings = {}
           self.counts = {}

       @profile_function
       def measure_function_performance(self, func_name, func, *args, **kwargs):
           """Measure execution time of a function"""
           start_time = time.time()
           result = func(*args, **kwargs)
           end_time = time.time()

           elapsed = end_time - start_time

           if func_name not in self.timings:
               self.timings[func_name] = []
               self.counts[func_name] = 0

           self.timings[func_name].append(elapsed)
           self.counts[func_name] += 1

           return result

       def get_performance_report(self):
           """Generate performance report"""
           report = {}
           for func_name, times in self.timings.items():
               report[func_name] = {
                   'avg_time': sum(times) / len(times),
                   'min_time': min(times),
                   'max_time': max(times),
                   'call_count': self.counts[func_name],
                   'total_time': sum(times)
               }
           return report
   ```

2. **Threading and Parallelization**:
   ```python
   # Optimized threading for humanoid AI-robot brain
   import threading
   import queue
   import time
   from concurrent.futures import ThreadPoolExecutor, as_completed

   class OptimizedBrainPipeline:
       def __init__(self):
           self.perception_queue = queue.Queue(maxsize=5)
           self.planning_queue = queue.Queue(maxsize=5)
           self.control_queue = queue.Queue(maxsize=5)

           # Thread pools for different subsystems
           self.perception_executor = ThreadPoolExecutor(max_workers=2)
           self.planning_executor = ThreadPoolExecutor(max_workers=2)
           self.control_executor = ThreadPoolExecutor(max_workers=1)

           # Start processing threads
           self.start_pipeline_threads()

       def start_pipeline_threads(self):
           """Start pipeline processing threads"""
           threading.Thread(target=self.perception_worker, daemon=True).start()
           threading.Thread(target=self.planning_worker, daemon=True).start()
           threading.Thread(target=self.control_worker, daemon=True).start()

       def perception_worker(self):
           """Worker for perception processing"""
           while True:
               try:
                   sensor_data = self.perception_queue.get(timeout=1.0)

                   # Process perception
                   processed_data = self.process_perception(sensor_data)

                   # Add to planning queue
                   try:
                       self.planning_queue.put_nowait(processed_data)
                   except queue.Full:
                       print("Planning queue full, dropping frame")

                   self.perception_queue.task_done()
               except queue.Empty:
                   continue

       def planning_worker(self):
           """Worker for planning processing"""
           while True:
               try:
                   perception_data = self.planning_queue.get(timeout=1.0)

                   # Process planning
                   plan = self.generate_plan(perception_data)

                   # Add to control queue
                   try:
                       self.control_queue.put_nowait(plan)
                   except queue.Full:
                       print("Control queue full, dropping plan")

                   self.planning_queue.task_done()
               except queue.Empty:
                   continue
   ```

## Humanoid-Specific Issues

### Balance and Stability

**Problem**: Humanoid robot exhibits unstable behavior during navigation

**Symptoms**:
- Robot falls over during movement
- Unstable walking patterns
- Balance controller conflicts with navigation

**Solutions**:
1. **Balance Controller Integration**:
   ```python
   # Balance-aware navigation controller
   import numpy as np
   from scipy import interpolate

   class BalanceAwareController:
       def __init__(self):
           self.balance_controller = self.initialize_balance_controller()
           self.com_estimator = CenterOfMassEstimator()
           self.ik_solver = InverseKinematicsSolver()

       def compute_stable_velocities(self, desired_twist):
           """Compute velocities that maintain balance"""
           # Get current state
           current_com = self.com_estimator.get_current_com()
           current_support_polygon = self.get_support_polygon()

           # Check if desired motion maintains balance
           if self.will_lose_balance(desired_twist, current_com, current_support_polygon):
               # Compute safe velocities that maintain balance
               safe_twist = self.compute_balance_preserving_velocities(
                   desired_twist, current_com, current_support_polygon
               )
               return safe_twist
           else:
               return desired_twist

       def will_lose_balance(self, twist, com, support_polygon):
           """Predict if motion will cause balance loss"""
           # Use inverted pendulum model or ZMP (Zero Moment Point) analysis
           predicted_com_position = self.predict_com_position(twist)
           return not self.is_com_in_support_polygon(predicted_com_position, support_polygon)

       def compute_balance_preserving_velocities(self, desired_twist, com, support_polygon):
           """Compute velocities that keep CoM in support polygon"""
           # Optimize velocities to stay close to desired while maintaining balance
           # This is a simplified approach - real implementation would be more complex
           pass
   ```

2. **Step Planning for Bipedal Locomotion**:
   ```python
   class StepPlanner:
       def __init__(self):
           self.step_length = 0.3  # meters
           self.step_width = 0.2   # meters (stance width)
           self.step_height = 0.1  # meters (foot clearance)

       def plan_next_step(self, robot_state, desired_velocity):
           """Plan next step considering balance and terrain"""
           # Calculate next step position based on desired velocity
           next_step_pos = self.calculate_step_position(
               robot_state, desired_velocity, self.step_length
           )

           # Verify step is safe (no obstacles, stable terrain)
           if self.is_step_safe(next_step_pos):
               return next_step_pos
           else:
               # Find alternative safe step
               alternative_step = self.find_alternative_safe_step(
                   next_step_pos, robot_state
               )
               return alternative_step

       def calculate_step_position(self, robot_state, desired_velocity, step_length):
           """Calculate where to place next foot"""
           # Based on current state and desired velocity
           # Consider robot's current stance, momentum, and balance
           pass
   ```

## Common Error Messages and Solutions

### Isaac Sim Errors

**Error**: `CUDA error: no kernel image is available for execution on the device`

**Cause**: GPU compute capability not supported or CUDA compilation issue

**Solution**:
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Ensure Isaac Sim was compiled for your GPU
# Download appropriate version or recompile with correct compute capability
```

**Error**: `Failed to initialize graphics context`

**Solution**:
```bash
# Try different graphics settings
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MESA_GL_VERSION_OVERRIDE=4.6

# Or run headless
./isaac-sim-headless.sh
```

### ROS 2 Connection Errors

**Error**: `Unable to load plugin libnav2_amcl/amcl: Cannot locate path`

**Solution**:
```bash
# Source ROS environment
source /opt/ros/humble/setup.bash
source install/setup.bash  # If using workspace

# Check if package is installed
ros2 pkg list | grep nav2

# Verify library path
echo $LD_LIBRARY_PATH
```

### Python Import Errors

**Error**: `ImportError: No module named 'omni.isaac.core'`

**Solution**:
```bash
# Ensure Isaac Sim Python path is set
export PYTHONPATH=$ISAAC_SIM_PATH/python:$PYTHONPATH

# Or activate Isaac Sim Python environment
cd $ISAAC_SIM_PATH
python3 -c "import omni.isaac.core"
```

## Debugging Tools and Techniques

### Isaac Sim Debugging

```python
# Isaac Sim debugging utilities
import omni
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom

class IsaacSimDebugger:
    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()

    def inspect_robot_state(self, robot_prim_path):
        """Inspect robot state in simulation"""
        robot_prim = self.stage.GetPrimAtPath(robot_prim_path)

        if not robot_prim.IsValid():
            print(f"Robot prim {robot_prim_path} not found!")
            return

        # Get all children prims (links, joints, etc.)
        for child in robot_prim.GetAllChildren():
            print(f"Child: {child.GetName()}, Type: {child.GetTypeName()}")

            # If it's a joint, inspect joint properties
            if "joint" in child.GetTypeName().lower():
                self.inspect_joint(child)

    def inspect_joint(self, joint_prim):
        """Inspect joint properties"""
        # Print joint properties
        pass

    def visualize_sensors(self, sensor_paths):
        """Visualize sensor fields of view"""
        for path in sensor_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                # Visualize sensor range, FOV, etc.
                pass
```

### ROS 2 Debugging

```bash
# ROS 2 debugging commands
ros2 doctor  # Check ROS 2 system health
ros2 run tf2_tools view_frames  # View TF tree
ros2 run rqt_graph rqt_graph  # Visualize node graph
ros2 bag record -a  # Record all topics for analysis
ros2 lifecycle list  # Check lifecycle node states
```

## Performance Monitoring

### System Resource Monitoring

```python
# Create resource_monitor.py
import psutil
import GPUtil
import time
import json
from datetime import datetime

class SystemResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.data_log = []

    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.monitoring_start_time = time.time()

        while self.monitoring:
            current_data = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }

            # Add GPU data if available
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                current_data.update({
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_percent': gpu.memoryUtil * 100,
                    'gpu_temperature': gpu.temperature
                })

            self.data_log.append(current_data)
            time.sleep(0.1)  # Monitor every 100ms

    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.monitoring = False
        self.save_monitoring_data()

    def save_monitoring_data(self):
        """Save monitoring data to file"""
        filename = f"system_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        print(f"Monitoring data saved to {filename}")

    def analyze_performance(self):
        """Analyze collected performance data"""
        if not self.data_log:
            print("No monitoring data available")
            return

        # Calculate averages
        cpu_avg = sum(d['cpu_percent'] for d in self.data_log) / len(self.data_log)
        mem_avg = sum(d['memory_percent'] for d in self.data_log) / len(self.data_log)

        print(f"Average CPU usage: {cpu_avg:.2f}%")
        print(f"Average memory usage: {mem_avg:.2f}%")

        # Check for bottlenecks
        high_cpu_periods = [d for d in self.data_log if d['cpu_percent'] > 80]
        high_mem_periods = [d for d in self.data_log if d['memory_percent'] > 80]

        print(f"High CPU periods (>80%): {len(high_cpu_periods)}")
        print(f"High memory periods (>80%): {len(high_mem_periods)}")
```

## Troubleshooting Checklist

### Pre-Deployment Checklist

- [ ] Isaac Sim launches without errors
- [ ] All required Isaac ROS packages are installed
- [ ] GPU drivers and CUDA are properly configured
- [ ] ROS 2 environment is sourced
- [ ] Network configuration allows ROS communication
- [ ] Robot URDF/USD model loads correctly
- [ ] All sensors publish data as expected
- [ ] Navigation stack parameters are tuned for humanoid
- [ ] Safety limits and constraints are configured
- [ ] Emergency stop mechanisms are in place

### Runtime Issue Diagnosis

When facing runtime issues:

1. **Check System Resources**
   - CPU and memory usage
   - GPU utilization and temperature
   - Disk space availability

2. **Verify Communications**
   - ROS 2 topics and services are active
   - TF tree is complete and consistent
   - Sensor data is flowing correctly

3. **Examine Logs**
   - Isaac Sim logs in `~/.nvidia-isaac/logs/`
   - ROS 2 node logs with `ros2 topic echo` and `ros2 service call`
   - System logs with `journalctl` or `dmesg`

4. **Validate Parameters**
   - Ensure all required parameters are set
   - Check parameter values are reasonable
   - Verify configuration files are properly loaded

5. **Test Components Individually**
   - Test perception pipeline separately
   - Test planning without execution
   - Test control without planning
   - Gradually integrate components

## Getting Help

### When to Seek Help

- Issues persist after trying troubleshooting steps
- Complex performance problems requiring optimization
- Hardware-specific problems with drivers or compatibility
- Integration issues between different software components
- Safety concerns with real robot deployment

### Resources

- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/latest/
- **Isaac ROS Documentation**: https://nvidia-isaac-ros.github.io/
- **ROS 2 Documentation**: https://docs.ros.org/
- **Isaac Sim Forums**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/sim/
- **ROS Answers**: https://answers.ros.org/

### Creating Effective Bug Reports

When reporting issues:

1. **Environment Details**: ROS 2 distribution, Isaac Sim version, GPU model, OS version
2. **Steps to Reproduce**: Clear sequence of actions that cause the issue
3. **Expected vs Actual**: What should happen vs what actually happens
4. **Error Messages**: Complete error output and logs
5. **Configuration Files**: Relevant launch files, parameter files, URDF
6. **Hardware Details**: Specifics about robot hardware and sensors

## Next Steps

After resolving issues with your AI-robot brain implementation, continue to [Module 4: Vision-Language-Action (VLA) & Conversational Robotics](../module4-vla/index.md) to learn about integrating vision, language, and action systems for conversational humanoid robots.