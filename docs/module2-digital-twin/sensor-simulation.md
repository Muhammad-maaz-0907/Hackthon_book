---
title: Sensor Simulation
sidebar_position: 6
---

# Sensor Simulation

Sensors are critical components in robotics, and accurate simulation of sensor data is essential for effective humanoid robotics development. This lesson covers how to simulate various sensor types in Gazebo and how to use them effectively in humanoid robotics applications.

## Introduction to Sensor Simulation

Sensor simulation in Gazebo involves modeling the physical properties, noise characteristics, and environmental interactions of real sensors. For humanoid robots, accurate sensor simulation is crucial for:

- **Perception**: Object detection, recognition, and localization
- **Navigation**: Path planning and obstacle avoidance
- **Balance**: Posture control and fall prevention
- **Interaction**: Human-robot interaction and manipulation

### Key Aspects of Sensor Simulation

1. **Geometric Properties**: Field of view, resolution, range
2. **Physical Properties**: Noise models, latency, drift
3. **Environmental Interactions**: Occlusion, reflections, interference
4. **Integration**: ROS 2 message types and interfaces

## Types of Sensors in Robotics

### Camera Sensors

Camera sensors simulate visual perception and are essential for humanoid robots:

#### Basic Camera Configuration

```xml
<sensor name="camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="head_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### Depth Camera Configuration

```xml
<sensor name="depth_camera" type="depth">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="depth_cam">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>320</width>
      <height>240</height>
      <format>L8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>5</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### RGB-D Camera Configuration

```xml
<sensor name="rgbd_camera" type="rgbd">
  <pose>0.1 0 0.1 0 0 0</pose>
  <camera name="rgbd_cam">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <depth_camera>
      <image>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </depth_camera>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LiDAR Sensors

LiDAR sensors provide 2D or 3D distance measurements and are crucial for navigation and mapping:

#### 2D LiDAR Configuration

```xml
<sensor name="laser_2d" type="ray">
  <pose>0.15 0 0.2 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle> <!-- -180 degrees -->
        <max_angle>3.14159</max_angle>   <!-- 180 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

#### 3D LiDAR Configuration

```xml
<sensor name="laser_3d" type="ray">
  <pose>0.2 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>1024</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
      <vertical>
        <samples>64</samples>
        <resolution>1</resolution>
        <min_angle>-0.5236</min_angle> <!-- -30 degrees -->
        <max_angle>0.3491</max_angle>  <!-- 20 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensors

Inertial Measurement Units (IMUs) are critical for balance and orientation in humanoid robots:

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
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### Force/Torque Sensors

Force/torque sensors are important for manipulation and contact detection:

```xml
<sensor name="ft_sensor" type="force_torque">
  <pose>0 0 0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
</sensor>
```

### GPS Sensors

For outdoor humanoid robots, GPS sensors provide location information:

```xml
<sensor name="gps" type="gps">
  <pose>0 0 1.0 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>1</update_rate>
  <gps>
    <position_sensing>
      <horizontal>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.1</stddev>
        </noise>
      </horizontal>
      <vertical>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.15</stddev>
        </noise>
      </vertical>
    </position_sensing>
  </gps>
</sensor>
```

## Sensor Placement for Humanoid Robots

### Head-Mounted Sensors

Head-mounted sensors provide the humanoid's "vision" of the environment:

```xml
<!-- Head-mounted camera for perception -->
<sensor name="head_camera" type="camera">
  <pose>0.05 0 0.05 0 0 0</pose> <!-- Position in head -->
  <camera name="head_cam">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
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

<!-- Head-mounted IMU for orientation -->
<sensor name="head_imu" type="imu">
  <pose>0 0 0.05 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
</sensor>
```

### Body-Mounted Sensors

Body-mounted sensors provide information about the robot's state:

```xml
<!-- Torso IMU for balance control -->
<sensor name="torso_imu" type="imu">
  <pose>0 0 0.2 0 0 0</pose> <!-- Center of torso -->
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x><noise type="gaussian"><stddev>2e-4</stddev></noise></x>
      <y><noise type="gaussian"><stddev>2e-4</stddev></noise></y>
      <z><noise type="gaussian"><stddev>2e-4</stddev></noise></z>
    </angular_velocity>
    <linear_acceleration>
      <x><noise type="gaussian"><stddev>1.7e-2</stddev></noise></x>
      <y><noise type="gaussian"><stddev>1.7e-2</stddev></noise></y>
      <z><noise type="gaussian"><stddev>1.7e-2</stddev></noise></z>
    </linear_acceleration>
  </imu>
</sensor>

<!-- Chest-mounted LiDAR for navigation -->
<sensor name="chest_lidar" type="ray">
  <pose>0.1 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Foot-Mounted Sensors

Foot-mounted sensors are crucial for walking and balance control:

```xml
<!-- Left foot force/torque sensor -->
<sensor name="left_foot_ft" type="force_torque">
  <pose>0 0 -0.025 0 0 0</pose> <!-- Center of foot -->
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
</sensor>

<!-- Right foot force/torque sensor -->
<sensor name="right_foot_ft" type="force_torque">
  <pose>0 0 -0.025 0 0 0</pose>
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
</sensor>
```

## Sensor Noise and Realism

### Adding Realistic Noise

Realistic sensor noise is crucial for humanoid robot simulation:

```xml
<!-- Camera with realistic noise -->
<sensor name="noisy_camera" type="camera">
  <camera name="realistic_cam">
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev> <!-- 1% noise -->
    </noise>
  </camera>
</sensor>

<!-- LiDAR with realistic noise -->
<sensor name="noisy_lidar" type="ray">
  <ray>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.02</stddev> <!-- 2cm noise -->
    </noise>
  </ray>
</sensor>

<!-- IMU with realistic bias and drift -->
<sensor name="realistic_imu" type="imu">
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1e-3</stddev>
          <bias_mean>1e-5</bias_mean>
          <bias_stddev>1e-6</bias_stddev>
        </noise>
      </x>
    </angular_velocity>
    <linear_acceleration>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1e-2</stddev>
          <bias_mean>1e-3</bias_mean>
          <bias_stddev>1e-4</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### Environmental Effects

Simulating environmental effects on sensors:

```xml
<!-- Camera affected by lighting conditions -->
<sensor name="light_sensitive_camera" type="camera">
  <camera name="adaptive_cam">
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <distortion>
      <k1>-0.177729</k1>
      <k2>0.220249</k2>
      <k3>-0.099798</k3>
      <p1>0.000118</p1>
      <p2>-0.000001</p2>
      <center>0.5 0.5</center>
    </distortion>
  </camera>
</sensor>
```

## Sensor Integration with ROS 2

### Standard ROS 2 Message Types

Gazebo sensors publish standard ROS 2 message types:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, CameraInfo
from cv_bridge import CvBridge
import numpy as np


class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Camera subscriber
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.camera_callback,
            10
        )

        # LiDAR subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # IMU subscriber
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.bridge = CvBridge()

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Process image data
            self.get_logger().info(f'Camera image received: {cv_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        # Convert ranges to numpy array
        ranges = np.array(msg.ranges)

        # Remove invalid readings
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Min distance: {min_distance:.2f}m')

    def imu_callback(self, msg):
        """Process IMU data"""
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        self.get_logger().info(
            f'IMU - Orientation: ({orientation.x:.3f}, {orientation.y:.3f}, {orientation.z:.3f}, {orientation.w:.3f})'
        )
```

### Sensor Configuration in Launch Files

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Sensor processor node
        Node(
            package='my_robot_perception',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ])
```

## Sensor Fusion for Humanoid Robotics

### Combining Multiple Sensors

Humanoid robots often need to combine data from multiple sensors:

```python
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Multiple sensor subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Fused state publisher
        self.fused_state_pub = self.create_publisher(HumanoidState, '/fused_state', 10)

        # Store sensor data
        self.last_imu = None
        self.last_camera = None
        self.last_lidar = None

        # Timer for fusion
        self.fusion_timer = self.create_timer(0.033, self.fusion_callback)  # ~30Hz

    def fusion_callback(self):
        """Combine sensor data into fused state"""
        if self.last_imu and self.last_camera and self.last_lidar:
            fused_state = HumanoidState()

            # Combine orientation from IMU
            fused_state.orientation = self.last_imu.orientation

            # Combine position estimates
            # (This is simplified - real fusion would use Kalman filters, etc.)

            # Publish fused state
            self.fused_state_pub.publish(fused_state)
```

## Performance Optimization

### Sensor Update Rates

Balancing accuracy with performance:

```xml
<!-- High-rate sensors for critical control -->
<sensor name="balance_imu" type="imu">
  <update_rate>200</update_rate> <!-- High rate for balance -->
</sensor>

<!-- Medium-rate sensors for navigation -->
<sensor name="navigation_lidar" type="ray">
  <update_rate>10</update_rate> <!-- Lower rate for navigation -->
</sensor>

<!-- Low-rate sensors for mapping -->
<sensor name="mapping_camera" type="camera">
  <update_rate>5</update_rate> <!-- Lowest rate for mapping -->
</sensor>
```

### Reducing Computational Load

```xml
<!-- Reduce resolution for performance -->
<sensor name="low_res_camera" type="camera">
  <camera name="fast_cam">
    <image>
      <width>320</width>  <!-- Lower resolution -->
      <height>240</height>
    </image>
  </camera>
</sensor>

<!-- Reduce scan samples for performance -->
<sensor name="fast_lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples> <!-- Half the samples -->
      </horizontal>
    </scan>
  </ray>
</sensor>
```

## Common Sensor Simulation Issues

### 1. Sensor Not Publishing Data

**Symptoms**: Sensor topics show no data
**Solutions**:
- Check sensor pose is valid
- Verify update rate is set
- Ensure sensor is not inside other objects
- Check collision geometry doesn't interfere

### 2. Unrealistic Sensor Data

**Symptoms**: Data doesn't match expected values
**Solutions**:
- Verify noise parameters
- Check sensor range limits
- Validate coordinate frames
- Review environmental conditions

### 3. Performance Issues

**Symptoms**: Low simulation speed with sensors
**Solutions**:
- Reduce update rates
- Lower resolution
- Simplify sensor models
- Use fewer sensors in simulation

## Best Practices for Humanoid Robotics

### 1. Sensor Redundancy

Use multiple sensors for critical functions:

```xml
<!-- Redundant orientation sensing -->
<sensor name="torso_imu" type="imu">
  <pose>0 0 0.2 0 0 0</pose>
</sensor>

<sensor name="head_imu" type="imu">
  <pose>0 0 0.5 0 0 0</pose>
</sensor>
```

### 2. Realistic Noise Models

Always include realistic noise:

```xml
<!-- Realistic IMU with bias and drift -->
<sensor name="realistic_imu" type="imu">
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
    </angular_velocity>
  </imu>
</sensor>
```

### 3. Proper Coordinate Frames

Ensure consistent coordinate frame conventions:

```xml
<!-- Use consistent coordinate frames -->
<!-- Standard: X forward, Y left, Z up -->
<sensor name="front_camera" type="camera">
  <pose>0.1 0 0.1 0 0 0</pose> <!-- Looking forward -->
</sensor>
```

### 4. Validation Against Real Sensors

Compare simulation with real sensor data:

```python
class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Subscribe to both simulated and real sensors
        self.sim_imu_sub = self.create_subscription(Imu, '/sim_imu/data', self.sim_imu_callback, 10)
        self.real_imu_sub = self.create_subscription(Imu, '/real_imu/data', self.real_imu_callback, 10)

        self.sim_data_buffer = []
        self.real_data_buffer = []

    def validate_sensor_data(self):
        """Compare simulated vs real sensor data"""
        # Calculate statistical differences
        # Log validation results
        pass
```

## Troubleshooting Sensor Issues

### Debugging Sensor Data

```bash
# Check if sensor topics are being published
ros2 topic list | grep sensor

# Echo sensor data to verify
ros2 topic echo /camera/image_raw

# Check sensor status
ros2 run rqt_plot rqt_plot
```

### Visualization Tools

```bash
# Visualize LiDAR data
ros2 run rviz2 rviz2

# Monitor IMU data
ros2 run rqt_plot rqt_plot
```

## Integration with Perception Systems

### Computer Vision Integration

```python
class VisionProcessor(Node):
    def __init__(self):
        super().__init__('vision_processor')

        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.object_pub = self.create_publisher(
            ObjectDetection, '/object_detections', 10
        )

    def image_callback(self, msg):
        """Process camera image for object detection"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run object detection
        detections = self.detect_objects(cv_image)

        # Publish detections
        detection_msg = ObjectDetection()
        detection_msg.objects = detections
        self.object_pub.publish(detection_msg)
```

### SLAM Integration

For humanoid robots that need mapping and localization:

```xml
<!-- Sensors for SLAM -->
<sensor name="slam_lidar" type="ray">
  <pose>0.1 0 0.3 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>1080</samples>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>20.0</max>
    </range>
  </ray>
</sensor>
```

## Next Steps

With a solid understanding of sensor simulation, continue to [Unity Overview](./unity-overview.md) to learn about Unity as an alternative simulation environment for high-fidelity visualization and human-robot interaction in humanoid robotics applications.