---
title: Worlds & Physics Settings
sidebar_position: 4
---

# Worlds & Physics Settings

Creating realistic simulation environments requires careful configuration of both the world structure and physics parameters. This lesson covers how to design effective simulation worlds and tune physics settings for humanoid robotics applications.

## World Design Principles

### Environment Design for Humanoid Robots

Humanoid robots operate in human environments, so simulation worlds should reflect this reality:

1. **Scale**: Environments should match human-scale dimensions
2. **Obstacles**: Include furniture, doorways, stairs, and other human-centric obstacles
3. **Surfaces**: Various surface types (carpet, tile, wood) with appropriate friction
4. **Lighting**: Realistic lighting conditions for sensor simulation
5. **Interactables**: Objects that robots might interact with

### World Categories

#### Indoor Worlds
- **Homes**: Living rooms, kitchens, bedrooms with furniture
- **Offices**: Desks, chairs, corridors, elevators
- **Public Spaces**: Hallways, lobbies, cafeterias

#### Outdoor Worlds
- **Urban**: Sidewalks, streets, crosswalks, urban furniture
- **Parks**: Grass, uneven terrain, trees, pathways
- **Industrial**: Warehouses, manufacturing floors, loading docks

## World File Structure

### Basic World Template

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_environment">
    <!-- Physics configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ground plane -->
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
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.0 0.0 0.0 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include models -->
    <include>
      <uri>model://my_humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Additional objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <!-- Model definition -->
    </model>
  </world>
</sdf>
```

## Physics Configuration

### Gravity Settings

For humanoid robotics, standard Earth gravity is typically used:

```xml
<gravity>0 0 -9.8</gravity>
```

However, for testing purposes, you might want to adjust gravity:

```xml
<!-- Reduced gravity for easier balance testing -->
<gravity>0 0 -1.62</gravity>  <!-- Moon gravity -->

<!-- Zero gravity for floating tests -->
<gravity>0 0 0</gravity>
```

### Time Step Configuration

Critical for humanoid stability:

```xml
<physics name="stable_humanoid" type="ode">
  <!-- Small time step for stability -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor of 1 for real-time simulation -->
  <real_time_factor>1</real_time_factor>

  <!-- High update rate for smooth control -->
  <real_time_update_rate>1000</real_time_update_rate>
</physics>
```

### Solver Configuration

For humanoid robots, stability is more important than speed:

```xml
<physics name="ode" type="ode">
  <ode>
    <solver>
      <!-- Use quick solver for real-time performance -->
      <type>quick</type>

      <!-- More iterations for stability (critical for humanoid balance) -->
      <iters>100</iters>

      <!-- Successive over-relaxation parameter -->
      <sor>1.3</sor>
    </solver>

    <constraints>
      <!-- Constraint force mixing - lower for stability -->
      <cfm>1e-5</cfm>

      <!-- Error reduction parameter - higher for faster error correction -->
      <erp>0.2</erp>

      <!-- Maximum contact correction velocity -->
      <contact_max_correcting_vel>100</contact_max_correcting_vel>

      <!-- Contact surface layer for stability -->
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Surface Properties for Humanoid Interaction

### Friction Settings

Different surfaces require different friction values for realistic humanoid locomotion:

```xml
<!-- High friction for stable walking -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>    <!-- Primary friction -->
      <mu2>1.0</mu2>  <!-- Secondary friction -->
    </ode>
  </friction>
</surface>

<!-- Low friction surface (ice, wet floor) -->
<surface>
  <friction>
    <ode>
      <mu>0.1</mu>
      <mu2>0.1</mu2>
    </ode>
  </friction>
</surface>

<!-- Anisotropic friction (directional) -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>
      <mu2>0.1</mu2>  <!-- Different friction in different directions -->
      <fdir1>1 0 0</fdir1>  <!-- Direction of mu2 -->
    </ode>
  </friction>
</surface>
```

### Damping and Compliance

For realistic contact with humanoid feet:

```xml
<surface>
  <bounce>
    <restitution_coefficient>0.01</restitution_coefficient>
    <threshold>100000</threshold>
  </bounce>
  <friction>
    <ode>
      <mu>0.8</mu>
      <mu2>0.8</mu2>
    </ode>
  </friction>
  <contact>
    <ode>
      <soft_cfm>0.000001</soft_cfm>
      <soft_erp>0.8</soft_erp>
      <kp>1e10</kp>  <!-- Contact stiffness -->
      <kd>1</kd>     <!-- Contact damping -->
      <max_vel>100</max_vel>
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

## World Examples for Humanoid Robotics

### Indoor Home Environment

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="home_environment">
    <!-- Physics optimized for humanoid -->
    <physics name="humanoid_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>1e-5</cfm>
          <erp>0.2</erp>
        </constraints>
      </ode>
    </physics>

    <!-- Sun light -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>

    <!-- Ground with carpet-like friction -->
    <model name="living_room_floor">
      <static>true</static>
      <link name="floor_link">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 8 0.1</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.7</mu>
                <mu2>0.7</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 8 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="sofa">
      <pose>2 -2 0 0 0 0</pose>
      <!-- Sofa model definition -->
    </model>

    <model name="coffee_table">
      <pose>0 0 0 0 0 0</pose>
      <!-- Table model definition -->
    </model>

    <!-- Starting position for humanoid -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>-2 0 1 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Stairs Environment

```xml
<!-- Stairs for testing humanoid locomotion -->
<model name="stairs">
  <static>true</static>
  <link name="stairs_link">
    <pose>0 0 0 0 0 0</pose>
    <!-- Individual steps -->
    <collision name="step_1">
      <pose>0 0 0.15 0 0 0</pose>
      <geometry>
        <box>
          <size>2 1.5 0.3</size>
        </box>
      </geometry>
    </collision>
    <visual name="step_1_visual">
      <pose>0 0 0.15 0 0 0</pose>
      <geometry>
        <box>
          <size>2 1.5 0.3</size>
        </box>
      </geometry>
      <material>
        <ambient>0.5 0.5 0.5 1</ambient>
        <diffuse>0.5 0.5 0.5 1</diffuse>
      </material>
    </visual>

    <collision name="step_2">
      <pose>0 0 0.45 0 0 0</pose>
      <geometry>
        <box>
          <size>2 1.5 0.3</size>
        </box>
      </geometry>
    </collision>
    <visual name="step_2_visual">
      <pose>0 0 0.45 0 0 0</pose>
      <geometry>
        <box>
          <size>2 1.5 0.3</size>
        </box>
      </geometry>
    </visual>
    <!-- Add more steps as needed -->
  </link>
</model>
```

## Performance Considerations

### Optimizing for Real-time Simulation

For humanoid robots that require real-time control:

```xml
<physics name="realtime_humanoid" type="ode">
  <!-- Slightly larger time step for performance -->
  <max_step_size>0.002</max_step_size>

  <!-- Lower solver iterations (less stable but faster) -->
  <ode>
    <solver>
      <iters>50</iters>  <!-- Reduced from 100 -->
    </solver>
  </ode>
</physics>
```

### Balancing Accuracy vs Performance

```xml
<!-- For high-fidelity validation -->
<physics name="high_fidelity" type="ode">
  <max_step_size>0.0005</max_step_size>
  <ode>
    <solver>
      <iters>200</iters>
    </solver>
  </ode>
</physics>

<!-- For rapid development iteration -->
<physics name="fast_development" type="ode">
  <max_step_size>0.01</max_step_size>
  <real_time_factor>2</real_time_factor>  <!-- Run faster than real-time -->
  <ode>
    <solver>
      <iters>20</iters>
    </solver>
  </ode>
</physics>
```

## Advanced Physics Concepts

### Contact Stiffness and Damping

For humanoid foot-ground contact:

```xml
<surface>
  <contact>
    <ode>
      <!-- High stiffness for solid contact -->
      <kp>1e9</kp>

      <!-- Appropriate damping ratio -->
      <kd>100</kd>

      <!-- Soft contact for stability -->
      <soft_cfm>1e-5</soft_cfm>
      <soft_erp>0.9</soft_erp>
    </ode>
  </contact>
</surface>
```

### Multi-Body Dynamics for Humanoid Systems

When simulating complex humanoid systems:

```xml
<physics name="humanoid_system" type="ode">
  <ode>
    <solver>
      <!-- More iterations for complex multi-body system -->
      <iters>150</iters>
    </solver>
    <constraints>
      <!-- Tighter constraints for accurate joint simulation -->
      <cfm>1e-6</cfm>
      <erp>0.1</erp>
    </constraints>
  </ode>
</physics>
```

## Testing Physics Configurations

### Validation Techniques

1. **Static Balance**: Test if humanoid can stand without falling
2. **Dynamic Motion**: Test walking, running, and other dynamic behaviors
3. **Contact Behavior**: Test foot-ground, hand-object interactions
4. **Stability**: Run simulations for extended periods to check for drift

### Common Physics Issues in Humanoid Simulation

#### Joint Drift
```xml
<!-- Increase ERP to reduce joint drift -->
<physics type="ode">
  <ode>
    <constraints>
      <erp>0.2</erp>  <!-- Higher ERP reduces drift -->
    </constraints>
  </ode>
</physics>
```

#### Unstable Contacts
```xml
<!-- Adjust contact parameters for stability -->
<surface>
  <contact>
    <ode>
      <soft_cfm>1e-4</soft_cfm>  <!-- Soft constraint force mixing -->
      <soft_erp>0.8</soft_erp>   <!-- Error reduction parameter -->
    </ode>
  </contact>
</surface>
```

#### Excessive Bouncing
```xml
<!-- Reduce bouncing with proper damping -->
<surface>
  <bounce>
    <restitution_coefficient>0.01</restitution_coefficient>
  </bounce>
</surface>
```

## Integration with ROS 2

### Physics Parameters via ROS 2

You can also configure some physics parameters through ROS 2:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class PhysicsParameterTuner(Node):
    def __init__(self):
        super().__init__('physics_parameter_tuner')

        # Publisher for physics parameters
        self.gravity_publisher = self.create_publisher(Float32, 'gravity_setting', 10)

        # Timer to periodically update physics parameters
        self.physics_timer = self.create_timer(1.0, self.update_physics_params)

    def update_physics_params(self):
        # Example: Adjust gravity for testing
        msg = Float32()
        msg.data = -9.8  # Earth gravity
        self.gravity_publisher.publish(msg)
```

## Best Practices for Humanoid Simulation

### 1. Start Simple
- Begin with basic environments
- Gradually add complexity
- Validate physics at each step

### 2. Use Realistic Parameters
- Match real-world values when possible
- Validate against physical robot behavior
- Document parameter choices and reasoning

### 3. Performance Monitoring
- Monitor simulation real-time factor
- Track physics update times
- Balance accuracy with performance needs

### 4. Iterative Testing
- Test basic balance first
- Add complexity gradually
- Validate each new feature

### 5. Documentation
- Document physics parameter choices
- Record validation results
- Note sim-to-real differences

## Troubleshooting Physics Issues

### Common Problems and Solutions

**Problem**: Robot falls through floor
- **Cause**: Collision geometry or physics parameters
- **Solution**: Check collision models and contact parameters

**Problem**: Robot is unstable or jittery
- **Cause**: Time step too large or solver iterations too low
- **Solution**: Reduce time step and increase solver iterations

**Problem**: Robot slides unrealistically
- **Cause**: Low friction coefficients
- **Solution**: Increase friction parameters

**Problem**: Simulation runs too slowly
- **Cause**: High physics accuracy requirements
- **Solution**: Optimize parameters for performance vs accuracy trade-off

## Next Steps

With a solid understanding of world design and physics configuration, continue to [URDF vs SDF for Simulation](./urdf-vs-sdf.md) to learn about the different robot description formats used in simulation and how to choose the right one for your humanoid robotics applications.