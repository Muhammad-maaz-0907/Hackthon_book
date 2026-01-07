---
title: Unity Overview
sidebar_position: 7
---

# Unity Overview

Unity is a powerful 3D development platform that provides high-fidelity visualization and simulation capabilities for robotics applications. This lesson covers Unity's role in robotics simulation, particularly for humanoid robotics, and how it compares to other simulation environments.

## Introduction to Unity for Robotics

Unity is a cross-platform game engine that has been increasingly adopted for robotics simulation due to its:
- **High-fidelity graphics**: Photorealistic rendering capabilities
- **Physics engine**: Realistic physics simulation with PhysX
- **Asset ecosystem**: Extensive library of 3D models and environments
- **Scripting support**: Flexible programming with C#
- **VR/AR capabilities**: Support for immersive interfaces

### Unity vs Traditional Robotics Simulation

| Aspect | Unity | Gazebo | Webots |
|--------|-------|--------|--------|
| **Graphics Quality** | Very High | Medium | Medium |
| **Physics Accuracy** | Good | Excellent | Good |
| **Robotics Integration** | Moderate | Excellent | Good |
| **Learning Curve** | Moderate | Moderate | Moderate |
| **Performance** | High | High | Medium |
| **Cost** | Free (personal) | Free | Free |

## Unity Robotics Ecosystem

### Unity Robotics Hub

Unity provides specialized tools for robotics development:

1. **Unity Robotics Package**: ROS/ROS 2 integration
2. **Unity Simulation Package**: Large-scale simulation capabilities
3. **Unity Perception Package**: Synthetic data generation for AI
4. **Unity ML-Agents**: Reinforcement learning framework

### ROS/ROS 2 Integration

Unity connects to ROS/ROS 2 through the Unity Robotics Package:

```csharp
using UnityEngine;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector;

public class UnityRobotController : MonoBehaviour
{
    // ROS connection
    private RosConnection ros;

    // Topics
    private string jointStatesTopic = "/joint_states";
    private string cameraTopic = "/camera/image_raw";

    void Start()
    {
        // Initialize ROS connection
        ros = RosConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(jointStatesTopic);

        // Subscribe to ROS topics
        ros.Subscribe<ImageMsg>(cameraTopic, OnCameraDataReceived);
    }

    void OnCameraDataReceived(ImageMsg imageMsg)
    {
        // Process camera data from ROS
        Debug.Log($"Received camera image: {imageMsg.width}x{imageMsg.height}");
    }

    void PublishJointStates()
    {
        // Create and publish joint state message
        var jointState = new JointStateMsg();
        jointState.name = new string[] { "joint1", "joint2", "joint3" };
        jointState.position = new double[] { 0.0, 0.5, -0.5 };
        jointState.velocity = new double[] { 0.0, 0.0, 0.0 };
        jointState.effort = new double[] { 0.0, 0.0, 0.0 };

        ros.Publish(jointStatesTopic, jointState);
    }
}
```

## Setting Up Unity for Robotics

### Installation Requirements

1. **Unity Hub**: Download from Unity's website
2. **Unity Editor**: Version 2021.3 LTS or newer recommended
3. **Unity Robotics Package**: Install via Package Manager
4. **ROS/ROS 2**: For communication with robotic systems

### Unity Robotics Package Installation

1. Open Unity Editor
2. Go to Window → Package Manager
3. Click the + button → Add package from git URL
4. Enter: `https://github.com/Unity-Technologies/Unity-Robotics-Hub.git`
5. Install the ROS TCP Connector package

## Unity Scene Architecture for Robotics

### Basic Scene Setup

```csharp
using UnityEngine;

public class RobotSceneManager : MonoBehaviour
{
    public GameObject robotPrefab;
    public Transform spawnPoint;
    public Material groundMaterial;

    void Start()
    {
        // Spawn robot at designated location
        if (robotPrefab != null && spawnPoint != null)
        {
            Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
        }

        // Set up environment
        SetupEnvironment();
    }

    void SetupEnvironment()
    {
        // Create ground plane
        GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.transform.position = Vector3.zero;
        ground.GetComponent<Renderer>().material = groundMaterial;

        // Add lighting
        SetupLighting();
    }

    void SetupLighting()
    {
        // Add directional light (sun)
        GameObject sun = new GameObject("Sun");
        sun.AddComponent<Light>();
        sun.GetComponent<Light>().type = LightType.Directional;
        sun.transform.rotation = Quaternion.Euler(50, -30, 0);
    }
}
```

### Robot Model Integration

Unity supports various robot model formats:

```csharp
using UnityEngine;

public class RobotModelController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public Transform[] joints;  // Array of joint transforms
    public string[] jointNames; // Corresponding joint names

    [Header("Joint Limits")]
    public float[] minAngles;
    public float[] maxAngles;

    void Start()
    {
        InitializeJoints();
    }

    void InitializeJoints()
    {
        // Validate joint configuration
        if (joints.Length != jointNames.Length)
        {
            Debug.LogError("Joint configuration mismatch!");
            return;
        }

        // Initialize joint limits
        for (int i = 0; i < joints.Length; i++)
        {
            ConfigureJoint(joints[i], minAngles[i], maxAngles[i]);
        }
    }

    void ConfigureJoint(Transform joint, float minAngle, float maxAngle)
    {
        // Configure joint constraints if using Unity's physics
        ConfigurableJoint configJoint = joint.GetComponent<ConfigurableJoint>();
        if (configJoint != null)
        {
            // Set angular limits
            SoftJointLimit limit = new SoftJointLimit();
            limit.limit = maxAngle;
            configJoint.highAngularXLimit = limit;

            limit.limit = minAngle;
            configJoint.lowAngularXLimit = limit;
        }
    }

    public void SetJointPositions(float[] positions)
    {
        if (positions.Length != joints.Length)
        {
            Debug.LogError("Position array length mismatch!");
            return;
        }

        for (int i = 0; i < joints.Length; i++)
        {
            // Apply joint position (simplified)
            joints[i].localRotation = Quaternion.Euler(0, positions[i] * Mathf.Rad2Deg, 0);
        }
    }
}
```

## Physics Simulation in Unity

### PhysX Configuration

Unity uses NVIDIA PhysX for physics simulation:

```csharp
using UnityEngine;

public class PhysicsConfigurator : MonoBehaviour
{
    [Header("Physics Settings")]
    public float gravity = -9.81f;
    public int solverIterations = 6;
    public int solverVelocityIterations = 1;

    void Start()
    {
        ConfigurePhysics();
    }

    void ConfigurePhysics()
    {
        // Set gravity
        Physics.gravity = new Vector3(0, gravity, 0);

        // Configure solver settings
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;

        // Set default physics material
        Physics.defaultMaterial = CreatePhysicsMaterial();
    }

    PhysicMaterial CreatePhysicsMaterial()
    {
        PhysicMaterial material = new PhysicMaterial("RobotMaterial");
        material.staticFriction = 0.5f;
        material.dynamicFriction = 0.4f;
        material.bounciness = 0.1f;
        material.frictionCombine = PhysicMaterialCombine.Average;
        material.bounceCombine = PhysicMaterialCombine.Average;

        return material;
    }
}
```

### Collision Detection

```csharp
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    void OnCollisionEnter(Collision collision)
    {
        // Handle collision with other objects
        Debug.Log($"Collision detected with {collision.gameObject.name}");

        // Get collision details
        foreach (ContactPoint contact in collision.contacts)
        {
            Debug.Log($"Contact point: {contact.point}");
            Debug.Log($"Contact normal: {contact.normal}");
            Debug.Log($"Contact force: {contact.thisCollider.name}");
        }
    }

    void OnCollisionStay(Collision collision)
    {
        // Continuous collision handling
        foreach (ContactPoint contact in collision.contacts)
        {
            // Handle ongoing contact
        }
    }

    void OnCollisionExit(Collision collision)
    {
        // Handle collision end
        Debug.Log($"Collision ended with {collision.gameObject.name}");
    }
}
```

## Sensor Simulation in Unity

### Camera Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Settings")]
    public Camera sensorCamera;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float updateRate = 30.0f; // Hz

    [Header("ROS Settings")]
    public string imageTopic = "/camera/image_raw";

    private RenderTexture renderTexture;
    private RosConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;

        // Create render texture for camera
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        sensorCamera.targetTexture = renderTexture;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishCameraImage();
            lastUpdateTime = Time.time;
        }
    }

    void PublishCameraImage()
    {
        // Read pixels from render texture
        RenderTexture.active = renderTexture;
        Texture2D texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS message format
        byte[] imageData = texture2D.EncodeToJPG();

        // Create and publish image message
        ImageMsg imageMsg = new ImageMsg();
        imageMsg.header = new std_msgs.HeaderMsg();
        imageMsg.header.stamp = new builtin_interfaces.TimeMsg(System.DateTime.Now);
        imageMsg.header.frame_id = "camera_frame";
        imageMsg.height = (uint)imageHeight;
        imageMsg.width = (uint)imageWidth;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(imageWidth * 3); // 3 bytes per pixel (RGB)
        imageMsg.data = imageData;

        ros.Publish(imageTopic, imageMsg);

        // Clean up
        Destroy(texture2D);
    }
}
```

### LiDAR Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLidarSensor : MonoBehaviour
{
    [Header("LiDAR Settings")]
    public float minRange = 0.1f;
    public float maxRange = 30.0f;
    public int horizontalSamples = 720;
    public float horizontalFOV = 360f; // degrees
    public float updateRate = 10.0f; // Hz

    [Header("ROS Settings")]
    public string scanTopic = "/scan";

    private RosConnection ros;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        updateInterval = 1.0f / updateRate;
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            PublishLidarScan();
            lastUpdateTime = Time.time;
        }
    }

    void PublishLidarScan()
    {
        LaserScanMsg scanMsg = new LaserScanMsg();
        scanMsg.header = new std_msgs.HeaderMsg();
        scanMsg.header.stamp = new builtin_interfaces.TimeMsg(System.DateTime.Now);
        scanMsg.header.frame_id = "lidar_frame";

        scanMsg.angle_min = -horizontalFOV * Mathf.Deg2Rad / 2f;
        scanMsg.angle_max = horizontalFOV * Mathf.Deg2Rad / 2f;
        scanMsg.angle_increment = (horizontalFOV * Mathf.Deg2Rad) / horizontalSamples;
        scanMsg.time_increment = 0;
        scanMsg.scan_time = 1.0f / updateRate;
        scanMsg.range_min = minRange;
        scanMsg.range_max = maxRange;

        // Perform raycasts to simulate LiDAR
        float[] ranges = new float[horizontalSamples];
        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = scanMsg.angle_min + i * scanMsg.angle_increment;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = float.PositiveInfinity; // No obstacle detected
            }
        }

        scanMsg.ranges = ranges;

        ros.Publish(scanTopic, scanMsg);
    }
}
```

## NVIDIA Isaac Integration

Unity works closely with NVIDIA Isaac for robotics simulation:

### Isaac Unity Plugin

NVIDIA provides plugins to enhance Unity's robotics capabilities:

1. **Isaac Sim**: High-fidelity simulation for AI and robotics
2. **Isaac ROS**: ROS 2 packages for NVIDIA hardware
3. **Isaac Apps**: Reference applications and examples

### Synthetic Data Generation

Unity can generate synthetic training data for AI:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SyntheticDataGenerator : MonoBehaviour
{
    [Header("Synthetic Data Settings")]
    public int datasetSize = 1000;
    public string datasetPath = "Assets/Datasets/";

    [Header("Annotation Settings")]
    public bool generateSegmentation = true;
    public bool generateDepth = true;
    public bool generateBboxes = true;

    public void GenerateDataset()
    {
        for (int i = 0; i < datasetSize; i++)
        {
            // Randomize environment
            RandomizeEnvironment();

            // Capture synthetic data
            CaptureSyntheticImage(i);
        }
    }

    void RandomizeEnvironment()
    {
        // Randomize lighting
        Light sunLight = GameObject.Find("Sun").GetComponent<Light>();
        sunLight.color = Random.ColorHSV();
        sunLight.intensity = Random.Range(0.5f, 2.0f);

        // Randomize object positions
        GameObject[] objects = GameObject.FindGameObjectsWithTag("RandomObject");
        foreach (GameObject obj in objects)
        {
            obj.transform.position = new Vector3(
                Random.Range(-5f, 5f),
                Random.Range(0.5f, 2f),
                Random.Range(-5f, 5f)
            );
        }
    }

    void CaptureSyntheticImage(int index)
    {
        // Capture RGB image
        // Capture segmentation mask
        // Capture depth map
        // Generate annotations
    }
}
```

## Humanoid Robotics Applications

### Character Animation Integration

Unity excels at humanoid character animation:

```csharp
using UnityEngine;
using UnityEngine.Animations;

public class HumanoidAnimationController : MonoBehaviour
{
    [Header("Animation Settings")]
    public Animator animator;
    public float walkingSpeed = 1.0f;
    public float turnSpeed = 90.0f;

    [Header("Balance Control")]
    public Transform centerOfMass;
    public float balanceThreshold = 0.1f;

    void Start()
    {
        if (animator != null)
        {
            // Configure center of mass
            if (centerOfMass != null)
            {
                animator.bodyPosition = centerOfMass.position;
            }
        }
    }

    void Update()
    {
        HandleMovement();
        CheckBalance();
    }

    void HandleMovement()
    {
        if (animator != null)
        {
            // Get input
            float moveX = Input.GetAxis("Horizontal");
            float moveZ = Input.GetAxis("Vertical");

            // Apply movement
            Vector3 movement = new Vector3(moveX, 0, moveZ).normalized;
            movement = transform.TransformDirection(movement);
            movement *= walkingSpeed * Time.deltaTime;

            transform.Translate(movement);

            // Turn based on input
            if (moveX != 0)
            {
                transform.Rotate(0, moveX * turnSpeed * Time.deltaTime, 0);
            }

            // Update animation parameters
            animator.SetFloat("Speed", movement.magnitude);
            animator.SetFloat("Direction", moveZ);
        }
    }

    void CheckBalance()
    {
        if (centerOfMass != null)
        {
            // Check if center of mass is within balance threshold
            float balanceError = Mathf.Abs(centerOfMass.position.x);
            if (balanceError > balanceThreshold)
            {
                // Trigger balance correction
                TriggerBalanceCorrection();
            }
        }
    }

    void TriggerBalanceCorrection()
    {
        if (animator != null)
        {
            // Trigger balance correction animation
            animator.SetTrigger("BalanceCorrection");
        }
    }
}
```

### VR/AR Integration for Humanoid Control

```csharp
using UnityEngine;
using UnityEngine.XR;

public class VRHumanoidController : MonoBehaviour
{
    [Header("VR Controllers")]
    public Transform leftController;
    public Transform rightController;
    public Transform headController;

    [Header("Humanoid Mapping")]
    public Transform humanoidHead;
    public Transform humanoidLeftHand;
    public Transform humanoidRightHand;

    void Update()
    {
        UpdateHumanoidPose();
    }

    void UpdateHumanoidPose()
    {
        // Map VR head to humanoid head
        if (headController != null && humanoidHead != null)
        {
            humanoidHead.position = headController.position;
            humanoidHead.rotation = headController.rotation;
        }

        // Map VR controllers to humanoid hands
        if (leftController != null && humanoidLeftHand != null)
        {
            humanoidLeftHand.position = leftController.position;
            humanoidLeftHand.rotation = leftController.rotation;
        }

        if (rightController != null && humanoidRightHand != null)
        {
            humanoidRightHand.position = rightController.position;
            humanoidRightHand.rotation = rightController.rotation;
        }
    }
}
```

## Performance Optimization

### Level of Detail (LOD)

Unity provides LOD systems for performance:

```csharp
using UnityEngine;

public class RobotLODController : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public float distance;
        public GameObject[] renderers;
    }

    public LODLevel[] lodLevels;
    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
    }

    void Update()
    {
        UpdateLOD();
    }

    void UpdateLOD()
    {
        if (mainCamera != null)
        {
            float distance = Vector3.Distance(transform.position, mainCamera.transform.position);

            for (int i = 0; i < lodLevels.Length; i++)
            {
                bool enable = distance <= lodLevels[i].distance;
                foreach (GameObject renderer in lodLevels[i].renderers)
                {
                    if (renderer != null)
                    {
                        renderer.SetActive(enable);
                    }
                }
            }
        }
    }
}
```

### Occlusion Culling

```csharp
using UnityEngine;

public class OcclusionCullingManager : MonoBehaviour
{
    [Header("Occlusion Settings")]
    public float updateInterval = 0.1f;
    public LayerMask occlusionMask;

    private float lastUpdate;
    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
        lastUpdate = Time.time;
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            UpdateOcclusion();
            lastUpdate = Time.time;
        }
    }

    void UpdateOcclusion()
    {
        // Unity has built-in occlusion culling system
        // This is handled automatically when enabled in build settings
    }
}
```

## Unity vs Gazebo for Humanoid Robotics

### Unity Advantages

1. **Photorealistic Graphics**: Excellent for perception training
2. **User Interface**: Better for human-robot interaction
3. **Asset Library**: Extensive 3D model library
4. **VR/AR Support**: Native support for immersive interfaces
5. **Animation System**: Powerful humanoid animation tools

### Gazebo Advantages

1. **Physics Accuracy**: More accurate physics simulation
2. **Robotics Integration**: Better ROS/ROS 2 integration
3. **Sensor Simulation**: More realistic sensor models
4. **Performance**: Better for large-scale simulation
5. **Open Source**: Free and open source

## Best Practices for Unity Robotics

### 1. Asset Management

- Use modular robot models
- Create reusable environment assets
- Implement proper naming conventions
- Use prefabs for consistency

### 2. Performance Optimization

- Use occlusion culling
- Implement LOD systems
- Optimize draw calls
- Use efficient physics settings

### 3. Integration with ROS

- Use proper message types
- Implement error handling
- Consider network latency
- Validate data consistency

### 4. Testing and Validation

- Compare with real robot data
- Validate sensor models
- Test in multiple environments
- Document simulation assumptions

## Troubleshooting Common Issues

### 1. ROS Connection Issues

**Problem**: Unity can't connect to ROS
**Solutions**:
- Check ROS TCP Connector settings
- Verify ROS network configuration
- Ensure ROS master is running
- Check firewall settings

### 2. Performance Issues

**Problem**: Low simulation frame rate
**Solutions**:
- Reduce scene complexity
- Use LOD systems
- Optimize draw calls
- Adjust physics settings

### 3. Physics Issues

**Problem**: Unstable or unrealistic physics
**Solutions**:
- Adjust solver iterations
- Check mass and inertia values
- Validate collision geometry
- Tune friction and damping

## Next Steps

With a solid understanding of Unity for robotics simulation, continue to [Module 2 Labs](./labs.md) to practice implementing these concepts in hands-on exercises that connect Unity simulation to actual ROS 2 systems and humanoid robotics applications.