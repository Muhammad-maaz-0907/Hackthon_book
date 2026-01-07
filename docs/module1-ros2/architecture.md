---
title: ROS 2 Architecture
sidebar_position: 2
---

# ROS 2 Architecture

Understanding the ROS 2 architecture is fundamental to working effectively with robotic systems. This lesson covers the core architectural concepts that make distributed robotics possible.

## The ROS 2 Ecosystem

ROS 2 is built on a distributed architecture that enables multiple processes (and potentially multiple machines) to communicate seamlessly. The architecture is designed around the concept of a "computational graph" that connects nodes through various communication mechanisms.

### Core Components

The ROS 2 architecture consists of several key components:

1. **Nodes**: Independent processes that perform computation
2. **Communication**: Mechanisms for nodes to exchange data
3. **Compositions**: Ways to organize nodes into logical units
4. **Parameters**: Configuration values that can be changed at runtime
5. **Actions**: Goal-oriented communication patterns

### The Computational Graph

The computational graph is the fundamental organizational principle of ROS 2. It consists of:

- **Nodes**: The computational units that perform work
- **Topics**: Named buses over which messages are sent
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous goal-oriented communication

## Nodes: The Basic Computational Unit

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 system.

### Node Characteristics

- **Process-based**: Each node runs as a separate process
- **Single-threaded by default**: Each node has a main thread for processing callbacks
- **Namespaced**: Nodes can be organized into namespaces for better organization
- **Resource-aware**: Nodes can be configured with QoS (Quality of Service) policies

### Node Lifecycle

Nodes follow a well-defined lifecycle:
1. **Unconfigured**: Node created but not yet configured
2. **Inactive**: Node configured but not yet active
3. **Active**: Node is running and processing callbacks
4. **Finalized**: Node is shut down and cleaned up

## Communication Patterns

ROS 2 provides several communication patterns to handle different types of data exchange:

### Topics (Publish/Subscribe)

Topics provide asynchronous, many-to-many communication:

```python
# Publisher
publisher = node.create_publisher(String, 'topic_name', 10)

# Subscriber
subscriber = node.create_subscription(String, 'topic_name', callback, 10)
```

**Use Cases**:
- Sensor data streaming (camera images, LiDAR scans, IMU data)
- Robot state broadcasting (joint states, odometry)
- Event notifications

**Characteristics**:
- Asynchronous communication
- Data is sent without waiting for acknowledgment
- Multiple publishers and subscribers can exist for the same topic
- Data loss possible if subscribers are slow

### Services (Request/Response)

Services provide synchronous, one-to-one communication:

```python
# Service Server
service = node.create_service(AddTwoInts, 'add_two_ints', handle_add_two_ints)

# Service Client
client = node.create_client(AddTwoInts, 'add_two_ints')
```

**Use Cases**:
- Requesting specific computations
- Configuration changes
- Synchronous operations that return results

**Characteristics**:
- Synchronous communication
- Request waits for response
- One-to-one communication pattern
- Request-response pattern with timeouts

### Actions (Goal/Result/Feedback)

Actions provide asynchronous, goal-oriented communication:

```python
# Action Server
action_server = ActionServer(node, Fibonacci, 'fibonacci', execute_callback)

# Action Client
action_client = ActionClient(node, Fibonacci, 'fibonacci')
```

**Use Cases**:
- Long-running operations (navigation, manipulation)
- Operations with intermediate feedback
- Tasks that can be preempted

**Characteristics**:
- Asynchronous communication
- Goal-oriented with feedback
- Can be preempted or canceled
- Goal-result-feedback pattern

## Quality of Service (QoS)

QoS policies allow you to specify the behavior of communication channels:

### Reliability Policy
- **Reliable**: All messages will be delivered (if possible)
- **Best Effort**: Messages may be lost (faster, less overhead)

### Durability Policy
- **Transient Local**: Late-joining subscribers receive last message
- **Volatile**: No messages stored for late joiners

### History Policy
- **Keep Last**: Maintain a fixed number of messages
- **Keep All**: Maintain all messages (use with caution)

## DDS Integration

ROS 2 uses Data Distribution Service (DDS) as its middleware layer. DDS provides the underlying communication infrastructure:

- **Vendor-neutral**: Multiple DDS implementations available (Fast DDS, Cyclone DDS, RTI Connext)
- **Real-time capable**: Designed for real-time systems
- **Distributed**: Handles communication across multiple machines
- **Configurable**: Extensive QoS options for different requirements

## Humanoid Robotics Considerations

In humanoid robotics, the ROS 2 architecture provides specific benefits:

### Modular Design
- Different subsystems (perception, planning, control) can be developed independently
- Easy integration of third-party libraries and tools
- Parallel development of different robot capabilities

### Scalability
- Multiple sensors and actuators can be managed efficiently
- Distributed processing across multiple computers
- Easy addition of new capabilities

### Safety and Reliability
- Fault isolation between different nodes
- Configurable QoS for safety-critical communications
- Standardized interfaces that promote robustness

## Best Practices

### Node Design
- Keep nodes focused on a single responsibility
- Use appropriate QoS settings for your application
- Handle errors gracefully and provide meaningful logging
- Use namespaces to organize related nodes

### Communication Design
- Choose the right communication pattern for your use case
- Consider bandwidth and latency requirements
- Use appropriate message types and structures
- Plan for system growth and complexity

## Next Steps

Now that you understand the ROS 2 architecture, continue to learn about the specific communication patterns in the [Nodes, Topics, Services, Actions](./nodes-topics-services-actions.md) lesson, where you'll dive deeper into implementation details and practical applications.