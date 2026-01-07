---
title: Nodes, Topics, Services, Actions
sidebar_position: 3
---

# Nodes, Topics, Services, Actions

This lesson provides a detailed exploration of the four fundamental communication patterns in ROS 2: Nodes, Topics, Services, and Actions. Understanding when and how to use each pattern is crucial for effective robotic system design.

## Nodes: The Foundation of ROS 2

Nodes are the fundamental building blocks of any ROS 2 system. They represent individual processes that perform computation and communicate with other nodes.

### Creating Nodes in Python

```python
import rclpy
from rclpy.node import Node

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
```

### Node Best Practices

- **Single Responsibility**: Each node should focus on a specific task
- **Error Handling**: Implement proper error handling and logging
- **Resource Management**: Clean up resources when the node shuts down
- **Parameter Usage**: Use parameters for configurable behavior

## Topics: Publish/Subscribe Communication

Topics enable asynchronous, one-to-many communication through a publish/subscribe pattern.

### Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):

    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.i += 1
```

### Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

### Topic Characteristics

- **Asynchronous**: Publishers don't wait for subscribers
- **Many-to-many**: Multiple publishers and subscribers can exist
- **Data Streaming**: Ideal for continuous data like sensor readings
- **No Guarantees**: Messages may be lost if QoS is set to best-effort

### Quality of Service (QoS) for Topics

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

publisher = node.create_publisher(String, 'topic', qos_profile)
```

## Services: Request/Response Communication

Services provide synchronous, one-to-one communication with request/response semantics.

### Service Definition

First, define the service in an `.srv` file (e.g., `AddTwoInts.srv`):
```
int64 a
int64 b
---
int64 sum
```

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
        self.get_logger().info(f'Returning {response.sum}')
        return response
```

### Service Client Implementation

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

### Service Characteristics

- **Synchronous**: Client waits for response
- **One-to-one**: Direct communication between client and server
- **Request-response**: Clear request and response structure
- **Blocking**: Client is blocked until response received

## Actions: Goal-Oriented Communication

Actions provide asynchronous, goal-oriented communication with feedback and result capabilities.

### Action Definition

Define the action in an `.action` file (e.g., `Fibonacci.action`):
```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

### Action Server Implementation

```python
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            self.get_logger().info(f'Publishing feedback: {feedback_msg.partial_sequence}')
            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        self.get_logger().info(f'Returning result: {result.sequence}')

        return result
```

### Action Client Implementation

```python
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
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
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback.partial_sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
```

### Action Characteristics

- **Asynchronous**: Non-blocking communication
- **Goal-oriented**: Designed for long-running tasks
- **Feedback**: Continuous feedback during execution
- **Cancelation**: Goals can be canceled during execution
- **Result**: Final result provided when goal completes

## When to Use Each Pattern

### Use Topics When:
- Streaming continuous data (sensors, robot state)
- Broadcasting information to multiple subscribers
- Real-time requirements with low latency
- Data loss is acceptable (best-effort QoS)

### Use Services When:
- Requesting a specific computation
- Need synchronous response
- Simple request-response pattern
- Configuration changes
- One-off operations

### Use Actions When:
- Long-running operations (navigation, manipulation)
- Need intermediate feedback
- Operations that can be canceled
- Goal-oriented behavior
- Asynchronous operations with results

## Humanoid Robotics Applications

### Topics in Humanoid Systems
- **Sensor Data**: IMU, camera, LiDAR, force/torque sensors
- **State Information**: Joint positions, velocities, efforts
- **Perception Results**: Object detection, pose estimation
- **Behavior Status**: Current behavior or state machine state

### Services in Humanoid Systems
- **Configuration**: Changing robot parameters
- **Calibration**: Sensor or actuator calibration
- **Simple Control**: Triggering specific behaviors
- **Query Systems**: Requesting current system status

### Actions in Humanoid Systems
- **Navigation**: Moving to specific locations with feedback
- **Manipulation**: Grasping objects with progress feedback
- **Walking**: Complex locomotion patterns
- **Speech**: Text-to-speech with progress feedback

## Performance Considerations

### Topic Performance
- Use appropriate message sizes to avoid network congestion
- Choose QoS settings based on reliability requirements
- Consider using intra-process communication for same-process nodes

### Service Performance
- Keep service calls lightweight to avoid blocking
- Use timeouts to prevent indefinite blocking
- Consider using actions for long-running operations

### Action Performance
- Design feedback messages to be lightweight
- Use appropriate goal validation to prevent resource exhaustion
- Implement proper cancelation handling

## Next Steps

With a solid understanding of all four communication patterns, you're ready to move on to [Practical ROS 2 Development](./practical-development.md) where you'll implement real-world examples using these patterns in the context of humanoid robotics applications.