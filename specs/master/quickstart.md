# Quickstart Guide: Vision-Language-Action (VLA) Systems

## Overview
This guide provides a quick introduction to implementing Vision-Language-Action (VLA) systems for humanoid robotics using ROS 2, Isaac ROS, and Nav2.

## Prerequisites
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill installed
- Python 3.11
- NVIDIA GPU (recommended for Isaac ROS perception)
- Microphone and camera for input

## Setup

### 1. Install ROS 2 Dependencies
```bash
# Install ROS 2 Humble and required packages
sudo apt update
sudo apt install ros-humble-desktop ros-humble-cv-bridge ros-humble-tf2-tools
sudo apt install ros-humble-nav2-bringup ros-humble-isaac-ros-common
```

### 2. Install Python Dependencies
```bash
pip3 install openai-whisper opencv-python numpy torch torchvision
```

### 3. Create Workspace
```bash
mkdir -p ~/vla_ws/src
cd ~/vla_ws
colcon build
source install/setup.bash
```

## Basic VLA System Components

### 1. Speech Processing Node
```python
# speech_processor.py
import rclpy
from rclpy.node import Node
import whisper

class SpeechProcessorNode(Node):
    def __init__(self):
        super().__init__('speech_processor')
        # Initialize Whisper model
        self.model = whisper.load_model("base")
        # Create publisher for speech commands
        self.speech_pub = self.create_publisher(SpeechCommand, 'speech_command', 10)
        # Audio input setup
        # ... implementation details

def main(args=None):
    rclpy.init(args=args)
    node = SpeechProcessorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Language Understanding Node
```python
# language_understanding.py
import rclpy
from rclpy.node import Node

class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding')
        # Subscribe to speech commands
        self.speech_sub = self.create_subscription(
            SpeechCommand, 'speech_command', self.speech_callback, 10)
        # Publish structured intents
        self.intent_pub = self.create_publisher(Intent, 'structured_intent', 10)

    def speech_callback(self, msg):
        # Process speech and generate intent
        intent = self.process_speech(msg.utterance)
        self.intent_pub.publish(intent)

def main(args=None):
    rclpy.init(args=args)
    node = LanguageUnderstandingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

## Running the Complete System

### 1. Launch VLA System
```bash
# Launch the complete VLA system
ros2 launch humanoid_robotics_book vla_system.launch.py
```

### 2. Test with Sample Commands
```bash
# Send a test speech command
ros2 topic pub /speech_command humanoid_robotics_book/msg/SpeechCommand \
  "utterance: 'Go to the kitchen' \
  confidence: 0.9 \
  language: 'en'"
```

## Key Concepts

### VLA Pipeline Flow
1. **Speech Processing**: Convert voice to text
2. **Language Understanding**: Interpret intent from text
3. **Perception Integration**: Ground language in visual context
4. **Intent Interpretation**: Plan actions from intent
5. **Safety Validation**: Ensure safe execution
6. **Action Execution**: Execute navigation, manipulation, or social actions
7. **Feedback Generation**: Provide multimodal feedback

### Message Types
- `SpeechCommand`: Raw speech transcription with confidence
- `Intent`: Structured interpretation of user intent
- `SceneGraph`: Semantic representation of environment
- `ActionPlan`: Sequence of executable actions
- `VLAAction`: Individual action with parameters

## Troubleshooting

### Common Issues
1. **No Audio Input**: Check microphone permissions and ROS audio setup
2. **High Latency**: Ensure sufficient GPU resources for perception
3. **Navigation Failures**: Verify map and localization setup
4. **Perception Errors**: Check camera calibration and lighting

### Performance Tips
- Use smaller Whisper models for faster processing
- Optimize Nav2 parameters for your environment
- Ensure proper Isaac ROS hardware acceleration
- Monitor system resources during operation

## Next Steps
- Implement perception integration with Isaac ROS
- Add safety validation for action plans
- Integrate with Nav2 for navigation execution
- Create custom behaviors for social interaction
- Add comprehensive testing and error handling