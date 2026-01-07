#!/usr/bin/env python3
"""
Feedback Generator Node for Vision-Language-Action (VLA) System

This node provides multimodal feedback to users, including text-to-speech,
visual feedback through LEDs/displays, gestural confirmation, and error
communication mechanisms for humanoid robots.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from typing import Dict, List, Optional, Tuple
import time
import math

# Import ROS 2 messages
from std_msgs.msg import Header, String, ColorRGBA
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt8MultiArray, Int32
from visualization_msgs.msg import Marker, MarkerArray

# Import custom messages
from humanoid_robotics_book.msg import ActionPlan, VLAAction, SceneGraph, SpeechCommand


class TextToSpeech:
    """Handles text-to-speech functionality for verbal feedback"""

    def __init__(self):
        self.volume = 0.8
        self.rate = 200  # words per minute
        self.pitch = 1.0
        self.voice = "default"
        self.is_enabled = True

    def speak(self, text: str, priority: float = 0.5):
        """Generate speech from text"""
        if not self.is_enabled:
            print(f"[TTS] Would speak: {text}")
            return True

        # In a real system, this would call a TTS service
        # For simulation, we'll just print the text
        print(f"[TTS] Speaking: {text}")
        time.sleep(len(text) * 0.05)  # Simulate speaking time
        return True

    def set_volume(self, volume: float):
        """Set speech volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))

    def set_rate(self, rate: int):
        """Set speech rate in words per minute"""
        self.rate = max(50, min(400, rate))


class VisualFeedback:
    """Handles visual feedback through LEDs and displays"""

    def __init__(self):
        self.led_colors = {
            'idle': ColorRGBA(r=0.2, g=0.2, b=0.2, a=1.0),      # Dim white
            'processing': ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0), # Orange
            'success': ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),    # Green
            'error': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),      # Red
            'attention': ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),  # Blue
            'listening': ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)   # Yellow
        }
        self.current_state = 'idle'

    def set_state(self, state: str):
        """Set the visual feedback state"""
        if state in self.led_colors:
            self.current_state = state
            color = self.led_colors[state]
            print(f"[Visual] LED state: {state}, RGB({color.r:.1f}, {color.g:.1f}, {color.b:.1f})")
            return True
        else:
            print(f"[Visual] Unknown state: {state}")
            return False

    def blink(self, state: str, count: int = 3, interval: float = 0.5):
        """Blink the LEDs with the specified state"""
        for i in range(count * 2):  # Each blink is on/off cycle
            if i % 2 == 0:
                self.set_state(state)
            else:
                self.set_state('idle')
            time.sleep(interval)

        # Return to original state
        self.set_state(self.current_state)

    def pulse(self, state: str, duration: float = 2.0):
        """Pulse the LEDs with the specified state"""
        start_time = time.time()
        while time.time() - start_time < duration:
            # Simple pulse effect by varying intensity
            t = (time.time() - start_time) * 2  # Speed up the pulse
            intensity = 0.5 + 0.5 * math.sin(t * math.pi)  # Pulse between 0.5 and 1.0
            print(f"[Visual] Pulsing {state} with intensity {intensity:.2f}")
            time.sleep(0.1)


class GesturalFeedback:
    """Handles gestural feedback through robot movements"""

    def __init__(self):
        self.gesture_map = {
            'acknowledge': self._execute_acknowledge,
            'confirm': self._execute_confirm,
            'error': self._execute_error_gesture,
            'success': self._execute_success_gesture,
            'attention': self._execute_attention_gesture
        }

    def execute_gesture(self, gesture_type: str) -> bool:
        """Execute a specific gesture"""
        if gesture_type in self.gesture_map:
            return self.gesture_map[gesture_type]()
        else:
            print(f"[Gesture] Unknown gesture type: {gesture_type}")
            return False

    def _execute_acknowledge(self) -> bool:
        """Execute acknowledgment gesture (small nod)"""
        print("[Gesture] Executing acknowledgment (small nod)")
        time.sleep(0.5)  # Simulate gesture execution
        return True

    def _execute_confirm(self) -> bool:
        """Execute confirmation gesture (slight head turn)"""
        print("[Gesture] Executing confirmation (head turn)")
        time.sleep(0.5)  # Simulate gesture execution
        return True

    def _execute_success_gesture(self) -> bool:
        """Execute success gesture (nod and slight wave)"""
        print("[Gesture] Executing success gesture")
        time.sleep(0.7)  # Simulate gesture execution
        return True

    def _execute_error_gesture(self) -> bool:
        """Execute error gesture (shake head)"""
        print("[Gesture] Executing error gesture (head shake)")
        time.sleep(0.6)  # Simulate gesture execution
        return True

    def _execute_attention_gesture(self) -> bool:
        """Execute attention gesture (wave or look)"""
        print("[Gesture] Executing attention gesture")
        time.sleep(0.5)  # Simulate gesture execution
        return True


class ErrorCommunicator:
    """Handles error communication and recovery feedback"""

    def __init__(self):
        self.error_messages = {
            'navigation_failed': "I couldn't reach that location safely.",
            'object_not_found': "I couldn't find the object you mentioned.",
            'grasp_failed': "I had trouble picking up that object.",
            'perception_error': "I'm having trouble seeing clearly right now.",
            'safety_violation': "I can't do that as it's not safe.",
            'communication_error': "I didn't understand that command.",
            'execution_timeout': "I'm still working on that task.",
            'joint_limit': "I can't move that way right now."
        }

    def communicate_error(self, error_type: str, details: str = "") -> bool:
        """Communicate an error to the user"""
        if error_type in self.error_messages:
            error_msg = self.error_messages[error_type]
            if details:
                error_msg += f" {details}"
        else:
            error_msg = f"I encountered an issue: {error_type}"

        print(f"[Error] Communicating: {error_msg}")

        # Use multiple feedback modalities
        tts = TextToSpeech()
        visual = VisualFeedback()
        gesture = GesturalFeedback()

        # Verbal feedback
        tts.speak(error_msg, priority=0.8)

        # Visual feedback
        visual.set_state('error')

        # Gestural feedback
        gesture.execute_gesture('error')

        return True

    def suggest_alternative(self, error_type: str, alternatives: List[str]) -> bool:
        """Suggest alternative actions when an error occurs"""
        if alternatives:
            suggestion = f"Would you like me to {alternatives[0]} instead?"
            print(f"[Suggestion] {suggestion}")

            tts = TextToSpeech()
            tts.speak(suggestion, priority=0.7)
            return True
        return False


class ContextManager:
    """Manages feedback context and history"""

    def __init__(self):
        self.feedback_history = []
        self.max_history = 10
        self.current_context = "default"

    def add_feedback(self, feedback_type: str, message: str, timestamp: Time = None):
        """Add feedback to history"""
        if timestamp is None:
            timestamp = Time()

        feedback_entry = {
            'type': feedback_type,
            'message': message,
            'timestamp': timestamp,
            'context': self.current_context
        }

        self.feedback_history.append(feedback_entry)

        # Limit history size
        if len(self.feedback_history) > self.max_history:
            self.feedback_history = self.feedback_history[-self.max_history:]

    def get_recent_feedback(self, count: int = 3) -> List[Dict]:
        """Get recent feedback entries"""
        return self.feedback_history[-count:] if len(self.feedback_history) >= count else self.feedback_history

    def set_context(self, context: str):
        """Set the current feedback context"""
        self.current_context = context


class FeedbackGeneratorNode(Node):
    def __init__(self):
        super().__init__('feedback_generator')

        # Initialize components
        self.tts = TextToSpeech()
        self.visual = VisualFeedback()
        self.gesture = GesturalFeedback()
        self.error_communicator = ErrorCommunicator()
        self.context_manager = ContextManager()

        # Store latest data
        self.latest_action_plan = None
        self.latest_scene_graph = None
        self.latest_speech_command = None

        # Publishers and Subscribers
        self.action_plan_sub = self.create_subscription(
            ActionPlan,
            'action_plan',
            self.action_plan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.navigation_status_sub = self.create_subscription(
            ActionPlan,
            'navigation_status',
            self.navigation_status_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.manipulation_status_sub = self.create_subscription(
            ActionPlan,
            'manipulation_status',
            self.manipulation_status_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.social_status_sub = self.create_subscription(
            ActionPlan,
            'social_behavior_status',
            self.social_status_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.speech_command_sub = self.create_subscription(
            SpeechCommand,
            'speech_command',
            self.speech_command_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for feedback status
        self.feedback_status_pub = self.create_publisher(
            String,
            'feedback_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for LED control
        self.led_pub = self.create_publisher(
            UInt8MultiArray,
            '/led_control',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for display control
        self.display_pub = self.create_publisher(
            String,
            '/display_text',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Feedback Generator node initialized.')

    def speech_command_callback(self, msg: SpeechCommand):
        """Handle incoming speech commands with feedback"""
        self.get_logger().info(f'Received speech command: {msg.utterance}')

        # Provide feedback that command was received
        self.visual.set_state('processing')
        self.tts.speak("I heard you. Processing your request.", priority=0.6)

        # Add to context
        self.context_manager.add_feedback('input', f"Heard: {msg.utterance}", msg.timestamp)

    def action_plan_callback(self, msg: ActionPlan):
        """Provide feedback for action plan execution"""
        self.get_logger().info(f'Received action plan with {len(msg.actions)} actions')

        # Provide feedback about the plan
        action_types = [action.action_type for action in msg.actions]
        unique_actions = list(set(action_types))

        if len(unique_actions) == 1:
            feedback_msg = f"I will {unique_actions[0].replace('_', ' ')} for you."
        else:
            feedback_msg = f"I will execute a plan with {len(msg.actions)} actions."

        self.tts.speak(feedback_msg, priority=0.7)
        self.visual.set_state('processing')

        # Add to context
        self.context_manager.add_feedback('plan', f"Plan with {len(msg.actions)} actions", msg.timestamp)

    def navigation_status_callback(self, msg: ActionPlan):
        """Provide feedback for navigation status"""
        if msg.status == "completed":
            self.visual.set_state('success')
            self.gesture.execute_gesture('success')
            self.tts.speak("I have reached the destination.", priority=0.7)
            self.context_manager.add_feedback('navigation', "Navigation completed", msg.timestamp)
        elif msg.status == "failed":
            self.error_communicator.communicate_error('navigation_failed')
            self.context_manager.add_feedback('navigation', "Navigation failed", msg.timestamp)

    def manipulation_status_callback(self, msg: ActionPlan):
        """Provide feedback for manipulation status"""
        if msg.status == "completed":
            self.visual.set_state('success')
            self.gesture.execute_gesture('success')
            self.tts.speak("I have successfully manipulated the object.", priority=0.7)
            self.context_manager.add_feedback('manipulation', "Manipulation completed", msg.timestamp)
        elif msg.status == "failed":
            self.error_communicator.communicate_error('grasp_failed')
            self.context_manager.add_feedback('manipulation', "Manipulation failed", msg.timestamp)

    def social_status_callback(self, msg: ActionPlan):
        """Provide feedback for social behavior status"""
        if msg.status == "completed":
            self.visual.set_state('success')
            self.gesture.execute_gesture('acknowledge')
            self.tts.speak("Social interaction completed.", priority=0.6)
            self.context_manager.add_feedback('social', "Social behavior completed", msg.timestamp)
        elif msg.status == "failed":
            self.tts.speak("I had trouble with the social interaction.", priority=0.6)
            self.context_manager.add_feedback('social', "Social behavior failed", msg.timestamp)

    def provide_task_feedback(self, task_status: str, task_description: str = ""):
        """Provide feedback about task execution"""
        if task_status == "started":
            self.visual.set_state('processing')
            if task_description:
                self.tts.speak(f"Starting {task_description}.", priority=0.5)
        elif task_status == "completed":
            self.visual.set_state('success')
            self.gesture.execute_gesture('success')
            if task_description:
                self.tts.speak(f"Completed {task_description}.", priority=0.7)
        elif task_status == "failed":
            self.error_communicator.communicate_error('execution_timeout', task_description)
        elif task_status == "in_progress":
            self.visual.set_state('processing')
            if task_description:
                self.tts.speak(f"Working on {task_description}.", priority=0.5)

    def provide_error_feedback(self, error_type: str, details: str = ""):
        """Provide comprehensive error feedback"""
        success = self.error_communicator.communicate_error(error_type, details)
        if success:
            self.context_manager.add_feedback('error', f"{error_type}: {details}", Time())

    def provide_confirmation_feedback(self, action_description: str):
        """Provide confirmation feedback for actions"""
        self.gesture.execute_gesture('confirm')
        self.tts.speak(f"OK, I will {action_description}.", priority=0.6)
        self.context_manager.add_feedback('confirmation', action_description, Time())

    def provide_attention_feedback(self):
        """Provide attention-getting feedback"""
        self.visual.set_state('attention')
        self.gesture.execute_gesture('attention')
        self.tts.speak("I'm ready to help you.", priority=0.8)
        self.context_manager.add_feedback('attention', "Attention requested", Time())


def main(args=None):
    rclpy.init(args=args)

    try:
        feedback_generator = FeedbackGeneratorNode()

        try:
            rclpy.spin(feedback_generator)
        except KeyboardInterrupt:
            pass
        finally:
            feedback_generator.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()