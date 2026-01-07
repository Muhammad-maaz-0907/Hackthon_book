#!/usr/bin/env python3
"""
VLA Main Application - Vision-Language-Action System for Humanoid Robotics

This is the main orchestrator for the complete VLA system, integrating all
components to process end-to-end voice commands, execute complex multi-step
tasks, handle error recovery, and manage the complete system workflow.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from typing import Dict, List, Optional, Tuple
import time
import threading
import queue
import traceback

# Import ROS 2 messages
from std_msgs.msg import String, Header
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

# Import custom messages
from humanoid_robotics_book.msg import SpeechCommand, Intent, SceneGraph, ActionPlan, VLAAction, SocialBehavior
from humanoid_robotics_book.srv import SafetyCheck


class VLAOrchestratorNode(Node):
    """
    Main orchestrator for the Vision-Language-Action system.
    Coordinates all VLA components to process voice commands end-to-end.
    """

    def __init__(self):
        super().__init__('vla_main')

        # System state tracking
        self.system_state = 'idle'  # idle, processing, executing, error
        self.active_task_id = None
        self.conversation_context = {}
        self.system_status = {
            'speech_processor_ready': False,
            'language_understanding_ready': False,
            'perception_ready': False,
            'safety_validator_ready': False,
            'navigation_ready': False,
            'manipulation_ready': False,
            'social_ready': False,
            'feedback_ready': False
        }

        # Queues for internal communication
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # Publishers for system coordination
        self.status_pub = self.create_publisher(
            String,
            'vla_system_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Subscribers for monitoring system status
        self.speech_sub = self.create_subscription(
            SpeechCommand,
            'speech_command',
            self.speech_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.intent_sub = self.create_subscription(
            Intent,
            'structured_intent',
            self.intent_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.scene_graph_sub = self.create_subscription(
            SceneGraph,
            'scene_graph',
            self.scene_graph_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

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
            SocialBehavior,
            'social_behavior_status',
            self.social_status_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Service client for safety validation
        self.safety_client = self.create_client(SafetyCheck, 'safety_check')
        while not self.safety_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for safety check service...')

        # Timer for system monitoring
        self.system_monitor_timer = self.create_timer(1.0, self.system_monitor_callback)

        self.get_logger().info('VLA Main Orchestrator initialized.')

    def speech_callback(self, msg: SpeechCommand):
        """Handle incoming speech commands"""
        self.get_logger().info(f'Received speech command: {msg.utterance}')

        # Update system state
        self.system_state = 'processing'
        self.publish_status()

        # Process the command through the pipeline
        self.process_speech_command(msg)

    def intent_callback(self, msg: Intent):
        """Handle incoming intents"""
        self.get_logger().info(f'Received intent: {msg.command_type}')

        # Convert intent to action plan
        self.generate_action_plan(msg)

    def scene_graph_callback(self, msg: SceneGraph):
        """Handle incoming scene graphs"""
        self.get_logger().info(f'Received scene graph with {len(msg.object_ids)} objects')

        # Update perception status
        self.system_status['perception_ready'] = True

    def action_plan_callback(self, msg: ActionPlan):
        """Handle incoming action plans"""
        self.get_logger().info(f'Received action plan with {len(msg.actions)} actions')

        # Validate the plan for safety
        self.validate_and_execute_plan(msg)

    def navigation_status_callback(self, msg: ActionPlan):
        """Handle navigation status updates"""
        self.get_logger().info(f'Navigation status: {msg.status}')

        if msg.status == 'completed':
            self.check_task_completion()

    def manipulation_status_callback(self, msg: ActionPlan):
        """Handle manipulation status updates"""
        self.get_logger().info(f'Manipulation status: {msg.status}')

        if msg.status == 'completed':
            self.check_task_completion()

    def social_status_callback(self, msg: SocialBehavior):
        """Handle social behavior status updates"""
        self.get_logger().info(f'Social behavior status: {msg.behavior_name}')

        # For now, just log the status
        pass

    def system_monitor_callback(self):
        """Monitor system status and health"""
        status_msg = String()
        status_msg.data = f"State: {self.system_state}, Components: {self.count_ready_components()}/8"
        self.status_pub.publish(status_msg)

        # Log system health
        self.get_logger().debug(f'System state: {self.system_state}')
        self.get_logger().debug(f'Ready components: {self.count_ready_components()}/8')

    def count_ready_components(self) -> int:
        """Count how many components are ready"""
        count = 0
        for ready in self.system_status.values():
            if ready:
                count += 1
        return count

    def process_speech_command(self, speech_cmd: SpeechCommand):
        """Process a speech command through the VLA pipeline"""
        try:
            self.get_logger().info(f'Processing speech command: {speech_cmd.utterance}')

            # Publish status update
            status_msg = String()
            status_msg.data = f"Processing: {speech_cmd.utterance}"
            self.status_pub.publish(status_msg)

            # The speech processing happens in the speech_processor node
            # We just need to ensure the pipeline is working
            self.get_logger().info('Speech command forwarded to processing pipeline')

        except Exception as e:
            self.get_logger().error(f'Error processing speech command: {e}')
            self.handle_error('speech_processing_error', str(e))

    def generate_action_plan(self, intent: Intent):
        """Generate an action plan from an intent"""
        try:
            self.get_logger().info(f'Generating action plan for intent: {intent.command_type}')

            # In a real system, this would be handled by the intent_interpreter
            # For simulation, we'll create a simple plan based on intent type
            plan = ActionPlan()
            plan.header.stamp = self.get_clock().now().to_msg()
            plan.plan_type = intent.command_type
            plan.original_command = intent.raw_command
            plan.confidence = intent.confidence
            plan.context_id = intent.context_id

            # Create actions based on intent type
            if intent.command_type == "navigation":
                action = VLAAction()
                action.action_type = "navigate_to"
                action.parameters = intent.entities
                action.priority = 0.8
                plan.actions = [action]

            elif intent.command_type == "manipulation":
                action = VLAAction()
                action.action_type = "grasp_object"
                action.parameters = intent.entities
                action.priority = 0.9
                plan.actions = [action]

            elif intent.command_type == "social":
                action = VLAAction()
                action.action_type = "wave"
                action.parameters = intent.entities
                action.priority = 0.7
                plan.actions = [action]

            else:
                # Default action for other intent types
                action = VLAAction()
                action.action_type = "listen"
                action.parameters = ["acknowledge"]
                action.priority = 0.5
                plan.actions = [action]

            plan.status = "planned"
            plan.estimated_duration = len(plan.actions) * 5.0  # 5 seconds per action

            # Publish the plan
            # In a real system, we'd have a publisher for action plans
            # For now, we'll just log that we created a plan
            self.get_logger().info(f'Generated action plan with {len(plan.actions)} actions')

        except Exception as e:
            self.get_logger().error(f'Error generating action plan: {e}')
            self.handle_error('action_plan_generation_error', str(e))

    def validate_and_execute_plan(self, plan: ActionPlan):
        """Validate and execute an action plan"""
        try:
            self.get_logger().info(f'Validating and executing plan with {len(plan.actions)} actions')

            # Create safety check request
            request = SafetyCheck.Request()
            request.action_plan = plan

            # Call safety service
            future = self.safety_client.call_async(request)

            # Wait for response (with timeout)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

            if future.result() is not None:
                response = future.result()

                if response.is_safe:
                    self.get_logger().info('Action plan validated as safe, proceeding with execution')
                    self.execute_action_plan(plan)
                else:
                    self.get_logger().warn(f'Action plan unsafe: {response.safety_issues}')
                    self.handle_safety_violation(plan, response)
            else:
                self.get_logger().error('Safety check service call failed')
                self.handle_error('safety_check_failed', 'Could not validate plan safety')

        except Exception as e:
            self.get_logger().error(f'Error validating and executing plan: {e}')
            self.handle_error('plan_validation_error', str(e))

    def execute_action_plan(self, plan: ActionPlan):
        """Execute an action plan"""
        try:
            self.get_logger().info(f'Executing action plan: {plan.plan_type}')
            self.system_state = 'executing'
            self.publish_status()

            # In a real system, this would trigger the execution nodes
            # For simulation, we'll just process each action
            for i, action in enumerate(plan.actions):
                self.get_logger().info(f'Executing action {i+1}/{len(plan.actions)}: {action.action_type}')

                # Simulate action execution
                success = self.simulate_action_execution(action)

                if not success:
                    self.get_logger().error(f'Action {i+1} failed: {action.action_type}')
                    self.handle_action_failure(action, plan)
                    return False

                # Small delay between actions
                time.sleep(0.5)

            # Plan completed successfully
            self.get_logger().info('Action plan completed successfully')
            plan.status = 'completed'
            self.system_state = 'idle'
            self.publish_status()

            return True

        except Exception as e:
            self.get_logger().error(f'Error executing action plan: {e}')
            self.handle_error('plan_execution_error', str(e))
            return False

    def simulate_action_execution(self, action: VLAAction) -> bool:
        """Simulate action execution - in real system, this would call actual execution nodes"""
        try:
            # Simulate different types of actions
            if action.action_type == "navigate_to":
                self.get_logger().info(f'Simulating navigation to: {action.parameters}')
                time.sleep(2)  # Simulate navigation time
            elif action.action_type == "grasp_object":
                self.get_logger().info(f'Simulating grasp of: {action.parameters}')
                time.sleep(1.5)  # Simulate grasp time
            elif action.action_type == "wave":
                self.get_logger().info('Simulating wave gesture')
                time.sleep(1)  # Simulate gesture time
            elif action.action_type == "speak":
                self.get_logger().info(f'Simulating speech: {action.parameters}')
                time.sleep(1)  # Simulate speech time
            else:
                self.get_logger().info(f'Simulating action: {action.action_type}')
                time.sleep(1)  # Default simulation time

            return True  # Simulate success

        except Exception as e:
            self.get_logger().error(f'Error simulating action execution: {e}')
            return False

    def handle_safety_violation(self, plan: ActionPlan, safety_response):
        """Handle safety violations in action plans"""
        self.get_logger().warn(f'Safety violation detected: {safety_response.safety_issues}')

        # In a real system, we would implement recovery strategies
        # For now, just log the issues and fail the plan
        plan.status = 'failed'
        self.system_state = 'error'

        # Suggest mitigation
        if safety_response.mitigation_suggestions:
            self.get_logger().info(f'Mitigation suggestions: {safety_response.mitigation_suggestions}')

        self.publish_status()

    def handle_action_failure(self, action: VLAAction, plan: ActionPlan):
        """Handle action failures during execution"""
        self.get_logger().error(f'Action failed: {action.action_type}')

        # Update plan status
        plan.status = 'failed'
        self.system_state = 'error'

        # In a real system, we would implement recovery strategies
        # For now, just log the failure

        self.publish_status()

    def handle_error(self, error_type: str, details: str):
        """Handle errors in the VLA system"""
        self.get_logger().error(f'VLA Error [{error_type}]: {details}')

        # Update system state
        self.system_state = 'error'
        self.publish_status()

        # In a real system, we would implement error recovery
        # For now, just log the error and return to idle after a delay
        time.sleep(2)  # Brief delay before returning to idle
        self.system_state = 'idle'
        self.publish_status()

    def check_task_completion(self):
        """Check if the current task is completed"""
        # In a real system, this would check if all actions in the current task are done
        # For simulation, we'll just log completion
        self.get_logger().info('Task completion check - all actions completed')

        # Return to idle state
        if self.system_state == 'executing':
            self.system_state = 'idle'
            self.publish_status()

    def publish_status(self):
        """Publish current system status"""
        status_msg = String()
        status_msg.data = f"State: {self.system_state}, Active Task: {self.active_task_id}"
        self.status_pub.publish(status_msg)


class VLALauncher:
    """Launcher for the complete VLA system"""

    def __init__(self):
        self.nodes = []

    def launch_system(self):
        """Launch all VLA components"""
        rclpy.init()

        # Create the orchestrator node
        orchestrator = VLAOrchestratorNode()
        self.nodes.append(orchestrator)

        # In a real system, we would launch all component nodes here
        # For this example, we're just running the orchestrator

        # Use multi-threaded executor to handle multiple callbacks
        executor = MultiThreadedExecutor(num_threads=4)

        for node in self.nodes:
            executor.add_node(node)

        try:
            self.get_logger().info('VLA System launched successfully')
            self.get_logger().info('Ready to process voice commands...')

            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown all VLA components"""
        self.get_logger().info('Shutting down VLA system...')

        for node in self.nodes:
            node.destroy_node()

        rclpy.shutdown()

        self.get_logger().info('VLA system shutdown complete')


def main(args=None):
    """Main entry point for the VLA system"""
    launcher = VLALauncher()
    launcher.launch_system()


if __name__ == '__main__':
    main()