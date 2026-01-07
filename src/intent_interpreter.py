#!/usr/bin/env python3
"""
Intent Interpreter Node for Vision-Language-Action (VLA) System

This node converts structured language intents into executable action plans
for humanoid robots, performing task decomposition, constraint checking,
action sequencing, and resource allocation.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

# Import ROS 2 messages
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time, Duration

# Import custom messages
# Note: We'll need to adjust these imports once the package is properly set up
# from humanoid_robotics_book.msg import Intent, SceneGraph, VLAAction, ActionPlan


class ActionType(Enum):
    """Types of actions that can be executed by the robot"""
    NAVIGATE_TO = "navigate_to"
    GRASP_OBJECT = "grasp_object"
    PLACE_OBJECT = "place_object"
    FOLLOW_PERSON = "follow_person"
    WAVE = "wave"
    PICK_UP = "pick_up"
    SPEAK = "speak"
    LISTEN = "listen"
    DETECT_OBJECT = "detect_object"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"


@dataclass
class Task:
    """Represents a task to be executed"""
    id: str
    name: str
    action_type: ActionType
    parameters: List[str]
    target_pose: Pose
    priority: float
    dependencies: List[str]
    estimated_duration: float


class RobotCapabilities:
    """Defines the capabilities of the humanoid robot"""

    def __init__(self):
        # Navigation capabilities
        self.can_navigate = True
        self.max_navigation_distance = 100.0  # meters
        self.min_navigation_distance = 0.1     # meters
        self.navigation_speed = 0.5            # m/s

        # Manipulation capabilities
        self.has_arms = True
        self.has_grippers = True
        self.max_reach = 1.5  # meters
        self.min_object_size = 0.01  # meters (1cm)
        self.max_object_weight = 2.0  # kg
        self.manipulation_speed = 0.3  # m/s

        # Perception capabilities
        self.can_detect_objects = True
        self.can_detect_people = True
        self.detection_range = 5.0  # meters

        # Social capabilities
        self.has_gestures = True
        self.has_speech = True


class TaskDecomposer:
    """Decomposes complex commands into subtasks"""

    def __init__(self):
        self.robot_capabilities = RobotCapabilities()

    def decompose_task(self, intent, scene_graph=None) -> List[Task]:
        """Decompose a high-level intent into executable tasks"""
        tasks = []

        if intent.command_type == "navigation":
            tasks = self._decompose_navigation_task(intent, scene_graph)
        elif intent.command_type == "manipulation":
            tasks = self._decompose_manipulation_task(intent, scene_graph)
        elif intent.command_type == "interaction":
            tasks = self._decompose_interaction_task(intent, scene_graph)
        elif intent.command_type == "social":
            tasks = self._decompose_social_task(intent, scene_graph)

        return tasks

    def _decompose_navigation_task(self, intent, scene_graph=None) -> List[Task]:
        """Decompose navigation tasks"""
        tasks = []

        # Find target location
        target_location = None
        for entity in intent.entities:
            if entity in ['kitchen', 'living room', 'bedroom', 'bathroom', 'office']:
                target_location = entity
                break

        if target_location:
            # For simulation, we'll use a predefined location
            pose = Pose()
            if target_location == 'kitchen':
                pose.position.x = 2.0
                pose.position.y = 1.0
            elif target_location == 'living room':
                pose.position.x = -1.0
                pose.position.y = 1.0
            elif target_location == 'bedroom':
                pose.position.x = 2.0
                pose.position.y = -1.0
            else:
                pose.position.x = 0.0
                pose.position.y = 0.0

            pose.orientation.w = 1.0

            task = Task(
                id=f"nav_{target_location}",
                name=f"Navigate to {target_location}",
                action_type=ActionType.NAVIGATE_TO,
                parameters=[target_location],
                target_pose=pose,
                priority=0.8,
                dependencies=[],
                estimated_duration=10.0  # 10 seconds
            )
            tasks.append(task)

        return tasks

    def _decompose_manipulation_task(self, intent, scene_graph=None) -> List[Task]:
        """Decompose manipulation tasks"""
        tasks = []

        # Identify target object
        target_object = None
        target_pose = Pose()

        if scene_graph and scene_graph.object_classes:
            # Find the object that matches the intent
            for i, obj_class in enumerate(scene_graph.object_classes):
                if any(entity in obj_class for entity in intent.entities):
                    target_object = scene_graph.object_ids[i]
                    target_pose = scene_graph.object_poses[i]
                    break

        if not target_object and intent.entities:
            # If no object found in scene graph, create a detection task first
            detect_task = Task(
                id="detect_object",
                name="Detect object",
                action_type=ActionType.DETECT_OBJECT,
                parameters=intent.entities,
                target_pose=Pose(),  # No specific target
                priority=0.9,
                dependencies=[],
                estimated_duration=5.0
            )
            tasks.append(detect_task)

            # Then grasp task
            grasp_task = Task(
                id="grasp_object",
                name="Grasp object",
                action_type=ActionType.GRASP_OBJECT,
                parameters=intent.entities,
                target_pose=target_pose,
                priority=0.9,
                dependencies=["detect_object"],
                estimated_duration=10.0
            )
            tasks.append(grasp_task)
        elif target_object:
            # Direct grasp task
            grasp_task = Task(
                id="grasp_object",
                name="Grasp object",
                action_type=ActionType.GRASP_OBJECT,
                parameters=[target_object],
                target_pose=target_pose,
                priority=0.9,
                dependencies=[],
                estimated_duration=10.0
            )
            tasks.append(grasp_task)

        return tasks

    def _decompose_interaction_task(self, intent, scene_graph=None) -> List[Task]:
        """Decompose interaction tasks"""
        tasks = []

        # For interaction tasks, we typically need to navigate to the person/object first
        if 'person' in intent.entities or 'human' in intent.entities:
            # Navigate to person
            navigate_task = Task(
                id="navigate_to_person",
                name="Navigate to person",
                action_type=ActionType.NAVIGATE_TO,
                parameters=["person"],
                target_pose=Pose(),  # Will be filled by navigation system
                priority=0.8,
                dependencies=[],
                estimated_duration=15.0
            )
            tasks.append(navigate_task)

            # Then interact (e.g., speak)
            speak_task = Task(
                id="speak",
                name="Speak to person",
                action_type=ActionType.SPEAK,
                parameters=["Hello, how can I help you?"],
                target_pose=Pose(),
                priority=0.7,
                dependencies=["navigate_to_person"],
                estimated_duration=5.0
            )
            tasks.append(speak_task)

        return tasks

    def _decompose_social_task(self, intent, scene_graph=None) -> List[Task]:
        """Decompose social tasks"""
        tasks = []

        if 'wave' in intent.raw_command.lower():
            wave_task = Task(
                id="wave",
                name="Wave gesture",
                action_type=ActionType.WAVE,
                parameters=[],
                target_pose=Pose(),
                priority=0.6,
                dependencies=[],
                estimated_duration=3.0
            )
            tasks.append(wave_task)

        return tasks


class ConstraintChecker:
    """Validates actions against robot capabilities and constraints"""

    def __init__(self):
        self.robot_capabilities = RobotCapabilities()

    def check_constraints(self, tasks: List[Task]) -> Tuple[bool, List[str]]:
        """Check if tasks are feasible given robot capabilities"""
        errors = []

        for task in tasks:
            action_type = task.action_type

            # Check navigation constraints
            if action_type == ActionType.NAVIGATE_TO:
                if not self.robot_capabilities.can_navigate:
                    errors.append("Robot cannot navigate")
                else:
                    # Check distance constraints
                    dist = self._calculate_distance(task.target_pose)
                    if dist > self.robot_capabilities.max_navigation_distance:
                        errors.append(f"Navigation distance {dist:.2f}m exceeds maximum {self.robot_capabilities.max_navigation_distance}m")
                    elif dist < self.robot_capabilities.min_navigation_distance:
                        errors.append(f"Navigation distance {dist:.2f}m is less than minimum {self.robot_capabilities.min_navigation_distance}m")

            # Check manipulation constraints
            elif action_type in [ActionType.GRASP_OBJECT, ActionType.PICK_UP]:
                if not self.robot_capabilities.has_arms:
                    errors.append("Robot has no arms for manipulation")
                elif not self.robot_capabilities.has_grippers:
                    errors.append("Robot has no grippers for grasping")
                else:
                    # Check reach constraints
                    dist = self._calculate_distance(task.target_pose)
                    if dist > self.robot_capabilities.max_reach:
                        errors.append(f"Object distance {dist:.2f}m exceeds maximum reach {self.robot_capabilities.max_reach}m")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _calculate_distance(self, pose: Pose) -> float:
        """Calculate distance to a pose from origin"""
        return ((pose.position.x ** 2 + pose.position.y ** 2 + pose.position.z ** 2) ** 0.5)


class ActionSequencer:
    """Orders actions appropriately based on dependencies and priorities"""

    def __init__(self):
        pass

    def sequence_actions(self, tasks: List[Task]) -> List[Task]:
        """Sequence tasks based on dependencies and priorities"""
        # Sort by priority (highest first) but respect dependencies
        sorted_tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)

        # Ensure dependencies are satisfied
        sequenced_tasks = []
        completed_tasks = set()

        # Process tasks respecting dependencies
        while len(sequenced_tasks) < len(sorted_tasks):
            scheduled = False
            for task in sorted_tasks:
                if task.id not in completed_tasks:
                    # Check if all dependencies are satisfied
                    all_deps_met = all(dep in completed_tasks for dep in task.dependencies)

                    if all_deps_met:
                        sequenced_tasks.append(task)
                        completed_tasks.add(task.id)
                        scheduled = True
                        break

            if not scheduled:
                # If no task was scheduled, there might be circular dependencies
                # For now, just add the first unscheduled task
                for task in sorted_tasks:
                    if task.id not in completed_tasks:
                        sequenced_tasks.append(task)
                        completed_tasks.add(task.id)
                        break

        return sequenced_tasks


class ResourceAllocator:
    """Manages allocation of robot resources for action execution"""

    def __init__(self):
        self.resources = {
            'navigation': True,
            'manipulation_arm_left': True,
            'manipulation_arm_right': True,
            'gripper_left': True,
            'gripper_right': True,
            'perception': True,
            'speech': True,
            'gestures': True
        }

    def allocate_resources(self, tasks: List[Task]) -> List[str]:
        """Allocate resources for the task plan"""
        allocated_resources = []

        for task in tasks:
            action_type = task.action_type

            if action_type in [ActionType.NAVIGATE_TO]:
                if self.resources['navigation']:
                    allocated_resources.append('navigation')
                    self.resources['navigation'] = False
            elif action_type in [ActionType.GRASP_OBJECT, ActionType.PICK_UP, ActionType.OPEN_GRIPPER, ActionType.CLOSE_GRIPPER]:
                # Prefer right arm/gripper
                if self.resources['gripper_right']:
                    allocated_resources.extend(['manipulation_arm_right', 'gripper_right'])
                    self.resources['manipulation_arm_right'] = False
                    self.resources['gripper_right'] = False
                elif self.resources['gripper_left']:
                    allocated_resources.extend(['manipulation_arm_left', 'gripper_left'])
                    self.resources['manipulation_arm_left'] = False
                    self.resources['gripper_left'] = False

        return allocated_resources

    def release_resources(self, resources: List[str]):
        """Release allocated resources"""
        for resource in resources:
            if resource in self.resources:
                self.resources[resource] = True


class IntentInterpreterNode(Node):
    def __init__(self):
        super().__init__('intent_interpreter')

        # Initialize components
        self.task_decomposer = TaskDecomposer()
        self.constraint_checker = ConstraintChecker()
        self.action_sequencer = ActionSequencer()
        self.resource_allocator = ResourceAllocator()

        # Store the latest scene graph for grounding
        self.latest_scene_graph = None

        # Publishers and Subscribers
        # self.intent_sub = self.create_subscription(
        #     Intent,
        #     'structured_intent',
        #     self.intent_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )
        #
        # self.scene_graph_sub = self.create_subscription(
        #     SceneGraph,
        #     'scene_graph',
        #     self.scene_graph_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )
        #
        # self.action_plan_pub = self.create_publisher(
        #     ActionPlan,
        #     'action_plan',
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )

        self.get_logger().info('Intent Interpreter node initialized.')

    def scene_graph_callback(self, msg):
        """Store the latest scene graph for grounding actions"""
        self.latest_scene_graph = msg

    def intent_callback(self, msg):
        """Process incoming intents and generate action plans"""
        self.get_logger().info(f'Received intent: {msg.command_type} with entities {msg.entities}')

        # Decompose the task into subtasks
        tasks = self.task_decomposer.decompose_task(msg, self.latest_scene_graph)

        if not tasks:
            self.get_logger().info('No tasks generated for intent')
            return

        # Check constraints
        is_valid, errors = self.constraint_checker.check_constraints(tasks)
        if not is_valid:
            self.get_logger().error(f'Constraint violations: {errors}')
            # Still proceed but with reduced confidence
            overall_confidence = 0.3
        else:
            overall_confidence = min(msg.confidence + 0.1, 1.0)  # Boost confidence slightly

        # Sequence the tasks
        sequenced_tasks = self.action_sequencer.sequence_actions(tasks)

        # Allocate resources
        allocated_resources = self.resource_allocator.allocate_resources(sequenced_tasks)

        # Create action plan
        # plan = ActionPlan()
        # plan.header.stamp = self.get_clock().now().to_msg()
        # plan.header.frame_id = 'map'
        # plan.timestamp = plan.header.stamp
        # plan.plan_id = str(uuid.uuid4())
        # plan.plan_type = msg.command_type
        #
        # # Convert tasks to VLAActions
        # plan.actions = []
        # for task in sequenced_tasks:
        #     action = VLAAction()
        #     action.action_type = task.action_type.value
        #     action.parameters = task.parameters
        #     action.target_pose = task.target_pose
        #     action.priority = task.priority
        #     action.is_optional = False
        #     action.timeout.sec = int(task.estimated_duration)
        #     action.timeout.nanosec = int((task.estimated_duration - int(task.estimated_duration)) * 1e9)
        #     action.preconditions = []
        #     action.effects = []
        #     plan.actions.append(action)
        #
        # plan.action_dependencies = []  # Could be computed from task dependencies
        # plan.estimated_duration = sum(task.estimated_duration for task in sequenced_tasks)
        # plan.confidence = overall_confidence
        # plan.resources_required = allocated_resources
        # plan.status = 'planned'  # Initially planned
        # plan.context_id = msg.context_id
        # plan.original_command = msg.raw_command
        #
        # # Add constraints
        # plan.execution_constraints = [1.0]  # Default execution parameters
        # plan.safety_constraints = ['maintain_balance', 'avoid_collisions']
        # plan.temporal_constraints = []

        # For now, just log the plan
        self.get_logger().info(f'Generated action plan with {len(sequenced_tasks)} tasks')
        for i, task in enumerate(sequenced_tasks):
            self.get_logger().info(f'  Task {i+1}: {task.action_type.value} - {task.name} (priority: {task.priority})')

        # Publish the action plan
        # self.action_plan_pub.publish(plan)
        self.get_logger().info(f'Published action plan with {len(sequenced_tasks)} tasks')

    def _estimate_duration(self, tasks: List[Task]) -> float:
        """Estimate total duration of the task plan"""
        total_duration = 0.0

        for task in tasks:
            # Estimate duration based on action type
            if task.action_type == ActionType.NAVIGATE_TO:
                total_duration += 10.0  # 10 seconds for navigation
            elif task.action_type in [ActionType.GRASP_OBJECT, ActionType.PICK_UP]:
                total_duration += 15.0  # 15 seconds for grasping
            elif task.action_type == ActionType.WAVE:
                total_duration += 5.0   # 5 seconds for waving
            elif task.action_type == ActionType.SPEAK:
                # Estimate based on text length
                total_duration += 2.0 + (len(task.parameters[0]) * 0.1 if task.parameters else 0)
            else:
                total_duration += 5.0   # Default 5 seconds

        return total_duration


def main(args=None):
    rclpy.init(args=args)

    try:
        intent_interpreter = IntentInterpreterNode()

        try:
            rclpy.spin(intent_interpreter)
        except KeyboardInterrupt:
            pass
        finally:
            intent_interpreter.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()