#!/usr/bin/env python3
"""
Social Behavior Executor Node for Vision-Language-Action (VLA) System

This node executes social behaviors and human-aware actions for humanoid robots,
including social navigation following, expressive behaviors, human attention
management, and social protocol adherence.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from typing import Dict, List, Optional, Tuple
import math
import time
import random

# Import ROS 2 messages and actions
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from builtin_interfaces.msg import Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

# Import custom messages
from humanoid_robotics_book.msg import ActionPlan, VLAAction, SceneGraph, SocialBehavior


class SocialNavigation:
    """Handles social navigation and human-aware movement"""

    def __init__(self):
        self.respectful_distance = 0.8  # meters
        self.following_distance = 1.2   # meters
        self.side_offset = 0.5          # meters to the side when following
        self.social_navigation_enabled = True

    def adjust_navigation_for_social_context(self, target_pose: Pose, robot_pose: Pose,
                                           target_type: str = "person") -> Tuple[Pose, str]:
        """Adjust navigation to be socially appropriate"""
        if not self.social_navigation_enabled:
            return target_pose, "direct_navigation"

        # Calculate distance to target
        dx = target_pose.position.x - robot_pose.position.x
        dy = target_pose.position.y - robot_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if target_type == "person":
            # If following a person, maintain appropriate distance
            if distance < self.following_distance:
                # Move back to maintain distance
                direction_x = dx / distance if distance > 0 else 1.0
                direction_y = dy / distance if distance > 0 else 0.0

                adjusted_pose = Pose()
                adjusted_pose.position.x = target_pose.position.x - direction_x * self.following_distance
                adjusted_pose.position.y = target_pose.position.y - direction_y * self.following_distance
                adjusted_pose.position.z = robot_pose.position.z
                adjusted_pose.orientation = robot_pose.orientation

                return adjusted_pose, "maintain_distance"
            else:
                # If too far, move closer
                if distance > self.following_distance * 1.5:
                    direction_x = dx / distance if distance > 0 else 1.0
                    direction_y = dy / distance if distance > 0 else 0.0

                    adjusted_pose = Pose()
                    adjusted_pose.position.x = robot_pose.position.x + direction_x * self.following_distance
                    adjusted_pose.position.y = robot_pose.position.y + direction_y * self.following_distance
                    adjusted_pose.position.z = robot_pose.position.z
                    adjusted_pose.orientation = robot_pose.orientation

                    return adjusted_pose, "move_closer"

        return target_pose, "direct_navigation"

    def calculate_social_path(self, start_pose: Pose, goal_pose: Pose, scene_graph: Optional[SceneGraph] = None) -> List[Pose]:
        """Calculate a socially appropriate path considering humans in the environment"""
        path = [start_pose, goal_pose]  # Simplified path

        if scene_graph:
            # Adjust path to respect human personal space
            for i, obj_class in enumerate(scene_graph.object_classes):
                if 'person' in obj_class.lower():
                    person_pose = scene_graph.object_poses[i]

                    # Check if path goes near the person
                    # For simplicity, we'll just add a waypoint to go around
                    if self._is_path_near_person(start_pose, goal_pose, person_pose, self.respectful_distance):
                        # Add a waypoint to go around the person
                        waypoint = Pose()
                        waypoint.position.x = person_pose.position.x + self.respectful_distance
                        waypoint.position.y = person_pose.position.y + self.respectful_distance
                        waypoint.position.z = person_pose.position.z
                        waypoint.orientation = goal_pose.orientation

                        path.insert(1, waypoint)

        return path

    def _is_path_near_person(self, start: Pose, end: Pose, person: Pose, threshold: float) -> bool:
        """Check if path between start and end goes near the person"""
        # Simplified check - in reality, this would be more complex
        # Calculate distance from person to line segment between start and end
        start_to_person = math.sqrt(
            (person.position.x - start.position.x)**2 +
            (person.position.y - start.position.y)**2
        )
        end_to_person = math.sqrt(
            (person.position.x - end.position.x)**2 +
            (person.position.y - end.position.y)**2
        )

        return min(start_to_person, end_to_person) < threshold


class ExpressiveBehaviors:
    """Manages expressive behaviors like gestures, head movements, etc."""

    def __init__(self):
        self.head_joints = ['head_pan', 'head_tilt']
        self.arm_joints = ['left_arm_shoulder', 'left_arm_elbow', 'right_arm_shoulder', 'right_arm_elbow']
        self.led_joints = []  # For LED expressions if available

        # Predefined gesture trajectories
        self.gesture_trajectories = {
            'wave': self._create_wave_trajectory(),
            'nod': self._create_nod_trajectory(),
            'point': self._create_point_trajectory(),
            'greeting': self._create_greeting_trajectory(),
            'acknowledge': self._create_acknowledge_trajectory()
        }

    def _create_wave_trajectory(self) -> JointTrajectory:
        """Create trajectory for waving gesture"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # Wave motion: move right arm up and down
        for i in range(4):  # 4 points for the wave
            point = JointTrajectoryPoint()
            point.time_from_start.sec = i

            if i % 2 == 0:
                # Raised position
                positions = [0.0, 0.0, 0.5, 0.0]  # Right arm raised
            else:
                # Lowered position
                positions = [0.0, 0.0, 0.0, 0.0]  # Right arm lowered

            point.positions = positions
            point.velocities = [0.0] * len(self.arm_joints)
            point.accelerations = [0.0] * len(self.arm_joints)

            trajectory.points.append(point)

        return trajectory

    def _create_nod_trajectory(self) -> JointTrajectory:
        """Create trajectory for nodding gesture"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.head_joints

        # Nod motion: move head up and down
        for i in range(3):  # 3 points for the nod
            point = JointTrajectoryPoint()
            point.time_from_start.sec = i

            if i == 1:  # Middle point - nod down
                positions = [0.0, -0.3]  # Head tilted down
            else:  # First and last - head neutral
                positions = [0.0, 0.0]  # Head neutral

            point.positions = positions
            point.velocities = [0.0] * len(self.head_joints)
            point.accelerations = [0.0] * len(self.head_joints)

            trajectory.points.append(point)

        return trajectory

    def _create_point_trajectory(self) -> JointTrajectory:
        """Create trajectory for pointing gesture"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # Pointing: extend right arm forward
        point1 = JointTrajectoryPoint()
        point1.time_from_start.sec = 0
        point1.positions = [0.0, 0.0, 0.0, 0.0]  # Initial position
        point1.velocities = [0.0] * len(self.arm_joints)

        point2 = JointTrajectoryPoint()
        point2.time_from_start.sec = 2
        point2.positions = [0.2, 0.0, 0.8, 0.5]  # Pointing position
        point2.velocities = [0.0] * len(self.arm_joints)

        trajectory.points = [point1, point2]
        return trajectory

    def _create_greeting_trajectory(self) -> JointTrajectory:
        """Create trajectory for greeting gesture"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints + self.head_joints

        # Combine wave and nod
        point1 = JointTrajectoryPoint()
        point1.time_from_start.sec = 0
        point1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Initial position
        point1.velocities = [0.0] * (len(self.arm_joints) + len(self.head_joints))

        point2 = JointTrajectoryPoint()
        point2.time_from_start.sec = 1
        point2.positions = [0.0, 0.0, 0.5, 0.0, 0.0, -0.2]  # Wave and slight nod
        point2.velocities = [0.0] * (len(self.arm_joints) + len(self.head_joints))

        point3 = JointTrajectoryPoint()
        point3.time_from_start.sec = 2
        point3.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Return to neutral
        point3.velocities = [0.0] * (len(self.arm_joints) + len(self.head_joints))

        trajectory.points = [point1, point2, point3]
        return trajectory

    def _create_acknowledge_trajectory(self) -> JointTrajectory:
        """Create trajectory for acknowledgment gesture"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.head_joints

        # Simple nod
        point1 = JointTrajectoryPoint()
        point1.time_from_start.sec = 0
        point1.positions = [0.0, 0.0]  # Neutral
        point1.velocities = [0.0] * len(self.head_joints)

        point2 = JointTrajectoryPoint()
        point2.time_from_start.sec = 1
        point2.positions = [0.0, -0.2]  # Slight nod
        point2.velocities = [0.0] * len(self.head_joints)

        point3 = JointTrajectoryPoint()
        point3.time_from_start.sec = 2
        point3.positions = [0.0, 0.0]  # Return to neutral
        point3.velocities = [0.0] * len(self.head_joints)

        trajectory.points = [point1, point2, point3]
        return trajectory

    def get_behavior_trajectory(self, behavior_type: str) -> Optional[JointTrajectory]:
        """Get the trajectory for a specific social behavior"""
        if behavior_type in self.gesture_trajectories:
            return self.gesture_trajectories[behavior_type]
        else:
            return None


class HumanAttentionManager:
    """Manages human attention and engagement"""

    def __init__(self):
        self.attention_threshold = 0.7  # Confidence threshold for attention detection
        self.max_attention_duration = 10.0  # seconds
        self.attention_history = {}  # Track attention over time

    def detect_attention(self, scene_graph: SceneGraph, robot_pose: Pose) -> List[Tuple[str, Pose, float]]:
        """Detect humans paying attention to the robot"""
        attention_targets = []

        if not scene_graph:
            return attention_targets

        for i, obj_class in enumerate(scene_graph.object_classes):
            if 'person' in obj_class.lower():
                person_pose = scene_graph.object_poses[i]
                person_id = scene_graph.object_ids[i] if scene_graph.object_ids else f"person_{i}"

                # Calculate if person is looking toward robot
                # This is simplified - in reality, this would use head pose estimation
                attention_confidence = self._calculate_attention_confidence(
                    person_pose, robot_pose, scene_graph
                )

                if attention_confidence > self.attention_threshold:
                    attention_targets.append((person_id, person_pose, attention_confidence))

        return attention_targets

    def _calculate_attention_confidence(self, person_pose: Pose, robot_pose: Pose, scene_graph: SceneGraph) -> float:
        """Calculate confidence that person is paying attention to robot"""
        # Simplified attention calculation
        # In reality, this would use head pose, gaze direction, etc.

        # Calculate distance - closer people more likely to pay attention
        dx = person_pose.position.x - robot_pose.position.x
        dy = person_pose.position.y - robot_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Distance factor (closer = higher attention)
        distance_factor = max(0.0, 1.0 - distance / 5.0)  # 5m max range

        # Random factor to simulate uncertainty
        random_factor = random.uniform(0.6, 1.0)

        return min(1.0, distance_factor * random_factor)

    def engage_attention(self, target_person_id: str, target_pose: Pose) -> SocialBehavior:
        """Create behavior to engage attention of a specific person"""
        behavior = SocialBehavior()
        behavior.header.stamp = Time()
        behavior.timestamp = behavior.header.stamp
        behavior.behavior_type = "attention_engagement"
        behavior.behavior_name = "attract_attention"
        behavior.target_type = "person"
        behavior.target_id = target_person_id
        behavior.target_pose = target_pose
        behavior.intensity = 0.8
        behavior.duration = 3.0
        behavior.social_context = "greeting"
        behavior.requires_acknowledgment = True
        behavior.priority = 0.7

        return behavior


class SocialProtocolAdherence:
    """Ensures adherence to social protocols and etiquette"""

    def __init__(self):
        self.social_rules = {
            'personal_space': 0.8,  # meters
            'greeting_distance': 1.5,  # meters
            'conversation_distance': 1.2,  # meters
            'attention_duration': 3.0  # seconds before looking away
        }

        self.cultural_contexts = {
            'default': {
                'greeting_style': 'wave',
                'eye_contact_duration': 3.0,
                'personal_space_multiplier': 1.0
            },
            'formal': {
                'greeting_style': 'nod',
                'eye_contact_duration': 2.0,
                'personal_space_multiplier': 1.2
            },
            'friendly': {
                'greeting_style': 'wave',
                'eye_contact_duration': 4.0,
                'personal_space_multiplier': 0.8
            }
        }

    def validate_behavior(self, behavior: SocialBehavior, scene_graph: Optional[SceneGraph] = None) -> Tuple[bool, List[str]]:
        """Validate that a behavior follows social protocols"""
        violations = []

        # Check personal space if relevant
        if behavior.target_type == "person" and scene_graph:
            for i, obj_id in enumerate(scene_graph.object_ids):
                if obj_id == behavior.target_id:
                    person_pose = scene_graph.object_poses[i]
                    robot_pose = Pose()  # This would be robot's actual pose in real system
                    dx = person_pose.position.x - robot_pose.position.x
                    dy = person_pose.position.y - robot_pose.position.y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance < self.social_rules['personal_space']:
                        violations.append(f"Violates personal space: {distance:.2f}m < {self.social_rules['personal_space']}m")

        # Check behavior appropriateness based on context
        if behavior.social_context == "greeting" and behavior.behavior_type not in ["greeting", "wave", "nod"]:
            violations.append(f"Behavior {behavior.behavior_type} not appropriate for greeting context")

        is_valid = len(violations) == 0
        return is_valid, violations

    def adjust_behavior_for_context(self, behavior: SocialBehavior, cultural_context: str = "default") -> SocialBehavior:
        """Adjust behavior based on cultural or social context"""
        if cultural_context in self.cultural_contexts:
            context_rules = self.cultural_contexts[cultural_context]

            # Adjust intensity based on formality
            if cultural_context == "formal":
                behavior.intensity = max(0.3, behavior.intensity * 0.7)
            elif cultural_context == "friendly":
                behavior.intensity = min(1.0, behavior.intensity * 1.2)

        return behavior


class SocialBehaviorExecutorNode(Node):
    def __init__(self):
        super().__init__('social_behavior_executor')

        # Initialize components
        self.social_navigation = SocialNavigation()
        self.expressive_behaviors = ExpressiveBehaviors()
        self.attention_manager = HumanAttentionManager()
        self.social_protocol = SocialProtocolAdherence()

        # Store latest data
        self.latest_scene_graph = None
        self.latest_action_plan = None
        self.latest_joint_state = None

        # Action clients for executing behaviors
        self.joint_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory'
        )

        # Callback group for handling multiple callbacks
        self.callback_group = ReentrantCallbackGroup()

        # Publishers and Subscribers
        self.action_plan_sub = self.create_subscription(
            ActionPlan,
            'action_plan',
            self.action_plan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.scene_graph_sub = self.create_subscription(
            SceneGraph,
            'scene_graph',
            self.scene_graph_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.social_behavior_sub = self.create_subscription(
            SocialBehavior,
            'social_behavior',
            self.social_behavior_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for social behavior status
        self.social_status_pub = self.create_publisher(
            SocialBehavior,
            'social_behavior_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'social_behavior_markers',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Social Behavior Executor node initialized.')

    def scene_graph_callback(self, msg: SceneGraph):
        """Store the latest scene graph for social decisions"""
        self.latest_scene_graph = msg

    def joint_state_callback(self, msg: JointState):
        """Store the latest joint states"""
        self.latest_joint_state = msg

    def action_plan_callback(self, msg: ActionPlan):
        """Process incoming action plans and execute social actions"""
        self.get_logger().info(f'Received action plan with {len(msg.actions)} actions')

        # Store the action plan
        self.latest_action_plan = msg

        # Check for social actions in the plan
        for i, action in enumerate(msg.actions):
            if action.action_type in ["wave", "greet", "acknowledge", "follow", "speak"]:
                self.get_logger().info(f'Executing social action {i+1}/{len(msg.actions)}: {action.action_type}')

                success = self.execute_social_action(action)

                if success:
                    self.get_logger().info(f'Social action {i+1} completed successfully')
                else:
                    self.get_logger().error(f'Social action {i+1} failed')

    def social_behavior_callback(self, msg: SocialBehavior):
        """Execute incoming social behavior requests"""
        self.get_logger().info(f'Executing social behavior: {msg.behavior_name} for {msg.target_type}')

        # Validate the behavior against social protocols
        is_valid, violations = self.social_protocol.validate_behavior(msg, self.latest_scene_graph)
        if not is_valid:
            self.get_logger().warn(f'Social behavior violates protocols: {violations}')
            # Still proceed but with caution

        # Adjust behavior based on context
        adjusted_behavior = self.social_protocol.adjust_behavior_for_context(
            msg, msg.cultural_context if msg.cultural_context else "default"
        )

        success = self.execute_social_behavior(adjusted_behavior)

        if success:
            self.get_logger().info(f'Social behavior {msg.behavior_name} completed successfully')
        else:
            self.get_logger().error(f'Social behavior {msg.behavior_name} failed')

    def execute_social_action(self, action: VLAAction) -> bool:
        """Execute a social action from an action plan"""
        behavior_type = self._map_action_to_behavior(action.action_type)
        if not behavior_type:
            self.get_logger().warn(f'Unknown social action type: {action.action_type}')
            return False

        # Create a social behavior from the action
        behavior = SocialBehavior()
        behavior.header.stamp = self.get_clock().now().to_msg()
        behavior.timestamp = behavior.header.stamp
        behavior.behavior_type = behavior_type
        behavior.behavior_name = action.action_type
        behavior.target_type = "person"  # Default assumption
        behavior.target_id = "closest_person"  # Will be determined at execution time
        behavior.intensity = 0.7
        behavior.duration = 2.0
        behavior.social_context = "interaction"
        behavior.priority = action.priority

        return self.execute_social_behavior(behavior)

    def _map_action_to_behavior(self, action_type: str) -> Optional[str]:
        """Map action types to behavior types"""
        action_to_behavior = {
            "wave": "gesture",
            "greet": "greeting",
            "acknowledge": "acknowledge",
            "follow": "follow",
            "speak": "attention_engagement"
        }
        return action_to_behavior.get(action_type)

    def execute_social_behavior(self, behavior: SocialBehavior) -> bool:
        """Execute a social behavior"""
        self.get_logger().info(f'Executing social behavior: {behavior.behavior_name}')

        # Get the appropriate trajectory for the behavior
        trajectory = self.expressive_behaviors.get_behavior_trajectory(behavior.behavior_type)

        if trajectory:
            # Execute the trajectory
            success = self.execute_trajectory(trajectory)
            if success:
                self.get_logger().info(f'Behavior trajectory executed successfully')
            else:
                self.get_logger().error(f'Behavior trajectory execution failed')
                return False
        else:
            # For navigation-based behaviors, handle differently
            if behavior.behavior_type == "follow":
                success = self.execute_follow_behavior(behavior)
            elif behavior.behavior_type == "greeting":
                success = self.execute_greeting_behavior(behavior)
            else:
                self.get_logger().warn(f'No trajectory defined for behavior: {behavior.behavior_type}')
                success = True  # Consider as success if no specific action needed

        # Publish status
        status_msg = behavior
        status_msg.header.stamp = self.get_clock().now().to_msg()
        self.social_status_pub.publish(status_msg)

        # Publish visualization markers
        self.publish_behavior_markers(behavior)

        return success

    def execute_follow_behavior(self, behavior: SocialBehavior) -> bool:
        """Execute follow behavior"""
        self.get_logger().info('Executing follow behavior')

        # In a real system, this would integrate with navigation
        # For simulation, we'll just return success
        return True

    def execute_greeting_behavior(self, behavior: SocialBehavior) -> bool:
        """Execute greeting behavior"""
        self.get_logger().info('Executing greeting behavior')

        # Combine wave and acknowledgment
        wave_traj = self.expressive_behaviors.get_behavior_trajectory('wave')
        ack_traj = self.expressive_behaviors.get_behavior_trajectory('acknowledge')

        success = True
        if wave_traj:
            success &= self.execute_trajectory(wave_traj)
        if ack_traj and success:
            success &= self.execute_trajectory(ack_traj)

        return success

    def execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """Execute a joint trajectory"""
        # Wait for action server
        if not self.joint_trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Joint trajectory action server not available')
            return False

        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # Send trajectory goal
        self.get_logger().info(f'Executing trajectory with {len(trajectory.points)} points')

        # Use async call to not block
        future = self.joint_trajectory_client.send_goal_async(goal_msg)

        # Wait for result (with timeout)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Trajectory goal accepted')

                # Get result
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    if result:
                        self.get_logger().info('Trajectory completed successfully')
                        return True
                    else:
                        self.get_logger().error('Trajectory failed')
                        return False
                else:
                    self.get_logger().error('Trajectory result future timed out')
                    return False
            else:
                self.get_logger().error('Trajectory goal was rejected')
                return False
        else:
            self.get_logger().error('Failed to send trajectory goal')
            return False

    def publish_behavior_markers(self, behavior: SocialBehavior):
        """Publish visualization markers for the behavior"""
        marker_array = MarkerArray()

        # Create marker for the behavior
        marker = Marker()
        marker.header = behavior.header
        marker.header.frame_id = 'base_link'  # or appropriate frame
        marker.ns = "social_behaviors"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose = behavior.target_pose
        marker.pose.position.z += 1.0  # Raise text above target

        marker.text = behavior.behavior_name
        marker.scale.z = 0.2  # Text size
        marker.color.a = 1.0  # Alpha
        marker.color.r = 1.0  # Red for social behaviors

        marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    try:
        social_behavior_executor = SocialBehaviorExecutorNode()

        try:
            rclpy.spin(social_behavior_executor)
        except KeyboardInterrupt:
            pass
        finally:
            social_behavior_executor.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()