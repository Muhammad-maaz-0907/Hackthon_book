#!/usr/bin/env python3
"""
Safety Validator Node for Vision-Language-Action (VLA) System

This node validates the safety of action plans before execution,
performing collision detection, balance validation, social norm checking,
and emergency handling for humanoid robots.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.service import Service

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

# Import ROS 2 messages
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid

# Import custom messages
# Note: We'll need to adjust these imports once the package is properly set up
# from humanoid_robotics_book.msg import ActionPlan, SceneGraph
# from humanoid_robotics_book.srv import SafetyCheck


class SafetyLevel(Enum):
    """Levels of safety risk"""
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class CollisionType(Enum):
    """Types of collisions to detect"""
    OBSTACLE = "obstacle"
    HUMAN = "human"
    ROBOT_SELF = "robot_self"
    ENVIRONMENT = "environment"


@dataclass
class CollisionRisk:
    """Represents a collision risk"""
    collision_type: CollisionType
    location: Pose
    distance: float
    severity: float  # 0.0 to 1.0
    timestamp: Time


@dataclass
class BalanceState:
    """Represents robot balance state"""
    center_of_mass: Point
    support_polygon: List[Point]  # Points defining the support base
    stability_margin: float  # Distance from CoM to edge of support polygon
    is_stable: bool


class CollisionDetector:
    """Detects potential collisions in action plans"""

    def __init__(self):
        self.robot_radius = 0.5  # meters, approximate humanoid size
        self.safe_distance = 0.8  # meters, minimum safe distance to obstacles
        self.human_safe_distance = 1.0  # meters, minimum safe distance to humans
        self.collision_threshold = 0.5  # meters, distance at which collision is imminent
        self.risk_threshold = 0.7  # Above this, action is considered unsafe

        # Store the latest environment data
        self.occupancy_grid = None
        self.laser_scan = None
        self.scene_graph = None

    def check_collision_risk(self, action_plan, scene_graph=None) -> Tuple[bool, List[str], float]:
        """Check for potential collisions in the action plan"""
        if scene_graph:
            self.scene_graph = scene_graph

        issues = []
        max_risk = 0.0

        # Analyze each action in the plan
        for action in action_plan.actions:
            if action.action_type == "navigate_to":
                risk, reason = self._check_navigation_collision(action.target_pose)
                if risk > max_risk:
                    max_risk = risk
                    if risk > self.risk_threshold:
                        issues.append(f"Navigation collision risk: {reason}")
            elif action.action_type in ["grasp_object", "pick_up", "place_object"]:
                risk, reason = self._check_manipulation_collision(action.target_pose)
                if risk > max_risk:
                    max_risk = risk
                    if risk > self.risk_threshold:
                        issues.append(f"Manipulation collision risk: {reason}")

        is_safe = max_risk < self.risk_threshold
        return is_safe, issues, max_risk

    def _check_navigation_collision(self, target_pose: Pose) -> Tuple[float, str]:
        """Check collision risk for navigation action"""
        if not self.scene_graph:
            # If no scene graph available, assume minimal risk
            return 0.1, "No scene data available, assuming minimal risk"

        # Check distance to obstacles in scene graph
        for i, obj_pose in enumerate(self.scene_graph.object_poses):
            # Calculate distance to object
            dx = target_pose.position.x - obj_pose.position.x
            dy = target_pose.position.y - obj_pose.position.y
            dz = target_pose.position.z - obj_pose.position.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            # Check if it's a human (higher safety requirement)
            if i < len(self.scene_graph.object_classes) and self.scene_graph.object_classes[i] in ['person', 'human']:
                if distance < self.human_safe_distance:
                    return 0.9, f"Too close to person: {distance:.2f}m < {self.human_safe_distance}m required"
            else:
                # Check if it's an obstacle
                if distance < self.safe_distance:
                    return 0.7, f"Too close to obstacle: {distance:.2f}m < {self.safe_distance}m required"

        # Check with occupancy grid if available
        if self.occupancy_grid:
            grid_risk, grid_reason = self._check_occupancy_grid_collision(target_pose)
            if grid_risk > 0:
                return grid_risk, grid_reason

        return 0.0, "No collision detected"

    def _check_occupancy_grid_collision(self, target_pose: Pose) -> Tuple[float, str]:
        """Check collision risk using occupancy grid"""
        # Convert pose to grid coordinates
        try:
            resolution = self.occupancy_grid.info.resolution
            origin_x = self.occupancy_grid.info.origin.position.x
            origin_y = self.occupancy_grid.info.origin.position.y

            grid_x = int((target_pose.position.x - origin_x) / resolution)
            grid_y = int((target_pose.position.y - origin_y) / resolution)

            # Check if coordinates are within grid bounds
            if (0 <= grid_x < self.occupancy_grid.info.width and
                0 <= grid_y < self.occupancy_grid.info.height):

                # Get the occupancy value (0: free, 100: occupied)
                idx = grid_y * self.occupancy_grid.info.width + grid_x
                occupancy = self.occupancy_grid.data[idx]

                if occupancy > 50:  # Threshold for "occupied"
                    distance_to_obstacle = occupancy / 100.0  # Scale risk based on occupancy
                    return min(distance_to_obstacle, 0.9), f"Occupancy grid shows obstacle at target location"
        except:
            # If occupancy grid processing fails, return minimal risk
            return 0.0, "Occupancy grid processing failed"

        return 0.0, "No collision detected in occupancy grid"

    def _check_manipulation_collision(self, target_pose: Pose) -> Tuple[float, str]:
        """Check collision risk for manipulation action"""
        if not self.scene_graph:
            # If no scene graph available, assume low risk
            return 0.2, "No scene data available, assuming low risk"

        # Check if manipulation target is near obstacles
        for i, obj_pose in enumerate(self.scene_graph.object_poses):
            # Calculate distance to object
            dx = target_pose.position.x - obj_pose.position.x
            dy = target_pose.position.y - obj_pose.position.y
            dz = target_pose.position.z - obj_pose.position.z
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            # For manipulation, we need more space
            manipulation_safe_distance = self.safe_distance * 0.7  # slightly less strict
            if distance < manipulation_safe_distance:
                return 0.6, f"Manipulation too close to obstacle: {distance:.2f}m < {manipulation_safe_distance}m required"

        return 0.0, "No collision detected"

    def update_occupancy_grid(self, msg):
        """Update with latest occupancy grid data"""
        self.occupancy_grid = msg

    def update_laser_scan(self, msg):
        """Update with latest laser scan data"""
        self.laser_scan = msg


class BalanceValidator:
    """Validates humanoid balance during action execution"""

    def __init__(self):
        self.max_com_displacement = 0.3  # meters, maximum center of mass displacement
        self.max_torque = 50.0  # Nm, maximum joint torque allowed
        self.balance_threshold = 0.8  # stability threshold
        self.stability_margin_threshold = 0.1  # meters, minimum stability margin

    def check_balance_safety(self, action_plan) -> Tuple[bool, List[str], float]:
        """Check if actions maintain humanoid balance"""
        issues = []
        max_risk = 0.0

        for action in action_plan.actions:
            if action.action_type in ["grasp_object", "pick_up", "place_object"]:
                # Manipulation actions can affect balance
                risk = self._estimate_manipulation_balance_risk(action)
                if risk > max_risk:
                    max_risk = risk
                    if risk > 0.3:  # Significant risk
                        issues.append(f"Balance risk during manipulation: {risk:.2f}")
            elif action.action_type == "navigate_to":
                # Check if navigation is on stable terrain (simplified)
                risk = self._estimate_navigation_balance_risk(action)
                if risk > max_risk:
                    max_risk = risk
                    if risk > 0.2:  # Moderate risk
                        issues.append(f"Balance risk during navigation: {risk:.2f}")

        is_safe = max_risk < 0.6  # Consider safe if max risk is below threshold
        return is_safe, issues, max_risk

    def _estimate_manipulation_balance_risk(self, action) -> float:
        """Estimate balance risk for manipulation action"""
        # Simplified balance risk based on target position
        # Higher risk for reaching far or high
        reach_distance = math.sqrt(
            action.target_pose.position.x**2 +
            action.target_pose.position.y**2 +
            action.target_pose.position.z**2
        )

        # Normalize risk based on reach distance
        if reach_distance > 1.2:  # Beyond typical reach
            return 0.7
        elif reach_distance > 0.8:  # Extended reach
            return 0.4
        else:  # Comfortable reach
            return 0.1

    def _estimate_navigation_balance_risk(self, action) -> float:
        """Estimate balance risk for navigation action"""
        # For now, assume navigation on flat surfaces is safe
        # In a real system, this would check terrain data
        return 0.1  # Low risk

    def calculate_balance_state(self, joint_positions: List[float], target_pose: Pose) -> BalanceState:
        """Calculate current balance state of the robot"""
        # Simplified calculation - in reality this would require forward kinematics
        # and center of mass calculation

        # Estimate center of mass based on joint positions and target
        com_x = np.mean([jp for jp in joint_positions if isinstance(jp, (int, float))]) if joint_positions else 0.0
        com_y = 0.0  # Assume robot is upright
        com_z = 0.8  # Typical humanoid CoM height

        # Simplified support polygon (rectangle under feet)
        support_points = [
            Point(x=-0.1, y=-0.1, z=0.0),  # Front left
            Point(x=0.1, y=-0.1, z=0.0),   # Front right
            Point(x=0.1, y=0.1, z=0.0),    # Back right
            Point(x=-0.1, y=0.1, z=0.0)    # Back left
        ]

        # Calculate distance from CoM projection to nearest edge of support polygon
        com_proj = Point(x=com_x, y=com_y, z=0.0)
        min_dist = float('inf')
        for point in support_points:
            dist = math.sqrt((com_proj.x - point.x)**2 + (com_proj.y - point.y)**2)
            min_dist = min(min_dist, dist)

        stability_margin = min_dist
        is_stable = stability_margin > self.stability_margin_threshold

        return BalanceState(
            center_of_mass=Point(x=com_x, y=com_y, z=com_z),
            support_polygon=support_points,
            stability_margin=stability_margin,
            is_stable=is_stable
        )


class SocialNormValidator:
    """Validates actions against social norms and etiquette"""

    def __init__(self):
        self.respectful_distance_to_humans = 0.8  # meters
        self.appropriate_gestures = ['wave', 'nod', 'point', 'greet']
        self.inappropriate_actions = ['stare', 'invade_space', 'ignore']

    def check_social_norms(self, action_plan, scene_graph=None) -> Tuple[bool, List[str], float]:
        """Check if actions follow social norms"""
        issues = []
        max_risk = 0.0

        for action in action_plan.actions:
            if action.action_type == "navigate_to" and scene_graph:
                # Check if navigation path respects human personal space
                risk, reason = self._check_navigation_social_norms(action.target_pose, scene_graph)
                if risk > max_risk:
                    max_risk = risk
                    if risk > 0:
                        issues.append(f"Social norm violation: {reason}")
            elif action.action_type in ["speak", "greet"]:
                # Check if speech is appropriate
                risk, reason = self._check_speech_social_norms(action)
                if risk > max_risk:
                    max_risk = risk
                    if risk > 0:
                        issues.append(f"Social norm violation: {reason}")

        is_safe = max_risk < 0.4  # Consider safe if max risk is below threshold
        return is_safe, issues, max_risk

    def _check_navigation_social_norms(self, target_pose: Pose, scene_graph) -> Tuple[float, str]:
        """Check if navigation respects social norms"""
        for i, obj_class in enumerate(scene_graph.object_classes):
            if obj_class in ['person', 'human']:
                person_pose = scene_graph.object_poses[i]
                dx = target_pose.position.x - person_pose.position.x
                dy = target_pose.position.y - person_pose.position.y
                distance = math.sqrt(dx*dx + dy*dy)

                if distance < self.respectful_distance_to_humans:
                    return 0.8, f"Path too close to person: {distance:.2f}m < {self.respectful_distance_to_humans}m"

        return 0.0, "Navigation respects social norms"

    def _check_speech_social_norms(self, action) -> Tuple[float, str]:
        """Check if speech action is socially appropriate"""
        # For now, assume all speech is appropriate
        # In a real system, this would analyze the content
        return 0.0, "Speech is socially appropriate"


class EmergencyHandler:
    """Handles emergency situations and safety-critical scenarios"""

    def __init__(self):
        self.emergency_stop_active = False
        self.emergency_scenarios = [
            'human_fall', 'fire_detected', 'obstacle_sudden_appearance',
            'robot_stuck', 'balance_loss', 'system_failure'
        ]

    def check_emergency_conditions(self, action_plan, scene_graph=None) -> Tuple[bool, List[str], float]:
        """Check for emergency conditions that would halt execution"""
        issues = []
        risk = 0.0

        # Check if any action would trigger emergency conditions
        if self.emergency_stop_active:
            issues.append("Emergency stop is active")
            risk = 1.0

        # In a real system, this would connect to emergency detection systems
        # For simulation, we'll assume no emergencies unless specifically triggered
        is_safe = risk < 0.9
        return is_safe, issues, risk

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False


class SafetyValidatorNode(Node):
    def __init__(self):
        super().__init__('safety_validator')

        # Initialize safety components
        self.collision_detector = CollisionDetector()
        self.balance_validator = BalanceValidator()
        self.social_norm_validator = SocialNormValidator()
        self.emergency_handler = EmergencyHandler()

        # Store the latest data
        self.latest_action_plan = None
        self.latest_scene_graph = None
        self.latest_occupancy_grid = None
        self.latest_laser_scan = None

        # Publishers and Subscribers
        # self.scene_graph_sub = self.create_subscription(
        #     SceneGraph,
        #     'scene_graph',
        #     self.scene_graph_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )
        #
        # self.occupancy_grid_sub = self.create_subscription(
        #     OccupancyGrid,
        #     '/map',
        #     self.occupancy_grid_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )
        #
        # self.laser_scan_sub = self.create_subscription(
        #     LaserScan,
        #     '/scan',
        #     self.laser_scan_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        # )
        #
        # # Service server for safety checks
        # self.safety_check_srv = self.create_service(
        #     SafetyCheck,
        #     'safety_check',
        #     self.safety_check_callback
        # )

        self.get_logger().info('Safety Validator node initialized.')

    def scene_graph_callback(self, msg):
        """Store the latest scene graph for safety validation"""
        self.latest_scene_graph = msg

    def occupancy_grid_callback(self, msg):
        """Store the latest occupancy grid for collision detection"""
        self.latest_occupancy_grid = msg
        self.collision_detector.update_occupancy_grid(msg)

    def laser_scan_callback(self, msg):
        """Store the latest laser scan for collision detection"""
        self.latest_laser_scan = msg
        self.collision_detector.update_laser_scan(msg)

    def safety_check_callback(self, request, response):
        """Handle safety check service requests"""
        action_plan = request.action_plan
        self.get_logger().info(f'Received safety check request for plan with {len(action_plan.actions)} actions')

        # Perform all safety checks
        collision_safe, collision_issues, collision_risk = self.collision_detector.check_collision_risk(
            action_plan, self.latest_scene_graph
        )

        balance_safe, balance_issues, balance_risk = self.balance_validator.check_balance_safety(action_plan)

        social_safe, social_issues, social_risk = self.social_norm_validator.check_social_norms(
            action_plan, self.latest_scene_graph
        )

        emergency_safe, emergency_issues, emergency_risk = self.emergency_handler.check_emergency_conditions(
            action_plan, self.latest_scene_graph
        )

        # Aggregate results
        all_issues = collision_issues + balance_issues + social_issues + emergency_issues
        max_risk = max(collision_risk, balance_risk, social_risk, emergency_risk)

        # Overall safety is safe only if all individual checks pass
        overall_safe = collision_safe and balance_safe and social_safe and emergency_safe

        # Prepare response
        response.is_safe = overall_safe
        response.safety_issues = all_issues
        response.risk_score = max_risk
        response.mitigation_suggestions = self._generate_mitigation_suggestions(
            collision_issues, balance_issues, social_issues
        )

        self.get_logger().info(f'Safety check result: Safe={overall_safe}, Risk={max_risk:.2f}, Issues={len(all_issues)}')

        return response

    def _generate_mitigation_suggestions(self, collision_issues: List[str], balance_issues: List[str], social_issues: List[str]) -> List[str]:
        """Generate suggestions for mitigating safety issues"""
        suggestions = []

        if collision_issues:
            suggestions.append("Reduce navigation speed near obstacles")
            suggestions.append("Increase safety margin around humans")
            suggestions.append("Verify object positions before manipulation")

        if balance_issues:
            suggestions.append("Lower manipulation speed for better stability")
            suggestions.append("Use both arms for heavy objects")
            suggestions.append("Maintain wider stance during manipulation")

        if social_issues:
            suggestions.append("Maintain respectful distance from humans")
            suggestions.append("Use appropriate social behaviors")
            suggestions.append("Announce intentions before action")

        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)

        return unique_suggestions

    def validate_action_plan(self, action_plan, scene_graph=None):
        """Validate an action plan using all safety checks"""
        # Perform all safety validations
        collision_safe, collision_issues, collision_risk = self.collision_detector.check_collision_risk(
            action_plan, scene_graph
        )

        balance_safe, balance_issues, balance_risk = self.balance_validator.check_balance_safety(action_plan)

        social_safe, social_issues, social_risk = self.social_norm_validator.check_social_norms(
            action_plan, scene_graph
        )

        emergency_safe, emergency_issues, emergency_risk = self.emergency_handler.check_emergency_conditions(
            action_plan, scene_graph
        )

        # Aggregate results
        all_issues = collision_issues + balance_issues + social_issues + emergency_issues
        max_risk = max(collision_risk, balance_risk, social_risk, emergency_risk)
        overall_safe = collision_safe and balance_safe and social_safe and emergency_safe

        return overall_safe, all_issues, max_risk


def main(args=None):
    rclpy.init(args=args)

    try:
        safety_validator = SafetyValidatorNode()

        try:
            rclpy.spin(safety_validator)
        except KeyboardInterrupt:
            pass
        finally:
            safety_validator.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()