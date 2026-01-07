#!/usr/bin/env python3
"""
Navigation Executor Node for Vision-Language-Action (VLA) System

This node executes navigation actions for humanoid robots by integrating
with Nav2 for path planning, obstacle avoidance, and goal execution.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from typing import Dict, List, Optional, Tuple
import math
import time

# Import ROS 2 messages and actions
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path

# Import custom messages
from humanoid_robotics_book.msg import ActionPlan, VLAAction, SceneGraph


class NavigationTranslator:
    """Translates voice commands and action plans to navigation goals"""

    def __init__(self):
        # Predefined locations for common commands
        self.predefined_locations = {
            'kitchen': Pose(position=Point(x=2.0, y=1.0, z=0.0), orientation=Quaternion(w=1.0)),
            'living_room': Pose(position=Point(x=-1.0, y=1.0, z=0.0), orientation=Quaternion(w=1.0)),
            'bedroom': Pose(position=Point(x=2.0, y=-1.0, z=0.0), orientation=Quaternion(w=1.0)),
            'bathroom': Pose(position=Point(x=-2.0, y=-1.0, z=0.0), orientation=Quaternion(w=1.0)),
            'office': Pose(position=Point(x=0.0, y=2.0, z=0.0), orientation=Quaternion(w=1.0)),
        }

        # Semantic location mapping
        self.semantic_mapping = {
            'kitchen': ['kitchen', 'cooking', 'food', 'refrigerator', 'stove'],
            'living_room': ['living room', 'sofa', 'couch', 'tv', 'relax'],
            'bedroom': ['bedroom', 'bed', 'sleep', 'bedroom'],
            'bathroom': ['bathroom', 'bath', 'shower', 'toilet'],
            'office': ['office', 'desk', 'computer', 'work'],
        }

    def translate_command_to_goal(self, command: str, scene_graph: Optional[SceneGraph] = None) -> Optional[Pose]:
        """Translate voice command to navigation goal pose"""
        command_lower = command.lower()

        # Check predefined locations
        for location, pose in self.predefined_locations.items():
            if location in command_lower:
                return pose

        # Check semantic mappings
        for location, keywords in self.semantic_mapping.items():
            if any(keyword in command_lower for keyword in keywords):
                return self.predefined_locations[location]

        # If scene graph is available, look for specific objects/locations
        if scene_graph:
            # Look for a person to follow
            for i, obj_class in enumerate(scene_graph.object_classes):
                if 'person' in obj_class.lower() and 'follow' in command_lower:
                    return scene_graph.object_poses[i]

        return None

    def extract_navigation_goal_from_action(self, action: VLAAction, scene_graph: Optional[SceneGraph] = None) -> Optional[Pose]:
        """Extract navigation goal from VLA action"""
        if action.action_type != "navigate_to":
            return None

        # If target pose is already specified, use it
        if action.target_pose.position.x != 0.0 or action.target_pose.position.y != 0.0:
            return action.target_pose

        # Otherwise, try to extract from parameters
        if action.parameters:
            for param in action.parameters:
                pose = self.translate_command_to_goal(param, scene_graph)
                if pose:
                    return pose

        return None


class ObstacleAvoider:
    """Handles obstacle avoidance during navigation"""

    def __init__(self):
        self.laser_data = None
        self.map_data = None
        self.safe_distance = 0.5  # meters
        self.avoidance_threshold = 0.7  # distance in meters for obstacle detection

    def update_laser_scan(self, laser_scan: LaserScan):
        """Update with latest laser scan data"""
        self.laser_data = laser_scan

    def update_map(self, occupancy_grid: OccupancyGrid):
        """Update with latest map data"""
        self.map_data = occupancy_grid

    def check_path_clear(self, start_pose: Pose, goal_pose: Pose) -> Tuple[bool, str]:
        """Check if path is clear of obstacles"""
        if not self.laser_data:
            return True, "No laser data available, assuming path clear"

        # Simple check: look for obstacles in front of robot
        min_range = min(self.laser_data.ranges) if self.laser_data.ranges else float('inf')

        if min_range < self.avoidance_threshold:
            return False, f"Obstacle detected at {min_range:.2f}m"

        return True, "Path is clear"

    def generate_avoidance_path(self, current_pose: Pose, goal_pose: Pose) -> Optional[Pose]:
        """Generate alternative path to avoid obstacles"""
        # In a real system, this would implement more sophisticated path planning
        # For simulation, we'll just slightly adjust the goal
        if not self.laser_data:
            return goal_pose

        # Simple avoidance: if obstacle ahead, try to go around
        if min(self.laser_data.ranges) < self.avoidance_threshold:
            # Adjust goal slightly to the side
            adjusted_goal = Pose()
            adjusted_goal.position.x = goal_pose.position.x + 0.5  # Move right
            adjusted_goal.position.y = goal_pose.position.y
            adjusted_goal.position.z = goal_pose.position.z
            adjusted_goal.orientation = goal_pose.orientation
            return adjusted_goal

        return goal_pose


class SocialNavigation:
    """Handles social navigation and human-aware behaviors"""

    def __init__(self):
        self.respectful_distance = 0.8  # meters
        self.social_navigation_enabled = True

    def adjust_path_for_social_norms(self, goal_pose: Pose, scene_graph: Optional[SceneGraph]) -> Pose:
        """Adjust navigation path to respect social norms"""
        if not scene_graph or not self.social_navigation_enabled:
            return goal_pose

        # Check for humans in the environment
        for i, obj_class in enumerate(scene_graph.object_classes):
            if 'person' in obj_class.lower():
                person_pose = scene_graph.object_poses[i]

                # Calculate distance to person
                dx = goal_pose.position.x - person_pose.position.x
                dy = goal_pose.position.y - person_pose.position.y
                distance = math.sqrt(dx*dx + dy*dy)

                # If too close, adjust path
                if distance < self.respectful_distance:
                    # Move the goal slightly away from the person
                    direction_x = goal_pose.position.x - person_pose.position.x
                    direction_y = goal_pose.position.y - person_pose.position.y
                    norm = math.sqrt(direction_x**2 + direction_y**2)

                    if norm > 0:
                        # Normalize and scale to respectful distance
                        direction_x = direction_x / norm * self.respectful_distance
                        direction_y = direction_y / norm * self.respectful_distance

                        adjusted_goal = Pose()
                        adjusted_goal.position.x = person_pose.position.x + direction_x
                        adjusted_goal.position.y = person_pose.position.y + direction_y
                        adjusted_goal.position.z = goal_pose.position.z
                        adjusted_goal.orientation = goal_pose.orientation

                        return adjusted_goal

        return goal_pose


class NavigationExecutorNode(Node):
    def __init__(self):
        super().__init__('navigation_executor')

        # Initialize components
        self.navigation_translator = NavigationTranslator()
        self.obstacle_avoider = ObstacleAvoider()
        self.social_navigation = SocialNavigation()

        # Store latest data
        self.latest_scene_graph = None
        self.latest_action_plan = None

        # Action client for Nav2
        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
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

        self.laser_scan_sub = self.create_subscription(
            LaserScan,
            '/scan',  # Standard laser scan topic
            self.laser_scan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',  # Standard map topic
            self.map_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for navigation status
        self.nav_status_pub = self.create_publisher(
            ActionPlan,
            'navigation_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Navigation Executor node initialized.')

    def scene_graph_callback(self, msg: SceneGraph):
        """Store the latest scene graph for navigation decisions"""
        self.latest_scene_graph = msg

    def laser_scan_callback(self, msg: LaserScan):
        """Update with latest laser scan data for obstacle detection"""
        self.obstacle_avoider.update_laser_scan(msg)

    def map_callback(self, msg: OccupancyGrid):
        """Update with latest map data for navigation"""
        self.obstacle_avoider.update_map(msg)

    def action_plan_callback(self, msg: ActionPlan):
        """Process incoming action plans and execute navigation actions"""
        self.get_logger().info(f'Received action plan with {len(msg.actions)} actions')

        # Store the action plan
        self.latest_action_plan = msg

        # Execute navigation actions in sequence
        for i, action in enumerate(msg.actions):
            if action.action_type == "navigate_to":
                self.get_logger().info(f'Executing navigation action {i+1}/{len(msg.actions)}')

                # Extract navigation goal from action
                goal_pose = self.navigation_translator.extract_navigation_goal_from_action(
                    action, self.latest_scene_graph
                )

                if goal_pose:
                    # Adjust for social norms
                    adjusted_goal = self.social_navigation.adjust_path_for_social_norms(
                        goal_pose, self.latest_scene_graph
                    )

                    # Check for obstacles and adjust if necessary
                    safe_goal = self.obstacle_avoider.generate_avoidance_path(
                        self.get_current_pose(), adjusted_goal
                    )

                    # Execute navigation
                    success = self.execute_navigation(safe_goal)

                    if success:
                        self.get_logger().info(f'Navigation action {i+1} completed successfully')
                    else:
                        self.get_logger().error(f'Navigation action {i+1} failed')
                        break  # Stop execution if navigation fails
                else:
                    self.get_logger().warn(f'Could not extract navigation goal from action {i+1}')

    def get_current_pose(self) -> Pose:
        """Get current robot pose (in simulation, return origin)"""
        # In a real system, this would get the current pose from localization
        # For simulation, return a default pose
        return Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(w=1.0))

    def execute_navigation(self, goal_pose: Pose) -> bool:
        """Execute navigation to the specified goal pose"""
        # Wait for action server
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available')
            return False

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose = goal_pose

        # Send navigation goal
        self.get_logger().info(f'Navigating to goal: ({goal_pose.position.x:.2f}, {goal_pose.position.y:.2f})')

        # Use async call to not block
        future = self.nav_to_pose_client.send_goal_async(goal_msg)

        # Wait for result (with timeout)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info('Navigation goal accepted')

                # Get result
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    if result:
                        self.get_logger().info('Navigation completed successfully')
                        return True
                    else:
                        self.get_logger().error('Navigation failed')
                        return False
                else:
                    self.get_logger().error('Navigation result future timed out')
                    return False
            else:
                self.get_logger().error('Navigation goal was rejected')
                return False
        else:
            self.get_logger().error('Failed to send navigation goal')
            return False

    def check_navigation_feasibility(self, goal_pose: Pose) -> Tuple[bool, str]:
        """Check if navigation to goal is feasible"""
        # Check if path is clear
        current_pose = self.get_current_pose()
        is_clear, reason = self.obstacle_avoider.check_path_clear(current_pose, goal_pose)

        if not is_clear:
            return False, f"Path not clear: {reason}"

        # Additional checks could go here
        # For example: check if goal is in map bounds, is on traversable terrain, etc.

        return True, "Navigation is feasible"


def main(args=None):
    rclpy.init(args=args)

    try:
        navigation_executor = NavigationExecutorNode()

        # Use multi-threaded executor to handle callbacks properly
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(navigation_executor)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            navigation_executor.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()