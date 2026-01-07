#!/usr/bin/env python3
"""
Manipulation Executor Node for Vision-Language-Action (VLA) System

This node executes manipulation actions for humanoid robots, handling
grasp planning from visual-language input, arm trajectory generation,
hand manipulation execution, and failure recovery mechanisms.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from typing import Dict, List, Optional, Tuple
import math
import time
import numpy as np

# Import ROS 2 messages and actions
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from builtin_interfaces.msg import Time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState

# Import custom messages
from humanoid_robotics_book.msg import ActionPlan, VLAAction, SceneGraph


class GraspPlanner:
    """Plans grasps for objects based on visual-language input"""

    def __init__(self):
        self.grasp_database = {
            'cup': {
                'approach': 'top_grasp',
                'gripper_width': 0.05,
                'grasp_height_offset': 0.05,
                'pre_grasp_distance': 0.1
            },
            'bottle': {
                'approach': 'side_grasp',
                'gripper_width': 0.04,
                'grasp_height_offset': 0.0,
                'pre_grasp_distance': 0.1
            },
            'box': {
                'approach': 'top_grasp',
                'gripper_width': 0.08,
                'grasp_height_offset': 0.05,
                'pre_grasp_distance': 0.15
            },
            'book': {
                'approach': 'side_grasp',
                'gripper_width': 0.03,
                'grasp_height_offset': 0.0,
                'pre_grasp_distance': 0.1
            }
        }

        self.approach_strategies = {
            'top_grasp': self._plan_top_grasp,
            'side_grasp': self._plan_side_grasp,
            'pinch_grasp': self._plan_pinch_grasp
        }

    def plan_grasp(self, object_class: str, object_pose: Pose, scene_graph: Optional[SceneGraph] = None) -> Optional[Tuple[Pose, float]]:
        """Plan a grasp for the specified object"""
        if object_class not in self.grasp_database:
            # Use default grasp if object type is unknown
            object_class = 'cup'  # Default to cup grasp

        grasp_info = self.grasp_database[object_class]
        approach_strategy = self.approach_strategies[grasp_info['approach']]

        grasp_pose = approach_strategy(object_pose, grasp_info)
        gripper_width = grasp_info['gripper_width']

        return grasp_pose, gripper_width

    def _plan_top_grasp(self, object_pose: Pose, grasp_info: Dict) -> Pose:
        """Plan a top-down grasp approach"""
        grasp_pose = Pose()
        grasp_pose.position.x = object_pose.position.x
        grasp_pose.position.y = object_pose.position.y
        grasp_pose.position.z = object_pose.position.z + grasp_info['grasp_height_offset']

        # Orient gripper to approach from above
        grasp_pose.orientation.x = 0.0
        grasp_pose.orientation.y = 0.707  # 90 degrees pitch
        grasp_pose.orientation.z = 0.0
        grasp_pose.orientation.w = 0.707

        return grasp_pose

    def _plan_side_grasp(self, object_pose: Pose, grasp_info: Dict) -> Pose:
        """Plan a side grasp approach"""
        grasp_pose = Pose()
        grasp_pose.position.x = object_pose.position.x - grasp_info['pre_grasp_distance']
        grasp_pose.position.y = object_pose.position.y
        grasp_pose.position.z = object_pose.position.z + grasp_info['grasp_height_offset']

        # Orient gripper to approach from side
        grasp_pose.orientation.x = 0.0
        grasp_pose.orientation.y = 0.0
        grasp_pose.orientation.z = 0.0
        grasp_pose.orientation.w = 1.0

        return grasp_pose

    def _plan_pinch_grasp(self, object_pose: Pose, grasp_info: Dict) -> Pose:
        """Plan a pinch grasp approach"""
        grasp_pose = Pose()
        grasp_pose.position.x = object_pose.position.x - grasp_info['pre_grasp_distance']
        grasp_pose.position.y = object_pose.position.y
        grasp_pose.position.z = object_pose.position.z + grasp_info['grasp_height_offset']

        # Orient gripper for pinch grasp
        grasp_pose.orientation.x = 0.0
        grasp_pose.orientation.y = 0.5
        grasp_pose.orientation.z = 0.0
        grasp_pose.orientation.w = 0.866  # 60 degrees pitch

        return grasp_pose


class ArmTrajectoryGenerator:
    """Generates trajectories for robot arm movement"""

    def __init__(self):
        self.arm_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
        self.max_velocity = 0.5  # rad/s
        self.max_acceleration = 0.2  # rad/s^2
        self.approach_distance = 0.2  # meters

    def generate_approach_trajectory(self, target_pose: Pose, current_pose: Pose) -> JointTrajectory:
        """Generate trajectory to approach the target pose"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # For simulation, we'll create a simple trajectory
        # In a real system, this would use inverse kinematics

        # Create intermediate points
        for i in range(5):  # 5 intermediate points
            point = JointTrajectoryPoint()

            # Interpolate between current and target
            t = i / 4.0  # 0.0 to 1.0
            point_time = Time()
            point_time.sec = int(t * 5)  # 5 seconds total
            point_time.nanosec = int((t * 5 - int(t * 5)) * 1e9)

            point.time_from_start = point_time

            # Simulated joint positions (in a real system, these would come from IK)
            joint_positions = []
            for j in range(len(self.arm_joints)):
                # Simple interpolation for simulation
                joint_pos = 0.1 * j + t * 0.5  # Simulated joint position
                joint_positions.append(joint_pos)

            point.positions = joint_positions
            point.velocities = [0.0] * len(self.arm_joints)  # Start/stop velocities
            point.accelerations = [0.0] * len(self.arm_joints)

            trajectory.points.append(point)

        return trajectory

    def generate_grasp_trajectory(self, grasp_pose: Pose) -> JointTrajectory:
        """Generate trajectory to execute the grasp"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # Pre-grasp position
        pre_grasp_point = JointTrajectoryPoint()
        pre_grasp_point.time_from_start.sec = 2
        pre_grasp_point.positions = [0.0, -0.5, 0.0, -0.5, 0.0]  # Pre-grasp configuration
        pre_grasp_point.velocities = [0.0] * len(self.arm_joints)
        pre_grasp_point.accelerations = [0.0] * len(self.arm_joints)

        # Grasp position
        grasp_point = JointTrajectoryPoint()
        grasp_point.time_from_start.sec = 4
        grasp_point.positions = [0.0, -0.2, 0.0, -0.8, 0.0]  # Grasp configuration
        grasp_point.velocities = [0.0] * len(self.arm_joints)
        grasp_point.accelerations = [0.0] * len(self.arm_joints)

        trajectory.points = [pre_grasp_point, grasp_point]
        return trajectory

    def generate_lift_trajectory(self, lift_height: float = 0.1) -> JointTrajectory:
        """Generate trajectory to lift the object"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joints

        # Lift position
        lift_point = JointTrajectoryPoint()
        lift_point.time_from_start.sec = 2
        lift_point.positions = [0.1, -0.3, 0.1, -0.9, 0.1]  # Lifted configuration
        lift_point.velocities = [0.0] * len(self.arm_joints)
        lift_point.accelerations = [0.0] * len(self.arm_joints)

        trajectory.points = [lift_point]
        return trajectory


class HandManipulator:
    """Controls hand/gripper manipulation"""

    def __init__(self):
        self.gripper_joints = ['left_gripper_finger_joint', 'right_gripper_finger_joint']
        self.gripper_open_position = 0.05  # meters (open)
        self.gripper_closed_position = 0.0  # meters (closed)
        self.gripper_effort = 10.0  # N (gripper effort)

    def generate_gripper_trajectory(self, open_gripper: bool, gripper_width: float = None) -> JointTrajectory:
        """Generate trajectory for gripper control"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.gripper_joints

        point = JointTrajectoryPoint()
        point.time_from_start.sec = 1

        if open_gripper:
            position = self.gripper_open_position
            if gripper_width is not None:
                position = max(position, gripper_width)
        else:
            position = self.gripper_closed_position

        point.positions = [position, position]  # Both fingers
        point.velocities = [0.0, 0.0]
        point.effort = [self.gripper_effort, self.gripper_effort]

        trajectory.points = [point]
        return trajectory


class FailureRecovery:
    """Handles manipulation failure recovery mechanisms"""

    def __init__(self):
        self.recovery_strategies = {
            'grasp_failure': self._recover_from_grasp_failure,
            'object_slip': self._recover_from_object_slip,
            'collision': self._recover_from_collision,
            'joint_limit': self._recover_from_joint_limit
        }
        self.max_recovery_attempts = 3

    def attempt_recovery(self, failure_type: str, context: Dict) -> bool:
        """Attempt to recover from a manipulation failure"""
        if failure_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[failure_type]
            return recovery_func(context)
        else:
            return False

    def _recover_from_grasp_failure(self, context: Dict) -> bool:
        """Recover from grasp failure by trying alternative grasp"""
        print("Attempting to recover from grasp failure...")
        # Try a different grasp approach
        # In a real system, this would try a different grasp strategy
        return True  # Simulated success

    def _recover_from_object_slip(self, context: Dict) -> bool:
        """Recover from object slip by adjusting grip force"""
        print("Attempting to recover from object slip...")
        # Increase grip force
        # In a real system, this would adjust gripper force
        return True  # Simulated success

    def _recover_from_collision(self, context: Dict) -> bool:
        """Recover from collision by replanning trajectory"""
        print("Attempting to recover from collision...")
        # Replan trajectory to avoid collision
        # In a real system, this would use collision avoidance
        return True  # Simulated success

    def _recover_from_joint_limit(self, context: Dict) -> bool:
        """Recover from joint limit by adjusting configuration"""
        print("Attempting to recover from joint limit...")
        # Adjust joint configuration
        # In a real system, this would find alternative configuration
        return True  # Simulated success


class ManipulationExecutorNode(Node):
    def __init__(self):
        super().__init__('manipulation_executor')

        # Initialize components
        self.grasp_planner = GraspPlanner()
        self.arm_trajectory_generator = ArmTrajectoryGenerator()
        self.hand_manipulator = HandManipulator()
        self.failure_recovery = FailureRecovery()

        # Store latest data
        self.latest_scene_graph = None
        self.latest_action_plan = None
        self.latest_joint_state = None

        # Action clients for trajectory execution
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory'
        )

        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            'gripper_controller/follow_joint_trajectory'
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

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Publisher for manipulation status
        self.manip_status_pub = self.create_publisher(
            ActionPlan,
            'manipulation_status',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Manipulation Executor node initialized.')

    def scene_graph_callback(self, msg: SceneGraph):
        """Store the latest scene graph for manipulation decisions"""
        self.latest_scene_graph = msg

    def joint_state_callback(self, msg: JointState):
        """Store the latest joint states"""
        self.latest_joint_state = msg

    def action_plan_callback(self, msg: ActionPlan):
        """Process incoming action plans and execute manipulation actions"""
        self.get_logger().info(f'Received action plan with {len(msg.actions)} actions')

        # Store the action plan
        self.latest_action_plan = msg

        # Execute manipulation actions in sequence
        for i, action in enumerate(msg.actions):
            if action.action_type in ["grasp_object", "pick_up", "place_object"]:
                self.get_logger().info(f'Executing manipulation action {i+1}/{len(msg.actions)}: {action.action_type}')

                success = False
                attempt = 0

                while not success and attempt < self.failure_recovery.max_recovery_attempts:
                    try:
                        if action.action_type == "grasp_object" or action.action_type == "pick_up":
                            success = self.execute_grasp_action(action)
                        elif action.action_type == "place_object":
                            success = self.execute_place_action(action)
                        else:
                            success = False
                            self.get_logger().warn(f'Unknown manipulation action type: {action.action_type}')

                        if not success:
                            self.get_logger().warn(f'Manipulation action failed, attempt {attempt + 1}')
                            recovery_context = {
                                'action': action,
                                'attempt': attempt,
                                'scene_graph': self.latest_scene_graph
                            }
                            recovery_success = self.failure_recovery.attempt_recovery(
                                'grasp_failure', recovery_context
                            )

                            if recovery_success:
                                attempt += 1
                            else:
                                break
                    except Exception as e:
                        self.get_logger().error(f'Error executing manipulation action: {e}')
                        success = False
                        break

                if success:
                    self.get_logger().info(f'Manipulation action {i+1} completed successfully')
                else:
                    self.get_logger().error(f'Manipulation action {i+1} failed after {attempt + 1} attempts')
                    break  # Stop execution if manipulation fails

    def execute_grasp_action(self, action: VLAAction) -> bool:
        """Execute a grasp action"""
        self.get_logger().info(f'Executing grasp action for object: {action.parameters}')

        # Find the object in the scene graph
        target_object_id = None
        target_object_class = None
        target_pose = Pose()

        if self.latest_scene_graph:
            for i, obj_id in enumerate(self.latest_scene_graph.object_ids):
                if any(param in obj_id for param in action.parameters) or \
                   any(param in self.latest_scene_graph.object_classes[i] for param in action.parameters):
                    target_object_id = obj_id
                    target_object_class = self.latest_scene_graph.object_classes[i]
                    target_pose = self.latest_scene_graph.object_poses[i]
                    break

        if not target_object_id:
            # If object not found in scene graph, use the target pose from action
            if action.target_pose.position.x != 0.0 or action.target_pose.position.y != 0.0:
                target_pose = action.target_pose
                target_object_class = "unknown_object"
            else:
                self.get_logger().error('Could not find target object for grasping')
                return False

        # Plan the grasp
        grasp_result = self.grasp_planner.plan_grasp(target_object_class, target_pose, self.latest_scene_graph)
        if not grasp_result:
            self.get_logger().error(f'Could not plan grasp for object class: {target_object_class}')
            return False

        grasp_pose, gripper_width = grasp_result

        # Generate trajectories
        approach_traj = self.arm_trajectory_generator.generate_approach_trajectory(grasp_pose, self.get_current_pose())
        grasp_traj = self.arm_trajectory_generator.generate_grasp_trajectory(grasp_pose)
        gripper_open_traj = self.hand_manipulator.generate_gripper_trajectory(open_gripper=True, gripper_width=gripper_width)
        gripper_close_traj = self.hand_manipulator.generate_gripper_trajectory(open_gripper=False)

        # Execute the manipulation sequence
        self.get_logger().info('Executing approach trajectory...')
        if not self.execute_trajectory(approach_traj, 'arm'):
            return False

        self.get_logger().info('Opening gripper...')
        if not self.execute_trajectory(gripper_open_traj, 'gripper'):
            return False

        self.get_logger().info('Executing grasp trajectory...')
        if not self.execute_trajectory(grasp_traj, 'arm'):
            return False

        self.get_logger().info('Closing gripper...')
        if not self.execute_trajectory(gripper_close_traj, 'gripper'):
            return False

        # Lift the object slightly
        lift_traj = self.arm_trajectory_generator.generate_lift_trajectory()
        self.get_logger().info('Lifting object...')
        if not self.execute_trajectory(lift_traj, 'arm'):
            return False

        self.get_logger().info('Grasp completed successfully')
        return True

    def execute_place_action(self, action: VLAAction) -> bool:
        """Execute a place action"""
        self.get_logger().info(f'Executing place action at: {action.target_pose}')

        # For simulation, we'll just open the gripper at the target location
        place_pose = action.target_pose if action.target_pose.position.x != 0.0 else self.get_current_pose()

        # Generate trajectory to place location
        place_traj = self.arm_trajectory_generator.generate_approach_trajectory(place_pose, self.get_current_pose())
        gripper_open_traj = self.hand_manipulator.generate_gripper_trajectory(open_gripper=True)

        # Execute the placement sequence
        self.get_logger().info('Moving to place location...')
        if not self.execute_trajectory(place_traj, 'arm'):
            return False

        self.get_logger().info('Opening gripper to release object...')
        if not self.execute_trajectory(gripper_open_traj, 'gripper'):
            return False

        self.get_logger().info('Place completed successfully')
        return True

    def get_current_pose(self) -> Pose:
        """Get current end-effector pose (in simulation, return a default pose)"""
        # In a real system, this would get the current pose from forward kinematics
        # For simulation, return a default pose
        return Pose(
            position=Point(x=0.5, y=0.0, z=0.8),
            orientation=Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)  # Looking down
        )

    def execute_trajectory(self, trajectory: JointTrajectory, controller_type: str) -> bool:
        """Execute a joint trajectory"""
        if controller_type == 'arm':
            client = self.arm_client
            action_name = 'arm_controller/follow_joint_trajectory'
        elif controller_type == 'gripper':
            client = self.gripper_client
            action_name = 'gripper_controller/follow_joint_trajectory'
        else:
            self.get_logger().error(f'Unknown controller type: {controller_type}')
            return False

        # Wait for action server
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'{action_name} action server not available')
            return False

        # Create trajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory

        # Send trajectory goal
        self.get_logger().info(f'Executing {controller_type} trajectory with {len(trajectory.points)} points')

        # Use async call to not block
        future = client.send_goal_async(goal_msg)

        # Wait for result (with timeout)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info(f'{controller_type} trajectory goal accepted')

                # Get result
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    if result:
                        self.get_logger().info(f'{controller_type} trajectory completed successfully')
                        return True
                    else:
                        self.get_logger().error(f'{controller_type} trajectory failed')
                        return False
                else:
                    self.get_logger().error(f'{controller_type} trajectory result future timed out')
                    return False
            else:
                self.get_logger().error(f'{controller_type} trajectory goal was rejected')
                return False
        else:
            self.get_logger().error(f'Failed to send {controller_type} trajectory goal')
            return False


def main(args=None):
    rclpy.init(args=args)

    try:
        manipulation_executor = ManipulationExecutorNode()

        try:
            rclpy.spin(manipulation_executor)
        except KeyboardInterrupt:
            pass
        finally:
            manipulation_executor.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()