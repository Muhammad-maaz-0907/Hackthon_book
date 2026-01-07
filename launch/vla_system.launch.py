from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for the complete VLA system."""

    return LaunchDescription([
        # Speech Processor Node
        Node(
            package='humanoid_robotics_book',
            executable='speech_processor',
            name='speech_processor',
            parameters=[
                {'sample_rate': 16000},
                {'chunk_size': 1024},
                {'model_size': 'base'},
                {'language': 'en'},
                {'noise_threshold': 0.01},
                {'silence_duration': 1.0}
            ],
            output='screen'
        ),

        # Language Understanding Node
        Node(
            package='humanoid_robotics_book',
            executable='language_understanding',
            name='language_understanding',
            parameters=[
                {'use_llm': False},
                {'confidence_threshold': 0.6}
            ],
            output='screen'
        ),

        # Perception Integration Node
        Node(
            package='humanoid_robotics_book',
            executable='perception_integrator',
            name='perception_integrator',
            output='screen'
        ),

        # Intent Interpreter Node
        Node(
            package='humanoid_robotics_book',
            executable='intent_interpreter',
            name='intent_interpreter',
            output='screen'
        ),

        # Safety Validator Node
        Node(
            package='humanoid_robotics_book',
            executable='safety_validator',
            name='safety_validator',
            output='screen'
        ),

        # Navigation Executor Node
        Node(
            package='humanoid_robotics_book',
            executable='navigation_executor',
            name='navigation_executor',
            output='screen'
        ),

        # Manipulation Executor Node
        Node(
            package='humanoid_robotics_book',
            executable='manipulation_executor',
            name='manipulation_executor',
            output='screen'
        ),

        # Social Behavior Executor Node
        Node(
            package='humanoid_robotics_book',
            executable='social_behavior_executor',
            name='social_behavior_executor',
            output='screen'
        ),

        # Feedback Generator Node
        Node(
            package='humanoid_robotics_book',
            executable='feedback_generator',
            name='feedback_generator',
            output='screen'
        ),

        # Context Manager Node
        Node(
            package='humanoid_robotics_book',
            executable='context_manager',
            name='context_manager',
            output='screen'
        ),

        # VLA Main Orchestrator Node
        Node(
            package='humanoid_robotics_book',
            executable='vla_main',
            name='vla_main',
            output='screen'
        )
    ])