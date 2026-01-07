#!/usr/bin/env python3
"""
Context Manager Node for Vision-Language-Action (VLA) System

This node maintains conversation and task context for humanoid robots,
including conversation history tracking, object reference resolution,
task state management, and context switching capabilities.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from typing import Dict, List, Optional, Tuple, Any
import time
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime

# Import ROS 2 messages
from std_msgs.msg import Header, String
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

# Import custom messages
from humanoid_robotics_book.msg import SpeechCommand, Intent, SceneGraph, ActionPlan


@dataclass
class ConversationEntry:
    """Represents a single conversation entry"""
    id: str
    timestamp: Time
    speaker: str  # 'human' or 'robot'
    text: str
    entities: List[str] = field(default_factory=list)
    intent: Optional[str] = None
    context_snapshot: Optional[Dict] = None


@dataclass
class TaskState:
    """Represents the state of a task"""
    id: str
    name: str
    status: str  # 'pending', 'in_progress', 'completed', 'failed'
    created_time: Time
    updated_time: Time
    progress: float  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)
    related_objects: List[str] = field(default_factory=list)


@dataclass
class ObjectReference:
    """Represents an object reference with resolution"""
    id: str
    name: str
    resolved_name: str  # What the user likely meant
    last_seen_pose: Optional[Pose] = None
    last_seen_time: Optional[Time] = None
    confidence: float = 1.0


class ConversationTracker:
    """Tracks conversation history and context"""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.entries: List[ConversationEntry] = []
        self.current_context_id = str(uuid.uuid4())

    def add_entry(self, speaker: str, text: str, entities: List[str] = None, intent: str = None):
        """Add a conversation entry"""
        entry = ConversationEntry(
            id=str(uuid.uuid4()),
            timestamp=Time(sec=int(time.time()), nanosec=0),
            speaker=speaker,
            text=text,
            entities=entities or [],
            intent=intent,
            context_snapshot=self._capture_context()
        )

        self.entries.append(entry)

        # Limit history size
        if len(self.entries) > self.max_history:
            self.entries = self.entries[-self.max_history:]

    def get_recent_entries(self, count: int = 5) -> List[ConversationEntry]:
        """Get recent conversation entries"""
        return self.entries[-count:] if len(self.entries) >= count else self.entries

    def get_context_for_pronouns(self) -> Dict[str, str]:
        """Get context for resolving pronouns like 'it', 'that', 'there'"""
        context = {}

        # Look for the last mentioned objects
        for entry in reversed(self.entries[-10:]):  # Look at last 10 entries
            if entry.entities:
                # Use the last mentioned entity as the referent for 'it'
                context['it'] = entry.entities[-1]
                context['that'] = entry.entities[-1]
                break

        return context

    def _capture_context(self) -> Dict:
        """Capture current context snapshot"""
        return {
            'timestamp': time.time(),
            'context_id': self.current_context_id,
            'entry_count': len(self.entries)
        }

    def create_new_context(self) -> str:
        """Create a new conversation context"""
        self.current_context_id = str(uuid.uuid4())
        return self.current_context_id


class ObjectReferenceResolver:
    """Resolves object references in user commands"""

    def __init__(self):
        self.object_references: Dict[str, ObjectReference] = {}
        self.name_aliases: Dict[str, str] = {}  # user_name -> resolved_name

    def add_object_reference(self, obj_id: str, name: str, resolved_name: str = None,
                           pose: Pose = None, confidence: float = 1.0):
        """Add an object reference"""
        if resolved_name is None:
            resolved_name = name

        obj_ref = ObjectReference(
            id=obj_id,
            name=name,
            resolved_name=resolved_name,
            last_seen_pose=pose,
            last_seen_time=Time(sec=int(time.time()), nanosec=0),
            confidence=confidence
        )

        self.object_references[obj_id] = obj_ref
        self.name_aliases[name.lower()] = resolved_name

    def resolve_reference(self, reference: str) -> Optional[str]:
        """Resolve an object reference to a specific object"""
        ref_lower = reference.lower()

        # Direct name match
        if ref_lower in self.name_aliases:
            return self.name_aliases[ref_lower]

        # Partial match
        for name, resolved_name in self.name_aliases.items():
            if ref_lower in name or name in ref_lower:
                return resolved_name

        # Check for pronouns
        if ref_lower in ['it', 'that', 'the object', 'the thing']:
            # Return the most recently seen object
            most_recent = None
            latest_time = 0
            for obj_ref in self.object_references.values():
                if obj_ref.last_seen_time and obj_ref.last_seen_time.sec > latest_time:
                    latest_time = obj_ref.last_seen_time.sec
                    most_recent = obj_ref.resolved_name
            return most_recent

        return None

    def update_object_pose(self, obj_id: str, pose: Pose):
        """Update the pose of a known object"""
        if obj_id in self.object_references:
            self.object_references[obj_id].last_seen_pose = pose
            self.object_references[obj_id].last_seen_time = Time(sec=int(time.time()), nanosec=0)

    def get_object_by_name(self, name: str) -> Optional[ObjectReference]:
        """Get an object reference by name"""
        for obj_ref in self.object_references.values():
            if obj_ref.name.lower() == name.lower() or obj_ref.resolved_name.lower() == name.lower():
                return obj_ref
        return None


class TaskStateManager:
    """Manages task states and progress"""

    def __init__(self):
        self.tasks: Dict[str, TaskState] = {}
        self.active_task_id: Optional[str] = None

    def create_task(self, name: str, dependencies: List[str] = None, related_objects: List[str] = None) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = TaskState(
            id=task_id,
            name=name,
            status='pending',
            created_time=Time(sec=int(time.time()), nanosec=0),
            updated_time=Time(sec=int(time.time()), nanosec=0),
            progress=0.0,
            dependencies=dependencies or [],
            related_objects=related_objects or []
        )

        self.tasks[task_id] = task
        return task_id

    def update_task_status(self, task_id: str, status: str, progress: float = None):
        """Update task status and optionally progress"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_time = Time(sec=int(time.time()), nanosec=0)
            if progress is not None:
                self.tasks[task_id].progress = progress

            # Set as active task if in progress
            if status == 'in_progress':
                self.active_task_id = task_id

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def get_active_task(self) -> Optional[TaskState]:
        """Get the currently active task"""
        if self.active_task_id and self.active_task_id in self.tasks:
            return self.tasks[self.active_task_id]
        return None

    def complete_task(self, task_id: str):
        """Mark a task as completed"""
        self.update_task_status(task_id, 'completed', 1.0)

    def fail_task(self, task_id: str):
        """Mark a task as failed"""
        self.update_task_status(task_id, 'failed', 0.0)

    def get_task_by_name(self, name: str) -> Optional[TaskState]:
        """Get a task by name"""
        for task in self.tasks.values():
            if task.name.lower() == name.lower():
                return task
        return None


class ContextSwitcher:
    """Manages context switching between different interaction modes"""

    def __init__(self):
        self.contexts: Dict[str, Dict] = {}
        self.current_context = "default"
        self.context_stack: List[str] = []

    def create_context(self, name: str, properties: Dict = None) -> str:
        """Create a new context with given properties"""
        context_id = str(uuid.uuid4())
        self.contexts[context_id] = {
            'name': name,
            'properties': properties or {},
            'created_time': time.time(),
            'active': False
        }
        return context_id

    def switch_context(self, context_id: str) -> bool:
        """Switch to a different context"""
        if context_id in self.contexts:
            # Deactivate current context
            if self.current_context in self.contexts:
                self.contexts[self.current_context]['active'] = False

            # Activate new context
            self.contexts[context_id]['active'] = True
            self.current_context = context_id
            return True
        return False

    def push_context(self, context_id: str):
        """Push current context to stack and switch to new context"""
        self.context_stack.append(self.current_context)
        self.switch_context(context_id)

    def pop_context(self):
        """Pop context from stack and return to previous context"""
        if self.context_stack:
            prev_context = self.context_stack.pop()
            self.switch_context(prev_context)

    def get_current_context_info(self) -> Dict:
        """Get information about the current context"""
        if self.current_context in self.contexts:
            return self.contexts[self.current_context]
        return {}


class ContextManagerNode(Node):
    def __init__(self):
        super().__init__('context_manager')

        # Initialize components
        self.conversation_tracker = ConversationTracker()
        self.object_resolver = ObjectReferenceResolver()
        self.task_manager = TaskStateManager()
        self.context_switcher = ContextSwitcher()

        # Initialize default contexts
        self.default_context = self.context_switcher.create_context("default", {"type": "general"})
        self.task_context = self.context_switcher.create_context("task", {"type": "task_execution"})
        self.social_context = self.context_switcher.create_context("social", {"type": "social_interaction"})

        # Store latest data
        self.latest_speech_command = None
        self.latest_intent = None
        self.latest_scene_graph = None
        self.latest_action_plan = None

        # Publishers and Subscribers
        self.speech_command_sub = self.create_subscription(
            SpeechCommand,
            'speech_command',
            self.speech_command_callback,
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

        # Publishers for context updates
        self.context_update_pub = self.create_publisher(
            String,
            'context_update',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.get_logger().info('Context Manager node initialized.')

    def speech_command_callback(self, msg: SpeechCommand):
        """Handle incoming speech commands and update context"""
        self.get_logger().info(f'Received speech command: {msg.utterance}')

        # Add to conversation history
        self.conversation_tracker.add_entry(
            speaker='human',
            text=msg.utterance,
            entities=[],
            intent=None
        )

        # Resolve any object references in the command
        resolved_command = self.resolve_command_references(msg.utterance)
        self.get_logger().info(f'Resolved command: {resolved_command}')

        # Update the latest speech command
        self.latest_speech_command = msg

    def intent_callback(self, msg: Intent):
        """Handle incoming intents and update context"""
        self.get_logger().info(f'Received intent: {msg.command_type} with entities {msg.entities}')

        # Add to conversation history with intent information
        self.conversation_tracker.add_entry(
            speaker='robot',
            text=f"Processing {msg.command_type} command",
            entities=msg.entities,
            intent=msg.command_type
        )

        # Update object references based on entities
        for entity in msg.entities:
            self.object_resolver.add_object_reference(
                obj_id=f"entity_{len(self.object_resolver.object_references)}",
                name=entity,
                resolved_name=entity
            )

        # Update latest intent
        self.latest_intent = msg

    def scene_graph_callback(self, msg: SceneGraph):
        """Handle incoming scene graphs and update object references"""
        self.get_logger().info(f'Received scene graph with {len(msg.object_ids)} objects')

        # Update object references with detected objects
        for i, obj_id in enumerate(msg.object_ids):
            obj_class = msg.object_classes[i] if i < len(msg.object_classes) else "unknown"
            obj_pose = msg.object_poses[i] if i < len(msg.object_poses) else None

            self.object_resolver.add_object_reference(
                obj_id=obj_id,
                name=obj_class,
                resolved_name=obj_class,
                pose=obj_pose,
                confidence=msg.object_confidences[i] if i < len(msg.object_confidences) else 1.0
            )

        # Update latest scene graph
        self.latest_scene_graph = msg

    def action_plan_callback(self, msg: ActionPlan):
        """Handle incoming action plans and update task context"""
        self.get_logger().info(f'Received action plan: {msg.plan_type} with {len(msg.actions)} actions')

        # Create or update task based on action plan
        task_id = self.task_manager.create_task(
            name=f"action_plan_{msg.plan_id}",
            related_objects=[action.parameters[0] if action.parameters else "unknown"
                           for action in msg.actions if action.parameters]
        )

        self.task_manager.update_task_status(task_id, 'in_progress', 0.0)

        # Update latest action plan
        self.latest_action_plan = msg

    def resolve_command_references(self, command: str) -> str:
        """Resolve object references in a command"""
        # Find all potential object references in the command
        words = command.lower().split()
        resolved_command = command

        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            resolved_obj = self.object_resolver.resolve_reference(clean_word)

            if resolved_obj and resolved_obj != clean_word:
                # Replace the reference in the command
                resolved_command = resolved_command.replace(
                    word, f"{word} (resolved to: {resolved_obj})"
                )

        return resolved_command

    def get_context_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current context"""
        return {
            'timestamp': time.time(),
            'conversation_entries': len(self.conversation_tracker.entries),
            'object_references': len(self.object_resolver.object_references),
            'tasks': len(self.task_manager.tasks),
            'active_task': self.task_manager.active_task_id,
            'current_context': self.context_switcher.current_context,
            'recent_conversations': [
                {'speaker': e.speaker, 'text': e.text, 'timestamp': e.timestamp.sec}
                for e in self.conversation_tracker.get_recent_entries(3)
            ]
        }

    def handle_pronoun_resolution(self, text: str) -> str:
        """Resolve pronouns in text based on conversation context"""
        # Get pronoun context
        pronoun_context = self.conversation_tracker.get_context_for_pronouns()

        # Replace pronouns in the text
        resolved_text = text
        for pronoun, referent in pronoun_context.items():
            resolved_text = re.sub(
                r'\b' + re.escape(pronoun) + r'\b',
                f"{pronoun} ({referent})",
                resolved_text,
                flags=re.IGNORECASE
            )

        return resolved_text

    def get_active_task_status(self) -> str:
        """Get the status of the active task"""
        active_task = self.task_manager.get_active_task()
        if active_task:
            return f"Active task: {active_task.name} ({active_task.status}, {active_task.progress*100:.1f}%)"
        else:
            return "No active task"

    def get_context_summary(self) -> str:
        """Get a summary of the current context"""
        active_task_status = self.get_active_task_status()
        recent_conv = self.conversation_tracker.get_recent_entries(2)
        recent_text = [f"{e.speaker}: {e.text}" for e in recent_conv]

        summary = f"Context Summary:\n"
        summary += f"- Active task: {active_task_status}\n"
        summary += f"- Objects tracked: {len(self.object_resolver.object_references)}\n"
        summary += f"- Conversation entries: {len(self.conversation_tracker.entries)}\n"
        summary += f"- Recent conversation: {recent_text}\n"
        summary += f"- Current context: {self.context_switcher.current_context}\n"

        return summary


def main(args=None):
    rclpy.init(args=args)

    try:
        context_manager = ContextManagerNode()

        try:
            rclpy.spin(context_manager)
        except KeyboardInterrupt:
            pass
        finally:
            context_manager.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()