#!/usr/bin/env python3
"""
Language Understanding Node for Vision-Language-Action (VLA) System

This node processes natural language commands using LLMs to extract
intents, entities, and context for robotic action planning.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import ROS 2 messages
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Time

# Import custom messages
# Note: We'll need to adjust these imports once the package is properly set up
# from humanoid_robotics_book.msg import SpeechCommand, Intent


class CommandType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    SOCIAL = "social"


@dataclass
class ExtractedIntent:
    command_type: CommandType
    entities: List[str]
    entity_poses: List[Pose]
    confidence: float
    context_id: str
    raw_command: str
    parameters: List[str]


class ContextManager:
    """Manages conversation and task context for language understanding"""

    def __init__(self):
        self.context_history = {}
        self.current_context_id = "default"
        self.max_history = 10  # Keep last 10 interactions

    def create_new_context(self) -> str:
        """Create a new context ID"""
        import uuid
        new_id = str(uuid.uuid4())
        self.current_context_id = new_id
        self.context_history[new_id] = {
            'interactions': [],
            'entities': {},
            'tasks': []
        }
        return new_id

    def add_interaction(self, command: str, intent: ExtractedIntent):
        """Add an interaction to the current context"""
        if self.current_context_id not in self.context_history:
            self.create_new_context()

        context = self.context_history[self.current_context_id]
        context['interactions'].append({
            'command': command,
            'intent': intent,
            'timestamp': rclpy.clock.Clock().now().to_msg()
        })

        # Limit history size
        if len(context['interactions']) > self.max_history:
            context['interactions'] = context['interactions'][-self.max_history:]

    def resolve_pronouns(self, text: str) -> str:
        """Resolve pronouns based on context (e.g., 'it' -> specific object)"""
        # Simple pronoun resolution based on last mentioned entities
        if self.current_context_id in self.context_history:
            context = self.context_history[self.current_context_id]
            if context['interactions']:
                # Get the last mentioned entities
                last_intent = context['interactions'][-1].get('intent')
                if last_intent and last_intent.entities:
                    last_entity = last_intent.entities[-1] if last_intent.entities else None
                    if last_entity:
                        text = re.sub(r'\bit\b', last_entity, text, flags=re.IGNORECASE)
                        text = re.sub(r'\bthat\b', last_entity, text, flags=re.IGNORECASE)
                        text = re.sub(r'\bthe\s+\w+\b', f'the {last_entity}', text, flags=re.IGNORECASE)
        return text

    def get_context(self) -> Dict:
        """Get the current context"""
        return self.context_history.get(self.current_context_id, {})


class EntityExtractor:
    """Extracts entities from natural language commands"""

    def __init__(self):
        # Common object categories for humanoid robotics
        self.object_categories = [
            'cup', 'bottle', 'box', 'chair', 'table', 'person', 'human',
            'robot', 'door', 'window', 'kitchen', 'living room', 'bedroom',
            'couch', 'sofa', 'book', 'phone', 'keys', 'wallet', 'food',
            'snack', 'drink', 'water', 'coffee', 'tea', 'mug', 'plate',
            'fork', 'spoon', 'knife', 'bowl', 'fridge', 'microwave',
            'counter', 'cabinet', 'drawer', 'light', 'switch', 'fan'
        ]

        # Common location descriptors
        self.location_descriptors = [
            'kitchen', 'living room', 'bedroom', 'bathroom', 'office',
            'dining room', 'hallway', 'corridor', 'entrance', 'exit',
            'near', 'by', 'next to', 'beside', 'in front of', 'behind',
            'left', 'right', 'front', 'back', 'up', 'down', 'over', 'under'
        ]

        # Common action verbs
        self.action_verbs = [
            'go', 'navigate', 'move', 'walk', 'bring', 'fetch', 'get',
            'take', 'pick up', 'grasp', 'hold', 'carry', 'deliver',
            'follow', 'accompany', 'assist', 'help', 'greet', 'meet',
            'introduce', 'talk to', 'speak to', 'look at', 'watch',
            'find', 'locate', 'search for', 'show', 'demonstrate'
        ]

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using pattern matching and NLP techniques"""
        entities = []
        text_lower = text.lower()

        # Extract known object categories
        for category in self.object_categories:
            if category in text_lower:
                # Check if it's not part of a larger word
                pattern = r'\b' + re.escape(category) + r'\b'
                if re.search(pattern, text_lower):
                    entities.append(category)

        # Extract locations
        for location in self.location_descriptors:
            if location in text_lower:
                pattern = r'\b' + re.escape(location) + r'\b'
                if re.search(pattern, text_lower):
                    entities.append(location)

        # Extract specific objects with descriptors
        # Pattern: "the red cup", "a big bottle", etc.
        object_pattern = r'(?:the|a|an)\s+(\w+\s+)?\w+\s+(?:cup|bottle|box|chair|table|person|human|book|phone|keys|wallet|food|drink|mug|plate|bowl)'
        matches = re.findall(object_pattern, text_lower)
        for match in matches:
            entities.extend(match.strip().split()[-2:])  # Get the last 2 words (descriptor and object)

        # Remove duplicates while preserving order
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)

        return unique_entities


class IntentClassifier:
    """Classifies intents from natural language commands"""

    def __init__(self):
        # Define patterns for different command types
        self.navigation_patterns = [
            r'go\s+to',
            r'move\s+to',
            r'walk\s+to',
            r'navigate\s+to',
            r'go\s+(?:kitchen|living room|bedroom|bathroom|office)',
            r'bring\s+me',
            r'get\s+(?:from|to)',
            r'follow',
            r'come\s+here',
            r'come\s+to'
        ]

        self.manipulation_patterns = [
            r'pick\s+up',
            r'grasp',
            r'hold',
            r'carry',
            r'take\s+(?!to|from)',
            r'get\s+(?!to|from)',
            r'bring\s+(?!to)',
            r'fetch',
            r'lift',
            r'grab',
            r'catch',
            r'collect',
            r'retrieve'
        ]

        self.interaction_patterns = [
            r'talk\s+to',
            r'speak\s+to',
            r'greet',
            r'meet',
            r'introduce',
            r'help',
            r'assist',
            r'look\s+at',
            r'watch',
            r'show',
            r'demonstrate',
            r'play'
        ]

        self.social_patterns = [
            r'wave',
            r'nod',
            r'point',
            r'gesture',
            r'bow',
            r'smile',
            r'hello',
            r'hi',
            r'goodbye',
            r'bye',
            r'please',
            r'thank you',
            r'thanks'
        ]

    def classify_intent(self, text: str) -> Tuple[CommandType, float]:
        """Classify the intent type with confidence score"""
        text_lower = text.lower()

        # Count matches for each category
        nav_matches = sum(1 for pattern in self.navigation_patterns if re.search(pattern, text_lower))
        manip_matches = sum(1 for pattern in self.manipulation_patterns if re.search(pattern, text_lower))
        inter_matches = sum(1 for pattern in self.interaction_patterns if re.search(pattern, text_lower))
        social_matches = sum(1 for pattern in self.social_patterns if re.search(pattern, text_lower))

        # Determine the dominant category
        max_matches = max(nav_matches, manip_matches, inter_matches, social_matches)

        if max_matches == 0:
            # Default to navigation if no clear pattern
            return CommandType.NAVIGATION, 0.5

        total_matches = nav_matches + manip_matches + inter_matches + social_matches
        confidence = max_matches / total_matches if total_matches > 0 else 0.5

        if nav_matches == max_matches:
            return CommandType.NAVIGATION, min(confidence, 0.9)
        elif manip_matches == max_matches:
            return CommandType.MANIPULATION, min(confidence, 0.9)
        elif inter_matches == max_matches:
            return CommandType.INTERACTION, min(confidence, 0.9)
        else:  # social_matches == max_matches
            return CommandType.SOCIAL, min(confidence, 0.9)


class LLMInterface:
    """Interface to Large Language Model for advanced understanding"""

    def __init__(self):
        # For this implementation, we'll simulate LLM responses
        # In a real system, this would connect to OpenAI API or local model
        self.enabled = False  # Set to True when actual LLM is available

    def process_command(self, command: str, context: Dict = None) -> Optional[ExtractedIntent]:
        """Process command using LLM for complex understanding"""
        if not self.enabled:
            return None

        # This is a placeholder for actual LLM integration
        # In a real implementation, this would call the LLM API
        try:
            # Simulate LLM processing
            import time
            time.sleep(0.1)  # Simulate processing time

            # Return simulated result
            intent = ExtractedIntent(
                command_type=CommandType.NAVIGATION,
                entities=['kitchen'],
                entity_poses=[],
                confidence=0.8,
                context_id='simulated',
                raw_command=command,
                parameters=[]
            )
            return intent
        except Exception as e:
            print(f"LLM processing failed: {e}")
            return None


class LanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('language_understanding')

        # Initialize components
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.context_manager = ContextManager()
        self.llm_interface = LLMInterface()

        # Parameters
        self.declare_parameter('use_llm', False)  # Whether to use LLM for processing
        self.declare_parameter('confidence_threshold', 0.6)  # Minimum confidence for processing

        self.use_llm = self.get_parameter('use_llm').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value

        # Publishers and Subscribers
        # self.speech_sub = self.create_subscription(
        #     SpeechCommand,
        #     'speech_command',
        #     self.speech_callback,
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )
        #
        # self.intent_pub = self.create_publisher(
        #     Intent,
        #     'structured_intent',
        #     QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # )

        self.get_logger().info('Language Understanding node initialized.')

    def speech_callback(self, msg):
        """Process incoming speech commands"""
        # This is a placeholder - in a real system this would receive SpeechCommand messages
        command_text = getattr(msg, 'utterance', str(msg))  # Fallback to string representation

        if hasattr(msg, 'confidence') and msg.confidence < 0.5:  # Basic confidence check
            self.get_logger().info(f'Ignoring low-confidence speech: {getattr(msg, "confidence", "unknown")}')
            return

        # Resolve pronouns based on context
        resolved_text = self.context_manager.resolve_pronouns(command_text)

        # Process with rule-based system first
        extracted_intent = self.extract_intent_from_text(resolved_text)

        # Optionally enhance with LLM if enabled and confidence is low
        if self.use_llm and extracted_intent.confidence < self.confidence_threshold:
            llm_intent = self.llm_interface.process_command(resolved_text, self.context_manager.get_context())
            if llm_intent and llm_intent.confidence > extracted_intent.confidence:
                extracted_intent = llm_intent

        # Validate confidence threshold
        if extracted_intent.confidence >= self.confidence_threshold:
            # Add to context
            self.context_manager.add_interaction(command_text, extracted_intent)

            # Publish the intent
            # self.publish_intent(extracted_intent, msg.header.stamp if hasattr(msg, 'header') else None)
            self.get_logger().info(f'Intent extracted: {extracted_intent.command_type.value} with entities {extracted_intent.entities}')
        else:
            self.get_logger().info(f'Intent confidence too low: {extracted_intent.confidence}')

    def extract_intent_from_text(self, text: str) -> ExtractedIntent:
        """Extract intent from text using rule-based approach"""
        # Classify intent
        command_type, type_confidence = self.intent_classifier.classify_intent(text)

        # Extract entities
        entities = self.entity_extractor.extract_entities(text)

        # Calculate overall confidence based on multiple factors
        entity_count = len(entities)
        has_entities = entity_count > 0
        has_clear_intent = type_confidence > 0.5

        # Calculate confidence score
        base_confidence = type_confidence
        if has_entities:
            base_confidence = min(base_confidence + 0.1 * entity_count, 0.9)
        if has_clear_intent:
            base_confidence = min(base_confidence + 0.1, 0.9)

        # Create empty poses for entities
        entity_poses = [Pose() for _ in entities]  # Will be filled by perception system

        return ExtractedIntent(
            command_type=command_type,
            entities=entities,
            entity_poses=entity_poses,
            confidence=base_confidence,
            context_id=self.context_manager.current_context_id,
            raw_command=text,
            parameters=[]
        )

    def publish_intent(self, extracted_intent: ExtractedIntent, timestamp: Time = None):
        """Publish the extracted intent as a ROS 2 message"""
        # This is a placeholder - in a real system this would publish Intent messages
        if timestamp is None:
            timestamp = self.get_clock().now().to_msg()

        # msg = Intent()
        # msg.header.stamp = timestamp
        # msg.header.frame_id = 'language_understanding'
        # msg.command_type = extracted_intent.command_type.value
        # msg.entities = extracted_intent.entities
        # msg.entity_poses = extracted_intent.entity_poses
        # msg.timestamp = timestamp
        # msg.context_id = extracted_intent.context_id
        # msg.raw_command = extracted_intent.raw_command
        # msg.confidence = extracted_intent.confidence
        # msg.parameters = extracted_intent.parameters
        #
        # self.intent_pub.publish(msg)
        self.get_logger().info(
            f'Published intent: {extracted_intent.command_type.value} '
            f'with entities {extracted_intent.entities} '
            f'(confidence: {extracted_intent.confidence:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)

    try:
        language_understanding = LanguageUnderstandingNode()

        try:
            rclpy.spin(language_understanding)
        except KeyboardInterrupt:
            pass
        finally:
            language_understanding.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()