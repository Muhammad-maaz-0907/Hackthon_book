# Data Model: Module 4 - Vision-Language-Action (VLA) Systems

## Core Message Definitions

### SpeechCommand.msg
```
std_msgs/Header header
string utterance          # The transcribed text from speech
float32 confidence        # Confidence score for the transcription (0.0 to 1.0)
string language           # Detected language code (e.g., 'en', 'es', 'fr')
builtin_interfaces/Time timestamp  # When the speech was processed
string[] alternatives     # Alternative transcriptions if available
```

### Intent.msg
```
std_msgs/Header header
string command_type      # Type: "navigation", "manipulation", "interaction", "social"
string[] entities        # Identified entities (objects, locations, people)
geometry_msgs/Pose[] entity_poses  # 3D poses of entities if available
builtin_interfaces/Time timestamp  # When the intent was processed
string context_id        # Context identifier for conversation management
string raw_command       # Original command text
float32 confidence      # Confidence in intent interpretation (0.0 to 1.0)
string[] parameters     # Additional parameters for the command
```

### SceneGraph.msg
```
std_msgs/Header header
builtin_interfaces/Time timestamp  # When the scene was captured

# Objects in the scene
string[] object_ids              # Unique identifiers for objects
string[] object_classes          # Object category (cup, person, chair, etc.)
geometry_msgs/Pose[] object_poses # 3D poses of objects
float32[] object_confidences     # Detection confidences
string[] object_attributes       # Additional attributes (color, size, etc.)

# Relationships between objects
string[] subject_ids             # Subject object IDs in relationships
string[] predicates              # Relationship types (left_of, right_of, on_top_of, etc.)
string[] object_ids_rel          # Object IDs in relationships
string[] relationship_confidences # Confidence in relationships

# Spatial regions
string[] region_ids              # Unique identifiers for spatial regions
string[] region_names            # Names of regions (kitchen, living room, etc.)
geometry_msgs/Polygon region_bounds # Boundaries of regions

# Affordances
string[] affordance_object_ids   # Objects that have affordances
string[] affordance_types        # Types of affordances (graspable, sitable, etc.)
geometry_msgs/Pose[] affordance_poses # Poses for affordance interaction
```

### VLAAction.msg
```
string action_type             # Type: "navigate_to", "grasp_object", "follow_person", etc.
string[] parameters            # Action-specific parameters (object_id, location, text, etc.)
geometry_msgs/Pose target_pose # Target pose for the action if applicable
float32 priority              # Priority of the action (0.0 to 1.0)
bool is_optional              # Whether the action is optional
builtin_interfaces/Duration timeout  # Timeout for the action completion
string[] preconditions        # Conditions that must be true before executing
string[] effects              # Effects of the action on the world state
```

### ActionPlan.msg
```
std_msgs/Header header
builtin_interfaces/Time timestamp  # When the plan was created

# Plan identification
string plan_id                    # Unique identifier for the plan
string plan_type                  # Type: "navigation", "manipulation", "interaction", "composite"

# Action sequence
VLAAction[] actions              # Sequential actions to execute
int32[] action_dependencies      # Dependencies between actions (action i depends on action j)

# Plan metadata
float32 estimated_duration       # Estimated time to complete the plan (in seconds)
float32 confidence              # Overall confidence in the plan (0.0 to 1.0)
string[] resources_required      # Resources needed for execution (arm, gripper, etc.)
string status                   # Current status: "planned", "executing", "completed", "failed"

# Constraints
float32[] execution_constraints   # Constraints for execution (speed, force, etc.)
string[] safety_constraints      # Safety constraints for the plan
string[] temporal_constraints    # Temporal constraints (before, after, etc.)

# Context
string context_id               # Context identifier from language understanding
string original_command         # Original command that generated this plan
```

### SocialBehavior.msg
```
std_msgs/Header header
builtin_interfaces/Time timestamp  # When the behavior was triggered

# Behavior identification
string behavior_type               # Type: "greeting", "wave", "nod", "gesture", "follow", "maintain_distance"
string behavior_name               # Specific behavior name (e.g., "hello_wave", "acknowledge_nod")

# Target information
string target_type                 # "person", "object", "location", "none"
string target_id                   # ID of the target entity
geometry_msgs/Pose target_pose     # Pose of the target if applicable

# Behavior parameters
float32 intensity                  # Intensity/speed of the behavior (0.0 to 1.0)
float32 duration                   # Duration of the behavior in seconds
string[] modifiers                 # Additional behavior modifiers (polite, friendly, formal, etc.)

# Social context
string social_context              # Context: "greeting", "farewell", "attention", "interaction", "navigation"
string cultural_context            # Cultural considerations if applicable
bool requires_acknowledgment       # Whether acknowledgment is expected from human

# Execution constraints
float32 priority                   # Priority of the behavior (0.0 to 1.0)
builtin_interfaces/Duration timeout  # Timeout for behavior completion
string[] preconditions            # Conditions that must be true before executing
string[] effects                  # Effects of the behavior on social state
```

## Service Definitions

### SafetyCheck.srv
```
# Request
humanoid_robotics_book/msg/ActionPlan action_plan  # The action plan to validate

---
# Response
bool is_safe                                    # Whether the action plan is safe to execute
string[] safety_issues                          # List of safety issues found
float32 risk_score                              # Overall risk score (0.0 to 1.0, where 0.0 is safe)
string[] mitigation_suggestions                 # Suggestions for making the plan safer
```

## Entity Relationships

### VLA System Data Flow
1. **Speech Processing**: `SpeechCommand` → Language Understanding
2. **Intent Processing**: `Intent` → Action Planning → `ActionPlan`
3. **Perception Integration**: `SceneGraph` → Intent Grounding → Action Planning
4. **Action Execution**: `ActionPlan` → Execution Modules → Status Updates
5. **Social Interaction**: `SocialBehavior` → Behavior Execution → Feedback

### Context Management
- **Conversation Context**: Links related `SpeechCommand`, `Intent`, and `ActionPlan` messages
- **Spatial Context**: Connects `SceneGraph` data with action targets in `ActionPlan`
- **Task Context**: Maintains state across multiple related actions in `ActionPlan`

## Validation Rules

### Message Validation
- `SpeechCommand.confidence` must be between 0.0 and 1.0
- `Intent.confidence` must be between 0.0 and 1.0
- `ActionPlan.estimated_duration` must be positive
- `VLAAction.priority` must be between 0.0 and 1.0
- Timestamps must be current (within system time window)

### State Transitions
- `ActionPlan.status` transitions: planned → executing → (completed | failed)
- `SocialBehavior` execution follows priority-based scheduling
- Safety validation required before action execution
- Context preservation across conversation turns