# SocialBehavior.msg
# Message for social behaviors in VLA system

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