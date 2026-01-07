# Intent.msg
# Message for structured intent representation in VLA system

std_msgs/Header header
string command_type      # Type: "navigation", "manipulation", "interaction", "social"
string[] entities        # Identified entities (objects, locations, people)
geometry_msgs/Pose[] entity_poses  # 3D poses of entities if available
builtin_interfaces/Time timestamp  # When the intent was processed
string context_id        # Context identifier for conversation management
string raw_command       # Original command text
float32 confidence      # Confidence in intent interpretation (0.0 to 1.0)
string[] parameters     # Additional parameters for the command