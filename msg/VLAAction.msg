# VLAAction.msg
# Message for individual actions in VLA system

string action_type             # Type: "navigate_to", "grasp_object", "follow_person", "wave", "pick_up", "place", "open_gripper", "close_gripper", "speak", "listen", "detect_object"
string[] parameters            # Action-specific parameters
geometry_msgs/Pose target_pose # Target pose for the action if applicable
float32 priority              # Priority of the action (0.0 to 1.0)
bool is_optional              # Whether the action is optional
builtin_interfaces/Duration timeout  # Timeout for the action completion
string[] preconditions        # Conditions that must be true before executing
string[] effects              # Effects of the action on the world state