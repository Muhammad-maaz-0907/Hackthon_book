# ActionPlan.msg
# Message for structured action planning in VLA system

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