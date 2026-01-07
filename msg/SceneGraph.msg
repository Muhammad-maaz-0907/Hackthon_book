# SceneGraph.msg
# Message for semantic scene representation in VLA system

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