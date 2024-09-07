import os

# Reserved keys used to handle ClassMethodNode in Ray DAG building.
PARENT_CLASS_NODE_KEY = "parent_class_node"
PREV_CLASS_METHOD_CALL_KEY = "prev_class_method_call"
BIND_INDEX_KEY = "bind_index"
IS_CLASS_METHOD_OUTPUT_KEY = "is_class_method_output"

# Reserved keys used to handle CollectiveGroupNode in Ray DAG building.
COLLECTIVE_GROUP_INPUT_NODES_KEY = "collective_group_input_nodes"
REDUCE_OP_KEY = "reduce_op"
# Reserved keys used to handle CollectiveOutputNode in Ray DAG building.
COLLECTIVE_OUTPUT_INPUT_NODE_KEY = "collective_output_input_node"
COLLECTIVE_GROUP_NODE_KEY = "collective_group_node"

# Reserved key to distinguish DAGNode type and avoid collision with user dict.
DAGNODE_TYPE_KEY = "__dag_node_type__"

# Feature flag to turn off the deadlock detection.
RAY_ADAG_ENABLE_DETECT_DEADLOCK = (
    os.environ.get("RAY_ADAG_ENABLE_DETECT_DEADLOCK", "1") == "1"
)
