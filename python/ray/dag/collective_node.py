import ray
from ray.dag.dag_node import DAGNode
from ray.dag.class_node import ClassMethodNode
from ray.dag.input_node import InputNode
from ray.dag.format_utils import get_dag_node_str
from ray.dag.constants import (
    PARENT_CLASS_NODE_KEY,
    PREV_CLASS_METHOD_CALL_KEY,
    BIND_INDEX_KEY,
    IS_CLASS_METHOD_OUTPUT_KEY,
)
from ray.util.annotations import DeveloperAPI

from typing import Any, Dict, List, Union, Tuple, Optional

# [TODO] Collective node.

class CollectiveNode(DAGNode):
    class_nodes: List[ClassMethodNode]


class CollectiveOutputNode:
    collective_node: CollectiveNode
    class_node: ClassMethodNode


# [TODO] Add typehint to nodes. Might need to add a new typehint for torch tensors and collectives.
