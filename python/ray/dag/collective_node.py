from weakref import ReferenceType
import ray
from ray.dag import (
    DAGNode,
    ClassMethodNode,
    ClassNode,
)
from ray.dag.constants import (
    PARENT_CLASS_NODE_KEY,
    BIND_INDEX_KEY,
    COLLECTIVE_GROUP_INPUT_NODES_KEY,
    REDUCE_OP_KEY,
    COLLECTIVE_OUTPUT_INPUT_NODE_KEY,
    COLLECTIVE_GROUP_NODE_KEY,
)
from ray.dag.format_utils import get_dag_node_str
from ray.util.annotations import DeveloperAPI
from ray.util.collective import types
from ray.experimental.channel.torch_tensor_type import TorchTensorType

from typing import Any, Dict, List, Union, Tuple, Optional


@DeveloperAPI
class CollectiveGroupNode(DAGNode):
    # [TODO] Comment.
    # [TODO] Pass collective function as a parameter: allreduce, allgather, etc.
    # For now, implement allreduce first.

    def __init__(
        self,
        method_name: str,
        method_args: Tuple[Any],
        method_kwargs: Dict[str, Any],
        method_options: Dict[str, Any],
        other_args_to_resolve: Dict[str, Any],
    ):
        self._bound_args = method_args or []
        self._bound_kwargs = method_kwargs or {}
        self._bound_options = method_options or {}
        self._method_name: str = method_name

        # [TODO] Determine which other args are needed.
        # Parse other_args_to_resolve and assign to variables
        self._parent_class_node: Union[
            ClassNode, ReferenceType["ray._private.actor.ActorHandle"]
        ] = other_args_to_resolve.get(PARENT_CLASS_NODE_KEY)
        # The index/order when bind() is called on this class method
        self._bind_index: Optional[int] = other_args_to_resolve.get(
            BIND_INDEX_KEY, None
        )

        # Get the input nodes for this collective operation and the reduce op.
        # [TODO] Only allreduce is implemented for now.
        # Enable other collective operations in the future.
        self._inp_nodes: Optional[List[ClassMethodNode]] = other_args_to_resolve.get(
            COLLECTIVE_GROUP_INPUT_NODES_KEY, None
        )
        if self._inps is None:
            raise ValueError("CollectiveGroupNode needs at least 1 input node")
        self._reduce_op: types.ReduceOp = other_args_to_resolve.get(
            REDUCE_OP_KEY, types.ReduceOp.SUM
        )
        self._type = TorchTensorType(transport=TorchTensorType.NCCL_ALLREDUCE)

        # The actor creation task dependency is encoded as the first argument,
        # and the ordering dependency as the second, which ensures they are
        # executed prior to this node.
        super().__init__(
            method_args,
            method_kwargs,
            method_options,
            other_args_to_resolve=other_args_to_resolve,
        )

    def _copy_impl(
        self,
        new_args: List[Any],
        new_kwargs: Dict[str, Any],
        new_options: Dict[str, Any],
        new_other_args_to_resolve: Dict[str, Any],
    ):
        return CollectiveGroupNode(
            self._method_name,
            new_args,
            new_kwargs,
            new_options,
            other_args_to_resolve=new_other_args_to_resolve,
        )

    def _execute_impl(self, *args, **kwargs):
        raise NotImplementedError("CollectiveNode is only supported for aDAG")

    def __str__(self) -> str:
        return get_dag_node_str(self, f"{self._method_name}()")

    def get_method_name(self) -> str:
        return self._method_name

    def _get_bind_index(self) -> int:
        return self._bind_index

    def _get_remote_method(self, method_name):
        method_body = getattr(self._parent_class_node, method_name)
        return method_body

    def _get_actor_handle(self) -> Optional["ray.actor.ActorHandle"]:
        if not isinstance(self._parent_class_node, ray.actor.ActorHandle):
            return None
        return self._parent_class_node


@DeveloperAPI
class CollectiveOutputNode:
    # [TODO] Comment.

    def __init__(
        self,
        method_name: str,
        method_args: Tuple[Any],
        method_kwargs: Dict[str, Any],
        method_options: Dict[str, Any],
        other_args_to_resolve: Dict[str, Any],
    ):
        self._bound_args = method_args or []
        self._bound_kwargs = method_kwargs or {}
        self._bound_options = method_options or {}
        self._method_name: str = method_name
        # Parse other_args_to_resolve and assign to variables
        self._parent_class_node: Union[
            ClassNode, ReferenceType["ray._private.actor.ActorHandle"]
        ] = other_args_to_resolve.get(PARENT_CLASS_NODE_KEY)
        # The index/order when bind() is called on this class method
        self._bind_index: Optional[int] = other_args_to_resolve.get(
            BIND_INDEX_KEY, None
        )

        # Get the input node and the collective group node.
        self._inp_node: Optional[ClassMethodNode] = other_args_to_resolve.get(
            COLLECTIVE_OUTPUT_INPUT_NODE_KEY, None
        )
        if self._inp_node is None:
            raise ValueError("CollectiveOutputNode must have an input node")
        self._collective_group_node: Optional[CollectiveGroupNode] = (
            other_args_to_resolve.get(COLLECTIVE_GROUP_NODE_KEY, None)
        )
        if self._collective_group_node is None:
            raise ValueError(
                "CollectiveOutputNode must be associated with a CollectiveGroupNode"
            )

        # The actor creation task dependency is encoded as the first argument,
        # and the ordering dependency as the second, which ensures they are
        # executed prior to this node.
        super().__init__(
            method_args,
            method_kwargs,
            method_options,
            other_args_to_resolve=other_args_to_resolve,
        )

    def _copy_impl(
        self,
        new_args: List[Any],
        new_kwargs: Dict[str, Any],
        new_options: Dict[str, Any],
        new_other_args_to_resolve: Dict[str, Any],
    ):
        return CollectiveGroupNode(
            self._method_name,
            new_args,
            new_kwargs,
            new_options,
            other_args_to_resolve=new_other_args_to_resolve,
        )

    def _execute_impl(self, *args, **kwargs):
        raise NotImplementedError("CollectiveOutputNode is only supported for aDAG")

    def __str__(self) -> str:
        return get_dag_node_str(self, f"{self._method_name}()")

    def get_method_name(self) -> str:
        return self._method_name

    def _get_bind_index(self) -> int:
        return self._bind_index

    def _get_remote_method(self, method_name):
        method_body = getattr(self._parent_class_node, method_name)
        return method_body

    def _get_actor_handle(self) -> Optional["ray.actor.ActorHandle"]:
        if not isinstance(self._parent_class_node, ray.actor.ActorHandle):
            return None
        return self._parent_class_node
