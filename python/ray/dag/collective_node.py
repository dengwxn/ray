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
    COLLECTIVE_OUTPUT_INPUT_NODE_KEY,
    COLLECTIVE_GROUP_KEY,
)
from ray.dag.format_utils import get_dag_node_str
from ray.util.annotations import DeveloperAPI
from ray.util.collective import types
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.experimental.channel.nccl_group import _NcclGroup
from ray.experimental.channel.torch_tensor_nccl_channel import _init_nccl_group

from typing import Any, Dict, List, Union, Tuple, Optional


@DeveloperAPI
class _CollectiveGroup:
    # [TODO] Comment.
    # [TODO] Pass collective function as a parameter: allreduce, allgather, etc.
    # For now, implement allreduce first.

    def __init__(
        self,
        input_nodes: List[DAGNode],
        reduce_op: types.ReduceOp,
        count: Optional[int] = None,
    ):
        # Get the nodes for this collective operation and the reduce op.
        # [TODO] Only allreduce is implemented for now.
        # Enable other collective operations in the future.
        self._nodes: List[Tuple[DAGNode, Optional[CollectiveOutputNode]]] = [
            (input_node, None) for input_node in input_nodes
        ]
        if len(self._nodes) == 0:
            raise ValueError("CollectiveGroup needs at least 1 input node")
        self._reduce_op = reduce_op
        if count is not None:
            self._type = TorchTensorType(
                _shape=count,
                transport=TorchTensorType.NCCL_ALLREDUCE,
                _direct_return=True,
            )
        else:
            self._type = TorchTensorType(
                transport=TorchTensorType.NCCL_ALLREDUCE,
                _direct_return=True,
            )
        self._nccl_group_id: Optional["_NcclGroup"] = None

    def _copy_impl(
        self,
        input_nodes: List[DAGNode],
        reduce_op: types.ReduceOp,
    ):
        return _CollectiveGroup(input_nodes, reduce_op)

    def _execute_impl(self, *args, **kwargs):
        raise NotImplementedError("CollectiveNode is only supported for aDAG")

    def __str__(self) -> str:
        return f"CollectiveGroup(inputs={self._nodes}, op={self._reduce_op}"

    def _set_output_node(self, idx: int, output_node: "CollectiveOutputNode") -> None:
        input_node, prev_output = self._nodes[idx]
        if prev_output is not None:
            raise ValueError(
                f"CollectiveOutputNode at index {idx} is already set for this group."
            )
        self._nodes[idx] = (input_node, output_node)

    def _init_nccl_collective_group(self) -> None:
        if self._nccl_group_id is not None:
            return

        actor_handles: List[Optional["ray.actor.ActorHandle"]] = [
            output._get_actor_handle() for (_, output) in self._nodes
        ]
        if None in actor_handles:
            raise ValueError("Each AllReduce participant needs an actor handle.")

        self._nccl_group_id = _init_nccl_group(actor_handles)
        self._type.set_nccl_group_id(self._nccl_group_id)


@DeveloperAPI
class CollectiveOutputNode(DAGNode):
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
        self._parent_class_node: Optional[
            Union[ClassNode, ReferenceType["ray._private.actor.ActorHandle"]]
        ] = other_args_to_resolve.get(PARENT_CLASS_NODE_KEY, None)
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
        if self._parent_class_node is None:
            self._parent_class_node = self._inp_node
        self._collective_group: Optional[_CollectiveGroup] = other_args_to_resolve.get(
            COLLECTIVE_GROUP_KEY, None
        )
        if self._collective_group is None:
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
        return _CollectiveGroup(
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
        if not isinstance(
            self._parent_class_node, (ray.actor.ActorHandle, ClassMethodNode)
        ):
            return None
        if isinstance(self._parent_class_node, ray.actor.ActorHandle):
            return self._parent_class_node
        return self._parent_class_node._get_actor_handle()
