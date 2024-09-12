from weakref import ReferenceType
import ray
from ray.dag import (
    DAGNode,
    ClassNode,
)
from ray.dag.constants import (
    PARENT_CLASS_NODE_KEY,
    BIND_INDEX_KEY,
    COLLECTIVE_GROUP_KEY,
)
from ray.dag.format_utils import get_dag_node_str
from ray.util.annotations import DeveloperAPI
from ray.util.collective import types
from ray.experimental.channel.torch_tensor_type import TorchTensorType
from ray.experimental.channel.torch_tensor_nccl_channel import _init_nccl_group

from typing import Any, Dict, List, Union, Tuple, Optional


class _CollectiveGroup:
    # [TODO] Comment.

    def __init__(
        self,
        input_nodes: List[DAGNode],
        op: types.ReduceOp,  # [TODO] General collective ops.
        count: Optional[int] = None,
    ):
        self._input_nodes: List[DAGNode] = input_nodes
        if len(self._input_nodes) == 0:
            raise ValueError("CollectiveGroup needs at least 1 input node")
        self._op = op
        if count is not None:
            self._type = TorchTensorType(
                _shape=count,
                transport="nccl",
                _direct_return=True,
            )
        else:
            self._type = TorchTensorType(
                transport="nccl",
                _direct_return=True,
            )
        self._nccl_group_id: Optional[str] = None

    def _copy_impl(
        self,
        input_nodes: List[DAGNode],
        reduce_op: types.ReduceOp,
    ):
        return _CollectiveGroup(input_nodes, reduce_op)

    def _execute_impl(self, *args, **kwargs):
        raise NotImplementedError("CollectiveNode is only supported for aDAG")

    def __str__(self) -> str:
        # [TODO] String representation.
        return f"CollectiveGroup(input_nodes={self._input_nodes}, op={self._op}, type={self._type}, nccl_group_id={self._nccl_group_id})"

    def _init_nccl_group(self) -> None:
        if self._nccl_group_id is not None:
            # If the NCCL group ID is already set, no need to reinitialize.
            return

        actor_handles: List[Optional["ray.actor.ActorHandle"]] = [
            input_node._get_actor_handle() for input_node in self._input_nodes
        ]
        if None in actor_handles:
            raise ValueError("Got an invalid actor handle from input nodes")

        self._nccl_group_id = _init_nccl_group(actor_handles)
        self._type.set_nccl_group_id(self._nccl_group_id)


@DeveloperAPI
class CollectiveOutputNode(DAGNode):
    # [TODO] Comment.

    def __init__(
        self,
        method_name: str,
        method_args: Tuple[
            DAGNode,
        ],
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
        assert isinstance(method_args, tuple) and isinstance(
            method_args[0], DAGNode
        ), "Expected a single input node"
        self._input_node = method_args[0]
        self._collective_group: _CollectiveGroup = other_args_to_resolve.get(
            COLLECTIVE_GROUP_KEY, None
        )
        if self._collective_group is None:
            raise ValueError("Expected a collective group")

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
        if not isinstance(self._parent_class_node, ray.actor.ActorHandle):
            return None
        return self._parent_class_node

    @property
    def collective_group(self) -> _CollectiveGroup:
        return self._collective_group
