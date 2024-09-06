import logging
from typing import List

from ray.dag import (
    ClassMethodNode,
    CollectiveNode,
    CollectiveOutputNode,
)
from ray.util.collective import types


logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comments for this class.

    def bind(
        self, class_nodes: List[ClassMethodNode], op: types.ReduceOp
    ) -> List[ClassMethodNode]:
        collective_node = CollectiveNode(
            method_name="collective",
            method_args=tuple(class_nodes, op),
            method_kwargs=dict(),
            method_options=dict(),
            other_args_to_resolve=dict(),
        )
        output_nodes: List[CollectiveOutputNode] = []
        for class_node in class_nodes:
            output_node = CollectiveOutputNode(
                method_name="collective_output",
                method_args=tuple(class_node, collective_node),
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve=dict(),
            )
            output_nodes.append(output_node)
        return output_nodes

    def __call__(
        self,
        tensor,
        group_name: str = "default",
        op: types.ReduceOp = types.ReduceOp.SUM,
    ):
        from ray.util.collective.collective import _allreduce

        return _allreduce(tensor, group_name, op)


allreduce = AllReduceWrapper()
