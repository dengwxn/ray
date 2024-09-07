import logging
from typing import List

from ray.dag import (
    COLLECTIVE_GROUP_INPUT_NODES_KEY,
    COLLECTIVE_GROUP_NODE_KEY,
    COLLECTIVE_OUTPUT_INPUT_NODE_KEY,
    REDUCE_OP_KEY,
    ClassMethodNode,
    CollectiveGroupNode,
    CollectiveOutputNode,
)
from ray.util.collective import types

logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comments for this class.

    def bind(
        self, class_nodes: List["ClassMethodNode"], op: types.ReduceOp
    ) -> List[CollectiveOutputNode]:
        collective_group_node = CollectiveGroupNode(
            method_name="collective_group",
            method_args=tuple(),
            method_kwargs=dict(),
            method_options=dict(),
            other_args_to_resolve={
                COLLECTIVE_GROUP_INPUT_NODES_KEY: class_nodes,
                REDUCE_OP_KEY: op,
            },
        )

        collective_output_nodes: List[CollectiveOutputNode] = []
        for class_node in class_nodes:
            output_node = CollectiveOutputNode(
                method_name="collective_output",
                method_args=tuple(),
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve={
                    COLLECTIVE_OUTPUT_INPUT_NODE_KEY: class_node,
                    COLLECTIVE_GROUP_NODE_KEY: collective_group_node,
                },
            )
            collective_output_nodes.append(output_node)

        return collective_output_nodes

    def __call__(
        self,
        tensor,
        group_name: str = "default",
        op: types.ReduceOp = types.ReduceOp.SUM,
    ):
        from ray.util.collective.collective import allreduce

        return allreduce(tensor, group_name, op)


allreduce = AllReduceWrapper()
