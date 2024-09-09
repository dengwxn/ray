import logging
from typing import List

from ray.dag.constants import COLLECTIVE_GROUP_KEY, COLLECTIVE_OUTPUT_INPUT_NODE_KEY
from ray.dag.dag_node import DAGNode
from ray.dag.collective_node import _CollectiveGroup, CollectiveOutputNode
from ray.util.collective import types

logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comments for this class.

    # [TODO] Change to DAGNode.
    def bind(
        self, input_nodes: List["DAGNode"], op: types.ReduceOp
    ) -> List[CollectiveOutputNode]:
        # Rename `CollectiveGroupNode` to private `_CollectiveGroup` for now.
        collective_group = _CollectiveGroup(input_nodes, op)

        collective_output_nodes: List[CollectiveOutputNode] = []
        for input_node in input_nodes:
            output_node = CollectiveOutputNode(
                method_name="__CollectiveOutputNode__",
                method_args=tuple(input_node),
                # [TODO] Check upstream and downstream are correct.
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve={
                    COLLECTIVE_OUTPUT_INPUT_NODE_KEY: input_node,
                    COLLECTIVE_GROUP_KEY: collective_group,
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
