import logging
from typing import List, Optional

from ray.dag.collective_node import CollectiveOutputNode, _CollectiveGroup
from ray.dag.constants import COLLECTIVE_GROUP_KEY, COLLECTIVE_OUTPUT_INPUT_NODE_KEY
from ray.dag.dag_node import DAGNode
from ray.util.collective import types

logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comments for this class.

    # [TODO] Change to DAGNode.
    def bind(
        self,
        input_nodes: List["DAGNode"],
        op: types.ReduceOp,
        count: Optional[int] = None,
    ) -> List[CollectiveOutputNode]:
        # Rename `CollectiveGroupNode` to private `_CollectiveGroup` for now.
        collective_group = _CollectiveGroup(input_nodes, op, count)

        collective_output_nodes: List[CollectiveOutputNode] = []
        for i, input_node in enumerate(input_nodes):
            output_node = CollectiveOutputNode(
                method_name="__CollectiveOutputNode__",
                method_args=(input_node,),  # [TODO:andy] Can it be empty tuple?
                # [TODO] Check upstream and downstream are correct.
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve={
                    COLLECTIVE_OUTPUT_INPUT_NODE_KEY: input_node,
                    COLLECTIVE_GROUP_KEY: collective_group,
                },
            )
            collective_group._set_output_node(i, output_node)
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
