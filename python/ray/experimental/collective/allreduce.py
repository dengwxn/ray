import logging
from typing import List, Optional

import ray
from ray.dag.collective_node import CollectiveOutputNode, _CollectiveGroup
from ray.dag.constants import (
    BIND_INDEX_KEY,
    COLLECTIVE_GROUP_KEY,
    PARENT_CLASS_NODE_KEY,
)
from ray.dag.dag_node import DAGNode
from ray.util.collective import types

logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comment.

    def bind(
        self,
        input_nodes: List["DAGNode"],
        op: types.ReduceOp,
        count: Optional[int] = None,
    ) -> List[CollectiveOutputNode]:
        collective_group = _CollectiveGroup(input_nodes, op, count)
        collective_output_nodes: List[CollectiveOutputNode] = []

        for input_node in input_nodes:
            actor_handle: Optional[
                "ray.actor.ActorHandle"
            ] = input_node._get_actor_handle()
            assert actor_handle
            output_node = CollectiveOutputNode(
                method_name="allreduce",  # [TODO] From op.
                method_args=(input_node,),
                method_kwargs=dict(),
                method_options=dict(),
                other_args_to_resolve={
                    PARENT_CLASS_NODE_KEY: actor_handle,
                    BIND_INDEX_KEY: actor_handle._ray_dag_bind_index,
                    COLLECTIVE_GROUP_KEY: collective_group,
                },
            )
            actor_handle._ray_dag_bind_index += 1
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
