import logging
from types import ModuleType
from typing import TYPE_CHECKING, List, Optional

import ray
from ray.dag import ClassMethodNode
from ray.exceptions import RayChannelError
from ray.util.collective import types

if TYPE_CHECKING:
    import cupy as cp
    import torch


# Logger for this module. It should be configured at the entry point
# into the program using Ray. Ray provides a default configuration at
# entry/init points.
logger = logging.getLogger(__name__)


class AllReduceWrapper:
    # [TODO] Comments for this class.
    def bind(
        self, class_nodes: List[ClassMethodNode], op: types.ReduceOp
    ) -> List[ClassMethodNode]:
        logger.info("Binding all reduce")
        # [TODO] Create CollectiveNode and CollectiveOutputNode.

    def __call__(
        self,
        tensor,
        group_name: str = "default",
        op: types.ReduceOp = types.ReduceOp.SUM,
    ):
        from ray.util.collective.collective import _allreduce

        return _allreduce(tensor, group_name, op)


allreduce = AllReduceWrapper()
