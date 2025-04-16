from typing import List, Tuple

import torch
import ray
from ray.dag import InputNode, MultiOutputNode

# COMPUTE_INTENSITY = 10000
COMPUTE_INTENSITY = 1000


@ray.remote
class GPipeWorker:
    def __init__(self, rank: int):
        self.rank = rank

    def get_input(self, size: int) -> torch.Tensor:
        assert self.rank == 0
        return torch.zeros(size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        for _ in range(COMPUTE_INTENSITY):
            t += 1
        return t

    def backward(self, t: torch.Tensor) -> torch.Tensor:
        for _ in range(COMPUTE_INTENSITY):
            t += 1
        return t

    def get_output(
        self, bw_result: torch.Tensor
    ) -> Tuple[float, torch.Size, torch.dtype]:
        assert self.rank == 0
        val = bw_result[0].item()
        assert torch.equal(
            bw_result,
            torch.full(bw_result.shape, fill_value=val, dtype=bw_result.dtype),
        )
        return (val, bw_result.shape, bw_result.dtype)


def generate_afab_schedules(
    num_microbatches: int, num_stages: int
) -> Tuple["ray.dag.CompiledDAG", List["ray.actor.ActorHandle"]]:
    """
    Generate a DAG for GPipe-style pipeline parallelism.

    Args:
        num_microbatches (int): The number of microbatches.
        num_stages (int): The number of stages.

    Returns:
        The compiled DAG and the list of GPipe workers.
    """
    actor_cls = GPipeWorker.options(num_cpus=1)
    workers = []
    for rank in range(num_stages):
        worker = actor_cls.remote(rank)
        workers.append(worker)

    rank0 = workers[0]

    with InputNode() as inp:
        batches = []
        for _ in range(num_microbatches):
            batch = rank0.get_input.bind(inp)
            for worker in workers:
                batch = worker.forward.bind(batch)
            batches.append(batch)

        for batch_idx in reversed(range(num_microbatches)):
            batch = batches[batch_idx]
            for worker in reversed(workers):
                batch = worker.backward.bind(batch)
            batch = rank0.get_output.bind(batch)
            batches[batch_idx] = batch

        dag = MultiOutputNode(batches)

    compiled_dag = dag.experimental_compile()

    return compiled_dag, workers
