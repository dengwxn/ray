import os

import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import ray
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode


@ray.remote(num_gpus=1)
class AllReduceWorker:
    def __init__(self):
        self.device = torch_utils.get_devices()[0]

    def get_tensor(self, unused):
        print(f"Ray allreduce actor start time: {time.perf_counter()}")
        return torch.ones(10, device=self.device) * 10


def run_ray_allreduce():
    workers = [AllReduceWorker.remote() for _ in range(2)]
    with InputNode() as inp:
        tensors = [worker.get_tensor.bind(inp) for worker in workers]
        for _ in range(10):
            tensors = allreduce.bind(tensors)
        dag = MultiOutputNode(tensors)

    compiled_dag = dag.experimental_compile()
    for i in range(10):
        start = time.perf_counter()
        ref = compiled_dag.execute(0)
        result = ray.get(ref)
        end = time.perf_counter()
        print(
            f"iteration {i} end-to-end time: {round((end - start) * 1e6)} us ({start, end})"
        )
    print(result)


def run_torch_allreduce():
    mp.spawn(run_torch_allreduce_per_process, nprocs=2)


def run_torch_allreduce_per_process(rank: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    dist.init_process_group("nccl", rank=rank, world_size=2)

    for it in range(10):
        tensor = torch.ones(10, device=f"cuda:{rank}") * 10
        times = []
        for _ in range(10):
            start = time.perf_counter()
            dist.all_reduce(tensor)
            end = time.perf_counter()
            times.append((start, end))
        print(tensor)
        elapses = [end - start for start, end in times]
        for i, elapse in enumerate(elapses):
            print(f"iteration {it} torch allreduce {i}: {round(elapse * 1e6)} us")

    dist.destroy_process_group()


def main():
    run_ray_allreduce()
    run_torch_allreduce()


if __name__ == "__main__":
    main()
