import logging
from typing import Any, Dict, List, Optional, Tuple

import fire
import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def get_timing_event() -> torch.cuda.Event:
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def get_elapse_us(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
    elapse_us = round(start.elapsed_time(end) * 1e3)
    return elapse_us


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, size_comp: int, num_comp: int, size_comm: int):
        self.device = torch.device("cuda:0")
        self.size_comp = size_comp
        self.num_comp = num_comp
        self.size_comm = size_comm

    def init_tensor(self, _):
        self.tensor_comp = torch.randn(1, self.size_comp, device=self.device)
        self.tensor_comm = torch.randn(1, self.size_comm, device=self.device)
        torch.cuda.synchronize()

    def init_tracing(self, _):
        self.ev_e2e_start = get_timing_event()

    def comp_tensor(self, _):
        self.ev_comp_tensor_start = get_timing_event()
        for _ in range(self.num_comp):
            self.tensor_comp += 1
        self.ev_comp_tensor_end = get_timing_event()
        return self.tensor_comp

    def comm_tensor(self, _):
        return self.tensor_comm

    def finish_tracing(self, _comp_tensor, _comm_tensor):
        self.ev_e2e_end = get_timing_event()
        torch.cuda.synchronize()
        elapses = dict()
        elapses["e2e"] = get_elapse_us(self.ev_e2e_start, self.ev_e2e_end)
        elapses["comp"] = get_elapse_us(
            self.ev_comp_tensor_start, self.ev_comp_tensor_end
        )
        return elapses


def benchmark(
    overlap: bool = False,
    num_iters: int = 50,
    size_comp: int = 1_00_000,
    num_comp: int = 12_000,
    size_comm: int = 1_000_000_000,
):
    num_iters_warmup = int(num_iters * 0.2)
    actors = [Actor.remote(size_comp, num_comp, size_comm) for _ in range(2)]

    with InputNode() as inp:
        init_tensors = [actor.init_tensor.bind(inp) for actor in actors]
        init_tracings = [
            actor.init_tracing.bind(init_tensor)
            for actor, init_tensor in zip(actors, init_tensors)
        ]
        comm_tensors = [
            actor.comm_tensor.bind(init_tracing)
            for actor, init_tracing in zip(actors, init_tracings)
        ]
        comp_tensors = [
            actor.comp_tensor.bind(init_tracing)
            for actor, init_tracing in zip(actors, init_tracings)
        ]
        ar_tensors = allreduce.bind(comm_tensors)
        elapses = [
            actor.finish_tracing.bind(comp_tensor, ar_tensor)
            for actor, comp_tensor, ar_tensor in zip(actors, comp_tensors, ar_tensors)
        ]
        dag = MultiOutputNode(elapses)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=overlap)

    # elapses_us = []
    for iter in range(num_iters):
        elapses = ray.get(compiled_dag.execute(None))
        if iter > num_iters_warmup and iter % 10 == 0:
            logger.info(f"Iteration: {iter}, elapses: {elapses}")
        # elapses_us.append(max(elapse_us))

    compiled_dag.teardown()

    # elapses_us = elapses_us[int(len(elapses_us) * 0.2) :]
    # elapse_us_avg = round(sum(elapses_us) / len(elapses_us))
    # logger.info(f"Elapse avg: {elapse_us_avg} us")


if __name__ == "__main__":
    ray.init()
    fire.Fire(benchmark)
