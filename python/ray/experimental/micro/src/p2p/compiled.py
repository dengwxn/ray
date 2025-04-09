import logging
from typing import Any, Dict, List, Optional, Tuple

import fire
import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allgather, reducescatter

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, size: int):
        self.size = size
        self.device = torch.device("cuda:0")

    def init_tracing(self, _):
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def get_elapse(self, _):
        self.end = torch.cuda.Event(enable_timing=True)
        self.end.record()
        torch.cuda.synchronize()
        elapse_us = round(self.start.elapsed_time(self.end) * 1e3)
        return elapse_us

    def init_tensor(self, _init1, _init2):
        self.tensor = torch.randn(1, self.size, device=self.device)

    def send_tensor(self, _):
        return self.tensor

    def recv_tensor(self, tensor):
        tensor += 1


def benchmark(
    num_iters: int = 50,
    size: int = 2**20,
):
    actors = [Actor.remote(size) for _ in range(2)]

    with InputNode() as inp:
        init_tracings = [actor.init_tracing.bind(inp) for actor in actors]
        init_tensor = actors[0].init_tensor.bind(init_tracings[0], init_tracings[1])
        send_tensor = (
            actors[0].send_tensor.bind(init_tensor).with_tensor_transport("nccl")
        )
        recv_tensor = actors[1].recv_tensor.bind(send_tensor)
        get_elapses = [actor.get_elapse.bind(recv_tensor) for actor in actors]
        dag = MultiOutputNode(get_elapses)

    compiled_dag = dag.experimental_compile()

    elapses_us = []
    for iter in range(num_iters):
        elapse_us = ray.get(compiled_dag.execute(None))
        elapses_us.append(max(elapse_us))
        if iter % 10 == 0:
            logger.info(f"Iteration: {iter}, elapse: {elapse_us} us")

    elapses_us = elapses_us[int(len(elapses_us) * 0.2) :]
    elapse_us_avg = round(sum(elapses_us) / len(elapses_us))
    logger.info(f"Elapse avg: {elapse_us_avg} us")


if __name__ == "__main__":
    ray.init()
    fire.Fire(benchmark)
