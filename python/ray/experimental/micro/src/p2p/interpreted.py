import logging
import time
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

    def init_tensor(self, _):
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

    elapses_us = []
    for iter in range(num_iters):
        start = time.perf_counter()
        init_tensor = ray.get(actors[0].init_tensor.remote(0))
        send_tensor = ray.get(actors[0].send_tensor.remote(init_tensor))
        recv_tensor = ray.get(actors[1].recv_tensor.remote(send_tensor))
        end = time.perf_counter()
        elapse_us = round((end - start) * 1e6)
        elapses_us.append(elapse_us)
        if iter % 10 == 0:
            logger.info(f"Iteration: {iter}, elapse: {elapse_us} us")

    elapses_us = elapses_us[int(len(elapses_us) * 0.2) :]
    elapse_us_avg = round(sum(elapses_us) / len(elapses_us))
    logger.info(f"Elapse avg: {elapse_us_avg} us")


if __name__ == "__main__":
    ray.init()
    fire.Fire(benchmark)
