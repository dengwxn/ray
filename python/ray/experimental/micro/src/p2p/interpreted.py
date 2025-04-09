import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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


def benchmark(args):
    actors = [Actor.remote(args["size"]) for _ in range(2)]

    for iter in range(args["num_iters"]):
        start = time.perf_counter()
        init_tensor = ray.get(actors[0].init_tensor.remote(0))
        send_tensor = ray.get(actors[0].send_tensor.remote(init_tensor))
        recv_tensor = ray.get(actors[1].recv_tensor.remote(send_tensor))
        end = time.perf_counter()
        elapse_us = round((end - start) * 1e6)
        logger.info(f"Iteration: {iter}, elapse: {elapse_us} us")


if __name__ == "__main__":
    args = dict()
    args["num_iters"] = 50
    args["size"] = 2**20
    ray.init()
    benchmark(args)
