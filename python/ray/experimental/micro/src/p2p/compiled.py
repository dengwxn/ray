import logging
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

    with InputNode() as inp:
        init_tensor = actors[0].init_tensor.bind(inp)
        send_tensor = (
            actors[0].send_tensor.bind(init_tensor).with_tensor_transport("nccl")
        )
        recv_tensor = actors[1].recv_tensor.bind(send_tensor)
        dag = recv_tensor

    compiled_dag = dag.experimental_compile()

    for iter in range(args["num_iters"]):
        compiled_dag.execute(None)
        if iter % 10 == 0:
            logger.info(f"Iteration: {iter}")


if __name__ == "__main__":
    args = dict()
    args["num_iters"] = 50
    args["size"] = 2**20
    ray.init()
    benchmark(args)
