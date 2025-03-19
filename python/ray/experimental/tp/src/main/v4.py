import logging
import os
from typing import Any, Dict, List

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist

import ray
from ..core.config import parse_args
from ..core.model import LLAMA_DEBUG as LLAMA
from ..core.model import TransformerTP as Transformer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


@ray.remote
class Actor:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl")

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Actor {rank}, device: {self.device}")

        model_parallel_size = 2
        fs_init.initialize_model_parallel(model_parallel_size)

        self.model_args = LLAMA
        self.batch_size = 1
        self.seq_len = 1024

        self.model = Transformer(self.model_args).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def init_training(self):
        torch.manual_seed(998244353)
        self.input = torch.randint(
            0,
            self.model_args.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )
        self.target = torch.randn(
            self.batch_size,
            self.seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
            device=self.device,
        )
        torch.cuda.synchronize()

    def forward(self):
        return self.model.forward(self.input, 0)

    def backward(self, logits):
        loss = self.criterion(logits, self.target)
        loss.backward()

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def clean(self):
        dist.destroy_process_group()


def init() -> List[Actor]:
    ray.init()
    actor_cls = Actor.options(num_gpus=1)
    world_size = 2
    master_addr = "127.0.0.1"
    master_port = 12345
    actors = [
        actor_cls.remote(
            rank=i,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
        )
        for i in range(world_size)
    ]
    return actors


def main(args, actors) -> None:
    for _ in range(2):
        ray.get([actor.init_training.remote() for actor in actors])
        actor_to_logits = ray.get([actor.forward.remote() for actor in actors])
        ray.get(
            [
                actor.backward.remote(logits)
                for actor, logits in zip(actors, actor_to_logits)
            ]
        )
        ray.get([actor.update.remote() for actor in actors])
    ray.get([actor.clean.remote() for actor in actors])


def clean(actors):
    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    actors = init()
    main(args, actors)
    clean(actors)
