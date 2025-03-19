import logging
import os
from typing import Any, Dict

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

    def train(self, args: Dict[str, Any]) -> None:
        model_args = LLAMA
        batch_size = 1
        seq_len = 1024

        model = Transformer(model_args).to(self.device)

        torch.manual_seed(998244353)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        torch.cuda.synchronize()

        for _ in range(2):
            input = torch.randint(
                0,
                model_args.vocab_size,
                (batch_size, seq_len),
                device=self.device,
            )
            target = torch.randn(
                batch_size,
                seq_len,
                model_args.vocab_size,
                requires_grad=True,
                device=self.device,
            )

            logits = model.forward(input, 0)

            loss = criterion(logits, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    def clean(self):
        dist.destroy_process_group()


def online(args) -> None:
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
    ray.get([actor.train.remote(LLAMA) for actor in actors])
    ray.get([actor.clean.remote() for actor in actors])

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    online(args)
