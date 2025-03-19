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
from ray.dag import InputNode, MultiOutputNode

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


@ray.remote
class Actor:
    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_actors_tp: int,
        master_addr: str,
        master_port: int,
        tracing: bool,
    ):
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.tracing = tracing

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(num_actors_tp)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl")

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Actor {rank}, device: {self.device}")

        model_parallel_size = 2
        fs_init.initialize_model_parallel(model_parallel_size)

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

    def forward(self, _):
        return self.model.forward(self.input, 0)

    def backward(self, logits):
        loss = self.criterion(logits, self.target)
        loss.backward()

    def update(self, _):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def clean(self):
        dist.destroy_process_group()


def init_actors(args: Dict[str, Any]) -> List[Actor]:
    ray.init()
    model_args = LLAMA
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    num_actors_tp = 2
    master_addr = "127.0.0.1"
    master_port = 12345
    tracing = args["tracing"]

    actor_cls = Actor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=i,
            num_actors_tp=num_actors_tp,
            master_addr=master_addr,
            master_port=master_port,
            tracing=tracing,
        )
        for i in range(num_actors_tp)
    ]

    return actors


def train(actors: List[Actor]) -> None:
    with InputNode() as inp:
        forwards = [actor.forward.bind(inp) for actor in actors]
        backwards = [
            actor.backward.bind(forward) for actor, forward in zip(actors, forwards)
        ]
        updates = [
            actor.update.bind(backward) for actor, backward in zip(actors, backwards)
        ]
        dag = MultiOutputNode(updates)
    compiled_dag = dag.experimental_compile()

    for _ in range(2):
        ray.get([actor.init_training.remote() for actor in actors])
        compiled_dag.execute(None)

    ray.get([actor.clean.remote() for actor in actors])


def clean(actors):
    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    actors = init_actors(args)
    train(actors)
    clean(actors)
