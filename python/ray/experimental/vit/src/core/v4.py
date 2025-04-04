import os
import logging
import fire
import torch
import torch.multiprocessing as mp

from actor import WorkerV4 as Worker
from dist import TorchDistributedConfig


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


# V4: Torch (no Ray)

# 4xH100
# WORLD_SIZE = 4
# CUDA_VISIBLE_DEVICES = "0,1,2,3"

# WORLD_SIZE = 3
# CUDA_VISIBLE_DEVICES = "1,2,3"

WORLD_SIZE = 2
CUDA_VISIBLE_DEVICES = "2,3"


def train(
    rank: int,
    model_name: str,
    bs_single: int,
    num_dp_vision: int,
    num_dp: int,
    num_iters: int,
) -> None:
    bs_global = bs_single * num_dp_vision

    torch.cuda.set_device(rank)

    actor = Worker(model_name, rank, num_dp)
    local_rank = rank
    world_size = num_dp
    local_world_size = world_size
    master_addr = "127.0.0.1"
    master_port = "8888"
    gpu_ids = [int(gpu_id) for gpu_id in CUDA_VISIBLE_DEVICES.split(",")]
    worker_config = TorchDistributedConfig(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        master_addr=master_addr,
        master_port=master_port,
        gpu_ids=gpu_ids,
    )
    actor.init_torch_distributed(worker_config)

    actor.init_fsdp_model()

    for i in range(num_iters):
        actor.init_training()
        actor.forward((i, bs_global))
        actor.backward()
        if rank == 0:
            logger.info(f"Iteration {i} finished")
        actor.finish_tracing()


def main(
    # model_name: str = "ViT-L-14",
    model_name: str = "ViT-bigG-14",
    bs_single: int = 16,
    num_dp_vision: int = 3,
    num_dp: int = 4,
    num_iters: int = 50,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    assert num_dp <= WORLD_SIZE
    mp.spawn(
        train,
        args=(model_name, bs_single, num_dp_vision, num_dp, num_iters),
        nprocs=WORLD_SIZE,
    )


if __name__ == "__main__":
    fire.Fire(main)
