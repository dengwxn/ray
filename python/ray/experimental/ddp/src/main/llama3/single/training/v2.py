import os
import time

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .....core.llama3.actor import Actor
from .....core.llama3.model import LLAMA_1B, LLAMA_3B, LLAMA_8B, TransformerBP


def main():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25670"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        model_parallel_size = 1
        initialize_model_parallel(model_parallel_size)

    local_rank = 0
    torch.cuda.set_device(local_rank)
    torch.manual_seed(998244353)

    model_args = LLAMA_1B
    actor = Actor(model_args)
    print(actor.model)

    n_epochs = 3

    for _ in range(n_epochs):
        actor.init_training()

        fw_start = time.perf_counter()
        actor.forward(None)
        fw_end = time.perf_counter()
        print(f"Forward time: {round((fw_end - fw_start) * 1e3)} ms")

        bw_start = time.perf_counter()
        actor.backward_all(None)
        bw_end = time.perf_counter()
        print(f"Backward time: {round((bw_end - bw_start) * 1e3)} ms")

        upd_start = time.perf_counter()
        actor.update_all(None)
        upd_end = time.perf_counter()
        print(f"Update time: {round((upd_end - upd_start) * 1e3)} ms")

        print()


if __name__ == "__main__":
    main()
