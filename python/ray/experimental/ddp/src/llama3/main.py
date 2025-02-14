import os

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .generation import Transformer
from .model import LLAMA_1B


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
    model = Transformer(model_args)
    print(model)


if __name__ == "__main__":
    main()
