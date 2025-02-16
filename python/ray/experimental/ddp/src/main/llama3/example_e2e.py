import os

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from ...core.llama3.generation import Transformer
from ...core.llama3.model import LLAMA_1B, LLAMA_3B, LLAMA_8B


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
    model = Transformer(model_args).to("cuda")
    print(model)

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, seq_len),
        device="cuda",
    )
    print(input_ids)

    target_ids = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, seq_len),
        device="cuda",
    )
    print(target_ids)

    logits = model.forward(input_ids, 0)  # shape: (batch_size, seq_len, vocab_size)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    print(loss)


if __name__ == "__main__":
    main()
