import logging
from typing import Any, Dict

import fairscale.nn.model_parallel.initialize as fs_init
import torch

from ..core.config import parse_args
from ..core.model import LLAMA_DEBUG as LLAMA
from ..core.model import TransformerTP as Transformer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init() -> None:
    torch.distributed.init_process_group(backend="nccl")
    model_parallel_size = 1
    fs_init.initialize_model_parallel(model_parallel_size)
    model_parallel_rank = fs_init.get_model_parallel_rank()
    model_parallel_group = fs_init.get_model_parallel_group()


def main(args: Dict[str, Any]) -> None:
    model_args = LLAMA
    batch_size = 1
    seq_len = 1024

    device = torch.device("cuda:0")
    model = Transformer(model_args).to(device)

    torch.manual_seed(998244353)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    torch.cuda.synchronize()

    for _ in range(2):
        input = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
            device=device,
        )
        target = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
            requires_grad=True,
            device=device,
        )

        logits = model.forward(input, 0)

        loss = criterion(logits, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    args = parse_args()
    init()
    main(args)
