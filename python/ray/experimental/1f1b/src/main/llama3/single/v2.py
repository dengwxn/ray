import logging
from typing import Any, Dict, List

import torch

from ....core.config import parse_args
from ....core.llama3.model import LLAMA_DEBUG as LLAMA
from ....core.llama3.model import Transformer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def main(args: Dict[str, Any]) -> None:
    model_args = LLAMA
    batch_size = 1
    seq_len = 1024
    device = torch.device("cuda:0")

    model = Transformer(model_args).to(device)
    logger.info(f"model: {model}")

    torch.manual_seed(998244353)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    torch.cuda.synchronize()

    def display(model):
        n_require_grad = 0
        n_grad_not_none = 0
        for p in model.parameters():
            if p.requires_grad:
                n_require_grad += 1
            if p.grad is not None:
                n_grad_not_none += 1
        logger.warning(
            f"n_require_grad: {n_require_grad}, n_grad_not_none: {n_grad_not_none}"
        )

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

        logits = model(input, 0)
        display(model)

        loss = criterion(logits, target)
        loss.backward()
        display(model)

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    args = parse_args()
    main(args)
