import logging
from typing import Any, Dict

import torch

from ....core.config import parse_args
from ....core.llama3.model import LLAMA_DEBUG as LLAMA
from ....core.llama3.model import TransformerPPV3 as Transformer

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

    model1 = Transformer(model_args, 0).to(device)
    model2 = Transformer(model_args, 0).to(device)

    torch.manual_seed(998244353)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-6)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-6)

    torch.cuda.synchronize()

    def display(title, model):
        n_require_grad = 0
        n_grad_not_none = 0
        for p in model.parameters():
            if p.requires_grad:
                n_require_grad += 1
            if p.grad is not None:
                n_grad_not_none += 1
        logger.warning(
            f"{title}, n_require_grad: {n_require_grad}, n_grad_not_none: {n_grad_not_none}"
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

        logits_1 = model1.forward_first(input, 0)
        display("model1", model1)

        logits_as_input = logits_1.detach().requires_grad_(True)
        logits_2 = model2.forward_second(input, 0, logits_as_input)
        display("model2", model2)

        loss = criterion(logits_2, target)
        loss.backward()
        display("model2", model2)

        assert logits_as_input.grad is not None
        logits_1.backward(logits_as_input.grad)
        display("model1", model1)

        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()


if __name__ == "__main__":
    args = parse_args()
    main(args)
