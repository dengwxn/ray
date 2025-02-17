import logging
from typing import Any

import torch

from ...core.llama3.model import TransformerBP

logger = logging.getLogger(__name__)


class Actor:
    def __init__(self, model_args):
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024
        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

    def forward(self, _) -> None:
        self.intermediates = []
        tokens = self.input_ids
        input, freqs_cis, mask = None, None, None
        for i, bp in enumerate(self.bparams):
            if i == 0:
                pred = bp.forward(tokens)
                freqs_cis, mask = bp.post_hook(tokens, pred)
            elif i < len(self.bparams) - 1:
                pred = bp.forward_transformer(input, 0, freqs_cis, mask)
            else:
                pred = bp.forward(bp.pre_hook(input))
            if i < len(self.bparams) - 1:
                input = pred.detach().requires_grad_(True)
                freqs_cis = freqs_cis.detach().requires_grad_(True)
                mask = mask.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))

    def backward(self, _, idx: int) -> torch.Tensor:
        if idx == len(self.bparams) - 1:
            loss = self.bparams[idx].criterion(
                self.intermediates[idx][0],
                self.target_ids,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
        grads = self.bparams[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        return grads

    def backward_all(self, _) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(_, i)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if grads_passed:
            grads_cat /= self.num_actors
        self.bparams[idx].update(grads_cat, grads_passed)

    def update_all(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.update(_, False, i)


class Actor_V1_5:
    def __init__(self, model_args):
        model_args.n_layers = 1
        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024
        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

    def forward(self, _) -> torch.Tensor:
        # [VERSION] Exclude per-bucket forward.
        logits = self.model.forward_bp_auto(self.input_ids)
        return logits

    def backward_all(self, logits) -> torch.Tensor:
        # [VERSION] Exclude per-bucket backward.
        loss = self.criterion(logits, self.target_ids)
        loss.backward()

    def update_all(self, _) -> None:
        # [VERSION] Exclude optimizer.
        self.optimizer.step()
        self.optimizer.zero_grad()
