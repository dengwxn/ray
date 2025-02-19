import logging

import torch

import ray
from ...core.llama3.model import TransformerBP

logger = logging.getLogger(__name__)


@ray.remote
class LlamaActor:
    def __init__(self, model_args, rank: int, num_partitions: int, num_actors: int):
        torch.cuda.set_device(0)
        torch.manual_seed(998244353)

        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        self.rank = rank
        assert len(self.bparams) == num_partitions
        self.num_actors = num_actors

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
            else:
                input = pred
            self.intermediates.append((pred, input))
        return pred

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

    def backward_aio(self, _) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(_, i)

    def copy(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if grads_passed:
            grads_cat /= self.num_actors
        self.bparams[idx].copy(grads_cat, grads_passed)

    def copy_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.copy(_, False, i)

    def step(self, idx: int) -> None:
        self.bparams[idx].step()

    def step_aio(self, _) -> None:
        # [NOTE] It is slower to use a single optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for i in reversed(range(len(self.bparams))):
            self.step(i)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        self.copy(grads_cat, grads_passed, idx)
        self.bparams[idx].step()

    def update_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.update(_, False, i)


class _Actor_V4:
    def __init__(self, model_args):
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
            else:
                input = pred
            self.intermediates.append((pred, input))
        return pred

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

    def backward_aio(self, _) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(_, i)

    def copy(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if grads_passed:
            grads_cat /= self.num_actors
        self.bparams[idx].copy(grads_cat, grads_passed)

    def copy_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.copy(_, False, i)

    def step(self, idx: int) -> None:
        self.bparams[idx].step()

    def step_aio(self, _) -> None:
        # [NOTE] It is slower to use a single optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for i in reversed(range(len(self.bparams))):
            self.step(i)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        self.copy(grads_cat, grads_passed, idx)
        self.bparams[idx].step()

    def update_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.update(_, False, i)


class _Actor_V2:
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
            else:
                input = pred
            self.intermediates.append((pred, input))
        return pred

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

    def backward_aio(self, logits) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(None, i)

    def update_aio(self, _) -> None:
        # [VERSION] Exclude optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for bparam in self.bparams:
            bparam.step(None, False)
