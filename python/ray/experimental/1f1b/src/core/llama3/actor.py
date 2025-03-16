import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

import ray
from ..common import millis_to_micros
from .model import Shard, TransformerBP

logger = logging.getLogger(__name__)


@ray.remote
class LlamaActor:
    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_partitions: int,
        num_actors: int,
        tracing: bool,
    ):
        self.seed = 998244353
        self.device = torch.device("cuda:0")

        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.num_partitions = num_partitions
        self.num_actors = num_actors
        self.tracing = tracing

        self.input: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.intermediates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.criterion = torch.nn.CrossEntropyLoss()

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

        torch.manual_seed(2025)
        self.model = TransformerBP(self.model_args).to(self.device)
        self.bparams = self.model.bparams

    def init_and_partition_model(self) -> List[List[Shard]]:
        assert self.rank < self.num_partitions
        if self.rank == 0:
            self.bparams_pp_range = (0, len(self.bparams) // 2)
        else:
            self.bparams_pp_range = (len(self.bparams) // 2, len(self.bparams))

    def init_training(self) -> None:
        torch.manual_seed(self.seed)
        self.seed += 1

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
        self.intermediates = []
        self.freqs_cis = None
        self.mask = None

        self.events: Dict[str, torch.cuda.Event] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "bw.loss.comp.starts": [],
            "bw.loss.comp.ends": [],
            "bw.loss.grad.starts": [],
            "bw.loss.grad.ends": [],
            "bw.grad.pre.starts": [],
            "bw.grad.pre.ends": [],
            "bw.grad.intra.starts": [],
            "bw.grad.intra.ends": [],
            "bw.grad.post.starts": [],
            "bw.grad.post.ends": [],
            "others.upd.starts": [],
            "others.upd.ends": [],
        }

        torch.cuda.synchronize()

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert key in self.events
        self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor finished iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        assert len(self.events["start"]) == 1
        assert len(self.events["end"]) == 1
        total = self.events["start"][0].elapsed_time(self.events["end"][0])

        def log(key: str, total_ms: float, count: int = 1) -> None:
            total_us = millis_to_micros(total_ms)
            self.elapses[key].append(total_us)
            if count == 1:
                logger.warning(
                    f"{key}: {total_us} us, percent: {round(total_ms / total * 100, 1)}%"
                )
            else:
                avg_us = round(total_us / count)
                logger.warning(
                    f"{key}: {total_us} us, avg: {avg_us} us, count: {count}, percent: {round(total_ms / total * 100, 1)}%"
                )

        log(
            "actor.total",
            total,
        )
        if self.tracing:
            assert len(self.events["bw.loss.comp.starts"]) == 1
            assert len(self.events["bw.loss.comp.ends"]) == 1
            assert len(self.events["bw.loss.grad.starts"]) == 1
            assert len(self.events["bw.loss.grad.ends"]) == 1

            fw_total = self.events["fw.starts"][0].elapsed_time(
                self.events["fw.ends"][-1]
            )
            bw_total = self.events["bw.loss.comp.starts"][0].elapsed_time(
                self.events["bw.grad.post.ends"][-1]
            )
            bw_loss = self.events["bw.loss.comp.starts"][0].elapsed_time(
                self.events["bw.loss.grad.ends"][0]
            )
            bw_grad = self.events["bw.grad.pre.starts"][0].elapsed_time(
                self.events["bw.grad.post.ends"][-1]
            )
            bw_grad_pre = sum(
                [
                    bw_grad_pre_start.elapsed_time(bw_grad_pre_end)
                    for bw_grad_pre_start, bw_grad_pre_end in zip(
                        self.events["bw.grad.pre.starts"],
                        self.events["bw.grad.pre.ends"],
                    )
                ]
            )
            bw_grad_intra = sum(
                [
                    bw_grad_intra_start.elapsed_time(bw_grad_intra_end)
                    for bw_grad_intra_start, bw_grad_intra_end in zip(
                        self.events["bw.grad.intra.starts"],
                        self.events["bw.grad.intra.ends"],
                    )
                ]
            )
            bw_grad_post = sum(
                [
                    bw_grad_post_start.elapsed_time(bw_grad_post_end)
                    for bw_grad_post_start, bw_grad_post_end in zip(
                        self.events["bw.grad.post.starts"],
                        self.events["bw.grad.post.ends"],
                    )
                ]
            )
            others_upd = sum(
                [
                    bw_upd_start.elapsed_time(bw_upd_end)
                    for bw_upd_start, bw_upd_end in zip(
                        self.events["others.upd.starts"],
                        self.events["others.upd.ends"],
                    )
                ]
            )

            log("fw.total", fw_total, len(self.events["fw.starts"]))
            log("bw.total", bw_total)
            log("bw.loss", bw_loss)
            log("bw.grad", bw_grad)
            log("bw.grad.pre", bw_grad_pre, len(self.events["bw.grad.pre.starts"]))
            log(
                "bw.grad.intra", bw_grad_intra, len(self.events["bw.grad.intra.starts"])
            )
            log("bw.grad.post", bw_grad_post, len(self.events["bw.grad.post.starts"]))
            log("others.upd", others_upd, len(self.events["others.upd.starts"]))
        logger.warning("")

    @classmethod
    def get_metrics(cls, tracing: bool) -> List[str]:
        if not tracing:
            return [
                "total",
                "actor.total",
            ]
        else:
            return [
                "total",
                "actor.total",
                "fw.total",
                "bw.total",
                "bw.loss",
                "bw.grad",
                "bw.grad.pre",
                "bw.grad.intra",
                "bw.grad.post",
                "others.upd",
            ]

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def get_input(self, _) -> torch.Tensor:
        assert self.input is not None
        return self.input

    def get_target(self, _) -> torch.Tensor:
        assert self.target is not None
        return self.target

    def forward_bparam(self, idx: int, input: torch.Tensor) -> torch.Tensor:
        if idx == 0:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        bp = self.bparams[idx]
        if idx == 0:
            pred = bp.forward(input)
            self.freqs_cis, self.mask = bp.post_hook(input, pred)
        elif idx < len(self.bparams) - 1:
            if self.freqs_cis is None:
                self.freqs_cis, self.mask = bp.post_embeddings(
                    self.seq_len, self.device, input
                )
            pred = bp.forward_transformer(input, 0, self.freqs_cis, self.mask)
        else:
            pred = bp.forward(bp.pre_hook(input))
            self.freqs_cis, self.mask = None, None

        pred.requires_grad_(True)
        self.intermediates.append((pred, None))

        if self.tracing:
            self.update_tracing("fw.ends")
        return pred

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.intermediates.append((None, input))
        for i in range(*self.bparams_pp_range):
            input = self.forward_bparam(i, input)
        return input

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Any:
        if self.tracing:
            self.update_tracing("bw.loss.comp.starts")
        loss = self.criterion(pred, target)
        if self.tracing:
            self.update_tracing("bw.loss.comp.ends")
        return loss

    def backward_loss(self, loss: Any) -> None:
        if self.tracing:
            self.update_tracing("bw.loss.grad.starts")
        loss.backward()
        if self.tracing:
            self.update_tracing("bw.loss.grad.ends")
        return None

    def backward_intra(self, idx: int, grad: torch.Tensor) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.grad.intra.starts")
        pred, _ = self.intermediates[idx]
        assert isinstance(pred, torch.Tensor), pred
        assert isinstance(grad, torch.Tensor), grad
        pred.backward(grad)
        if self.tracing:
            self.update_tracing("bw.grad.intra.ends")
        return None

    def backward(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if self.rank == 0:
            self.backward_intra(-1, tensor)
            return None
        else:
            target = self.get_target(None)
            loss = self.compute_loss(tensor, target)
            self.backward_loss(loss)
            for pred, _ in self.intermediates[1:]:
                logger.warning(f"pred.grad: {pred.grad}")
                # [TODO] Why is everything none?
            _, input = self.intermediates[0]
            grad = input.grad
            assert isinstance(grad, torch.Tensor), grad
            # [TODO] Why does this assert fail?
            return grad

    def update(self, idx: int, grad: torch.Tensor, grad_passed: bool) -> None:
        if self.tracing:
            self.update_tracing("others.upd.starts")
        if grad_passed:
            grad /= self.num_actors
        self.bparams[idx].update(grad, grad_passed)
        if self.tracing:
            self.update_tracing("others.upd.ends")
        if idx == 0:
            self.update_tracing("end")

    # [TODO] Get visibility of IO.
