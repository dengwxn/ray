import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

import ray
from ..common import millis_to_micros
from .model import TransformerPP as Transformer

logger = logging.getLogger(__name__)


@ray.remote
class LlamaActor:
    @dataclass
    class BatchParameter:
        # criterion: torch.nn.CrossEntropyLoss
        # optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits_as_output: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_batches: int,
        num_partitions: int,
        num_actors: int,
        tracing: bool,
    ):
        torch.autograd.set_detect_anomaly(True)

        self.seed = 998244353
        self.device = torch.device(f"cuda:0")

        logger.info(f"Rank {rank}: device {self.device}")
        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.num_batches = num_batches
        self.num_partitions = num_partitions
        self.num_actors = num_actors
        self.tracing = tracing

        self.input: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None

        # manual pipeline
        model = Transformer(model_args, rank)
        layers_per_rank = model_args.n_layers // num_partitions
        if rank == 0:
            for _ in range(layers_per_rank):
                del model.layers[0]
            model.norm = None
            model.output = None
        else:
            model.tok_embeddings = None
            for _ in range(layers_per_rank):
                del model.layers[layers_per_rank]
        model.to(self.device)
        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        # microbatch metadata
        self.bparams: List[LlamaActor.BatchParameter] = []
        for i in range(num_batches):
            torch.manual_seed(2025 + i)
            bparam = self.BatchParameter()
            self.bparams.append(bparam)

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

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

        self.num_batches_forwarded = 0
        self.num_batches_updated = 0
        for bparam in self.bparams:
            bparam.logits_as_input = None
            bparam.logits_as_output = None
        
        self.grad_acc = None

        self.events: Dict[str, Any] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "bw.starts": [],
            "bw.ends": [],
            "upd.starts": [],
            "upd.ends": [],
        }

        torch.cuda.synchronize()

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
                "upd.total",
            ]

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert key in self.events
        self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor {self.rank} finished iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        assert len(self.events["start"]) == 1
        assert len(self.events["end"]) == 1
        total = self.events["start"][0].elapsed_time(self.events["end"][0])
        print("iter total: ", round(total))

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
            fw_total = sum(
                [
                    fw_start.elapsed_time(fw_end)
                    for fw_start, fw_end in zip(
                        self.events["fw.starts"],
                        self.events["fw.ends"],
                    )
                ]
            )
            bw_total = sum(
                [
                    bw_start.elapsed_time(bw_end)
                    for bw_start, bw_end in zip(
                        self.events["bw.starts"],
                        self.events["bw.ends"],
                    )
                ]
            )
            upd_total = sum(
                [
                    bw_upd_start.elapsed_time(bw_upd_end)
                    for bw_upd_start, bw_upd_end in zip(
                        self.events["upd.starts"],
                        self.events["upd.ends"],
                    )
                ]
            )

            log("fw.total", fw_total, len(self.events["fw.starts"]))
            log("bw.total", bw_total, len(self.events["bw.starts"]))
            log("upd.total", upd_total, len(self.events["upd.starts"]))
        logger.warning("")

    def forward(
        self, idx: int, logits_as_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        print(f"rank_{self.rank}.fwd batch {idx}")

        self.num_batches_forwarded += 1
        if self.num_batches_forwarded == 1:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")


        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        if self.rank == 0:
            logits_as_output = self.model.forward_first(self.input)
            bparam.logits_as_output = logits_as_output
            logits_as_input = logits_as_output.detach()
            output = logits_as_input
        else:
            assert logits_as_input is not None
            logits_as_input = logits_as_input.to(self.device).requires_grad_(True)
            bparam.logits_as_input = logits_as_input
            logits_as_output = self.model.forward_second(
                self.input, logits_as_input
            )
            output = logits_as_output

        if self.tracing:
            self.update_tracing("fw.ends")
        return output

    def backward_first(self, idx: int, logits: torch.Tensor) -> torch.Tensor:
        print(f"rank_{self.rank}.bwd_first batch {idx}")
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        loss = self.criterion(logits, self.target)
        loss.backward()
        grad = bparam.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_intra(self, idx: int, prev_grad: torch.Tensor) -> None:
        print(f"rank_{self.rank}.bwd_intra batch {idx}")
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        assert prev_grad is not None
        prev_grad = prev_grad.to(self.device)

        bparam.logits_as_output.backward(prev_grad)
        grad = bparam.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_last(self, idx: int, prev_grad: torch.Tensor) -> None:
        print(f"rank_{self.rank}.bwd_last batch {idx}")
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        assert prev_grad is not None
        prev_grad = prev_grad.to(self.device)

        bparam.logits_as_output.backward(prev_grad)

        return None

    def backward(self, idx: int, data: torch.Tensor) -> Optional[torch.Tensor]:
        print(f"rank_{self.rank}.fwd batch {idx}")

        if self.tracing:
            self.update_tracing("bw.starts")

        if self.rank == self.num_actors - 1:
            output = self.backward_first(idx, data)
        elif self.rank == 0:
            output = self.backward_last(idx, data)
        else:
            output = self.backward_intra(idx, data)

        # if self.grad_acc:
        #     self.grad_acc += output
        # else:
        #     self.grad_acc = output
        # self.optimizer.step()
        
        if self.tracing:
            self.update_tracing("bw.ends")
        return output

    def update(self, idx, data) -> None:
        print(f"rank_{self.rank}.upd batch {idx}")

        if self.tracing:
            self.update_tracing("upd.starts")

        # assert idx < len(self.bparams)
        # bparam = self.bparams[idx]
    
        self.num_batches_updated += 1
        if self.num_batches_updated == self.num_batches:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.update_tracing("end")

        if self.tracing:
            self.update_tracing("upd.ends")

        return data


class LlamaActorOff:
    @dataclass
    class BatchParameter:
        model: Transformer
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits_as_output: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_batches: int,
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
        self.num_batches = num_batches
        self.num_partitions = num_partitions
        self.num_actors = num_actors
        self.tracing = tracing

        self.input: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None

        self.bparams: List[LlamaActor.BatchParameter] = []
        for i in range(num_batches):
            torch.manual_seed(2025 + i)
            model = Transformer(model_args, rank).to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            bparam = self.BatchParameter(model, criterion, optimizer)
            self.bparams.append(bparam)

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self) -> None:
        torch.manual_seed(self.seed)
        self.seed += 1

        self.num_batches_forwarded = 0
        self.num_batches_updated = 0
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

        for bparam in self.bparams:
            bparam.logits_as_input = None
            bparam.logits_as_output = None

        self.events: Dict[str, Any] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "bw.starts": [],
            "bw.ends": [],
            "upd.starts": [],
            "upd.ends": [],
        }

        torch.cuda.synchronize()

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
                "upd.total",
            ]

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert key in self.events
        self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor {self.rank} finished iteration {self.it}")
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
            fw_total = sum(
                [
                    fw_start.elapsed_time(fw_end)
                    for fw_start, fw_end in zip(
                        self.events["fw.starts"],
                        self.events["fw.ends"],
                    )
                ]
            )
            bw_total = sum(
                [
                    bw_start.elapsed_time(bw_end)
                    for bw_start, bw_end in zip(
                        self.events["bw.starts"],
                        self.events["bw.ends"],
                    )
                ]
            )
            upd_total = sum(
                [
                    bw_upd_start.elapsed_time(bw_upd_end)
                    for bw_upd_start, bw_upd_end in zip(
                        self.events["upd.starts"],
                        self.events["upd.ends"],
                    )
                ]
            )

            log("fw.total", fw_total, len(self.events["fw.starts"]))
            log("bw.total", bw_total, len(self.events["bw.starts"]))
            log("upd.total", upd_total, len(self.events["upd.starts"]))
        logger.warning("")

    def forward(
        self, idx: int, logits_as_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.num_batches_forwarded += 1
        if self.num_batches_forwarded == 1:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        if self.rank == 0:
            logits_as_output = bparam.model.forward_first(self.input, 0)
            bparam.logits_as_output = logits_as_output
            logits_as_input = logits_as_output.detach()
            output = logits_as_input
        else:
            assert logits_as_input is not None
            logits_as_input = logits_as_input.to(self.device).requires_grad_(True)
            bparam.logits_as_input = logits_as_input
            logits_as_output = bparam.model.forward_second(
                self.input, 0, logits_as_input
            )
            output = logits_as_output

        if self.tracing:
            self.update_tracing("fw.ends")
        return output

    def backward_loss(self, idx: int, logits: torch.Tensor) -> torch.Tensor:
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        loss = bparam.criterion(logits, self.target)
        loss.backward()
        grad = bparam.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_intra(self, idx: int, grad: torch.Tensor) -> None:
        assert idx < len(self.bparams)
        bparam = self.bparams[idx]
        assert grad is not None
        grad = grad.to(self.device)

        bparam.logits_as_output.backward(grad)

    def backward(self, idx: int, data: torch.Tensor) -> Optional[torch.Tensor]:
        if self.tracing:
            self.update_tracing("bw.starts")

        if self.rank == 0:
            output = self.backward_intra(idx, data)
        else:
            output = self.backward_loss(idx, data)

        if self.tracing:
            self.update_tracing("bw.ends")
        return output

    def update(self, idx: int, _backward) -> None:
        if self.tracing:
            self.update_tracing("upd.starts")

        assert idx < len(self.bparams)
        bparam = self.bparams[idx]

        bparam.optimizer.step()
        bparam.optimizer.zero_grad()

        if self.tracing:
            self.update_tracing("upd.ends")
        self.num_batches_updated += 1
        if self.num_batches_updated == self.num_batches:
            self.update_tracing("end")
