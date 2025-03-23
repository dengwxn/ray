import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist

import ray
from .common import millis_to_micros
from .model import TransformerTP, TransformerTP2PP


@ray.remote
class ActorTP:
    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank: int,
        num_actors_tp: int,
        master_addr: str,
        master_port: int,
        tracing: bool,
    ):
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.tracing = tracing

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(num_actors_tp)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl")

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Actor {rank}, device: {self.device}")

        model_parallel_size = 2
        fs_init.initialize_model_parallel(model_parallel_size)

        self.model = TransformerTP(self.model_args).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self):
        torch.manual_seed(998244353)
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
        torch.cuda.synchronize()

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

    def forward(self, _) -> torch.Tensor:
        self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        logits = self.model.forward(self.input, 0)

        if self.tracing:
            self.update_tracing("fw.ends")
        return logits

    def backward(self, logits) -> None:
        if self.tracing:
            self.update_tracing("bw.starts")

        loss = self.criterion(logits, self.target)
        loss.backward()

        if self.tracing:
            self.update_tracing("bw.ends")

    def update(self, _) -> None:
        if self.tracing:
            self.update_tracing("upd.starts")

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.tracing:
            self.update_tracing("upd.ends")
        self.update_tracing("end")

    def clean(self) -> None:
        dist.destroy_process_group()


@ray.remote
class ActorTP2PP:
    @dataclass
    class BatchParameter:
        model: TransformerTP2PP
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits_as_output: Optional[torch.Tensor] = None

    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank_tp: int,
        num_tp: int,
        master_addr: str,
        master_port: int,
        rank_pp: int,
        num_pp_batches: int,
        tracing: bool,
    ):
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank_tp = rank_tp
        self.rank_pp = rank_pp
        self.num_pp_batches = num_pp_batches
        self.tracing = tracing

        os.environ["RANK"] = str(rank_tp)
        os.environ["WORLD_SIZE"] = str(num_tp)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl")

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Actor rank_tp: {rank_tp}, rank_pp: {rank_pp}, device: {self.device}")

        model_parallel_size = 2
        fs_init.initialize_model_parallel(model_parallel_size)

        self.bparams: List[ActorTP2PP.BatchParameter] = []
        for i in range(num_pp_batches):
            torch.manual_seed(2025 + i)
            model = TransformerTP2PP(model_args, rank_pp).to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            bparam = self.BatchParameter(model, criterion, optimizer)
            self.bparams.append(bparam)

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self):
        torch.manual_seed(998244353)

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
        logger.warning(f"Actor {self.rank_pp} finished iteration {self.it}")
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

        if self.rank_pp == 0:
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

        if self.rank_pp == 0:
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
        if self.num_batches_updated == self.num_pp_batches:
            self.update_tracing("end")

    def clean(self) -> None:
        dist.destroy_process_group()
