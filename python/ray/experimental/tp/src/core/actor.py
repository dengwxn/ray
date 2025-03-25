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
from .model import (
    TransformerTP,
    TransformerTP2DP,
    TransformerTP2PP,
    TransformerTP2PP4DP,
)


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
    class BatchModel:
        model: TransformerTP2PP
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None

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

        self.batches: List[ActorTP2PP.BatchModel] = []
        for i in range(num_pp_batches):
            torch.manual_seed(2025 + i)
            model = TransformerTP2PP(model_args, rank_pp).to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            model = self.BatchModel(model, criterion, optimizer)
            self.batches.append(model)

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
        for batch in self.batches:
            batch.logits_as_input = None
            batch.logits = None

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
        logger.warning(
            f"Actor ({self.rank_tp}, {self.rank_pp}) finished iteration {self.it}"
        )
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

        assert idx < len(self.batches)
        batch = self.batches[idx]

        if self.rank_pp == 0:
            logits = batch.model.forward_first(self.input, 0)
            batch.logits = logits
            logits_as_input = logits.detach()
            output = logits_as_input
        else:
            assert logits_as_input is not None
            logits_as_input = logits_as_input.requires_grad_(True)
            batch.logits_as_input = logits_as_input
            logits = batch.model.forward_second(self.input, 0, logits_as_input)
            output = logits

        if self.tracing:
            self.update_tracing("fw.ends")
        return output

    def backward_loss(self, idx: int, logits: torch.Tensor) -> torch.Tensor:
        assert idx < len(self.batches)
        batch = self.batches[idx]

        loss = batch.criterion(logits, self.target)
        loss.backward()
        grad = batch.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_intra(self, idx: int, grad: torch.Tensor) -> None:
        assert idx < len(self.batches)
        batch = self.batches[idx]
        assert grad is not None

        batch.logits.backward(grad)

    def backward(self, idx: int, data: torch.Tensor) -> Optional[torch.Tensor]:
        if self.tracing:
            self.update_tracing("bw.starts")

        if self.rank_pp == 0:
            output = self.backward_intra(idx, data)
        else:
            output = self.backward_loss(idx, data)

        if self.tracing:
            self.update_tracing("bw.ends")
            self.update_tracing("upd.starts")

        assert idx < len(self.batches)
        batch = self.batches[idx]

        batch.optimizer.step()
        batch.optimizer.zero_grad()

        if self.tracing:
            self.update_tracing("upd.ends")
        self.num_batches_updated += 1
        if self.num_batches_updated == self.num_pp_batches:
            self.update_tracing("end")

        return output

    def clean(self) -> None:
        dist.destroy_process_group()


@ray.remote
class ActorTP2DP:
    def __init__(
        self,
        model_args,
        batch_size: int,
        seq_len: int,
        rank_tp: int,
        num_actors_tp: int,
        master_addr: str,
        master_port: int,
        rank_dp: int,
        num_actors_dp: int,
        num_parts_dp: int,
        tracing: bool,
    ):
        self.model_args = model_args
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank_tp = rank_tp
        self.rank_dp = rank_dp
        self.num_actors_dp = num_actors_dp
        self.num_parts_dp = num_parts_dp
        self.tracing = tracing

        os.environ["RANK"] = str(rank_tp)
        os.environ["WORLD_SIZE"] = str(num_actors_tp)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend="nccl")

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Actor rank_tp: {rank_tp}, rank_dp: {rank_dp}, device: {self.device}")

        model_parallel_size = 2
        fs_init.initialize_model_parallel(model_parallel_size)

        self.model = TransformerTP2DP(self.model_args).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self):
        torch.manual_seed(998244353)

        self.model.inters_dp = []

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

        self.events: Dict[str, Any] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "bw.grad.loss.starts": [],
            "bw.grad.loss.ends": [],
            "bw.grad.intra.starts": [],
            "bw.grad.intra.ends": [],
            "others.upd.starts": [],
            "others.upd.ends": [],
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
                "bw.grad.loss.total",
                "bw.grad.intra.total",
                "others.upd.total",
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
        logger.warning(
            f"Actor ({self.rank_tp}, {self.rank_dp}) finished iteration {self.it}"
        )
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
            bw_grad_loss_total = sum(
                [
                    bw_grad_loss_start.elapsed_time(bw_grad_loss_end)
                    for bw_grad_loss_start, bw_grad_loss_end in zip(
                        self.events["bw.grad.loss.starts"],
                        self.events["bw.grad.loss.ends"],
                    )
                ]
            )
            bw_grad_intra_total = sum(
                [
                    bw_grad_intra_start.elapsed_time(bw_grad_intra_end)
                    for bw_grad_intra_start, bw_grad_intra_end in zip(
                        self.events["bw.grad.intra.starts"],
                        self.events["bw.grad.intra.ends"],
                    )
                ]
            )
            others_upd_total = sum(
                [
                    others_upd_start.elapsed_time(others_upd_end)
                    for others_upd_start, others_upd_end in zip(
                        self.events["others.upd.starts"],
                        self.events["others.upd.ends"],
                    )
                ]
            )

            log("fw.total", fw_total, len(self.events["fw.starts"]))
            log("bw.grad.loss.total", bw_grad_loss_total)
            log(
                "bw.grad.intra.total",
                bw_grad_intra_total,
                len(self.events["bw.grad.intra.starts"]),
            )
            log(
                "others.upd.total",
                others_upd_total,
                len(self.events["others.upd.starts"]),
            )
        logger.warning("")

    def forward(self, _) -> torch.Tensor:
        self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        tokens = self.input
        for i, part in enumerate(self.model.parts_dp):
            if i == 0:
                logits, freqs_cis, mask = part.forward(tokens)
            elif i < self.num_parts_dp - 1:
                logits = part.forward(logits_as_input, 0, freqs_cis, mask)
            else:
                logits = part.forward(logits_as_input)
            if i < self.num_parts_dp - 1:
                logits_as_input = logits.detach().requires_grad_(True)
            else:
                logits_as_input = logits
            self.model.inters_dp.append((logits, logits_as_input))

        if self.tracing:
            self.update_tracing("fw.ends")
        return logits

    def backward_loss(self, logits: torch.Tensor) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.grad.loss.starts")

        target = self.target
        loss = self.criterion(logits, target)
        loss.backward()
        part = self.model.parts_dp[-1]
        flat_grad = part.get_flat_grad()

        if self.tracing:
            self.update_tracing("bw.grad.loss.ends")
        return flat_grad

    def backward_intra(self, idx: int, _) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.grad.intra.starts")

        logits, logits_as_input = self.model.inters_dp[idx]
        grad = logits_as_input.grad
        assert logits is not None
        assert grad is not None
        logits.backward(grad)
        part = self.model.parts_dp[idx]
        flat_grad = part.get_flat_grad()

        if self.tracing:
            self.update_tracing("bw.grad.intra.ends")
        return flat_grad

    def update(self, idx: int, flat_grad: torch.Tensor, flat_grad_passed: bool) -> None:
        if self.tracing:
            self.update_tracing("others.upd.starts")

        if flat_grad_passed:
            flat_grad /= self.num_actors_dp
        part = self.model.parts_dp[idx]
        part.update(flat_grad, flat_grad_passed)

        if self.tracing:
            self.update_tracing("others.upd.ends")
        if idx == 0:
            self.update_tracing("end")

    def clean(self) -> None:
        dist.destroy_process_group()


@ray.remote
class ActorTP2PP4DP:
    @dataclass
    class BatchModel:
        model: TransformerTP2PP4DP
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.AdamW
        logits_as_input: Optional[torch.Tensor] = None
        logits: Optional[torch.Tensor] = None

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

        self.batches: List[ActorTP2PP4DP.BatchModel] = []
        for i in range(num_pp_batches):
            torch.manual_seed(2025 + i)
            model = TransformerTP2PP4DP(model_args, rank_pp).to(self.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
            batch = self.BatchModel(model, criterion, optimizer)
            self.batches.append(batch)

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
        for batch in self.batches:
            batch.logits_as_input = None
            batch.logits = None

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
        logger.warning(
            f"Actor ({self.rank_tp}, {self.rank_pp}) finished iteration {self.it}"
        )
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
        self, idx_batch: int, logits_as_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self.num_batches_forwarded += 1
        if self.num_batches_forwarded == 1:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        assert idx_batch < len(self.batches)
        batch = self.batches[idx_batch]

        if self.rank_pp == 0:
            logits = batch.model.forward_first(self.input)
            batch.logits = logits
            logits_as_input = logits.detach()
            output = logits_as_input
        else:
            assert logits_as_input is not None
            logits_as_input = logits_as_input.requires_grad_(True)
            batch.logits_as_input = logits_as_input
            logits = batch.model.forward_second(self.input, logits_as_input)
            output = logits

        if self.tracing:
            self.update_tracing("fw.ends")
        return output

    def backward_loss(self, idx: int, logits: torch.Tensor) -> torch.Tensor:
        assert idx < len(self.batches)
        batch = self.batches[idx]

        loss = batch.criterion(logits, self.target)
        loss.backward()
        grad = batch.logits_as_input.grad

        assert grad is not None
        return grad

    def backward_intra(self, idx: int, grad: torch.Tensor) -> None:
        assert idx < len(self.batches)
        batch = self.batches[idx]
        assert grad is not None

        batch.logits.backward(grad)

    def backward(self, idx: int, data: torch.Tensor) -> Optional[torch.Tensor]:
        if self.tracing:
            self.update_tracing("bw.starts")

        if self.rank_pp == 0:
            output = self.backward_intra(idx, data)
        else:
            output = self.backward_loss(idx, data)

        if self.tracing:
            self.update_tracing("bw.ends")
            self.update_tracing("upd.starts")

        assert idx < len(self.batches)
        batch = self.batches[idx]

        batch.optimizer.step()
        batch.optimizer.zero_grad()

        if self.tracing:
            self.update_tracing("upd.ends")
        self.num_batches_updated += 1
        if self.num_batches_updated == self.num_pp_batches:
            self.update_tracing("end")

        return output

    def clean(self) -> None:
        dist.destroy_process_group()
