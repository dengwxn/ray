import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

import ray
from ..common import ms_to_micros
from .model import BucketParameter, Shard, shard_model


@ray.remote
class LinearActor:
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        num_units: int,
        num_actors: int,
        device: torch.device,
        tracing: bool,
    ):
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_actors = num_actors
        self.device = device
        self.tracing = tracing

        self.shards: List[Shard] = []
        self.input: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.intermediates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.criterion = torch.nn.MSELoss()

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_and_shard_model(self) -> List[List[Shard]]:
        num_shards = self.num_actors
        fsdp_units = [
            BucketParameter(
                layer_size=self.layer_size,
                num_layers=self.num_layers // self.num_units,
                device=self.device,
            )
            for _ in range(self.num_units)
        ]
        actor_to_shards = [[] for _ in range(self.num_units)]
        for unit in fsdp_units:
            shards = shard_model(unit, num_shards)
            for rank, shard in enumerate(shards):
                actor_to_shards[rank].append(shard)
        return actor_to_shards

    def set_shards(self, shards: List[Shard]) -> None:
        self.shards = [shard.to(self.device) for shard in shards]

    def init_training(self) -> None:
        self.input = torch.randn(
            (1, self.layer_size),
            device=self.device,
            requires_grad=True,
        )
        self.target = torch.randn(
            (1, self.layer_size),
            device=self.device,
        )
        self.intermediates = []

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if key not in self.events:
            self.events[key] = event
        else:
            assert isinstance(self.events[key], list)
            self.events[key].append(event)

    def init_tracing(self) -> None:
        # [TODO] update_tracing.
        self.events: Dict[str, torch.cuda.Event] = {
            "forward_starts": [],
            "forward_ends": [],
            "backward_starts": [],
            "backward_ends": [],
            "update_starts": [],
            "update_ends": [],
        }

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor finished iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        total = self.events["start"].elapsed_time(self.events["end"])

        def log(key: str, elapse_ms: float):
            elapse_us = ms_to_micros(elapse_ms)
            self.elapses[key].append(elapse_us)
            logger.warning(
                f"{key} elapse: {elapse_us} us, percent: {round(elapse_ms / total * 100, 1)}%"
            )

        log(
            "actor.total",
            total,
        )
        if self.tracing:
            log(
                "fw.total",
                self.events["forward_starts"][0].elapsed_time(
                    self.events["forward_ends"][-1]
                ),
            )
            bw_total = self.events["backward_starts"][0].elapsed_time(
                self.events["update_ends"][-1]
            )
            bw_backward = sum(
                [
                    self.events["backward_starts"][i].elapsed_time(
                        self.events["backward_ends"][i]
                    )
                    for i in range(self.num_units)
                ]
            )
            bw_update = sum(
                [
                    self.events["update_starts"][i].elapsed_time(
                        self.events["update_ends"][i]
                    )
                    for i in range(self.num_units)
                ]
            )
            bw_others = bw_total - bw_backward - bw_update
            log("bw.total", bw_total)
            log("bw.backward", bw_backward)
            log("bw.others", bw_others)
            log("bw.update", bw_update)
        logger.warning("")

    def fetch_weights(self) -> List[torch.Tensor]:
        weights = [shard.sharded_param for shard in self.shards]
        return weights

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def get_input(self, _) -> torch.Tensor:
        assert self.input is not None
        return self.input

    def get_target(self, _) -> torch.Tensor:
        assert self.target is not None
        return self.target

    def get_shard(self, idx: int, _) -> torch.Tensor:
        assert self.shards
        return self.shards[idx].sharded_param

    def forward(self, idx: int, flat_param: torch.Tensor, input: torch.Tensor) -> None:
        if idx == 0:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("forward_starts")
        shard = self.shards[idx]
        shard.set_flat_param(flat_param)
        pred = shard.forward(input)
        if idx < len(self.models) - 1:
            pred_as_input = pred.detach().requires_grad_(True)
        else:
            pred_as_input = pred
        self.intermediates.append((pred, pred_as_input))
        if idx < len(self.models) - 1:
            shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("forward_ends")
        return pred_as_input

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # [TODO] update_tracing.
        return self.criterion(pred, target)

    def backward_loss(self, loss: torch.Tensor) -> None:
        # [TODO] update_tracing.
        loss.backward()
        shard = self.shards[-1]
        flat_grad = shard.get_flat_grad()
        shard.free_peer_shards()
        return flat_grad

    def backward(self, idx: int, flat_param: torch.Tensor) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("backward_starts")
        shard = self.shards[idx]
        shard.set_flat_param(flat_param)
        pred, pred_as_input = self.intermediates[idx]
        grad = pred_as_input.grad
        pred.backward(grad)
        flat_grad = shard.get_flat_grad()
        shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("backward_ends")
        return flat_grad

    def update(self, idx: int, grad: torch.Tensor, grad_passed: bool) -> None:
        if self.tracing:
            self.update_tracing("update_starts")
        if grad_passed:
            grad /= self.num_actors
        self.shards[idx].update(grad, grad_passed)
        if self.tracing:
            self.update_tracing("update_ends")
        if idx == 0:
            self.update_tracing("end")
