import logging
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from common import count_params, random_seed
from model import TextEncoder, VisionEncoder, init_optimizer, parallelize_2d
from open_clip.loss import ClipLoss

import ray
from ray.air._internal.util import find_free_port
from ray.air.util.torch_dist import _init_torch_distributed

logger = logging.getLogger(__name__)


def millis_to_micros(millis: float) -> int:
    return round(millis * 1e3)


class BaseWorker:
    def __init__(self) -> None:
        self.name = "Actor"

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def get_metadata(self):
        return {
            "gpu_ids": ray.get_gpu_ids(),
            "address": ray.util.get_node_ip_address(),
            "port": find_free_port(),
        }

    def init_torch_distributed(self, dist_config):
        self.dist_config = dist_config
        _init_torch_distributed(
            init_method="env", backend="nccl", **asdict(dist_config)
        )
        logger.info(f"Rank {self.dist_config.rank}: Initialized")
        if self.dist_config.rank == 0:
            logger.info(asdict(self.dist_config))

    def init_training(self):
        self.events: Dict[str, Any] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "fw.text.starts": [],
            "fw.text.ends": [],
            "fw.vision.starts": [],
            "fw.vision.ends": [],
            "bw.starts": [],
            "bw.ends": [],
            "bw.text.starts": [],
            "bw.text.ends": [],
            "bw.vision.starts": [],
            "bw.vision.ends": [],
            "upd.starts": [],
            "upd.ends": [],
        }

    @classmethod
    def get_metrics(cls) -> List[str]:
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
        logger.warning(f"{self.name} {self.rank} finished iteration {self.it}")
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

        log("actor.total", total)
        log("fw.total", fw_total)
        log("bw.total", bw_total)
        log("upd.total", upd_total)

        if self.name == "WorkerV3":
            fw_text_total = sum(
                [
                    fw_start.elapsed_time(fw_end)
                    for fw_start, fw_end in zip(
                        self.events["fw.text.starts"],
                        self.events["fw.text.ends"],
                    )
                ]
            )
            bw_text_total = sum(
                [
                    bw_start.elapsed_time(bw_end)
                    for bw_start, bw_end in zip(
                        self.events["bw.text.starts"],
                        self.events["bw.text.ends"],
                    )
                ]
            )
            fw_vision_total = sum(
                [
                    fw_start.elapsed_time(fw_end)
                    for fw_start, fw_end in zip(
                        self.events["fw.vision.starts"],
                        self.events["fw.vision.ends"],
                    )
                ]
            )
            bw_vision_total = sum(
                [
                    bw_start.elapsed_time(bw_end)
                    for bw_start, bw_end in zip(
                        self.events["bw.vision.starts"],
                        self.events["bw.vision.ends"],
                    )
                ]
            )
            log("fw.text.total", fw_text_total)
            log("bw.text.total", bw_text_total)
            log("fw.vision.total", fw_vision_total)
            log("bw.vision.total", bw_vision_total)

        logger.warning("")


@ray.remote(num_gpus=1)
class TextWorker(BaseWorker):
    def __init__(
        self, model_name, num_dp, num_tp, num_dp_vision, seed=998244353
    ) -> None:
        super().__init__()
        self.name = "Text"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = TextEncoder(model_name)
        self.num_params = count_params(self.model)
        self.num_dp = num_dp
        self.num_tp = num_tp
        self.num_dp_vision = num_dp_vision
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)

    def init_fsdp_model(self):
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank)

        self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert (
            self.world_size == self.num_tp * self.num_dp
        ), "world size must be equal to tp size * dp size"

        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.num_dp,
                self.num_tp,
                text=True,
                vision=False,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to(self.device)
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0
        self.logit_scale.to(self.device)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_text = bs_global // self.num_dp
        logger.warning(f"bs_text: {bs_text}")

        torch.manual_seed(i)
        texts = torch.randint(0, 49408, (bs_text, 77))
        return texts

    def forward(self, inputs):
        self.update_tracing("start")
        self.update_tracing("fw.starts")

        texts = self.load_batch(inputs).to(self.device)
        with torch.autocast(device_type="cuda"):
            self.text_features = self.model(texts)
        feats = self.text_features.detach()

        self.update_tracing("fw.ends")
        return feats

    def backward(self, vision_features):
        self.update_tracing("bw.starts")

        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                vision_features.cuda(), self.text_features, self.logit_scale
            )
        self.loss.backward()

        self.update_tracing("bw.ends")
        self.update_tracing("upd.starts")

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_tracing("upd.ends")
        self.update_tracing("end")
        return None

    def get_rank(self):
        return self.dp_rank, self.tp_rank

    def get_num_params(self):
        return self.num_params

    def get_device_name(self):
        return torch.cuda.get_device_name()

    def reduce_activations(self, *activations):
        # Concat the activations of DP workers
        return torch.cat(activations, dim=0)

    def scatter_activations(self, activations):
        # Scatter the activations to each vision dp group
        result = torch.chunk(activations, self.num_dp_vision)
        return (*result,)

    def echo(self, input):
        return input


class TextWorkerV2(BaseWorker):
    def __init__(
        self, model_name, dp_size, tp_size, vision_dp_size, seed=998244353
    ) -> None:
        super().__init__()
        self.name = "Text"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = TextEncoder(model_name).to(self.device)
        self.num_params = count_params(self.model)
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.vision_dp_size = vision_dp_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def init_fsdp_model(self):
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank)

        self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert (
            self.world_size == self.tp_size * self.dp_size
        ), "world size must be equal to tp size * dp size"

        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.dp_size,
                self.tp_size,
                text=True,
                vision=False,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to("cuda")
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0
        self.logit_scale.to("cuda")

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_text = bs_global // self.dp_size
        logger.warning(f"bs_text: {bs_text}")

        torch.manual_seed(i)
        texts = torch.randint(0, 49408, (bs_text, 77))
        return texts

    def forward(self, inputs):
        self.update_tracing("start")
        self.update_tracing("fw.starts")

        texts = self.load_batch(inputs).to(self.device)
        # with torch.autocast(device_type="cuda"):
        self.text_features = self.model(texts)
        feats = self.text_features.detach()

        self.update_tracing("fw.ends")
        return feats

    def backward(self, vision_features):
        self.update_tracing("bw.starts")

        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                vision_features.cuda(), self.text_features, self.logit_scale
            )
        self.loss.backward()

        self.update_tracing("bw.ends")
        self.update_tracing("upd.starts")

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_tracing("upd.ends")
        self.update_tracing("end")
        return None


class TextWorkerV3(BaseWorker):
    def __init__(self, model_name, num_dp, seed=998244353) -> None:
        super().__init__()
        self.name = "Text"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = TextEncoder(model_name).to(self.device)
        self.num_params = count_params(self.model)
        self.num_dp = num_dp
        self.num_tp = 1
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)

    def init_fsdp_model(self):
        self.world_size = self.num_tp * self.num_dp
        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.num_dp,
                self.num_tp,
                text=True,
                vision=False,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to(self.device)
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0
        self.logit_scale.to(self.device)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_text = bs_global // self.num_dp
        logger.warning(f"bs_text: {bs_text}")

        torch.manual_seed(i)
        texts = torch.randint(0, 49408, (bs_text, 77))
        return texts

    def forward(self, inputs):
        # self.update_tracing("start")
        # self.update_tracing("fw.starts")

        texts = self.load_batch(inputs).to(self.device)
        self.text_features = self.model(texts)
        feats = self.text_features.detach()

        # self.update_tracing("fw.ends")
        return feats

    def backward(self, vision_features):
        # self.update_tracing("bw.starts")

        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        self.loss = self.clip_loss_fn(
            vision_features.cuda(), self.text_features, self.logit_scale
        )
        self.loss.backward()

        # self.update_tracing("bw.ends")
        # self.update_tracing("upd.starts")

        self.optimizer.step()
        self.optimizer.zero_grad()

        # self.update_tracing("upd.ends")
        # self.update_tracing("end")
        return None


@ray.remote(num_gpus=1)
class VisionWorker(BaseWorker):
    def __init__(self, model_name, num_dp, num_tp, num_dp_text, seed=998244353) -> None:
        super().__init__()
        self.name = "Vision"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = VisionEncoder(model_name)
        self.num_params = count_params(self.model)
        self.num_dp = num_dp
        self.num_tp = num_tp
        self.num_dp_text = num_dp_text
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)

    def init_fsdp_model(self):
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank)

        self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert (
            self.world_size == self.num_tp * self.num_dp
        ), "world size must be equal to tp size * dp size"

        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.num_dp,
                self.num_tp,
                text=False,
                vision=True,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to(self.device)
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0
        self.logit_scale.to(self.device)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_vision = bs_global // self.num_dp
        logger.warning(f"bs_vision: {bs_vision}")

        torch.manual_seed(i)
        images = torch.randn(bs_vision, 3, 224, 224)
        return images

    def forward(self, inputs):
        self.update_tracing("start")
        self.update_tracing("fw.starts")

        images = self.load_batch(inputs).to(self.device)
        with torch.autocast(device_type="cuda"):
            self.vision_features = self.model(images)
        feats = self.vision_features.detach()

        self.update_tracing("fw.ends")
        return feats

    def backward(self, text_features):
        self.update_tracing("bw.starts")

        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                self.vision_features, text_features.cuda(), self.logit_scale
            )
        self.loss.backward()

        self.update_tracing("bw.ends")
        self.update_tracing("upd.starts")

        # num_grads = 0
        # num_params = 0
        # for param in self.model.parameters():
        #     num_params += 1
        #     if param.grad is not None:
        #         num_grads += 1
        # logger.warning(
        #     f"{self.name} {self.rank} num_grads: {num_grads}, num_params: {num_params}"
        # )

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_tracing("upd.ends")
        self.update_tracing("end")
        return None

    def get_rank(self):
        return self.dp_rank, self.tp_rank

    def get_num_params(self):
        return self.num_params

    def get_device_name(self):
        return torch.cuda.get_device_name()

    def reduce_activations(self, *activations):
        # Concat the activations of DP workers
        return torch.cat(activations, dim=0)

    def scatter_activations(self, activations):
        # Scatter the activations to each text dp group
        result = torch.chunk(activations, self.num_dp_text)
        return (*result,)

    def echo(self, input):
        return input


class VisionWorkerV2(BaseWorker):
    def __init__(
        self, model_name, dp_size, tp_size, text_dp_size, seed=998244353
    ) -> None:
        super().__init__()
        self.name = "Vision"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = VisionEncoder(model_name).to(self.device)
        self.num_params = count_params(self.model)
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.text_dp_size = text_dp_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def init_fsdp_model(self):
        self.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.rank)

        self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert (
            self.world_size == self.tp_size * self.dp_size
        ), "world size must be equal to tp size * dp size"

        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.dp_size,
                self.tp_size,
                text=False,
                vision=True,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to("cuda")
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_vision = bs_global // self.dp_size
        logger.warning(f"bs_vision: {bs_vision}")

        torch.manual_seed(i)
        images = torch.randn(bs_vision, 3, 224, 224)
        return images

    def forward(self, inputs):
        self.update_tracing("start")
        self.update_tracing("fw.starts")

        images = self.load_batch(inputs).to(self.device)
        # with torch.autocast(device_type="cuda"):
        self.vision_features = self.model(images)
        feats = self.vision_features.detach()

        self.update_tracing("fw.ends")
        return feats

    def backward(self, text_features):
        self.update_tracing("bw.starts")

        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                self.vision_features, text_features.cuda(), self.logit_scale
            )
        self.loss.backward()

        self.update_tracing("bw.ends")
        self.update_tracing("upd.starts")

        # num_grads = 0
        # num_params = 0
        # for param in self.model.parameters():
        #     num_params += 1
        #     if param.grad is not None:
        #         num_grads += 1
        # logger.warning(
        #     f"{self.name} {self.rank} num_grads: {num_grads}, num_params: {num_params}"
        # )

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_tracing("upd.ends")
        self.update_tracing("end")
        return None


class VisionWorkerV3(BaseWorker):
    def __init__(self, model_name, num_dp, seed=998244353) -> None:
        super().__init__()
        self.name = "Vision"
        self.device = torch.device("cuda:0")

        random_seed(seed)
        self.model = VisionEncoder(model_name).to(self.device)
        self.num_params = count_params(self.model)
        self.num_dp = num_dp
        self.num_tp = 1
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)

    def init_fsdp_model(self):
        self.world_size = self.num_tp * self.num_dp
        if self.world_size > 1:
            self.model, self.device_mesh = parallelize_2d(
                self.model,
                self.num_dp,
                self.num_tp,
                text=False,
                vision=True,
            )
            self.tp_rank = self.device_mesh["tp"].get_local_rank()
            self.dp_rank = self.device_mesh["dp"].get_local_rank()
        else:
            self.model.to(self.device)
            self.device_mesh = None
            self.tp_rank = 0
            self.dp_rank = 0
        self.logit_scale.to(self.device)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )

    def load_batch(self, inputs):
        i, bs_global = inputs
        bs_vision = bs_global // self.num_dp
        logger.warning(f"bs_vision: {bs_vision}")

        torch.manual_seed(i)
        images = torch.randn(bs_vision, 3, 224, 224)
        return images

    def forward(self, inputs):
        # self.update_tracing("start")
        # self.update_tracing("fw.starts")

        images = self.load_batch(inputs).to(self.device)
        self.vision_features = self.model(images)
        feats = self.vision_features.detach()

        # self.update_tracing("fw.ends")
        return feats

    def backward(self, text_features):
        # self.update_tracing("bw.starts")

        self.loss = self.clip_loss_fn(
            self.vision_features, text_features.cuda(), self.logit_scale
        )
        self.loss.backward()

        # self.update_tracing("bw.ends")
        # self.update_tracing("upd.starts")

        self.optimizer.step()
        self.optimizer.zero_grad()

        # self.update_tracing("upd.ends")
        # self.update_tracing("end")
        return None


class WorkerV2(BaseWorker):
    def __init__(self, model_name, dp_size, tp_size, seed=998244353):
        super().__init__()
        self.name = "WorkerV2"
        self.text = TextWorkerV2(model_name, dp_size, tp_size, dp_size, seed=seed)
        self.vision = VisionWorkerV2(model_name, dp_size, tp_size, dp_size, seed=seed)

    def forward(self, inputs):
        text_acts = self.text.forward(inputs)
        visions_acts = self.vision.forward(inputs)
        return text_acts, visions_acts

    def backward(self, acts):
        text_acts, vision_acts = acts
        self.text.backward(vision_acts)
        self.vision.backward(text_acts)


@ray.remote(num_gpus=1)
class WorkerV3(BaseWorker):
    def __init__(self, model_name, num_dp, seed=998244353):
        super().__init__()
        self.name = "WorkerV3"
        self.rank = 4  # [TODO]

        self.text = TextWorkerV3(model_name, num_dp, seed=seed)
        self.vision = VisionWorkerV3(model_name, num_dp, seed=seed)

        self.text_acts = None
        self.vision_acts = None

    def init_fsdp_model(self):
        self.text.init_fsdp_model()
        self.vision.init_fsdp_model()

    def forward(self, inputs):
        self.update_tracing("start")
        self.update_tracing("fw.starts")
        self.update_tracing("fw.text.starts")

        self.text_acts = self.text.forward(inputs)

        self.update_tracing("fw.text.ends")
        self.update_tracing("fw.vision.starts")

        self.vision_acts = self.vision.forward(inputs)

        self.update_tracing("fw.vision.ends")
        self.update_tracing("fw.ends")

    def backward(self):
        self.update_tracing("bw.starts")
        self.update_tracing("upd.starts")  # [TODO]
        self.update_tracing("bw.text.starts")

        self.text.backward(self.vision_acts)

        self.update_tracing("bw.text.ends")
        self.update_tracing("bw.vision.starts")

        self.vision.backward(self.text_acts)

        self.update_tracing("bw.vision.ends")
        self.update_tracing("upd.ends")  # [TODO]
        self.update_tracing("bw.ends")
        self.update_tracing("end")
