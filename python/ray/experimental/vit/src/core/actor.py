import logging
import os

import numpy as np
import torch
import torch.nn as nn
from common import count_params, random_seed
from dist import BaseWorker
from model import TextEncoder, VisionEncoder, init_optimizer, parallelize_2d
from open_clip.loss import ClipLoss

import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class VisionWorker(BaseWorker):
    def __init__(self, model_name, dp_size, tp_size, text_dp_size, seed=123) -> None:
        super().__init__()
        random_seed(seed)
        self.model = VisionEncoder(model_name)
        self.num_params = count_params(self.model)
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.text_dp_size = text_dp_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )
        self.device_set = False

    def init_parallel_strategy(self):
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
        i, global_batch_size = inputs
        vision_batch_size = global_batch_size // self.dp_size

        torch.manual_seed(i)
        images = torch.randn(vision_batch_size, 3, 224, 224)
        return images

    def forward(self, inputs):
        if not self.device_set:
            torch.cuda.set_device(self.rank)
            self.device_set = True

        images = self.load_batch(inputs).to(device="cuda")
        with torch.autocast(device_type="cuda"):
            self.vision_features = self.model(images)

        return self.vision_features.detach()

    def backward(self, text_features):
        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                self.vision_features, text_features.cuda(), self.logit_scale
            )
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
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
        result = torch.chunk(activations, self.text_dp_size)
        return (*result,)

    def echo(self, input):
        return input


@ray.remote(num_gpus=1)
class TextWorker(BaseWorker):
    def __init__(self, model_name, dp_size, tp_size, vision_dp_size, seed=123) -> None:
        super().__init__()
        random_seed(seed)
        self.model = TextEncoder(model_name)
        self.num_params = count_params(self.model)
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.vision_dp_size = vision_dp_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_loss_fn = ClipLoss(cache_labels=True)
        self.optimizer = init_optimizer(
            list(self.model.named_parameters()) + [("logit_scale", self.logit_scale)]
        )
        self.device_set = False

    def init_parallel_strategy(self):
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
        i, global_batch_size = inputs
        text_batch_size = global_batch_size // self.dp_size

        torch.manual_seed(i)
        texts = torch.randint(0, 49408, (text_batch_size, 77))
        return texts

    def forward(self, inputs):
        if not self.device_set:
            torch.cuda.set_device(self.rank)
            self.device_set = True

        texts = self.load_batch(inputs).to(device="cuda")
        with torch.autocast(device_type="cuda"):
            self.text_features = self.model(texts)

        return self.text_features.detach()

    def backward(self, vision_features):
        if isinstance(vision_features, tuple):
            vision_features = vision_features[0]
        with torch.autocast(device_type="cuda"):
            self.loss = self.clip_loss_fn(
                vision_features.cuda(), self.text_features, self.logit_scale
            )
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
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
        result = torch.chunk(activations, self.vision_dp_size)
        return (*result,)

    def echo(self, input):
        return input
