from typing import List, Tuple

import torch

import ray
from .model import ModelElement


@ray.remote
class ModelActor:
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        num_models: int,
        device: torch.device,
    ):
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_models = num_models
        self.device = device

        self.models = [
            ModelElement(
                layer_size=layer_size,
                num_layers=num_layers // num_models,
                device=device,
            )
            for _ in range(num_models)
        ]
        self.intermediates: List[torch.Tensor, torch.Tensor] = []

    def init_weights(self) -> None:
        torch.manual_seed(998244353)
        for model in self.models:
            model.init_weights()
            model = model.to(model.device)

    def init_training(self) -> None:
        self.models[0].x = torch.randn(
            1,
            self.models[0].layer_size,
            requires_grad=True,
        ).to(
            self.models[0].device,
        )
        self.models[-1].y = torch.randn(
            1,
            self.models[-1].layer_size,
        ).to(
            self.models[-1].device,
        )

    def fetch_weights(self) -> List[torch.Tensor]:
        weights = []
        for model in self.models:
            weights.extend(model.fetch_weights())
        return weights

    def forward(self, _) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        self.intermediates = []
        input = self.models[0].x
        for i, model in enumerate(self.models):
            pred = model.forward(input)
            if i < len(self.models) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        return self.intermediates

    def backward(self, _, idx: int) -> torch.Tensor:
        if idx == len(self.models) - 1:
            loss = self.models[idx].criterion(
                self.intermediates[idx][0],
                self.models[idx].y,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
        grads = self.models[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        return grads

    def update(self, _, idx: int) -> None:
        # [TODO] Use the passed in grads to update the model.
        # 1. grads /= self.world_size
        # 2. grads -> reshape grads
        # 3. grads -> layer.weight.grad
        self.models[idx].update()
