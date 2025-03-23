from typing import List, Optional, Tuple

import torch
from torch.nn.utils import parameters_to_vector


class BucketParameter(torch.nn.Module):
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.device = device
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    layer_size,
                    layer_size,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.relu_layers = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(num_layers)]
        )

        self.x = None
        self.y = None
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.linear_layers.parameters(),
            lr=1e-3,
        )

    def init_weights(self) -> None:
        with torch.no_grad():
            for layer in self.linear_layers:
                torch.nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def set_weights(self, weights: List[torch.Tensor]) -> None:
        for layer, weight in zip(self.linear_layers, weights):
            layer.weight.data = weight

    def fetch_weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach() for layer in self.linear_layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, relu in zip(self.linear_layers, self.relu_layers):
            y = linear(x)
            z = relu(y)
            x = z
        return z

    def backward(
        self,
        loss: Optional[torch.Tensor] = None,
        pred: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss is not None:
            assert pred is None
            loss.backward()
        elif pred is not None:
            assert grad is not None
            pred.backward(grad)

        grads_cat = parameters_to_vector(
            [layer.weight.grad for layer in self.linear_layers]
        )
        return grads_cat

    def update(self, grads_cat: torch.Tensor, grads_passed: bool) -> None:
        if grads_passed:
            offset = 0
            for layer in self.linear_layers:
                size = layer.weight.numel()
                grad = grads_cat[offset : offset + size].reshape(layer.weight.shape)
                layer.weight.grad = grad
                offset += size
            del grads_cat

        self.optimizer.step()
        self.optimizer.zero_grad()


class Shard(torch.nn.Module):
    # Simulate FSDP sharding.

    def __init__(
        self,
        model: torch.nn.Module,
        sharded_param: torch.Tensor,
        model_metadata: List[Tuple[torch.Size, int]],
    ) -> None:
        super().__init__()

        self.model = model
        self.sharded_param = torch.nn.Parameter(sharded_param, requires_grad=True)
        self.model_metadata = model_metadata
        self.optimizer = torch.optim.SGD([self.sharded_param], lr=1e-3)

    def set_flat_param(self, flat_param: torch.Tensor) -> None:
        _set_flat_param(self.model, flat_param, self.model_metadata)

    def get_flat_grad(self) -> torch.Tensor:
        flat_grad = parameters_to_vector(
            [param.grad for param in self.model.parameters()]
        )
        return flat_grad

    def free_peer_shards(self) -> None:
        _free_peer_shards(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def update(self, grad: torch.Tensor, grad_passed: bool) -> None:
        if grad_passed:
            self.sharded_param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()


def shard_model(model: torch.nn.Module, num_shards: int) -> List[Shard]:
    param = parameters_to_vector(model.parameters())
    padding = (num_shards - param.numel() % num_shards) % num_shards
    if padding > 0:
        param = torch.cat(
            [param, torch.zeros(padding, dtype=param.dtype, device=param.device)]
        )
    sharded_param_size = param.numel() // num_shards
    sharded_params = [
        param[i : i + sharded_param_size].reshape(-1)
        for i in range(0, param.numel(), sharded_param_size)
    ]
    model_metadata = [(param.shape, param.numel()) for param in model.parameters()]
    _free_peer_shards(model)
    shards = [
        Shard(model, sharded_param, model_metadata) for sharded_param in sharded_params
    ]
    return shards


def _set_flat_param(
    model: torch.nn.Module,
    flat_param: torch.Tensor,
    model_metadata: List[Tuple[torch.Size, int]],
) -> None:
    offset = 0
    for param, (shape, numel) in zip(model.parameters(), model_metadata):
        param.data = flat_param[offset : offset + numel].reshape(shape)
        offset += numel


def _free_peer_shards(model: torch.nn.Module) -> None:
    def get_first_param():
        for param in model.parameters():
            return param
        raise ValueError("Expected parameters")

    # Assume all parameters have the same dtype and device.
    first_param = get_first_param()
    dtype = first_param.dtype
    device = first_param.device
    empty_tensor = torch.empty(0, dtype=dtype, device=device)
    for param in model.parameters():
        param.data = empty_tensor
        param.grad = None


class LinearModel(torch.nn.Module):
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        num_units: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        if num_layers % num_units != 0:
            raise ValueError(f"{num_layers=} must be divisible by {num_units=}")

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.device = device
        self.buckets = torch.nn.ModuleList(
            [
                BucketParameter(
                    layer_size,
                    num_layers // num_units,
                    device,
                )
                for _ in range(num_units)
            ]
        )

        self.x = None
        self.y = None
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bucket in self.buckets:
            x = bucket(x)
        return x

    def init_weights(self) -> None:
        torch.manual_seed(2025)
        for bucket in self.buckets:
            bucket: BucketParameter
            bucket.init_weights()

    def fetch_weights(self) -> List[torch.Tensor]:
        weights = []
        for bucket in self.buckets:
            bucket: BucketParameter
            weights.extend(bucket.fetch_weights())
        return weights
