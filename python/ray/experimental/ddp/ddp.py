import logging
import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the demo DDP model."""

    # Model config.
    num_layers: int
    layer_size: int  # The layer is a square.
    # Training config.
    dtype: torch.dtype
    it: int
    lr: int
    # Distributed config.
    num_actors: int


SEED = 42


class LayeredModel(torch.nn.Module):
    """A model that is a chain of (linear, relu) layers."""

    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        lr: float,
    ):
        torch.manual_seed(SEED)

        super(LayeredModel, self).__init__()

        self.layers: List[torch.nn.Module] = []
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Linear(
                    layer_size, layer_size, device=device, dtype=dtype, bias=False
                )
            )
            self.layers.append(torch.nn.ReLU())
        self.layers: nn.ModuleList = nn.ModuleList(self.layers)
        self.inputs: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.lr: float = lr
        self.criterion = nn.MSELoss()
        self.optimizers: List[optim.SGD] = [
            optim.SGD(self.layers[2 * i].parameters(), lr=self.lr)
            for i in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        self.inputs.append(x)
        linear_layer: torch.nn.Linear = self.layers[2 * layer_idx]
        y: torch.Tensor = linear_layer(x)
        relu_activation: torch.nn.Module = self.layers[2 * layer_idx + 1]
        z: torch.Tensor = relu_activation(y)
        self.activations.append(z)
        return z

    def backward_layer(
        self, grad: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = self.activations[layer_idx]
        x: torch.Tensor = self.inputs[layer_idx]
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        W: torch.Tensor = layer.weight
        optimizer = self.optimizers[layer_idx]
        optimizer.zero_grad()
        z.backward(gradient=grad, retain_graph=True, inputs=[W, x])
        return x.grad, W.grad

    def update_layer(self, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        [param for param in layer.parameters()][0].grad = grad
        optimizer = self.optimizers[layer_idx]
        optimizer.step()
        return layer.weight


@ray.remote
class RayDDPWorker:
    """An actor class wrapper around the pytorch model."""

    def __init__(
        self,
        num_layers: int,
        layer_size: int,
        world_size: int,
        dtype: torch.dtype,
        lr: float,
    ):
        self.num_layers = num_layers
        self.device = torch_utils.get_devices()[0]

        self.model: LayeredModel = LayeredModel(
            layer_size, num_layers, self.device, dtype, lr
        )
        self.world_size: int = world_size

    def start_train(self) -> None:
        self.model.zero_grad()
        self.model.inputs = []
        self.model.activations = []

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.start_train()
        x = x.to(self.device)
        y = y.to(self.device)
        for i in range(self.num_layers):
            x = self.model.forward_layer(x, i)
        pred = x
        loss: torch.Tensor = self.model.criterion(pred, y)
        loss.backward(retain_graph=True, inputs=[pred])
        return pred.grad, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bp_grad, _ = grad
        bp_grad = bp_grad.to(self.device)
        return self.model.backward_layer(bp_grad, layer_idx)

    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.to(self.device)
        grad /= self.world_size
        return self.model.update_layer(grad, layer_idx)

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        """
        Gathers all weights across different layers.

        Args:
            updates: A tuple of any number of updated weights.
        """
        return updates

    def get_grad_to_reduce(
        self, grad: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        When an allreduce binds a class method output with `num_returns > 1`,
        an error is thrown. This is a workaround.
        See: https://github.com/ray-project/ray/issues/48522
        """
        _, reduce_grad = grad
        return reduce_grad


def generate_input_output(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate input `x` and output `y` for training."""
    layer_size = config.layer_size
    num_actors = config.num_actors
    dtype = config.dtype

    shape = (num_actors * layer_size, layer_size)
    numel = shape[0] * shape[1]

    x = torch.arange(numel, dtype=dtype, requires_grad=True) * 0.1
    x = x.reshape(shape)
    y = torch.arange(numel, dtype=dtype) * 0.1
    y = y.reshape(shape)

    return x, y


def run_torch(config):
    """Run pytorch without DDP and return the weights."""
    device = "cuda:0"
    model = LayeredModel(
        config.layer_size, config.num_layers, device, config.dtype, config.lr
    )
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    torch_weights = []
    elapses = []
    for i in range(config.it):
        x, y = generate_input_output(config)
        x = x.to(device)
        y = y.to(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(pred, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        cur_iter_weights = []
        for i in range(0, len(model.layers), 2):
            layer: torch.nn.Linear = model.layers[i]
            cur_iter_weights.append(torch.clone(layer.weight))
        torch_weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    print_elapses(elapses, "torch")

    return torch_weights


def run_torch_ddp(config: Config):
    """Run DDP with `torch.distributed`."""
    n_gpus = torch.cuda.device_count()
    assert (
        n_gpus >= config.num_actors
    ), f"Requires at least {config.num_actors} GPUs to run, but got {n_gpus}"
    world_size = config.num_actors
    mp.set_start_method("spawn", force=True)
    with mp.Manager() as manager:
        return_dict = manager.dict()
        mp.spawn(
            torch_ddp_per_process,
            args=(world_size, return_dict, config),
            nprocs=world_size,
            join=True,
        )
        torch_ddp_weights = get_torch_ddp_weights_per_device(return_dict, world_size)
        return torch_ddp_weights


def torch_ddp_per_process(rank, world_size, return_dict, config: Config):
    """Run DDP with `torch.distributed` in 1 process. Check and compare the weights."""
    print(f"Running torch DDP on rank {rank}.")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # create model and move it to GPU with id rank
    model = LayeredModel(
        config.layer_size, config.num_layers, f"cuda:{rank}", config.dtype, config.lr
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = model.criterion
    optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)

    num_actors = config.num_actors

    torch_ddp_weights = []
    elapses = []
    for i in range(config.it):
        x, y = generate_input_output(config)
        xs = torch.tensor_split(x, num_actors)
        ys = torch.tensor_split(y, num_actors)
        x = xs[rank]
        y = ys[rank]
        x = x.to(rank)
        y = y.to(rank)
        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = ddp_model(x)
        labels = y
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        cur_iter_weights = []
        for i in range(0, len(model.layers), 2):
            layer: torch.nn.Linear = model.layers[i]
            cur_iter_weights.append(torch.clone(layer.weight))
        torch_ddp_weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    print_elapses(elapses, f"torch ddp #{rank}")

    dist.destroy_process_group()

    def detach_all(tensors_across_iters: List[List[torch.Tensor]]):
        """
        Detach all tensors in order to pass tensors through
        `torch.multiprocessing.spawn`.
        """
        all_iters_detached = []
        for single_iter_tensors in tensors_across_iters:
            detached = []
            for tensor in single_iter_tensors:
                detached.append(tensor.detach().cpu())
            all_iters_detached.append(detached)
        return all_iters_detached

    return_dict[rank] = detach_all(torch_ddp_weights)


def get_torch_ddp_weights_per_device(return_dict, world_size: int):
    weights_across_devices = list(dict(return_dict).values())
    assert len(weights_across_devices) == world_size
    weights_per_device = weights_across_devices[0]
    # Weights on each device.
    for i in range(1, world_size):
        cur_device_weights = weights_across_devices[i]
        assert len(weights_per_device) == len(cur_device_weights)
        # Weights per iteration on each device.
        for j in range(len(weights_per_device)):
            assert len(weights_per_device[j]) == len(cur_device_weights[j])
            # Weights per layer in each iteration.
            for k in range(len(weights_per_device[j])):
                assert torch.equal(weights_per_device[j][k], cur_device_weights[j][k])
    return weights_per_device


def run_adag_ddp(config: Config):
    """
    Run DDP using aDAG and allreduce in Ray. Return the updated weights after each
    iteration.
    """
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < config.num_actors:
        raise ValueError(f"Needs at least {config.num_actors} GPUs")

    actor_cls = RayDDPWorker.options(num_gpus=1)
    num_layers, layer_size = config.num_layers, config.layer_size
    num_actors = config.num_actors
    actors = [
        actor_cls.remote(num_layers, layer_size, num_actors, config.dtype, config.lr)
        for _ in range(num_actors)
    ]

    with InputNode() as inp:
        losses = []
        for i, actor in enumerate(actors):
            x = inp[i]
            y = inp[num_actors + i]
            losses.append(actor.forward.bind(x, y))
        output = []
        grads = losses
        for j in reversed(range(num_layers)):
            for i, actor in enumerate(actors):
                grads[i] = actor.backward.bind(j, grads[i])
            reduced_grads = allreduce.bind(
                [
                    actor.get_grad_to_reduce.bind(grads[i])
                    for i, actor in enumerate(actors)
                ]
            )
            updates = [
                actor.update.bind(j, reduced_grad)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            output.append(updates)
        ends = [
            actor.finish_train.bind(
                *[output[j][i] for j in reversed(range(num_layers))]
            )
            for i, actor in enumerate(actors)
        ]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()
    it = config.it
    adag_ddp_weights = []
    elapses = []
    for i in range(it):
        x, y = generate_input_output(config)
        xs = torch.tensor_split(x, num_actors)
        ys = torch.tensor_split(y, num_actors)
        x = time.perf_counter()
        ref = compiled_dag.execute(*xs, *ys)
        cur_iter_weights = ray.get(ref)
        end = time.perf_counter()
        adag_ddp_weights.append(cur_iter_weights)
        elapse = end - x
        elapses.append(elapse)

    print_elapses(elapses, "adag ddp")
    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    adag_ddp_weights = get_adag_ddp_weights_per_device(
        adag_ddp_weights, config.num_actors
    )
    return adag_ddp_weights


def get_adag_ddp_weights_per_device(
    weights: List[List[Tuple[torch.Tensor]]], num_actors: int
):
    """
    Check that the model is consistent across all ranks after each iteration
    of training. Deduplicate the weights after the checks.

    Args:
        weights: Weights from running aDAG DDP for a number of iterations.
            `weights[i]` is the weights across all actors for the ith iteration.
    """
    weights_per_device = []
    for single_iter_weights in weights:
        per_device_len = len(single_iter_weights[0])
        for i in range(1, num_actors):
            for j in range(per_device_len):
                assert torch.equal(
                    single_iter_weights[0][j], single_iter_weights[i][j]
                ), f"{single_iter_weights[0][j]} vs. {single_iter_weights[i][j]}"
        weights_per_device.append(single_iter_weights[0])
    return weights_per_device


def compare_weights(W1, W2, desc, allow_error=False):
    """
    Compare the weights after each iteration across different training approaches.

    Args:
        W1: Weights after each iteration from one approach.
        W2: Weights after each iteration from the other approach.
        desc: Description of approaches
        allow_error: Whether small errors are allowed.
    """
    assert len(W1) == len(W2)
    # w1, w2 are weights after a single iteration for the 1st and 2nd approaches,
    # respectively.
    max_diff = 0
    for w1, w2 in zip(W1, W2):
        assert len(w1) == len(w2)
        # t1, t2 are weights of a single layer.
        for t1, t2 in zip(w1, w2):
            t1 = t1.to("cpu")
            t2 = t2.to("cpu")
            if not allow_error:
                assert torch.allclose(
                    t1, t2
                ), f"{desc} max diff: {torch.max(torch.abs(t1 - t2).flatten())}"
            elif not torch.allclose(t1, t2):
                max_diff = max(max_diff, torch.max(torch.abs(t1 - t2).flatten()))

    if max_diff != 0:
        print(f"{desc} max diff: {max_diff}")


def print_elapses(elapses, desc):
    """Print individual elapses and their average."""

    def s_to_us(seconds):
        """Converts seconds to microseconds (rounded)."""
        return round(seconds * 1e6)

    for i, elapse in enumerate(elapses):
        print(f"{desc} #{i} elapse={s_to_us(elapse)} us")
    total = sum(elapses)
    avg = total / len(elapses)
    print(f"{desc} avg: {s_to_us(avg)} us")
    total -= elapses[0]
    avg = total / (len(elapses) - 1)
    print(f"{desc} avg w/o 1st iter: {s_to_us(avg)} us")


def main(config: Config) -> None:
    """Compare aDAG DDP weights with pytorch and pytorch DDP weights."""
    adag_ddp_weights = run_adag_ddp(config)
    torch_weights = run_torch(config)
    compare_weights(
        adag_ddp_weights, torch_weights, "adag ddp vs torch", allow_error=True
    )
    torch_ddp_weights = run_torch_ddp(config)
    compare_weights(adag_ddp_weights, torch_ddp_weights, "adag ddp vs torch ddp")


def parse_config() -> Config:
    str_to_dtype = {
        "float32": torch.float32,
        "float": torch.float,  # alias for float32
        "float64": torch.float64,
        "double": torch.double,  # alias for float64
        "float16": torch.float16,
        "half": torch.half,  # alias for float16
    }
    parser = argparse.ArgumentParser(
        description="DDP demo (aDAG DDP vs torch vs torch DDP)"
    )
    parser.add_argument("--num-layers", type=int, default=2, help="number of layers")
    parser.add_argument(
        "--layer-size",
        type=int,
        default=10,
        help="size of a layer (each layer is a square)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(str_to_dtype.keys()),
        default="float32",
        help="data type of tensors",
    )
    parser.add_argument("--it", type=int, default=20, help="number of iterations")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--num-actors", type=int, default=2, help="number of actors")
    args = parser.parse_args()
    config = Config(
        args.num_layers,
        args.layer_size,
        str_to_dtype[args.dtype],
        args.it,
        args.lr,
        args.num_actors,
    )
    return config


if __name__ == "__main__":
    main(parse_config())


# 1. baseline (identify performance overhead)
# 2. profile
# 3. y axis: throughput/iter; x axis: increasing model size (keeping layers constant)
