import logging
import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Optional
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
    num_layers: int = 2
    layer_size: int = 10  # The layer is a square.
    # Training config.
    dtype: torch.dtype = torch.float32
    it: int = 20
    lr: int = 5e-4
    # Distributed config.
    num_actors: int = 2


CONFIG = Config()


def set_seed(seed):
    """Set the RNG seeds to get deterministic output."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)


class DDPModel(ABC):
    """A layered model that uses distributed data parallel."""

    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self._device = torch_utils.get_devices()[0]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def train(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        input = self.start_train(X)
        for i in range(self.num_layers):
            input = self.forward(i, input)
        pred = input
        grad = self.loss(pred, Y)
        updates = []
        for i in reversed(range(self.num_layers)):
            grad, grad_update = self.backward(i, grad)
            updates.append(self.update(i, grad_update))
        return self.finish_train(*updates)

    @abstractmethod
    def start_train(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        raise NotImplementedError

    @abstractmethod
    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def get_grad_to_reduce(
        self, grad: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _, reduce_grad = grad
        return reduce_grad


class Model(torch.nn.Module):
    """A model that is a chain of (linear, relu) layers."""

    def __init__(
        self, layer_size: int, num_layers: int, device: torch.device, dtype: torch.dtype
    ):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        super(Model, self).__init__()

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
        self.lr: float = CONFIG.lr
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

    def update_layer(
        self, grad: torch.Tensor, layer_idx: int, return_weight: bool = True
    ) -> torch.Tensor:
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        [param for param in layer.parameters()][0].grad = grad
        optimizer = self.optimizers[layer_idx]
        optimizer.step()
        return layer.weight if return_weight else None


@ray.remote
class TorchDDPModel(DDPModel):
    """An actor class wrapper around the pytorch model."""

    def __init__(self, num_layers: int, layer_size: int):
        super().__init__(num_layers)

        self._model: Model = Model(layer_size, num_layers, self._device, torch.float32)

    def start_train(self, x: torch.Tensor) -> torch.Tensor:
        self._model.zero_grad()
        self._model.inputs = []
        self._model.activations = []
        self._model.loss = None
        self._model.pred = None
        return x.to(self._device)

    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self._device)
        return self._model.forward_layer(input, layer_idx)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        y = y.to(self._device)
        pred = pred.to(self._device)
        loss: torch.Tensor = self._model.criterion(pred, y)
        loss.backward(retain_graph=True, inputs=[pred])
        return pred.grad, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bp_grad, _ = grad
        bp_grad = bp_grad.to(self._device)
        return self._model.backward_layer(bp_grad, layer_idx)

    def update(
        self, layer_idx: int, grad: torch.Tensor, return_weight: bool = True
    ) -> torch.Tensor:
        grad = grad.to(self._device)
        return self._model.update_layer(grad, layer_idx)

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        return updates


def generate_x_y(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
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


def print_elapses(elapses, desc):
    """Print individual elapses and their average."""
    for i, elapse in enumerate(elapses):
        print(f"{desc} #{i} elapse={elapse}")
    total = sum(elapses)
    avg = total / len(elapses)
    print(f"{desc} avg: {avg}")
    total -= elapses[0]
    avg = total / (len(elapses) - 1)
    print(f"{desc} avg w/o 1st iter: {avg}")


def measure_ray_perf(model: Type[DDPModel]):
    """
    Measure the performance of Ray DDP with aDAG and allreduce.
    For performance, only serialize the tensors to be allreduced (WIP).
    """
    actor_cls = model.options(num_gpus=1)
    num_layers, layer_size = CONFIG.num_layers, CONFIG.layer_size
    num_actors = CONFIG.num_actors
    actors = [actor_cls.remote(num_layers, layer_size) for _ in range(num_actors)]

    with InputNode() as inp:
        losses = []
        for i, actor in enumerate(actors):
            x = inp[i]
            y = inp[num_actors + i]
            start = actor.start_train.bind(x)
            forwards = [start]
            for j in range(num_layers):
                forwards.append(actor.forward.bind(j, forwards[-1]))
            loss = actor.loss.bind(forwards[-1], y)
            losses.append(loss)
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
                actor.update.bind(j, reduced_grad, False)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            for update in updates:
                output.append(update)
        dag = MultiOutputNode(output)

    compiled_dag = dag.experimental_compile()
    it = CONFIG.it
    results = []
    elapses = []
    for i in range(it):
        x, y = generate_x_y(CONFIG)
        xs = torch.tensor_split(x, num_actors)
        ys = torch.tensor_split(y, num_actors)
        start = time.perf_counter()
        ref = compiled_dag.execute(*xs, *ys)
        result = ray.get(ref)
        end = time.perf_counter()
        results.append(result)
        elapse = end - start
        elapses.append(elapse)

    print_elapses(elapses, "ray")
    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)


def deduplicate_ray_results(results: List[List[Tuple[torch.Tensor]]]):
    """
    Check that the model is consistent across all ranks after each iteration
    of training. Deduplicate the results after the checks.

    Args:
        results: Results from running Ray DDP. `results[i]` is the weights
            across all actors for the ith iteration.
    """
    deduplicated = []
    for result in results:
        part_len = len(result[0])
        for i in range(1, CONFIG.num_actors):
            for j in range(part_len):
                assert torch.equal(result[0][j], result[i][j])
        deduplicated.append(result[0])
    return deduplicated


def ray_inference(results):
    """Run inference with the model weights after training with Ray DDP."""
    device = "cpu"
    weights = results[-1]
    model = Model(CONFIG.layer_size, CONFIG.num_layers, device, CONFIG.dtype)
    for i in range(0, len(model.layers), 2):
        model.layers[i].weight = nn.Parameter(weights[i // 2].to(device))
    x, y = generate_x_y(CONFIG)
    print(f"x: {x}")
    print(f"y: {y}")
    pred = model(x)
    loss = model.criterion(pred, y)
    print(f"ray pred: {pred}")
    print(f"ray loss: {loss}")


def detach_all(results: List[List[torch.Tensor]]):
    """
    Detach all tensors in order to pass tensors through `torch.multiprocessing.spawn`.
    """
    detached = []
    for result in results:
        d = []
        for r in result:
            d.append(r.detach())
        detached.append(d)
    return detached


def compare_results(actual, expected, desc):
    """
    Compare the actual and expected results.
    The results are model weights after each iteration.
    [TODO] Figure out why weights are different.
    Maybe 1) compare loss/accuracy if weight difference exists but is small,
    or 2) rename to `compare_weights`. Same for other methods.
    """
    assert len(actual) == len(expected)
    max_diff_all = 0
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert len(a) == len(e)
        max_diff = 0
        for t_a, t_e in zip(a, e):
            t_a = t_a.to("cpu")
            t_e = t_e.to("cpu")
            diff = torch.max(torch.abs(t_a - t_e).flatten())
            if diff > max_diff:
                max_diff = diff
        print(f"{desc} it #{i} max diff: {max_diff}")
        if max_diff > max_diff_all:
            max_diff_all = max_diff
    print(f"{desc} max diff across its: {max_diff_all}")


def check_correctness(results) -> None:
    """Compare Ray results with pytorch basic and pytorch.distributed results."""
    results = deduplicate_ray_results(results)
    ray_inference(results)
    auto = check_correctness_torch_basic()
    compare_results(results, auto, "ray vs auto")
    results = detach_all(results)
    auto = detach_all(auto)
    check_correctness_torch_distributed(auto, results)


def check_correctness_ray(model: Type[DDPModel]):
    """
    Check the correctness of DDP using aDAG and allreduce in Ray.
    Return the updated weights after each iteration.
    """
    actor_cls = model.options(num_gpus=1)
    num_layers, layer_size = CONFIG.num_layers, CONFIG.layer_size
    num_actors = CONFIG.num_actors
    actors = [actor_cls.remote(num_layers, layer_size) for _ in range(num_actors)]

    with InputNode() as inp:
        losses = []
        for i, actor in enumerate(actors):
            x = inp[i]
            y = inp[num_actors + i]
            start = actor.start_train.bind(x)
            forwards = [start]
            for j in range(num_layers):
                forwards.append(actor.forward.bind(j, forwards[-1]))
            loss = actor.loss.bind(forwards[-1], y)
            losses.append(loss)
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
    it = CONFIG.it
    results = []
    for i in range(it):
        x, y = generate_x_y(CONFIG)
        xs = torch.tensor_split(x, num_actors)
        ys = torch.tensor_split(y, num_actors)
        ref = compiled_dag.execute(*xs, *ys)
        result = ray.get(ref)
        results.append(result)

    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)

    return results


def check_correctness_torch_basic():
    """Run basic pytorch without DDP and return the weights."""
    device = "cuda:0"
    model = Model(CONFIG.layer_size, CONFIG.num_layers, device, CONFIG.dtype)
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    results = []
    elapses = []
    for i in range(CONFIG.it):
        x, y = generate_x_y(CONFIG)
        x = x.to(device)
        y = y.to(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(pred, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        result = []
        for i in range(0, len(model.layers), 2):
            layer: torch.nn.Linear = model.layers[i]
            result.append(torch.clone(layer.weight))
        results.append(result)

        elapse = end - start
        elapses.append(elapse)

    print_elapses(elapses, "auto")

    x, y = generate_x_y(CONFIG)
    x = x.to(device)
    y = y.to(device)
    model = Model(CONFIG.layer_size, CONFIG.num_layers, device, CONFIG.dtype).to(device)
    for i in range(0, len(model.layers), 2):
        model.layers[i].weight = nn.Parameter(results[-1][i // 2])
    pred = model(y)
    loss = model.criterion(pred, y)
    print(f"auto pred: {pred}")
    print(f"auto loss: {loss}")

    return results


def check_correctness_torch_distributed(auto_res, ray_res):
    """Run DDP with `torch.distributed`. Check and compare the results."""
    n_gpus = torch.cuda.device_count()
    assert (
        n_gpus >= CONFIG.num_actors
    ), f"Requires at least {CONFIG.num_actors} GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    torch_dist_spawn(torch_dist_run, world_size, auto_res, ray_res)


def torch_dist_spawn(demo_fn, world_size, auto_res, ray_res):
    mp.spawn(
        demo_fn, args=(world_size, auto_res, ray_res), nprocs=world_size, join=True
    )


def torch_dist_run(rank, world_size, auto_res, ray_res):
    """Run DDP with `torch.distributed`."""
    print(f"Running basic DDP example on rank {rank}.")
    torch_dist_setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Model(
        CONFIG.layer_size, CONFIG.num_layers, f"cuda:{rank}", CONFIG.dtype
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = model.criterion
    optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)

    num_actors = CONFIG.num_actors

    results = []
    elapses = []
    for i in range(CONFIG.it):
        x, y = generate_x_y(CONFIG)
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

        result = []
        for i in range(0, len(model.layers), 2):
            layer: torch.nn.Linear = model.layers[i]
            result.append(torch.clone(layer.weight))
        results.append(result)

        elapse = end - start
        elapses.append(elapse)

    print_elapses(elapses, f"dist #{rank}")

    x, y = generate_x_y(CONFIG)
    x = x.to(rank)
    y = y.to(rank)
    model = Model(
        CONFIG.layer_size, CONFIG.num_layers, f"cuda:{rank}", CONFIG.dtype
    ).to(rank)
    for i in range(0, len(model.layers), 2):
        model.layers[i].weight = nn.Parameter(results[-1][i // 2])
    pred = model(y)
    loss = loss_fn(pred, y)
    print(f"dist #{rank} pred: {pred}")
    print(f"dist #{rank} loss: {loss}")

    torch_dist_cleanup()
    compare_results(auto_res, results, "auto vs dist")
    compare_results(ray_res, results, "ray vs dist")


def torch_dist_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def torch_dist_cleanup():
    dist.destroy_process_group()


def main() -> None:
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < CONFIG.num_actors:
        print(f"Needs at least {CONFIG.num_actors} GPUs")
        return

    results = check_correctness_ray(TorchDDPModel)
    measure_ray_perf(TorchDDPModel)

    ray.shutdown()

    check_correctness(results)


if __name__ == "__main__":
    main()

# 0. cleanup
# 1. baseline (identify performance overhead)
# 2. multiple steps
