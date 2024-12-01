import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
from .core.actor import RayDDPWorker
from .core.common import generate_input_output, secs_to_micros
from .core.config import Config, parse_config
from .core.correctness import (
    compare_weights,
    get_ray_ddp_weights_per_device,
    get_torch_ddp_weights_per_device,
)
from .core.model import LayeredModel
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


def run_torch(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run PyTorch without DDP.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked, and
        the average elapse across all iterations.
    """

    # To ensure that the model parameters are initialized in the same way across
    # different training methods (PyTorch without DDP, PyTorch with DDP, and Ray
    # with DDP), the model must be initialized on GPU.
    device = "cuda:0"
    model = LayeredModel(
        config.layer_size,
        config.num_layers,
        device,
        config.dtype,
        config.learning_rate,
    )
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    weights: Optional[List[List[torch.Tensor]]] = None
    if config.check_correctness:
        weights = []
    elapses: List[float] = []
    for i in range(config.num_iters):
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

        if config.check_correctness:
            cur_iter_weights: List[torch.Tensor] = []
            for i in range(0, len(model.layers), 2):
                layer: torch.nn.Linear = model.layers[i]
                cur_iter_weights.append(torch.clone(layer.weight))
            weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    avg_elapse = print_elapses(elapses, "torch")
    return weights, avg_elapse


def run_torch_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run DDP with PyTorch.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked,
        and the average elapse across all iterations.
    """
    n_gpus = torch.cuda.device_count()
    assert (
        n_gpus >= config.num_actors
    ), f"Requires at least {config.num_actors} GPUs to run, but got {n_gpus}"
    world_size = config.num_actors

    mp.set_start_method("spawn", force=True)
    # Use a multiprocessing manager to share data across devices.
    with mp.Manager() as manager:
        weights_dict = None
        if config.check_correctness:
            weights_dict = manager.dict()
        elapses_dict = manager.dict()
        # [TODO] Is spawn the only way for multiprocessing?
        mp.spawn(
            run_torch_ddp_per_process,
            args=(world_size, weights_dict, elapses_dict, config),
            nprocs=world_size,
            join=True,
        )
        weights = None
        if config.check_correctness:
            weights = get_torch_ddp_weights_per_device(weights_dict, world_size)
        avg_elapse = get_torch_ddp_e2e_elapse(elapses_dict)
        return weights, avg_elapse


def run_torch_ddp_per_process(
    rank: int,
    world_size: int,
    weights_dict: Optional[Dict[int, List[List[torch.Tensor]]]],
    elapses_dict: Dict[int, int],
    config: Config,
) -> None:
    """
    Run DDP with PyTorch in one process.

    Args:
        rank: The rank of this process.
        world_size: The total number of processes.
        weights_dict: A dictionary to store all weights of this process.
        elapses_dict: A dictionary to store the average elapse of this process.
        config: Configurations. If config.check_correctness, store the weights
            into weights_dict.
    """

    if config.check_correctness:
        assert weights_dict is not None

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    # Initialize the process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model on GPU with id rank.
    model = LayeredModel(
        config.layer_size,
        config.num_layers,
        f"cuda:{rank}",
        config.dtype,
        config.learning_rate,
    )
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = model.criterion
    optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)

    num_actors = config.num_actors

    weights: Optional[List[List[torch.Tensor]]] = None
    if config.check_correctness:
        weights = []
    elapses = []
    for i in range(config.num_iters):
        x, y = generate_input_output(config)
        x = torch.tensor_split(x, num_actors)[rank].to(rank)
        y = torch.tensor_split(y, num_actors)[rank].to(rank)
        start = time.perf_counter()
        optimizer.zero_grad()
        prediction: torch.Tensor = ddp_model(x)
        loss: torch.Tensor = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        if config.check_correctness:
            cur_iter_weights = []
            for i in range(0, len(model.layers), 2):
                layer: torch.nn.Linear = model.layers[i]
                cur_iter_weights.append(torch.clone(layer.weight))
            weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    avg_elapse = print_elapses(elapses, f"torch ddp rank: {rank}", rank)

    # Destroy the process group.
    dist.destroy_process_group()

    def detach_all(
        weights_across_iters: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """
        Detach all tensors in order to pass tensors across devices.
        If a tensor is not detached, serialization will fail.

        Args:
            weights_across_iters: Weights of all layers across all iterations.

        Returns:
            Detached weights of all layers across all iterations.
        """
        all_iters_detached: List[List[torch.Tensor]] = []
        for single_iter_tensors in weights_across_iters:
            detached: List[torch.Tensor] = []
            for tensor in single_iter_tensors:
                detached.append(tensor.detach().cpu())
            all_iters_detached.append(detached)
        return all_iters_detached

    if config.check_correctness:
        weights_dict[rank] = detach_all(weights)
    elapses_dict[rank] = avg_elapse


def get_torch_ddp_e2e_elapse(elapses_dict: Dict[int, int]) -> int:
    """
    Get the end-to-end elapse for all devices. The maximum elapse across all
    devices is used to approximate the end-to-end elapse.

    Args:
        elapses_dict: Dictionary that maps ranks (device ids) to its average
            elapse across iterations.

    Returns:
        The approximate end-to-end elapse for all devices.
    """
    return max(elapses_dict.values())


def run_ray_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run DDP using compiled graph and allreduce in Ray.

    Args:
        config: Model and training configurations.

    Returns:
        Per-device weights of all layers after each iteration if correctness is checked,
        and the average end-to-end elapse.
    """
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < config.num_actors:
        raise ValueError(f"Needs at least {config.num_actors} GPUs")

    actor_cls = RayDDPWorker.options(num_gpus=1)
    num_layers, layer_size = config.num_layers, config.layer_size
    num_actors = config.num_actors
    actors = [
        actor_cls.remote(
            num_layers,
            layer_size,
            num_actors,
            config.dtype,
            config.learning_rate,
            config.check_correctness,
            config.check_breakdown,
        )
        for _ in range(num_actors)
    ]

    with InputNode() as inp:
        grads = [actor.forward.bind(inp) for actor in actors]
        output = []
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

    x, y = generate_input_output(config)
    xs = torch.tensor_split(x, num_actors)
    ys = torch.tensor_split(y, num_actors)
    move_tensor_refs = [
        actor.tensor_to_device.remote(xs[i], ys[i]) for i, actor in enumerate(actors)
    ]
    ray.get(move_tensor_refs)

    weights = None
    if config.check_correctness:
        weights = []
    elapses = []
    for i in range(config.num_iters):
        start = time.perf_counter()
        # Use None as a placeholder.
        ref = compiled_dag.execute(None)
        # [TODO] Print timestamp before ray.get.
        # If correctness is not checked, the result is None.
        cur_iter_weights = ray.get(ref)
        end = time.perf_counter()
        if config.check_correctness:
            weights.append(cur_iter_weights)
        elapse = end - start
        elapses.append(elapse)

    avg_elapse = print_elapses(elapses, "ray ddp")
    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    if config.check_correctness:
        weights = get_ray_ddp_weights_per_device(weights, config.num_actors)
    return weights, avg_elapse


def print_elapses(elapses: List[float], name: str, rank: Optional[int] = None) -> int:
    """
    Print individual elapses and their average.

    Args:
        elapses: List of elapses for all iterations
        name: Name of the approach (Ray DDP, torch, or torch DDP).
        rank: Rank in torch DDP.

    Returns:
        avg: Average elapse without first iteration.
    """

    logger.info(name)
    for i, elapse in enumerate(elapses):
        if rank:
            logger.info(
                f"Iteration: {i}, rank: {rank}, elapse: {secs_to_micros(elapse)} us"
            )
        else:
            logger.info(f"Iteration: {i}, elapse: {secs_to_micros(elapse)} us")
    total = sum(elapses)
    avg = total / len(elapses)
    logger.info(f"Average elapse: {secs_to_micros(avg)} us")
    total -= elapses[0]
    avg = total / (len(elapses) - 1)
    avg = secs_to_micros(avg)
    logger.info(f"Average elapse without iteration 0: {avg} us")
    return avg


def main(config: Config) -> None:
    """
    Run and compare the performance of Ray DDP, PyTorch, and PyTorch DDP.
    Correctness of Ray DDP is checked if specified. Save the average elapses
    across iterations of all approaches to the output file.

    Args:
        config: Model and training configurations, as well as whether to check
            correctness and the output file path.
    """
    ray_ddp_weights, ray_ddp_elapse = run_ray_ddp(config)
    torch_weights, torch_elapse = run_torch(config)
    torch_ddp_weights, torch_ddp_elapse = run_torch_ddp(config)
    if config.check_correctness:
        compare_weights(
            ray_ddp_weights, torch_weights, "ray ddp vs torch", allow_error=True
        )
        compare_weights(ray_ddp_weights, torch_ddp_weights, "ray ddp vs torch ddp")
    with open(config.output_file, "w") as file:
        file.write("ray-ddp,torch,torch-ddp\n")
        file.write(f"{ray_ddp_elapse},{torch_elapse},{torch_ddp_elapse}\n")


if __name__ == "__main__":
    main(parse_config())
