import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import generate_input_output, print_elapses
from .config import Config
from .correctness import get_torch_ddp_weights
from .model import LayeredModel


def run_torch_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run PyTorch DDP.

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
            weights = get_torch_ddp_weights(weights_dict, world_size)
        max_elapse = max(elapses_dict.values())
        return weights, max_elapse


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
