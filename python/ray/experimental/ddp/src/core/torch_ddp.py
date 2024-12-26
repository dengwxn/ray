import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import (
    generate_input_output,
    log_elapses,
    log_torch_ddp_elapses,
    secs_to_micros,
)
from .config import Config
from .correctness import get_torch_ddp_weights
from .model import LayeredModel


def run_torch_ddp(cfg: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run PyTorch DDP.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked,
        and the average elapse across all iterations.
    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= cfg.num_actors
    world_size = cfg.num_actors

    mp.set_start_method("spawn", force=True)

    # Use a multiprocessing manager to share data across devices.
    with mp.Manager() as manager:
        ranks_to_weights = None
        if cfg.check_correctness:
            ranks_to_weights = manager.dict()
        ranks_to_elapses = manager.dict()

        mp.spawn(
            spwan_torch_ddp,
            args=(world_size, ranks_to_weights, ranks_to_elapses, cfg),
            nprocs=world_size,
            join=True,
        )

        if cfg.check_correctness:
            ranks_to_weights_clone = dict(ranks_to_weights)
        ranks_to_elapses_clone = dict(ranks_to_elapses)

    log_torch_ddp_elapses(
        list(ranks_to_elapses_clone.values()), cfg.output_path, cfg.output_prefix
    )

    weights = None
    if cfg.check_correctness:
        weights = get_torch_ddp_weights(ranks_to_weights_clone, world_size)
    elapse = max(
        log_elapses(
            elapses["total"],
            f"Running torch ddp on rank {rank}...",
        )
        for rank, elapses in ranks_to_elapses_clone.items()
    )

    return weights, elapse


def spwan_torch_ddp(
    rank: int,
    world_size: int,
    ranks_to_weights: Optional[Dict[int, List[List[torch.Tensor]]]],
    ranks_to_elapses: Dict[int, int],
    cfg: Config,
) -> None:
    """
    Spawn a PyTorch DDP process.

    Args:
        rank: Rank of the process.
        world_size: Number of processes.
        ranks_to_weights: Weights of all layers after each iteration across
            all processes.
        ranks_to_elapses: Elapses of all iterations across all processes.
        cfg: Model and training configurations. If correctness is checked,
            ranks_to_weights is not None and will be updated.
    """

    if cfg.check_correctness:
        assert ranks_to_weights is not None

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    try:
        logger = logging.getLogger(__name__)

        # Initialize the process group.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Create model on the device.
        device = f"cuda:{rank}"
        model = LayeredModel(
            cfg.layer_size,
            cfg.num_layers,
            device,
            cfg.dtype,
            cfg.learning_rate,
        )
        ddp_model = DDP(model, device_ids=[rank])
        optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)

        weights = None
        if cfg.check_correctness:
            weights: List[List[torch.Tensor]] = []
        elapses = defaultdict(list)

        for it in range(cfg.num_iters):
            if rank == 0:
                logger.info(f"Start iteration {it}...")

            x, y = generate_input_output(cfg)
            x = torch.tensor_split(x, cfg.num_actors)[rank].to(rank)
            y = torch.tensor_split(y, cfg.num_actors)[rank].to(rank)

            dist.barrier()
            start = time.perf_counter()
            optimizer.zero_grad()

            forward_start = time.perf_counter()
            pred: torch.Tensor = ddp_model(x)
            forward_end = time.perf_counter()

            loss_compute_start = time.perf_counter()
            loss: torch.Tensor = model.criterion(pred, y)
            loss_compute_end = time.perf_counter()

            backward_start = time.perf_counter()
            loss.backward()
            backward_end = time.perf_counter()

            update_start = time.perf_counter()
            optimizer.step()
            update_end = time.perf_counter()

            dist.barrier()
            end = time.perf_counter()

            if cfg.check_correctness:
                iter_weights: List[torch.Tensor] = []
                for i in range(0, len(model.layers), 2):
                    layer: torch.nn.Linear = model.layers[i]
                    iter_weights.append(torch.clone(layer.weight))
                weights.append(iter_weights)

            if rank == 0:
                logger.info(f"Finish iteration {it}")
            total = end - start

            def log(key: str, elapse: float):
                elapses[key].append(secs_to_micros(elapse))
                if rank == 0:
                    logger.info(
                        f"{key} elapse: {secs_to_micros(elapse)} us, percent: {round(elapse / total * 100, 1)}%"
                    )

            log("total", total)
            log("fw.total", forward_end - forward_start)
            log("loss.compute", loss_compute_end - loss_compute_start)
            log("bw.bw_ar", backward_end - backward_start)
            log("bw.update", update_end - update_start)
    finally:
        # Destroy the process group.
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses

    if cfg.check_correctness:
        ranks_to_weights[rank] = detach(weights)


def detach(
    weights: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Detach all tensors in order to pass tensors across devices. If a tensor is
    not detached, serialization will fail.

    Args:
        weights: Weights of all layers across all iterations.

    Returns:
        Detached weights of all layers across all iterations.
    """
    return [
        [tensor.detach().cpu() for tensor in iter_weights] for iter_weights in weights
    ]
