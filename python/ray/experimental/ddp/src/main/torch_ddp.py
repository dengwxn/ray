import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ..core.common import secs_to_micros
from ..core.config import parse_args
from ..core.mp.actor import ModelElement
from ..core.torch_ddp import run_torch_ddp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def run_torch_ddp(
    args: Dict[str, Any]
) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= args["num_actors"]
    world_size = args["num_actors"]

    mp.set_start_method("spawn", force=True)

    # Use a multiprocessing manager to share data across devices.
    with mp.Manager() as manager:
        ranks_to_elapses = manager.dict()

        mp.spawn(
            spwan_torch_ddp,
            args=(world_size, ranks_to_elapses, args),
            nprocs=world_size,
            join=True,
        )


def spwan_torch_ddp(
    rank: int,
    world_size: int,
    ranks_to_elapses: Dict[int, int],
    args: Dict[str, Any],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    try:
        logger = logging.getLogger(__name__)

        # Initialize the process group.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Create model on the device.
        device = f"cuda:{rank}"
        model = ModelElement(
            args["layer_size"],
            args["num_layers"],
            device,
        )
        model = model.to(model.device)
        ddp_model = DDP(model, device_ids=[rank])

        elapses = defaultdict(list)

        torch.manual_seed(998244353)
        model.init_weights()
        model = model.to(model.device)

        for it in range(args["num_epochs"]):
            if rank == 0:
                logger.info(f"Start iteration {it}...")

            model.x = torch.randn(
                1,
                model.layer_size,
                requires_grad=True,
            ).to(
                model.device,
            )
            model.y = torch.randn(
                1,
                model.layer_size,
            ).to(
                model.device,
            )

            dist.barrier()
            start = time.perf_counter()

            forward_start = time.perf_counter()
            pred = ddp_model(model.x)
            forward_end = time.perf_counter()

            loss_compute_start = time.perf_counter()
            loss = model.criterion(pred, model.y)
            loss_compute_end = time.perf_counter()

            backward_start = time.perf_counter()
            loss.backward()
            backward_end = time.perf_counter()

            update_start = time.perf_counter()
            model.optimizer.step()
            model.optimizer.zero_grad()
            update_end = time.perf_counter()

            barrier_start = time.perf_counter()
            dist.barrier()
            end = time.perf_counter()

            if rank == 0:
                logger.info(f"Finish iteration {it}")
            total = end - start

            def log(key: str, elapse: float):
                elapses[key].append(secs_to_micros(elapse))
                logger.info(
                    f"rank: {rank}, {key} elapse: {secs_to_micros(elapse)} us, percent: {round(elapse / total * 100, 1)}%"
                )

            log("total", total)
            log("fw.total", forward_end - forward_start)
            log("loss.compute", loss_compute_end - loss_compute_start)
            log("bw.bw_ar", backward_end - backward_start)
            log("bw.update", update_end - update_start)
            log("barrier", end - barrier_start)
    finally:
        # Destroy the process group.
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses


if __name__ == "__main__":
    args = parse_args()
    run_torch_ddp(args)
