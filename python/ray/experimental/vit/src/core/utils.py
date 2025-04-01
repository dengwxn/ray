import logging
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh


def get_logger():
    """Create and return a configured logger."""
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def random_seed(seed, rank=0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_clip_params(model):
    vision_params = count_params(model.visual)
    text_params = (
        count_params(model.transformer)
        + count_params(model.token_embedding)
        + count_params(model.ln_final)
    )
    return vision_params, text_params


def count_model_params(model):
    counts = Counter()
    for p in model.parameters():
        counts["n_params"] += 1
        counts["total_params"] += p.numel()
        if isinstance(p, DTensor):
            counts["n_dtensors"] += 1
            counts["total_dtensor_params"] += p.numel()
    return ", ".join([f"{k}: {v:,}" for k, v in counts.items()])


def master_log(dp_rank, tp_rank, msg, logger):
    """helper function to log only on tp_rank 0"""
    if tp_rank == 0:
        logger.info(f"[dp{dp_rank}-tp{tp_rank}] {msg}")


def rank_log(dp_rank, tp_rank, msg, logger):
    """helper function to log only on all ranks"""
    logger.info(f"[dp{dp_rank}-tp{tp_rank}] {msg}")


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """verification that we have at least 2 gpus to run dist examples"""
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    return has_cuda and gpu_count >= min_gpus
