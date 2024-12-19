import logging
from typing import List, Optional, Tuple

import torch

from .config import Config


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds (rounded).
    """
    return round(secs * 1e6)


def generate_input_output(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate input `x` and output `y` for training.

    Args:
        config: Model and training configurations.

    Returns:
        Input `x` and ground truth `y`.
    """
    layer_size = config.layer_size
    num_actors = config.num_actors
    dtype = config.dtype

    shape = (num_actors * layer_size, layer_size)
    numel = shape[0] * shape[1]

    x = torch.arange(numel, dtype=dtype, requires_grad=True) / numel
    x = x.reshape(shape)
    y = torch.arange(numel, dtype=dtype) / numel
    y = y.reshape(shape)

    return x, y


def log_elapses(elapses: List[float], header: str, rank: Optional[int] = None) -> int:
    """
    Log individual elapses and their average.

    Args:
        elapses: List of elapses for all iterations
        header: Header for the log.
        rank: Rank in torch DDP.

    Returns:
        avg: Average elapse without first iteration.
    """

    logger = logging.getLogger(__name__)
    logger.info(header)
    for i, elapse in enumerate(elapses):
        if rank is not None:
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
