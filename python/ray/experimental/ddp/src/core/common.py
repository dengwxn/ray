import csv
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds.
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
        mean: Elapse mean after first iteration.
    """

    # logger = logging.getLogger(__name__)
    # logger.info(header)
    # for i, elapse in enumerate(elapses):
    #     if rank is not None:
    #         logger.info(
    #             f"Iteration: {i}, rank: {rank}, elapse: {secs_to_micros(elapse)} us"
    #         )
    #     else:
    #         logger.info(f"Iteration: {i}, elapse: {secs_to_micros(elapse)} us")
    mean = sum(elapses[1:]) / (len(elapses) - 1)
    mean = secs_to_micros(mean)
    # logger.info(f"Elapse mean after iteration 0: {mean} us")
    return mean


def log_ray_elapses(
    actors_to_elapses: List[Dict[str, Any]],
    output_path: str,
    warmup: float = 0.2,
) -> None:
    metrics = [
        "total",
        "fw.total",
        "loss.compute",
        "loss.backward",
        "bw.total",
        "bw.backward",
        "bw.allreduce",
        "bw.update",
    ]

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each actor's data
    for idx, metrics_to_elapses in enumerate(actors_to_elapses):
        metrics_to_elapses = metrics_to_elapses[int(warmup * len(metrics_to_elapses)) :]
        filename = f"{output_path}/actor_{idx}.csv"

        # Calculate statistics for each metric
        total_mean = (
            np.mean(metrics_to_elapses["total"]) if metrics_to_elapses["total"] else 0
        )

        # Write statistics to CSV file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "mean", "std", "cv", "percent"])

            for metric in metrics:
                values = np.array(metrics_to_elapses[metric])
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean * 100 if mean > 0 else 0
                percent = (mean / total_mean * 100) if total_mean > 0 else 0

                writer.writerow(
                    [
                        metric,
                        round(mean),
                        round(std),
                        round(cv, 1),
                        round(percent, 1),
                    ]
                )


def log_torch_ddp_elapses(
    ranks_to_elapses: List[Dict[str, Any]],
    output_path: str,
    warmup: float = 0.2,
) -> None:
    metrics = [
        "total",
        "fw.total",
        "loss.compute",
        "bw.bw_ar",
        "bw.update",
    ]

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each actor's data
    for idx, metric_values in enumerate(ranks_to_elapses):
        for metric in metrics:
            assert metric in metric_values
            metric_values[metric] = metric_values[metric][
                int(warmup * len(metric_values[metric])) :
            ]

        filename = f"{output_path}/rank_{idx}.csv"

        # Calculate statistics for each metric
        total_mean = np.mean(metric_values["total"]) if metric_values["total"] else 0

        # Write statistics to CSV file
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "mean", "std", "cv", "percent"])

            for metric in metrics:
                values = np.array(metric_values[metric])
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean * 100 if mean > 0 else 0
                percent = (mean / total_mean * 100) if total_mean > 0 else 0

                writer.writerow(
                    [
                        metric,
                        round(mean),
                        round(std),
                        round(cv, 1),
                        round(percent, 1),
                    ]
                )
