import csv
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import ray
from ray.dag.compiled_dag_node import CompiledDAG


def get_start_time() -> float:
    return time.perf_counter()


def get_end_time(sync: bool = True) -> float:
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter()


def get_timing_event_torch() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds.
    """
    return round(secs * 1e6)


def millis_to_micros(millis: float) -> int:
    """
    Converts milliseconds to microseconds.
    """
    return round(millis * 1e3)


def log_elapses_to_csv(
    ranks_to_elapses: List[Dict[str, Any]],
    output_path: str,
    latency_prefix: str,
    metrics: List[str],
    aliases: Optional[List[str]] = None,
    warmup: float = 0.2,
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if aliases is None:
        aliases = [None] * len(metrics)

    # Process each rank's data
    for idx, metric_values in enumerate(ranks_to_elapses):
        output_file = f"{output_path}/{latency_prefix}_rank{idx}.csv"

        for metric in metrics:
            assert metric in metric_values
            metric_values[metric] = metric_values[metric][
                int(warmup * len(metric_values[metric])) :
            ]

        # Calculate statistics for each metric
        total_mean = (
            np.mean(metric_values["actor.total"]) if metric_values["actor.total"] else 0
        )

        # Write statistics to CSV file
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "mean", "std", "cv", "percent"])

            for metric, alias in zip(metrics, aliases):
                if alias is None:
                    alias = metric
                values = np.array(metric_values[metric])
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean * 100 if mean > 0 else 0
                percent = (mean / total_mean * 100) if total_mean > 0 else 0

                writer.writerow(
                    [
                        alias,
                        round(mean),
                        round(std),
                        round(cv, 1),
                        round(percent, 1),
                    ]
                )


def generate_1f1b_dag(
    workers: List[Any],
    num_microbatches: int,
    num_lead_microbatches: int,
) -> CompiledDAG:
    num_workers = len(workers)
    with ray.dag.InputNode() as inp:
        fwd_queues = [[] for _ in range(num_workers)]
        bwd_queues = [[] for _ in range(num_workers)]
        # Once a worker's counter reaches 0, it cannot execute another fwd until it
        # executes a bwd first.
        fwd_counter = [num_lead_microbatches - i for i in range(num_workers)]
        # All of the done batches.
        done = []

        # FWD on worker 0.
        # input_data = workers[0].read_input.bind(inp)
        # for i in range(num_microbatches):
        #     fwd_queues[0].append(input_data)
        for i in range(num_microbatches):
            fwd_queues[0].append(inp)

        while len(done) < num_microbatches:
            for i, worker in enumerate(workers):
                if fwd_counter[i] > 0 and fwd_queues[i]:
                    b = fwd_queues[i].pop(0)
                    b = worker.forward_pp.bind(b)
                    if i < num_workers - 1:
                        fwd_queues[i + 1].append(b)
                        # Use NCCL channel for communication between workers.
                        b.with_tensor_transport(transport="nccl")
                    else:
                        bwd_queues[i].append(b)
                    fwd_counter[i] -= 1
                elif bwd_queues[i]:
                    b = bwd_queues[i].pop(0)
                    b = worker.backward_pp.bind(b)
                    if i > 0:
                        bwd_queues[i - 1].append(b)
                        # Use NCCL channel for communication between workers.
                        b.with_tensor_transport(transport="nccl")
                    else:
                        done.append(b)
                    fwd_counter[i] += 1
        dag = ray.dag.MultiOutputNode(done)
    compiled_dag = dag.experimental_compile()
    return compiled_dag
