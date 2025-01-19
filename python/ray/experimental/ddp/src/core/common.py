import csv
import os
from typing import Any, Dict, List

import numpy as np


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds.
    """
    return round(secs * 1e6)


def log_elapses_to_csv(
    ranks_to_elapses: List[Dict[str, Any]],
    output_path: str,
    latency_prefix: str,
    metrics: List[str],
    warmup: float = 0.2,
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each rank's data
    for idx, metric_values in enumerate(ranks_to_elapses):
        output_file = f"{output_path}/{latency_prefix}_rank{idx}.csv"

        for metric in metrics:
            assert metric in metric_values
            metric_values[metric] = metric_values[metric][
                int(warmup * len(metric_values[metric])) :
            ]

        # Calculate statistics for each metric
        total_mean = np.mean(metric_values["total"]) if metric_values["total"] else 0

        # Write statistics to CSV file
        with open(output_file, "w", newline="") as csvfile:
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
