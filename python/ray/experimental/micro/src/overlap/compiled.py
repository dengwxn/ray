import csv
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import fire
import torch

import ray
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def get_timing_event() -> torch.cuda.Event:
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def get_elapse_us(start: torch.cuda.Event, end: torch.cuda.Event) -> int:
    elapse_us = round(start.elapsed_time(end) * 1e3)
    return elapse_us


def get_avg(data: List[int]) -> int:
    avg = round(sum(data) / len(data))
    return avg


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, size_comp: int, num_comp: int, size_comm: int):
        self.device = torch.device("cuda:0")
        self.size_comp = size_comp
        self.num_comp = num_comp
        self.size_comm = size_comm

    def init_tensor(self, _):
        self.tensor_comp = torch.randn(1, self.size_comp, device=self.device)
        self.tensor_comm = torch.randn(1, self.size_comm, device=self.device)
        torch.cuda.synchronize()

    def init_tracing(self, _):
        self.ev_e2e_start = get_timing_event()

    def comp_tensor(self, _):
        self.ev_comp_tensor_start = get_timing_event()
        for _ in range(self.num_comp):
            self.tensor_comp += 1
        self.ev_comp_tensor_end = get_timing_event()
        return self.tensor_comp

    def comm_tensor(self, _):
        return self.tensor_comm

    def finish_tracing(self, _comp_tensor, _comm_tensor):
        self.ev_e2e_end = get_timing_event()
        torch.cuda.synchronize()
        elapses = dict()
        elapses["e2e"] = get_elapse_us(self.ev_e2e_start, self.ev_e2e_end)
        elapses["comp"] = get_elapse_us(
            self.ev_comp_tensor_start, self.ev_comp_tensor_end
        )
        return elapses


def benchmark_container(
    overlap: bool = False,
    num_iters: int = 50,
    size_comp: int = 1_00_000,
    num_comp: int = 12_000,
    size_comm: int = 1_000_000_000,
):
    ray.init()
    num_iters_warmup = int(num_iters * 0.2)
    actors = [Actor.remote(size_comp, num_comp, size_comm) for _ in range(2)]

    with InputNode() as inp:
        init_tensors = [actor.init_tensor.bind(inp) for actor in actors]
        init_tracings = [
            actor.init_tracing.bind(init_tensor)
            for actor, init_tensor in zip(actors, init_tensors)
        ]
        comm_tensors = [
            actor.comm_tensor.bind(init_tracing)
            for actor, init_tracing in zip(actors, init_tracings)
        ]
        comp_tensors = [
            actor.comp_tensor.bind(init_tracing)
            for actor, init_tracing in zip(actors, init_tracings)
        ]
        ar_tensors = allreduce.bind(comm_tensors)
        elapses = [
            actor.finish_tracing.bind(comp_tensor, ar_tensor)
            for actor, comp_tensor, ar_tensor in zip(actors, comp_tensors, ar_tensors)
        ]
        dag = MultiOutputNode(elapses)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=overlap)

    key_to_elapses = defaultdict(list)
    for iter in range(num_iters):
        actor_to_elapses = ray.get(compiled_dag.execute(None))
        if iter > num_iters_warmup and iter % 10 == 0:
            logger.info(f"Iteration: {iter}, elapses: {actor_to_elapses}")
        actor_elapse = {
            key: get_avg([elapses[key] for elapses in actor_to_elapses])
            for key in actor_to_elapses[0]
        }
        for key, value in actor_elapse.items():
            key_to_elapses[key].append(value)

    compiled_dag.teardown()

    results = {
        "overlap": overlap,
        "num_iters": num_iters,
        "size_comp": size_comp,
        "num_comp": num_comp,
        "size_comm": size_comm,
    }
    for key, elapses in key_to_elapses.items():
        elapses = elapses[int(len(elapses) * 0.2) :]
        elapse_avg = get_avg(elapses)
        results[key] = elapse_avg
    logger.info(f"results: {results}")

    ray.shutdown()
    return results


def benchmark_single():
    fire.Fire(benchmark_container)


def benchmark_multi():
    # Create a timestamped filename for the CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/titan_n2/overlap/{timestamp}.csv"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Create/open the CSV file
    with open(csv_filename, "w", newline="") as csvfile:
        # Define the CSV header
        fieldnames = [
            "num_iters",
            "size_comp",
            "num_comp",
            "size_comm",
            "overlap_off",
            "overlap_off_comp",
            "overlap_on",
            "ratio_off_comp",
            "ratio_off_comm",
            "ratio_overlap_theory",
            "ratio_overlap",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        num_comps = [3_000 * i for i in range(1, 11)]
        for num_comp in num_comps:
            results_ov_off = benchmark_container(
                overlap=False,
                num_comp=num_comp,
            )
            results_ov_on = benchmark_container(
                overlap=True,
                num_comp=num_comp,
            )
            ratio_off_comp = round(
                results_ov_off["comp"] / results_ov_off["e2e"] * 100, 2
            )
            ratio_off_comm = round(100 - ratio_off_comp, 2)
            ratio_overlap_theory = round(
                100 / max(ratio_off_comp, ratio_off_comm) * 100, 2
            )
            ratio_overlap = round(results_ov_off["e2e"] / results_ov_on["e2e"] * 100, 2)

            # Organize results
            results = {
                "num_iters": results_ov_off["num_iters"],
                "size_comp": results_ov_off["size_comp"],
                "num_comp": results_ov_off["num_comp"],
                "size_comm": results_ov_off["size_comm"],
                "overlap_off": results_ov_off["e2e"],
                "overlap_off_comp": results_ov_off["comp"],
                "overlap_on": results_ov_on["e2e"],
                "ratio_off_comp": ratio_off_comp,
                "ratio_off_comm": ratio_off_comm,
                "ratio_overlap_theory": ratio_overlap_theory,
                "ratio_overlap": ratio_overlap,
            }

            # Write the results to CSV
            writer.writerow(results)

            # Still log to console for visibility
            logger.info(f"results overlap: {results}")

        logger.info(f"Benchmark results written to {os.path.abspath(csv_filename)}")


if __name__ == "__main__":
    # benchmark_single()
    benchmark_multi()
