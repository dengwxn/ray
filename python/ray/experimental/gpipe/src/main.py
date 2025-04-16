from argparse import Namespace, ArgumentParser
import time
import ray
import numpy as np

from .afab import generate_afab_schedules
from .common import secs_to_micros


def run_gpipe(
    num_microbatches: int,
    num_stages: int,
    num_trials: int,
    tensor_size: int,
    warmup: int = 2,
    verbose: bool = False,
) -> None:
    """
    Run a GPipe-style pipeline parallelism example.

    Args:
        num_microbatches (int): The number of microbatches.
        num_stages (int): The number of stages.
        num_trials (int): The number of trials to run.
        tensor_size (int): The size of the tensor.
        warmup (int): The number of warmup iterations.
        verbose (bool): Whether to print results.
    """
    if warmup < 0:
        raise ValueError("Warmup must be non-negative.")
    if warmup >= num_trials:
        raise ValueError("Warmup must be less than the number of trials.")

    compiled_dag, workers = generate_afab_schedules(num_microbatches, num_stages)
    elapses = []
    for trial in range(num_trials):
        # Execute the compiled DAG
        start = time.perf_counter()
        ref = compiled_dag.execute(tensor_size)
        result = ray.get(ref, timeout=100)
        end = time.perf_counter()
        elapse = secs_to_micros(end - start)
        elapses.append(elapse)
        print(f"[Trial #{trial + 1}] {elapse=}us")
    elapses = np.array(elapses[warmup:])
    mean, std = np.mean(elapses), np.std(elapses)
    print(f"{num_trials=}, {num_microbatches=}, {num_stages=}, {tensor_size=}")
    print(f"elapse (us): {mean=}, {std=}")
    tput = num_microbatches * num_stages / mean * 1e6
    print(f"{tput=} * 1e-6")
    print("num_microbatches, num_stages, tput")
    print(f"{num_microbatches},{num_stages},{tput:.2f}")


def parse_args() -> Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="GPipe Config")
    parser.add_argument(
        "--num_microbatches", type=int, required=True, help="Number of microbatches"
    )
    parser.add_argument(
        "--num_stages", type=int, required=True, help="Number of stages"
    )
    parser.add_argument(
        "--num_trials", type=int, default=10, help="Number of trials to run"
    )
    parser.add_argument(
        "--tensor_size", type=int, required=True, help="Size of the tensor"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to print results"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init()
    try:
        run_gpipe(
            num_microbatches=args.num_microbatches,
            num_stages=args.num_stages,
            num_trials=args.num_trials,
            tensor_size=args.tensor_size,
            warmup=args.warmup,
            verbose=args.verbose,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
