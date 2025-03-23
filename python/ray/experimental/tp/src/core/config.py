import argparse
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--layer-size",
        type=int,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
    )
    parser.add_argument(
        "--num-batches",
        type=int,
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
    )
    parser.add_argument(
        "--num-actors",
        type=int,
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--latency-prefix",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
    )
    parser.add_argument(
        "--model-file",
        type=str,
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
    )
    args = parser.parse_args()
    args = vars(args)
    return args
