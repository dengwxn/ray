import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

import fire

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_experiment(config):
    """
    Run a single experiment with the specified configuration.

    Args:
        config (dict): Experiment configuration

    Returns:
        tuple: (success, output_path, log_file)
    """
    # Set timezone
    os.environ["TZ"] = "America/Los_Angeles"

    # Disable ray log deduplication
    os.environ["RAY_DEDUP_LOGS"] = "0"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["output_path"] = os.path.join(config["output_path"], timestamp)
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)

    # Save config to a JSON file in output_path
    config_file = os.path.join(output_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config: {config}")
    logger.info(f"Saved config to {config_file}")

    logger.info(f"Running {output_path}...")

    # Set up experiment parameters
    batch_size = config["batch_size"]
    seq_len = config["seq_len"]
    num_actors = config["num_actors"]
    num_iters = config["num_iters"]
    latency_prefix = "latency"
    log_file = os.path.join(output_path, f"actors.log")

    # Build command
    module_path = config["module_path"]
    command = [
        "python",
        "-m",
        module_path,
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--num-actors",
        str(num_actors),
        "--num-iters",
        str(num_iters),
        "--output-path",
        output_path,
        "--latency-prefix",
        latency_prefix,
    ]
    if "overlap" in config:
        command += ["--overlap", str(config["overlap"])]

    # Redirect output to log file
    with open(log_file, "w") as f:
        process = subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    status = process.returncode
    if status != 0:
        logger.error(f"WA, status: {status}, log: {log_file}")
        return False, output_path, log_file

    logger.info(f"AC, log: {log_file}")
    return True, output_path, log_file


def benchmark_multi(
    folder: str = "titan_n2",
    model: str = "LLAMA_1B",
    num_actors: int = 2,
    num_iters: int = 50,
    batch_size: int = 1,
    seq_len: int = 1024,
    overlap: Optional[bool] = None,
):
    # Get current working directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    if not current_dir.endswith("python/ray/experimental/fsdp"):
        logger.error("Not running from python/ray/experimental/fsdp directory")
        sys.exit(1)

    # Define template variations
    variations = [
        # # Torch configurations
        # {
        #     "framework": "torch",
        #     "settings": [
        #         # {"cc": "off", "fp": "off", "num_actors": 1},
        #         # {"cc": "off", "fp": "on", "num_actors": 1},
        #         {"cc": "on", "fp": "on", "pf": "on", "num_actors": num_actors},
        #     ],
        # },
        # Ray configurations
        {
            "framework": "ray",
            "settings": [
                # {"cc": "off", "ov": "off", "num_actors": 1},
                # {"cc": "on", "ov": "off", "num_actors": num_actors},
                {"cc": "on", "ov": "on", "num_actors": num_actors},
            ],
        },
        # # Deepspeed configurations
        # {
        #     "framework": "deepspeed",
        #     "settings": [
        #         {"zero": "on", "num_actors": num_actors},
        #     ],
        # },
    ]

    # Generate experiments from template
    experiments = []
    for variant in variations:
        framework = variant["framework"]
        for setting in variant["settings"]:
            # Build path components based on settings
            components = []
            for key, value in setting.items():
                if key != "num_actors":  # num_actors is not part of the path
                    components.append(f"{key}_{value}")

            # Construct paths
            output_path = f"results/{folder}/llama3/{framework}/{'/'.join(components)}"
            module_path = f"ray.experimental.fsdp.src.main.llama3.{framework}.{'.'.join(components)}"

            # Create experiment config
            experiment = {
                "output_path": output_path,
                "module_path": module_path,
                "model": model,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_actors": num_actors,
                "num_iters": num_iters,
            }

            if overlap is not None:
                # Add overlap if specified
                # for o in [True, False]:
                for o in [False]:
                    experiment["overlap"] = o
                    experiments.append(experiment)
            else:
                experiments.append(experiment)

    # Run experiments sequentially
    for config in experiments:
        run_experiment(config)


def benchmark_v1(
    folder: str = "v1",
    model: str = "LLAMA_8B",
    num_actors_l: List[int] = [2, 4, 6],
    num_iters: int = 50,
    batch_size: int = 1,
    seq_len: int = 1024,
):
    for num_actors in num_actors_l:
        benchmark_multi(
            folder=folder,
            model=model,
            num_actors=num_actors,
            num_iters=num_iters,
            batch_size=batch_size,
            seq_len=seq_len,
            overlap=False,
        )


def benchmark_v2(
    folder: str = "v2",
    model: str = "LLAMA_1B",
    # num_actors_l: List[int] = [2, 4, 6],
    num_actors_l: List[int] = [2],
    num_iters: int = 50,
    batch_size: int = 1,
    seq_len: int = 1024,
):
    for num_actors in num_actors_l:
        benchmark_multi(
            folder=folder,
            model=model,
            num_actors=num_actors,
            num_iters=num_iters,
            batch_size=batch_size,
            seq_len=seq_len,
        )


if __name__ == "__main__":
    # fire.Fire(benchmark_multi)
    fire.Fire(benchmark_v1)
    # fire.Fire(benchmark_v2)
