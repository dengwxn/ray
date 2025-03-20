import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_command(command, env=None):
    """
    Run a shell command and return its exit code, stdout, and stderr.

    Args:
        command (list): Command to run as a list of arguments
        env (dict, optional): Environment variables to set

    Returns:
        tuple: (exit_code, stdout, stderr)
    """
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            env=env if env else os.environ.copy(),  # Pass environment
        )

        exit_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr

        if stdout:
            logger.info(f"Command stdout: {stdout}")
        if stderr:
            logger.error(f"Command stderr: {stderr}")

        return exit_code, stdout, stderr
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return -1, "", str(e)


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

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Disable ray log deduplication
    os.environ["RAY_DEDUP_LOGS"] = "0"

    # Create output directory
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)

    # Clean existing files
    for file in os.listdir(output_path):
        if file.endswith(".csv") or file.endswith(".log"):
            os.remove(os.path.join(output_path, file))

    logger.info(f"Running experiment for {output_path}...")

    # Set up experiment parameters
    batch_size = config.get("batch_size", 1)
    seq_len = config.get("seq_len", 1024)
    num_partitions = config.get("num_partitions", 18)
    num_actors = config.get("num_actors", 2)
    num_iters = config.get("num_iters", 20)
    latency_prefix = config.get("latency_prefix", timestamp)
    model_prefix = os.path.join(output_path, f"{timestamp}_model")
    log_file = os.path.join(output_path, f"{timestamp}.log")

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
        "--num-partitions",
        str(num_partitions),
        "--num-actors",
        str(num_actors),
        "--num-iters",
        str(num_iters),
        "--output-path",
        output_path,
        "--latency-prefix",
        latency_prefix,
        "--model-prefix",
        model_prefix,
        "--tracing",
    ]

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
        logger.error(f"Experiment failed with status {status} and log {log_file}")
        return False, output_path, log_file

    logger.info(f"Experiment succeeded with log {log_file}")
    return True, output_path, log_file


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run experiments with customizable parameters"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for experiments",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for experiments",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=18,
        help="Number of partitions",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=2,
        help="Number of actors",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="Number of iterations",
    )
    args = parser.parse_args()

    # Get current working directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")

    # Experiment configurations
    experiments = [
        {
            "output_path": "results/barbell_n2/llama3/ray/cc_off/ov_off/exp_self",
            "module_path": "ray.experimental.fsdp.src.main.llama3.ray.cc_off.ov_off",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_actors": 1,
            "num_iters": args.num_iters,
        },
        {
            "output_path": "results/barbell_n2/llama3/ray/cc_on/ov_off/exp_self",
            "module_path": "ray.experimental.fsdp.src.main.llama3.ray.cc_on.ov_off",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_actors": args.num_actors,
            "num_iters": args.num_iters,
        },
        {
            "output_path": "results/barbell_n2/llama3/ray/cc_on/ov_on/exp_self",
            "module_path": "ray.experimental.fsdp.src.main.llama3.ray.cc_on.ov_on",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_actors": args.num_actors,
            "num_iters": args.num_iters,
        },
        {
            "output_path": "results/barbell_n2/llama3/torch/cc_off/fp_on/exp_self",
            "module_path": "ray.experimental.fsdp.src.main.llama3.torch.cc_off.fp_on",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_actors": 1,
            "num_iters": args.num_iters,
        },
        {
            "output_path": "results/barbell_n2/llama3/torch/cc_on/fp_on/pf_on/exp_self",
            "module_path": "ray.experimental.fsdp.src.main.llama3.torch.cc_on.fp_on.pf_on",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "num_actors": args.num_actors,
            "num_iters": args.num_iters,
        },
    ]

    # Check if we're in the correct directory
    if not os.getcwd().endswith("python/ray/experimental/fsdp"):
        logger.error("Not running from python/ray/experimental/fsdp directory")
        sys.exit(1)

    # Run experiments sequentially
    for config in experiments:
        run_experiment(config)


if __name__ == "__main__":
    main()
