import logging

from .core.config import Config, parse_config
from .core.correctness import compare_weights
from .core.ray_ddp import run_ray_ddp
from .core.torch import run_torch
from .core.torch_ddp import run_torch_ddp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


def main(config: Config) -> None:
    """
    Run and compare the performance of Ray DDP, PyTorch, and PyTorch DDP.
    Correctness of Ray DDP is checked if specified. Save the average elapses
    across iterations of all approaches to the output file.

    Args:
        config: Model and training configurations, as well as whether to check
            correctness and the output file path.
    """
    torch_weights, torch_elapse = run_torch(config)
    torch_ddp_weights, torch_ddp_elapse = run_torch_ddp(config)
    ray_ddp_weights, ray_ddp_elapse = run_ray_ddp(config)
    if config.check_correctness:
        compare_weights(
            torch_weights,
            ray_ddp_weights,
            "ray ddp vs torch",
            allow_error=True,
        )
        compare_weights(
            torch_ddp_weights,
            ray_ddp_weights,
            "ray ddp vs torch ddp",
        )
    with open(config.output_path, "w") as file:
        file.write("torch,torch-ddp,ray-ddp\n")
        file.write(f"{torch_elapse},{torch_ddp_elapse},{ray_ddp_elapse}\n")


if __name__ == "__main__":
    config = parse_config()
    main(config)
