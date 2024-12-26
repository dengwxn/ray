import logging

from .core.config import Config, parse_config
from .core.ray_ddp import run_ray_ddp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


def main(config: Config) -> None:
    run_ray_ddp(config)


if __name__ == "__main__":
    config = parse_config()
    main(config)
