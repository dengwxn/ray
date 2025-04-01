import logging

import fire
from actor import TextWorker, VisionWorker, WorkerV3
from common import random_seed
from dist import initialize_dist_group

import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def main(
    model_name: str = "ViT-L-14",
    # model_name: str = "ViT-bigG-14",
    batch_size: int = 16,
    num_iters: int = 2,
):
    actor = WorkerV3.remote(model_name, 1, 1)

    for i in range(num_iters):
        ray.get(actor.forward.remote((i, batch_size)))
        ray.get(actor.backward.remote())
        logger.info(f"Iteration {i} finished")


if __name__ == "__main__":
    fire.Fire(main)
