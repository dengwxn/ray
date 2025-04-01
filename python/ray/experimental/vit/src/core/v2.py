import logging

import fire
from actor import TextWorker, VisionWorker, WorkerV2
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
    num_iters: int = 50,
):
    random_seed(998244353)

    actor = WorkerV2(model_name, 1, 1)

    for i in range(num_iters):
        acts = actor.forward((i, batch_size))
        actor.backward(acts)


if __name__ == "__main__":
    fire.Fire(main)
