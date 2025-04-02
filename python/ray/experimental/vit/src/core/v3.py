import logging

import fire
from actor import WorkerV3 as Worker
from dist import init_torch_distributed

import ray

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
    num_dp = 4

    actors = [Worker.remote(model_name, num_dp, 1) for _ in range(num_dp)]
    init_torch_distributed(actors)
    ray.get([actor.init_fsdp_model.remote() for actor in actors])

    for i in range(num_iters):
        ray.get([actor.forward.remote((i, batch_size)) for actor in actors])
        ray.get([actor.backward.remote() for actor in actors])
        logger.info(f"Iteration {i} finished")


if __name__ == "__main__":
    fire.Fire(main)
