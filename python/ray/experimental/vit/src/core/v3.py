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
    # model_name: str = "ViT-L-14",
    model_name: str = "ViT-bigG-14",
    bs_single: int = 16,
    num_dp_vision: int = 3,
    num_dp_text: int = 1,
    num_dp: int = 4,
    num_iters: int = 50,
):
    bs_global = bs_single * max(num_dp_vision, num_dp_text)

    actors = [Worker.remote(model_name, num_dp) for _ in range(num_dp)]
    init_torch_distributed(actors)
    ray.get([actor.init_fsdp_model.remote() for actor in actors])

    for i in range(num_iters):
        ray.get([actor.init_training.remote() for actor in actors])

        ray.get([actor.forward.remote((i, bs_global)) for actor in actors])
        ray.get([actor.backward.remote() for actor in actors])

        logger.info(f"Iteration {i} finished")
        ray.get([actor.finish_tracing.remote() for actor in actors])


if __name__ == "__main__":
    fire.Fire(main)
