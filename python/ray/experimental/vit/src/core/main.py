import logging

import fire
from actor import TextWorker, VisionWorker
from common import random_seed
from dist import init_torch_distributed

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
    # https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv
    model_name: str = "ViT-L-14",
    # model_name: str = "ViT-bigG-14",
    batch_size: int = 16,
    vision_num_tp: int = 1,
    vision_num_dp: int = 3,
    text_num_tp: int = 1,
    text_num_dp: int = 1,
    num_iters: int = 50,
):
    random_seed(998244353)
    assert vision_num_tp == text_num_tp == 1

    global_batch_size = batch_size * max(vision_num_dp, text_num_dp)
    vision_batch_size = global_batch_size // vision_num_dp
    text_batch_size = global_batch_size // text_num_dp
    logger.info(
        f"Global batch size: {global_batch_size}, vision batch size: {vision_batch_size}, text batch size: {text_batch_size}"
    )

    vision_actors = [
        VisionWorker.remote(model_name, vision_num_dp, vision_num_tp, text_num_dp)
        for _ in range(vision_num_dp * vision_num_tp)
    ]
    init_torch_distributed(vision_actors)
    ray.get([worker.init_parallel_strategy.remote() for worker in vision_actors])

    text_actors = [
        TextWorker.remote(model_name, text_num_dp, text_num_tp, vision_num_dp)
        for _ in range(text_num_dp * text_num_tp)
    ]
    init_torch_distributed(text_actors)
    ray.get([worker.init_parallel_strategy.remote() for worker in text_actors])

    vision_params = ray.get(vision_actors[0].get_num_params.remote())
    text_params = ray.get(text_actors[0].get_num_params.remote())
    logger.info(f"Vision params: {vision_params:,}, text params: {text_params:,}")

    vision_device = ray.get(vision_actors[0].get_device_name.remote())
    text_device = ray.get(text_actors[0].get_device_name.remote())
    logger.info(f"Vision device name: {vision_device}, text device name: {text_device}")

    with InputNode() as input_node:
        # forward
        text_acts = [actor.forward.bind(input_node) for actor in text_actors]
        vision_acts = [actor.forward.bind(input_node) for actor in vision_actors]

        # reduce activations to global batch size
        reduced_text_acts = text_actors[0].reduce_activations.bind(*text_acts)
        reduced_vision_acts = vision_actors[0].reduce_activations.bind(*vision_acts)

        # scatter activations to the other dp groups
        scattered_text_acts = (
            text_actors[0]
            .scatter_activations.options(num_returns=vision_num_dp)
            .bind(reduced_text_acts)
        )
        scattered_vision_acts = (
            vision_actors[0]
            .scatter_activations.options(num_returns=text_num_dp)
            .bind(reduced_vision_acts)
        )
        if not isinstance(scattered_vision_acts, list):
            assert text_num_dp == 1
            scattered_vision_acts = [scattered_vision_acts]

        # with_tensor_transport for NCCL transport
        scattered_text_acts = [
            to_vision_act.with_tensor_transport("nccl")
            for to_vision_act in scattered_text_acts
        ]
        scattered_vision_acts = [
            to_text_act.with_tensor_transport("nccl")
            for to_text_act in scattered_vision_acts
        ]

        # backward
        outputs = []
        for i, actor in enumerate(text_actors):
            outputs.append(actor.backward.bind(scattered_vision_acts[i]))
        for i, actor in enumerate(vision_actors):
            outputs.append(actor.backward.bind(scattered_text_acts[i]))

        dag = MultiOutputNode(outputs)

    dag = dag.experimental_compile(_submit_timeout=480)

    for i in range(num_iters):
        ray.get([actor.init_training.remote() for actor in text_actors])
        ray.get([actor.init_training.remote() for actor in vision_actors])

        ray.get(dag.execute((i, global_batch_size)))

        if (i + 1) % 10 == 0:
            logger.info(f"steps: {i+1}")

        ray.get([actor.finish_tracing.remote() for actor in text_actors])
        ray.get([actor.finish_tracing.remote() for actor in vision_actors])


if __name__ == "__main__":
    fire.Fire(main)
