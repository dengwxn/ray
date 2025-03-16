import logging
from typing import Any, Dict, List

import ray
from .....core.common import (
    generate_1f1b_dag,
    get_end_time,
    get_start_time,
    log_elapses_to_csv,
)
from .....core.config import parse_args
from .....core.llama3.actor import LlamaActor
from .....core.llama3.model import LLAMA_DEBUG as LLAMA

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[LlamaActor]:
    model_args = LLAMA
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    num_partitions = args["num_partitions"]
    num_actors = args["num_actors"]
    tracing = args["tracing"]

    actor_cls = LlamaActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=i,
            num_partitions=num_partitions,
            num_actors=num_actors,
            tracing=tracing,
        )
        for i in range(num_actors)
    ]

    return actors


def train(
    actors: List[LlamaActor],
    num_partitions: int,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    compiled_dag = generate_1f1b_dag(actors, 2, 2)
    compiled_dag.visualize(channel_details=True)

    for actor in actors:
        ray.get(actor.init_and_partition_model.remote())

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if save_model:
            for i, actor in enumerate(actors):
                weights = ray.get(actor.fetch_weights.remote())
                for idx, weight in enumerate(weights):
                    logger.info(f"actor: {i}, layer: {idx}, shard: {weight}")

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

        break

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    metrics = LlamaActor.get_metrics(tracing)
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
    )

    if save_model:
        model_file = f"{model_prefix}.log"
        with open(model_file, "w") as f:
            for weight in weights:
                f.write(f"{weight}\n")
        for i, actor in enumerate(actors):
            weights = ray.get(actor.fetch_weights.remote())
            model_file = f"{model_prefix}_{i}.log"
            with open(model_file, "w") as f:
                for weight in weights:
                    f.write(f"{weight}\n")


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train(
        actors,
        args["num_partitions"],
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["tracing"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
