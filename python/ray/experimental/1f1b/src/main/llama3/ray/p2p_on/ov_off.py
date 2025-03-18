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
from ray.dag import InputNode, MultiOutputNode

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
    num_batches = args["num_batches"]
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
            num_batches=num_batches,
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
    assert len(actors) == 2

    with InputNode() as inp:
        b1_fw1 = actors[0].forward.bind(0, inp).with_tensor_transport(transport="nccl")

        # PP
        b2_fw1 = actors[0].forward.bind(1, inp).with_tensor_transport(transport="nccl")
        b1_fw2 = actors[1].forward.bind(0, b1_fw1)

        b1_bw1 = (
            actors[1].backward.bind(0, b1_fw2).with_tensor_transport(transport="nccl")
        )
        b1_upd1 = actors[1].update.bind(0, inp)

        # PP
        b1_bw2 = actors[0].backward.bind(0, b1_bw1)
        b1_upd2 = actors[0].update.bind(0, b1_bw2)
        b2_fw2 = actors[1].forward.bind(1, b2_fw1)

        b2_bw1 = (
            actors[1].backward.bind(1, b2_fw2).with_tensor_transport(transport="nccl")
        )
        b2_upd1 = actors[1].update.bind(1, inp)

        b2_bw2 = actors[0].backward.bind(1, b2_bw1)
        b2_upd2 = actors[0].update.bind(1, b2_bw2)

        updates = [b1_upd1, b1_upd2, b2_upd1, b2_upd2]
        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile()
    # compiled_dag.visualize(channel_details=True)

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

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
