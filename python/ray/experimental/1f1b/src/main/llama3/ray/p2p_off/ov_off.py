import logging
from typing import Any, Dict, List

from .....core.common import (
    generate_1f1b_dag,
    get_end_time,
    get_start_time,
    log_elapses_to_csv,
)
from .....core.config import parse_args
from .....core.llama3.actor import LlamaActorOff as LlamaActor
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
    num_batches = args["num_batches"]
    num_partitions = args["num_partitions"]
    num_actors = args["num_actors"]
    tracing = args["tracing"]

    actors = [
        LlamaActor(
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

    def execute():
        b1_fw1 = actors[0].forward(0, None)

        # PP
        b2_fw1 = actors[0].forward(1, None)
        b1_fw2 = actors[1].forward(0, b1_fw1)

        b1_bw1 = actors[1].backward(0, b1_fw2)
        b1_upd1 = actors[1].update(0, None)

        # PP
        b1_bw2 = actors[0].backward(0, b1_bw1)
        b1_upd2 = actors[0].update(0, b1_bw2)
        b2_fw2 = actors[1].forward(1, b2_fw1)

        b2_bw1 = actors[1].backward(1, b2_fw2)
        b2_upd1 = actors[1].update(1, None)

        b2_bw2 = actors[0].backward(1, b2_bw1)
        b2_upd2 = actors[0].update(1, b2_bw2)

        _updates = [b1_upd1, b1_upd2, b2_upd1, b2_upd2]

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            actor.init_training()

        start = get_start_time()
        execute()
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            actor.finish_tracing()

    actors_to_elapses = [actor.fetch_traces() for actor in actors]
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
