import logging
from typing import Any, Dict, List

import ray
from ...core.actor import ActorTP as Actor
from ...core.common import get_end_time, get_start_time, log_elapses_to_csv
from ...core.config import parse_args
from ...core.model import LLAMA_DEBUG as LLAMA
from ray.dag import InputNode, MultiOutputNode

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[Actor]:
    ray.init()
    model_args = LLAMA
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    num_actors_tp = 2
    master_addr = "127.0.0.1"
    master_port = 12345
    tracing = args["tracing"]

    actor_cls = Actor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            model_args=model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            rank=i,
            num_actors_tp=num_actors_tp,
            master_addr=master_addr,
            master_port=master_port,
            tracing=tracing,
        )
        for i in range(num_actors_tp)
    ]

    return actors


def train(
    actors: List[Actor],
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        forwards = [actor.forward.bind(inp) for actor in actors]
        backwards = [
            actor.backward.bind(forward) for actor, forward in zip(actors, forwards)
        ]
        updates = [
            actor.update.bind(backward) for actor, backward in zip(actors, backwards)
        ]
        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile()

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

    ray.get([actor.clean.remote() for actor in actors])

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    metrics = Actor.get_metrics(tracing)
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
    )


def clean(actors):
    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    actors = init_actors(args)
    train(
        actors,
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args["tracing"],
    )
    clean(actors)
