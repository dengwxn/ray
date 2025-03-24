import logging
from typing import Any, Dict, List

import ray
from ...core.actor import ActorTP2PP as Actor
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
    num_tp = 1
    master_addr = "127.0.0.1"
    num_pp = 2
    num_pp_batches = 2
    tracing = args["tracing"]

    actor_cls = Actor.options(num_gpus=1)
    pp_to_tp_actors: List[List[Actor]] = []
    for i in range(num_pp):
        master_port = 12345 + i
        actors = []
        for j in range(num_tp):
            actors.append(
                actor_cls.remote(
                    model_args=model_args,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rank_tp=j,
                    num_tp=num_tp,
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_pp=i,
                    num_pp_batches=num_pp_batches,
                    tracing=tracing,
                )
            )
        pp_to_tp_actors.append(actors)

    return pp_to_tp_actors


def train(
    pp_to_tp_actors: List[List[Actor]],
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        b1_fw1s = [
            tp_actor.forward.bind(0, inp).with_tensor_transport(transport="nccl")
            for tp_actor in pp_to_tp_actors[0]
        ]

        b2_fw1s = [
            tp_actor.forward.bind(1, inp).with_tensor_transport(transport="nccl")
            for tp_actor in pp_to_tp_actors[0]
        ]
        b1_fw2s = [
            tp_actor.forward.bind(0, b1_fw1)
            for tp_actor, b1_fw1 in zip(pp_to_tp_actors[1], b1_fw1s)
        ]

        b1_bw1s = [
            tp_actor.backward.bind(0, b1_fw2).with_tensor_transport(transport="nccl")
            for tp_actor, b1_fw2 in zip(pp_to_tp_actors[1], b1_fw2s)
        ]

        b1_bw2s = [
            tp_actor.backward.bind(0, b1_bw1)
            for tp_actor, b1_bw1 in zip(pp_to_tp_actors[0], b1_bw1s)
        ]
        b2_fw2s = [
            tp_actor.forward.bind(1, b2_fw1)
            for tp_actor, b2_fw1 in zip(pp_to_tp_actors[1], b2_fw1s)
        ]

        b2_bw1s = [
            tp_actor.backward.bind(1, b2_fw2).with_tensor_transport(transport="nccl")
            for tp_actor, b2_fw2 in zip(pp_to_tp_actors[1], b2_fw2s)
        ]

        b2_bw2s = [
            tp_actor.backward.bind(1, b2_bw1)
            for tp_actor, b2_bw1 in zip(pp_to_tp_actors[0], b2_bw1s)
        ]

        updates = b1_bw2s + b2_bw2s
        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile()

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for tp_actors in pp_to_tp_actors:
            for actor in tp_actors:
                ray.get(actor.init_training.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for tp_actors in pp_to_tp_actors:
            for actor in tp_actors:
                ray.get(actor.finish_tracing.remote())

    for tp_actors in pp_to_tp_actors:
        for actor in tp_actors:
            ray.get(actor.clean.remote())

    actors_to_elapses = []
    for tp_actors in pp_to_tp_actors:
        for actor in tp_actors:
            actors_to_elapses.append(ray.get(actor.fetch_traces.remote()))
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    metrics = Actor.get_metrics(tracing)
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
    )


if __name__ == "__main__":
    args = parse_args()
    pp_to_tp_actors = init_actors(args)

    train(
        pp_to_tp_actors,
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args["tracing"],
    )

    for tp_actors in pp_to_tp_actors:
        for actor in tp_actors:
            ray.kill(actor)
    ray.shutdown()
