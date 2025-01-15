import logging
import time
import os
from typing import Any, Dict, List, Set, Callable
import csv

import ray
from ..core.common import log_elapses_to_csv, secs_to_micros
from ..core.config import parse_args
from ..core.mp.actor_resnet import ResnetActor
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.channel.torch_tensor_nccl_channel import (
    _destroy_nccl_group,
    _init_nccl_group,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[ResnetActor]:
    num_models = args["num_models"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    check_tracing = args["check_tracing"]

    actor_cls = ResnetActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            rank=i,
            num_models=num_models,
            num_actors=num_actors,
            device=device,
            check_tracing=check_tracing,
        )
        for i in range(num_actors)
    ]

    return actors


def train_cot(
    actors: List[ResnetActor],
    num_models: int,
    num_epochs: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    check_tracing: bool,
    trace_timeline: bool = False,
) -> None:
    nccl_group_id = _init_nccl_group(actors)
    for actor in actors:
        ray.get(actor.set_nccl_group.remote(nccl_group_id))

    with InputNode() as inp:
        actors_to_forwards = [actor.forward.bind(inp) for actor in actors]
        actors_to_backwards = actors_to_forwards
        outputs = []

        actors_to_backwards = [
            actor.backward.bind(actors_to_backwards[j], num_models - 1)
            for j, actor in enumerate(actors)
        ]
        for i in reversed(range(num_models)):
            grads_allreduced = [
                actor.allreduce.bind(grad_to_reduce)
                for actor, grad_to_reduce in zip(actors, actors_to_backwards)
            ]
            if i > 0:
                actors_to_backwards = [
                    actor.backward.bind(actors_to_backwards[j], i - 1)
                    for j, actor in enumerate(actors)
                ]
            actors_to_updates = [
                actor.update.bind(grads_allreduced[j], True, i)
                for j, actor in enumerate(actors)
            ]
            outputs.extend(actors_to_updates)

        dag = MultiOutputNode(outputs)

    BATCH_SIZE = 32

    total_elapses: List[int] = []
    for epoch in range(num_epochs):
        for actor in actors:
            ray.get(actor.init_training.remote(BATCH_SIZE))
            ray.get(actor.init_tracing.remote())

        start = time.perf_counter()
        ray.get(dag.execute(None))
        end = time.perf_counter()

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")
            total_elapses.append(round((end - start) * 1e6))

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    if not check_tracing:
        metrics = [
            "total",
            "actor.total",
        ]
    else:
        metrics = [
            "total",
            "actor.total",
            "fw.total",
            "bw.total",
            "bw.backward",
            "bw.allreduce",
            "bw.others",
            "bw.update",
            "serialization",
            "deserialization",
        ]
        if trace_timeline:
            log_timeline(
                len(actors),
                output_path,
                latency_prefix,
            )
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

    _destroy_nccl_group(nccl_group_id)

    time.sleep(1)


def log_timeline(num_actors: int, output_path: str, latency_prefix: str) -> None:
    timeline: List[Dict[str, Any]] = ray.timeline()
    logger.warning(f"timeline length: {len(timeline)}")
    tids: Set[str] = {
        event["tid"] for event in timeline if event["cat"].startswith("task")
    }
    assert len(tids) == num_actors
    actor_to_time: Dict[str, Dict[str, float]] = {tid: dict() for tid in tids}

    def extract_time(
        tid: str, key: str, pred: Callable[[Dict[str, Any]], bool]
    ) -> None:
        total_time = 0
        for event in timeline:
            if event["tid"] == tid and pred(event):
                total_time += event["dur"]
        actor_to_time[tid][key] = total_time

    for tid in tids:
        extract_time(
            tid,
            "task:deserialize_arguments",
            lambda event: event["cat"] == "task:deserialize_arguments",
        )
        extract_time(
            tid,
            "task:store_outputs",
            lambda event: event["cat"] == "task:store_outputs",
        )
        extract_time(
            tid,
            "task:execute",
            lambda event: event["cat"] == "task:execute",
        )
        extract_time(
            tid,
            "task:total",
            lambda event: event["cat"].startswith("task::"),
        )

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Process each actor's data
    for actor, time_dict in actor_to_time.items():
        output_file = f"{output_path}/{latency_prefix}_{actor}.csv"

        # Write statistics to CSV file
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "elapse", "percent"])
            total = time_dict["task:total"]
            for cat, elapse in time_dict.items():
                percent = elapse / total * 100
                writer.writerow(
                    [
                        cat,
                        round(elapse),
                        round(percent, 1),
                    ]
                )


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train_cot(
        actors,
        args["num_models"],
        args["num_epochs"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["check_tracing"],
        args.get("trace_timeline", False),
    )

    for actor in actors:
        ray.kill(actor)
    if args["trace_timeline"]:
        ray.timeline(filename="{}/timeline.json".format(args["output_path"]))
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
