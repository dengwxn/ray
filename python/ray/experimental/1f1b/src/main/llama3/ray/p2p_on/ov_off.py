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


def build_1f1b_dag(actors: List[LlamaActor], num_microbatches=4, num_lead_microbatches=4):
    """
    Constructs and compiles a 1F1B DAG for pipeline parallelism.

    Args:
        workers (list): List of Ray actor handles, each representing a pipeline stage.
        num_microbatches (int): Number of microbatches to pipeline per batch.
        num_lead_microbatches (int): Number of leading microbatches to maintain pipeline balance.

    Returns:
        A compiled DAG ready for execution.
    """
    x_idx = 0
    y_idx = 1
    num_workers = len(actors)
    fwd_queues = [[] for _ in range(num_workers)]
    bwd_queues = [[] for _ in range(num_workers)]
    fwd_counter = [num_lead_microbatches - p for p in range(num_workers)]
    done = []
    with ray.dag.InputNode() as inp:
      # Load the queue for worker 0 with all input micro-batches
      for idx in range(num_microbatches):
        fwd_queues[0].append([idx, inp])
      while len(done) < num_microbatches:
        for k, worker in enumerate(actors):
            if fwd_counter[k] > 0 and fwd_queues[k]:
                idx, mb = fwd_queues[k].pop(0) 
                if k < num_workers - 1:
                    mb = worker.forward.bind(idx, mb).with_tensor_transport(transport="nccl")
                    fwd_queues[k + 1].append([idx, mb])
                else:  
                    mb = worker.forward.bind(idx, mb)
                    bwd_queues[k].append([idx, mb])
                fwd_counter[k] -= 1
            elif bwd_queues[k]:
                idx, mb = bwd_queues[k].pop()
                if k > 0:
                    mb = worker.backward.bind(idx, mb)
                    mb = worker.update.bind(idx, mb).with_tensor_transport(transport="nccl")
                    bwd_queues[k - 1].append([idx, mb])
                else:
                    mb = worker.backward.bind(idx, mb)
                    mb = worker.update.bind(idx, mb)
                    done.append(mb)
                fwd_counter[k] += 1
      dag = ray.dag.MultiOutputNode(done)
    return dag.experimental_compile()


def train(
    actors: List[LlamaActor],
    num_microbatches: int,
    num_partitions: int,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    compiled_dag = build_1f1b_dag(actors, num_microbatches=num_microbatches, num_lead_microbatches=num_microbatches)
    compiled_dag.visualize(filename="compiled_graph",channel_details=True)

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())

        start = get_start_time()
        ray.get(compiled_dag.execute(None))
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
        args["num_batches"],
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
