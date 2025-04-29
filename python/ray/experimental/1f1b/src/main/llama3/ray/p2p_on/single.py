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
from .....core.llama3.actor import SingleLlamaActor
from .....core.llama3.model import LLAMA_DEBUG as LLAMA
from ray.dag import InputNode, MultiOutputNode

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init_actor(args: Dict[str, Any]) -> SingleLlamaActor:
    model_args = LLAMA
    model_args.n_layers = 8
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    tracing = args["tracing"]

    actor_cls = SingleLlamaActor.options(num_gpus=1)
    actor = actor_cls.remote(
            model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            tracing=tracing,
        )

    return actor


def train(
    actor: SingleLlamaActor,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        dag = actor.forward.bind(inp)
        dag = actor.backward.bind(dag)
    compiled_dag = dag.experimental_compile()

    total_elapses: List[int] = []
    for iter in range(num_iters):
        ray.get(actor.init_training.remote())
        
        start = get_start_time()
        # ray cg
        compiled_dag.execute(None)
        # ray
        # out = actor.backward.remote(actor.forward.remote(None))
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)
        logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
        total_elapses.append(elapse_us)

        ray.get(actor.finish_tracing.remote())

    elapses = ray.get(actor.fetch_traces.remote())
    elapses["total"] = total_elapses
    metrics = SingleLlamaActor.get_metrics(tracing)
    log_elapses_to_csv(
        [elapses],
        output_path,
        latency_prefix,
        metrics,
    )


def main(args: Dict[str, Any]) -> None:
    ray.init()
    actor = init_actor(args)

    train(
        actor,
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["tracing"],
    )

    ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
