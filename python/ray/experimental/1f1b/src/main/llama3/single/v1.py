import logging
from typing import Any, Dict, List

from ....core.common import get_end_time, get_start_time
from ....core.config import parse_args
from ....core.llama3.model import LLAMA_DEBUG as LLAMA
from ....core.llama3.model import ActorV1 as LlamaActor

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

    actors = [
        LlamaActor(
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
    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            actor.init_training()

        start = get_start_time()

        input = actors[0].get_input(None)
        b1_fw1 = actors[0].forward(input)
        b1_fw2 = actors[1].forward(input, b1_fw1)
        b1_bw1 = actors[1].backward(b1_fw2)
        b1_bw2 = actors[0].backward(b1_bw1)

        outputs = [b1_bw2]
        logger.warning(f"iter: {iter}, outputs: {outputs}")

        # b2_fw1 = actors[0].forward(input)
        # b2_fw2 = actors[1].forward(input, b2_fw1)
        # b2_bw1 = actors[1].backward(b2_fw2)
        # b2_bw2 = actors[0].backward(b2_bw1)

        # outputs = [b1_bw2, b2_bw2]

        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        break


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
