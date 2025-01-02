import logging
import time
from typing import Any, Dict, List

import ray
from ..core.config import parse_args
from ..core.mp.actor import ModelActor
from ray.dag import InputNode

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[ModelActor]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    num_models = args["num_models"]
    device = "cuda:0"

    actor_cls = ModelActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            layer_size=layer_size,
            num_layers=num_layers,
            num_models=num_models,
            device=device,
        )
    ]

    return actors


def train_cot(
    actors: List[ModelActor], num_models: int, num_epochs: int, model_file: str
) -> None:
    assert len(actors) == 1, "Only one actor is supported for now"
    actor = actors[0]

    with InputNode() as inp:
        forward = actor.forward.bind(inp)
        backward = actor.backward.bind(forward, -1)
        for i in reversed(range(num_models - 1)):
            backward = actor.backward.bind(backward, i)
        dag = backward

    compiled_dag = dag.experimental_compile()
    ray.get(actor.init_weights.remote())

    for epoch in range(num_epochs):
        ray.get(actor.init_training.remote())

        start = time.perf_counter()
        compiled_dag.execute(None)
        end = time.perf_counter()

        weights = ray.get(actor.fetch_weights.remote())
        for idx, weight in enumerate(weights):
            logger.info(f"layer: {idx}, weight: {weight}")

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for weight in weights:
            f.write(f"{weight}\n")

    time.sleep(1)


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train_cot(
        actors,
        args["num_models"],
        args["num_epochs"],
        args["model_file"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
