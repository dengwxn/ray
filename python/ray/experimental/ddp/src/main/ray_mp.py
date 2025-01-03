import logging
import time
from typing import Any, Dict, List

import ray
from ..core.config import parse_args
from ..core.mp.actor import ModelActor
from ray.dag import InputNode, MultiOutputNode

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
    num_actors = args["num_actors"]
    device = "cuda:0"

    actor_cls = ModelActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            layer_size=layer_size,
            num_layers=num_layers,
            num_models=num_models,
            device=device,
        )
        for _ in range(num_actors)
    ]

    return actors


def train_cot(
    actors: List[ModelActor],
    num_models: int,
    num_epochs: int,
    model_prefix: str,
) -> None:
    with InputNode() as inp:
        actors_to_forwards = [actor.forward.bind(inp) for actor in actors]
        actors_to_backwards = actors_to_forwards
        outputs = []
        for i in reversed(range(num_models)):
            actors_to_backwards = [
                actor.backward.bind(actors_to_backwards[j], i)
                for j, actor in enumerate(actors)
            ]
            actors_to_updates = [
                actor.update.bind(actors_to_backwards[j], i)
                for j, actor in enumerate(actors)
            ]
            outputs.extend(actors_to_updates)
        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile()
    for actor in actors:
        ray.get(actor.init_weights.remote())

    for epoch in range(num_epochs):
        for actor in actors:
            ray.get(actor.init_training.remote())

        start = time.perf_counter()
        compiled_dag.execute(None)
        end = time.perf_counter()

        weights = ray.get(actors[0].fetch_weights.remote())
        for idx, weight in enumerate(weights):
            logger.info(f"layer: {idx}, weight: {weight}")

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")

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

    time.sleep(1)


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train_cot(
        actors,
        args["num_models"],
        args["num_epochs"],
        args["model_prefix"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
