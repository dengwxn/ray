import time
from typing import List, Optional, Tuple

import torch

import ray
from .actor import RayDDPWorker
from .common import generate_input_output, log_elapses
from .config import Config
from .correctness import get_ray_ddp_weights
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


def run_ray_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run DDP using Ray Compiled Graphs.

    Args:
        config: Model and training configurations.

    Returns:
        Per-device weights of all layers after each iteration if correctness is checked,
        and the average end-to-end elapse.
    """
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < config.num_actors:
        raise ValueError(f"Needs at least {config.num_actors} GPUs")

    actor_cls = RayDDPWorker.options(num_gpus=1)
    num_layers, layer_size = config.num_layers, config.layer_size
    num_actors = config.num_actors
    actors = [
        actor_cls.remote(
            num_layers,
            layer_size,
            num_actors,
            config.dtype,
            config.learning_rate,
            config.check_correctness,
            config.check_breakdown,
        )
        for _ in range(num_actors)
    ]

    with InputNode() as inp:
        grads = [actor.forward.bind(inp) for actor in actors]
        output = []
        for j in reversed(range(num_layers)):
            for i, actor in enumerate(actors):
                grads[i] = actor.backward.bind(j, grads[i])
            reduced_grads = allreduce.bind(
                [
                    actor.get_grad_to_reduce.bind(grads[i])
                    for i, actor in enumerate(actors)
                ]
            )
            updates = [
                actor.update.bind(j, reduced_grad)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            output.append(updates)
        ends = [
            actor.finish_train.bind(
                *[output[j][i] for j in reversed(range(num_layers))]
            )
            for i, actor in enumerate(actors)
        ]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()

    x, y = generate_input_output(config)
    xs = torch.tensor_split(x, num_actors)
    ys = torch.tensor_split(y, num_actors)
    move_tensor_refs = [
        actor.tensor_to_device.remote(xs[i], ys[i]) for i, actor in enumerate(actors)
    ]
    ray.get(move_tensor_refs)

    weights = None
    if config.check_correctness:
        weights = []
    elapses = []
    for i in range(config.num_iters):
        start = time.perf_counter()
        # Use None as a placeholder.
        ref = compiled_dag.execute(None)
        # [TODO] Print timestamp before ray.get.
        # If correctness is not checked, the result is None.
        cur_iter_weights = ray.get(ref)
        end = time.perf_counter()
        if config.check_correctness:
            weights.append(cur_iter_weights)
        elapse = end - start
        elapses.append(elapse)

    avg_elapse = log_elapses(
        elapses,
        "Running ray ddp...",
    )
    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    if config.check_correctness:
        weights = get_ray_ddp_weights(weights, config.num_actors)
    return weights, avg_elapse
