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


def run_ray_ddp(cfg: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run Ray DDP using Ray Compiled Graphs.

    Args:
        config: Model and training configurations.

    Returns:
        Per-device weights of all layers after each iteration if correctness is checked,
        and the average end-to-end elapse.
    """
    ray.init()
    num_gpus = sum(node["Resources"].get("GPU", 0) for node in ray.nodes())
    assert num_gpus >= cfg.num_actors

    actor_cls = RayDDPWorker.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            cfg.layer_size,
            cfg.num_layers,
            cfg.num_actors,
            cfg.dtype,
            cfg.learning_rate,
            cfg.check_correctness,
            cfg.check_breakdown,
        )
        for _ in range(cfg.num_actors)
    ]

    with InputNode() as inp:
        grads = [actor.forward.bind(inp) for actor in actors]
        outputs = []
        for j in reversed(range(cfg.num_layers)):
            grads = [actor.backward.bind(j, grads[i]) for i, actor in enumerate(actors)]
            grads_allreduced = allreduce.bind(
                [
                    actor.get_grad_to_reduce.bind(grads[i])
                    for i, actor in enumerate(actors)
                ]
            )
            updates = [
                actor.update.bind(j, grad)
                for actor, grad in zip(actors, grads_allreduced)
            ]
            outputs.append(updates)
        ends = [
            actor.finish_train.bind(
                *[outputs[j][i] for j in reversed(range(cfg.num_layers))]
            )
            for i, actor in enumerate(actors)
        ]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()

    weights = None
    if cfg.check_correctness:
        weights = []
    elapses = []

    x, y = generate_input_output(cfg)
    xs = torch.tensor_split(x, cfg.num_actors)
    ys = torch.tensor_split(y, cfg.num_actors)
    tensor_to_device_refs = [
        actor.tensor_to_device.remote(xs[i], ys[i]) for i, actor in enumerate(actors)
    ]
    ray.get(tensor_to_device_refs)

    for _ in range(cfg.num_iters):
        start = time.perf_counter()
        ref = compiled_dag.execute(None)
        iter_weights = ray.get(ref)  # [TODO] Print timestamp before ray.get.
        end = time.perf_counter()

        if cfg.check_correctness:
            weights.append(iter_weights)

        elapse = end - start
        elapses.append(elapse)

    compiled_dag.teardown()
    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    if cfg.check_correctness:
        weights = get_ray_ddp_weights(weights, cfg.num_actors)
    avg_elapse = log_elapses(
        elapses,
        "Running ray ddp...",
    )
    return weights, avg_elapse
