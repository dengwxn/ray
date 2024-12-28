import time

import torch

from .actor_offline import RayDDPWorkerOffline
from .common import generate_input_output, log_ray_offline_elapses
from .config import Config


def run_ray_ddp_offline(cfg: Config) -> None:
    assert cfg.num_actors == 1, "Sanity check for offline case"

    actor = RayDDPWorkerOffline(
        cfg.layer_size,
        cfg.num_layers,
        cfg.num_actors,
        cfg.dtype,
        cfg.learning_rate,
        cfg.check_correctness,
        cfg.check_breakdown,
    )
    elapses = []

    x, y = generate_input_output(cfg)
    xs = torch.tensor_split(x, cfg.num_actors)[0]
    ys = torch.tensor_split(y, cfg.num_actors)[0]
    actor.tensor_to_device(xs, ys)

    def execute(actor: RayDDPWorkerOffline) -> None:
        grads = actor.forward(None)
        # outputs = []
        for j in reversed(range(cfg.num_layers)):
            grads = actor.backward_layer(j, grads)
            # [TODO] Focus on backward_layer.
            # # [TODO] `allreduce` unverified.
            # # grads_allreduced = allreduce(
            # #     actor.get_grad_to_reduce(grads)
            # # )
            # grads_allreduced = actor.get_grad_to_reduce(grads)
            # updates = actor.update_layer(j, grads_allreduced)
            # outputs.append(updates)
        # ends = actor.finish_train(*outputs)
        actor.finish_tracing()

    for _ in range(cfg.num_iters):
        start = time.perf_counter()
        execute(actor)
        end = time.perf_counter()
        elapse = end - start
        elapses.append(elapse)

    actors_to_traces = actor.fetch_traces()
    log_ray_offline_elapses(actors_to_traces, cfg.output_path, cfg.output_prefix)
