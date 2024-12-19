import time
from typing import List, Optional, Tuple

import torch
import torch.optim as optim

from .common import generate_input_output, log_elapses
from .config import Config
from .model import LayeredModel


def run_torch(cfg: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run PyTorch.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked, and
        the average elapse across all iterations.
    """

    # To ensure that the model parameters are initialized in the same way across
    # different training methods (Torch, Torch DDP, Ray DDP), the model must be
    # initialized on GPU.
    device = "cuda:0"
    model = LayeredModel(
        cfg.layer_size,
        cfg.num_layers,
        device,
        cfg.dtype,
        cfg.learning_rate,
    )
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    weights: Optional[List[List[torch.Tensor]]] = None
    if cfg.check_correctness:
        weights = []
    elapses: List[float] = []

    for _ in range(cfg.num_iters):
        x, y = generate_input_output(cfg)
        x = x.to(device)
        y = y.to(device)

        start = time.perf_counter()
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = model.criterion(pred, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        if cfg.check_correctness:
            iter_weights: List[torch.Tensor] = []
            for i in range(0, len(model.layers), 2):
                layer: torch.nn.Linear = model.layers[i]
                iter_weights.append(torch.clone(layer.weight))
            weights.append(iter_weights)

        elapse = end - start
        elapses.append(elapse)

    avg_elapse = log_elapses(
        elapses,
        "Running torch...",
    )
    return weights, avg_elapse
