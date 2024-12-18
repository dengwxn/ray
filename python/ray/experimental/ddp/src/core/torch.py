import time
from typing import List, Optional, Tuple

import torch
import torch.optim as optim

from .common import generate_input_output, print_elapses
from .config import Config
from .model import LayeredModel


def run_torch(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
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
        config.layer_size,
        config.num_layers,
        device,
        config.dtype,
        config.learning_rate,
    )
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    weights: Optional[List[List[torch.Tensor]]] = None
    if config.check_correctness:
        weights = []
    elapses: List[float] = []
    for i in range(config.num_iters):
        x, y = generate_input_output(config)
        x = x.to(device)
        y = y.to(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(pred, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        if config.check_correctness:
            cur_iter_weights: List[torch.Tensor] = []
            for i in range(0, len(model.layers), 2):
                layer: torch.nn.Linear = model.layers[i]
                cur_iter_weights.append(torch.clone(layer.weight))
            weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    avg_elapse = print_elapses(elapses, "torch")
    return weights, avg_elapse
