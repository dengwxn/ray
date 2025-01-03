import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..core.config import parse_args
from ..core.mp.model import ModelElement

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_models(args: Dict[str, Any]) -> List[ModelElement]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    device = "cuda:0"
    num_models = args["num_models"]

    models = [
        ModelElement(
            layer_size=layer_size,
            num_layers=num_layers // num_models,
            device=device,
        )
        for _ in range(num_models)
    ]

    return models


def init_weights(models: List[ModelElement]) -> None:
    torch.manual_seed(998244353)
    for model in models:
        model.init_weights()
        model = model.to(model.device)


def init_training(models: List[ModelElement]) -> None:
    # Generate input for first model and target for last model
    models[0].x = torch.randn(
        1,
        models[0].layer_size,
        requires_grad=True,
    ).to(
        models[0].device,
    )
    models[-1].y = torch.randn(
        1,
        models[-1].layer_size,
    ).to(
        models[-1].device,
    )


def forward(
    models: List[ModelElement],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    intermediates = []
    input = models[0].x
    for i, model in enumerate(models):
        pred = model.forward(input)
        if i < len(models) - 1:
            input = pred.detach().requires_grad_(True)
        else:
            input = pred
        intermediates.append((pred, input))
    return intermediates


def backward(
    models: List[ModelElement],
    intermediates: List[Tuple[torch.Tensor, torch.Tensor]],
    idx: int,
) -> None:
    if idx == len(models) - 1:
        loss = models[idx].criterion(
            intermediates[idx][0],
            models[idx].y,
        )
        pred = None
        grad = None
    else:
        loss = None
        pred, input = intermediates[idx]
        grad = input.grad
    models[idx].backward(
        loss=loss,
        pred=pred,
        grad=grad,
    )


def update(
    models: List[ModelElement],
    idx: int,
) -> None:
    models[idx].update()


def train_cot(
    models: List[ModelElement],
    num_epochs: int,
    model_file: str,
) -> None:
    init_weights(models)

    for epoch in range(num_epochs):
        start = time.perf_counter()

        init_training(models)

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {models[0].x}")
        logger.info(f"target: {models[-1].y}")

        # Forward pass through all models
        intermediates = forward(models)
        logger.info(f"prediction: {intermediates[-1][0]}")

        # Propagate gradients backward through intermediate models
        for i in reversed(range(len(intermediates))):
            backward(
                models,
                intermediates,
                i,
            )

        for i in reversed(range(len(intermediates))):
            update(
                models,
                i,
            )

        end = time.perf_counter()

        # Log updated weights
        for model_idx, model in enumerate(models):
            weights = model.fetch_weights()
            for layer_idx, layer_weight in enumerate(weights):
                logger.info(
                    f"model: {model_idx}, layer: {layer_idx}, weight: {layer_weight}"
                )

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for model in models:
            weights = model.fetch_weights()
            for weight in weights:
                f.write(f"{weight}\n")


def main(args: Dict[str, Any]) -> None:
    models = init_models(args)

    train_cot(
        models,
        args["num_epochs"],
        args["model_file"],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
