import logging
import time
from typing import Any, Dict, List

import torch

from ..core.config import parse_args
from ..core.mp.model import ModelElement

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


def train_cot(models: List[ModelElement], num_epochs: int, model_file: str) -> None:
    num_models = len(models)

    for epoch in range(num_epochs):
        start = time.perf_counter()

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

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {models[0].x}")
        logger.info(f"target: {models[-1].y}")

        # Forward pass through all models
        intermediate_outputs = []
        input = models[0].x

        for i, model in enumerate(models):
            pred = model.forward(input)
            if i < num_models - 1:
                # Detach intermediate outputs to create separate computation graphs
                input = pred.detach().requires_grad_(True)
                intermediate_outputs.append((pred, input))
            else:
                # Last model's output
                pred_final = pred

        logger.info(f"prediction: {pred_final}")

        # Backward pass and optimization starting from the last model
        # Initialize with the final loss
        loss = models[-1].criterion(pred_final, models[-1].y)
        models[-1].backward(
            loss=loss,
        )
        models[-1].update()

        # Propagate gradients backward through intermediate models
        for i in range(num_models - 2, -1, -1):
            pred, pred_detached = intermediate_outputs[i]
            models[i].backward(
                pred=pred,
                grad=pred_detached.grad,
            )
            models[i].update()

        end = time.perf_counter()

        # Log updated weights
        for model_idx, model in enumerate(models):
            for layer_idx, layer in enumerate(model.linear_layers):
                logger.info(
                    f"model: {model_idx}, layer: {layer_idx}, weight: {layer.weight}"
                )

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for model in models:
            for layer in model.linear_layers:
                f.write(f"{layer.weight}\n")


def main(args: Dict[str, Any]) -> None:
    logger.info("Welcome to Downton Abbey!")

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

    torch.manual_seed(998244353)
    for model in models:
        model.init_weights()
        model = model.to(model.device)

    num_epochs = args["num_epochs"]
    train_cot(
        models,
        num_epochs,
        args["model_file"],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
