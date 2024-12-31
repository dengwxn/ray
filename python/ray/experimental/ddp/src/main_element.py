import logging
import time
from typing import Any, Dict, List

import torch

from .core.config import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


class ElementModel(torch.nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.device = "cuda:0"
        self.size = 1024
        self.num_layers = num_layers
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.size,
                    self.size,
                    bias=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.relu_layers = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(self.num_layers)]
        )

        self.x = None
        self.y = None

        self.inputs = []
        self.activations = []
        self.gradients = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.linear_layers.parameters(), lr=0.01)

    def init_weights(self) -> None:
        with torch.no_grad():
            for layer in self.linear_layers:
                torch.nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs = []
        self.activations = []
        self.gradients = []

        for linear, relu in zip(self.linear_layers, self.relu_layers):
            self.inputs.append(x)
            y = linear(x)
            z = relu(y)
            self.activations.append(z)
            x = z

        return z

    def backward_aio(self, pred: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss = self.criterion(pred, self.y)
        loss.backward()
        self.optimizer.step()

    def backward_layers(self, idxs: List[int]) -> None:
        if len(idxs) == 1 and idxs[0] == len(self.linear_layers):
            z = self.activations[-1]
            loss = self.criterion(z, self.y)
            loss.backward(
                retain_graph=True,
                inputs=[z],
            )
            self.gradients.append(z.grad)
        else:
            z = self.activations[idxs[0]]
            gradient = self.gradients[-1]
            retain_graph = idxs[-1] != 0
            inputs = []
            for i in idxs:
                x = self.inputs[i]
                w = self.linear_layers[i].weight
                inputs.extend([x, w])
            z.backward(
                gradient=gradient,
                retain_graph=retain_graph,
                inputs=inputs,
            )
            self.gradients.append(x.grad)

            logger.info(f"layers: {idxs}")
            for input in inputs:
                logger.info(f"input grad: {input.grad}")


def train_sequential(model: ElementModel, num_epochs: int, model_file: str) -> None:
    for epoch in range(num_epochs):
        start = time.perf_counter()

        model.x = torch.randn(1, model.size, requires_grad=True).to(model.device)
        model.y = torch.randn(1, model.size).to(model.device)

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {model.x}")
        logger.info(f"target: {model.y}")

        # Forward pass
        pred = model.forward(model.x)
        logger.info(f"prediction: {pred}")

        # Backward and update all in one
        model.backward_aio(pred)

        end = time.perf_counter()

        logger.info("updated weights:")
        for idx, layer in enumerate(model.linear_layers):
            logger.info(f"layer {idx} weight: {layer.weight}")

        if epoch > 0:
            logger.warning(f"epoch: {epoch} elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for layer in model.linear_layers:
            f.write(f"{layer.weight}\n")


def train_cot(models: List[ElementModel], num_epochs: int, model_file: str) -> None:
    assert len(models) == 2, "Only support two models for now"

    for epoch in range(num_epochs):
        start = time.perf_counter()

        models[0].x = torch.randn(1, models[0].size, requires_grad=True).to(
            models[0].device
        )
        models[1].y = torch.randn(1, models[1].size).to(models[1].device)

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {models[0].x}")
        logger.info(f"target: {models[1].y}")

        # Forward pass
        pred1 = models[0].forward(models[0].x)
        pred1_detached = pred1.detach().requires_grad_(True)
        pred2 = models[1].forward(pred1_detached)
        logger.info(f"prediction: {pred2}")

        # Backward and update for all models
        loss = models[1].criterion(pred2, models[1].y)
        loss.backward()
        models[1].optimizer.step()
        models[1].optimizer.zero_grad()

        pred1.backward(pred1_detached.grad)
        models[0].optimizer.step()
        models[0].optimizer.zero_grad()

        end = time.perf_counter()

        logger.info("updated weights:")
        for model in models:
            for idx, layer in enumerate(model.linear_layers):
                logger.info(f"layer {idx} weight: {layer.weight}")

        if epoch > 0:
            logger.warning(f"epoch: {epoch} elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for model in models:
            for layer in model.linear_layers:
                f.write(f"{layer.weight}\n")


def main(args: Dict[str, Any]) -> None:
    logger.info("Welcome to Downton Abbey!")

    num_layers = 32
    num_epochs = 10

    if args["mode"] == "sequential":
        model = ElementModel(
            num_layers=num_layers,
        )

        torch.manual_seed(998244353)
        model.init_weights()
        model = model.to(model.device)

        train_sequential(
            model,
            num_epochs,
            args["model_file"],
        )
    elif args["mode"] == "cot":
        num_models = 2
        models = [
            ElementModel(
                num_layers=num_layers // num_models,
            )
            for _ in range(num_models)
        ]

        torch.manual_seed(998244353)
        for model in models:
            model.init_weights()
            model = model.to(model.device)

        train_cot(
            models,
            num_epochs,
            args["model_file"],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
