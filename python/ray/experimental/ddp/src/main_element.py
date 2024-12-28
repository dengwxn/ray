import logging
from typing import Any, Dict, List

import torch

from .core.config import parse_args

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


class ElementModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.size = 1024
        self.num_layers = 4
        self.linear_layers = [
            torch.nn.Linear(
                self.size,
                self.size,
                bias=False,
            )
            for _ in range(self.num_layers)
        ]
        self.relu_layers = [torch.nn.ReLU() for _ in range(self.num_layers)]
        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.relu_layers = torch.nn.ModuleList(self.relu_layers)

        self.x = None
        self.y = None

        self.inputs = []
        self.activations = []
        self.gradients = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.linear_layers.parameters(), lr=0.01)

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

    def backward_and_update_all_layers(self, pred: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.criterion(pred, self.y)
        loss.backward()
        self.optimizer.step()

    def backward_layer(self, idx: int) -> None:
        if idx == len(self.linear_layers):
            z = self.activations[-1]
            loss = self.criterion(z, self.y)
            loss.backward(
                retain_graph=True,
                inputs=[z],
            )
            self.gradients.append(z.grad)
        else:
            z = self.activations[idx]
            x = self.inputs[idx]
            w = self.linear_layers[idx].weight
            grad = self.gradients[-1]
            retain_graph = idx != 0
            z.backward(
                gradient=grad,
                retain_graph=retain_graph,
                inputs=[x, w],
            )
            self.gradients.append(x.grad)

            logger.info(f"layer: {idx}, input grad: {x.grad}, weight grad: {w.grad}")

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
        model.x = torch.randn(1, model.size, requires_grad=True)
        model.y = torch.randn(1, model.size)

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {model.x}")
        logger.info(f"target: {model.y}")

        # Forward pass
        pred = model.forward(model.x)
        logger.info(f"prediction: {pred}")

        # Backward pass and update
        model.backward_and_update_all_layers(pred)

        logger.info("updated weights:")
        for idx, layer in enumerate(model.linear_layers):
            logger.info(f"layer {idx} weight: {layer.weight}")

    with open(model_file, "w") as f:
        for layer in model.linear_layers:
            f.write(f"{layer.weight}\n")


def train_checkpoint(model: ElementModel, num_epochs: int, model_file: str) -> None:
    for epoch in range(num_epochs):
        model.zero_grad()
        model.x = torch.randn(1, model.size, requires_grad=True)
        model.y = torch.randn(1, model.size)

        logger.info(f"epoch: {epoch}")
        logger.info(f"input: {model.x}")
        logger.info(f"target: {model.y}")

        # Forward pass
        pred = model.forward(model.x)
        logger.info(f"prediction: {pred}")

        # Backward pass and update
        model.backward_layers([len(model.linear_layers)])
        model.backward_layers([3, 2])
        model.backward_layers([1, 0])

        # Update weights
        model.optimizer.step()

        logger.info("updated weights:")
        for idx, layer in enumerate(model.linear_layers):
            logger.info(f"layer {idx} weight: {layer.weight}")

    with open(model_file, "w") as f:
        for layer in model.linear_layers:
            f.write(f"{layer.weight}\n")


def main(args: Dict[str, Any]) -> None:
    torch.manual_seed(998244353)

    logger.info("Welcome to Downton Abbey!")

    model = ElementModel()
    num_epochs = 4

    if args["mode"] == "sequential":
        train_sequential(model, num_epochs, args["model_file"])
    elif args["mode"] == "checkpoint":
        train_checkpoint(model, num_epochs, args["model_file"])


if __name__ == "__main__":
    args = parse_args()
    main(args)
