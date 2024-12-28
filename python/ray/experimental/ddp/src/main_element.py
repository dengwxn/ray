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
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 2
        self.num_layers = 2
        self.linear_layers = [
            torch.nn.Linear(
                self.input_size,
                self.hidden_size,
                bias=False,
            ),
            torch.nn.Linear(
                self.hidden_size,
                self.output_size,
                bias=False,
            ),
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
                torch.nn.init.uniform_(layer.weight, -1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs = []
        self.activations = []

        for linear, relu in zip(self.linear_layers, self.relu_layers):
            self.inputs.append(x)
            y = linear(x)
            z = relu(y)
            self.activations.append(z)
            x = z

        return z

    def backward_and_update_all_layers(self, pred: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss = self.criterion(pred, self.y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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


def train_sequential(model: ElementModel, num_epochs: int) -> None:
    for epoch in range(num_epochs):
        # Generate new random input and target
        model.x = torch.randn(1, model.input_size)
        model.y = torch.randn(1, model.output_size)

        logger.info(f"\nEpoch {epoch}")
        logger.info(f"Input x: {model.x}")
        logger.info(f"Target y: {model.y}")

        # Forward pass
        pred = model.forward(model.x)
        logger.info(f"Prediction: {pred}")

        # Backward pass and update
        model.backward_and_update_all_layers(pred)

        # logger updated weights
        logger.info("\nUpdated weights:")
        for idx, layer in enumerate(model.linear_layers):
            logger.info(f"Layer {idx} weight:\n{layer.weight}")


def train_checkpoint(model: ElementModel, num_epochs: int) -> None:
    for epoch in range(num_epochs):
        model.zero_grad()
        model.x = torch.randn(1, model.input_size, requires_grad=True)
        model.y = torch.randn(1, model.output_size)

        logger.info(f"Epoch {epoch}")
        logger.info(f"Input x: {model.x}")
        logger.info(f"Target y: {model.y}")

        # Forward pass
        pred = model.forward(model.x)
        logger.info(f"Prediction: {pred}")

        # # Backward pass and update
        # model.backward_layer(len(model.linear_layers))
        # for idx in reversed(range(model.num_layers)):
        #     model.backward_layer(idx)

        # Backward pass and update
        model.backward_layers([len(model.linear_layers)])
        model.backward_layers([1, 0])

        model.optimizer.step()

        # logger updated weights
        logger.info("\nUpdated weights:")
        for idx, layer in enumerate(model.linear_layers):
            logger.info(f"Layer {idx} weight:\n{layer.weight}")


def main(args: Dict[str, Any]) -> None:
    torch.manual_seed(998244353)

    logger.info("Welcome to Downton Abbey!")

    model = ElementModel()
    num_epochs = 2

    logger.info("Initial weights:")
    for idx, layer in enumerate(model.linear_layers):
        logger.info(f"Layer {idx} weight:\n{layer.weight}")

    if args["mode"] == "sequential":
        train_sequential(model, num_epochs)
    elif args["mode"] == "checkpoint":
        train_checkpoint(model, num_epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
