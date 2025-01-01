from typing import List, Optional, Tuple

import torch


class ModelElement(torch.nn.Module):
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.device = device
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    layer_size,
                    layer_size,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.relu_layers = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(num_layers)]
        )

        self.x = None
        self.y = None
        self.inputs: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.linear_layers.parameters(), lr=0.01)

    def init_weights(self) -> None:
        # [TODO] Init seed.
        with torch.no_grad():
            for layer in self.linear_layers:
                torch.nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

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

    def backward(
        self,
        loss: Optional[torch.Tensor] = None,
        pred: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
    ) -> None:
        if loss is not None:
            assert pred is None
            loss.backward()
        elif pred is not None:
            assert grad is not None
            pred.backward(grad)

    def update(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
