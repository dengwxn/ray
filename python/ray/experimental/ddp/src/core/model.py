from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class LayeredModel(torch.nn.Module):
    """
    A model that is a chain of (linear, relu) layers.

    Args:
        layer_size: Size of each layer. Each layer is a square (n * n).
        num_layers: Number of layers in the model.
        device: Device the model is on.
        dtype: Data type of the parameters in the model.
        lr: Learning rate for the optimizer.
    """

    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        lr: float,
    ):
        torch.manual_seed(998244353)

        super(LayeredModel, self).__init__()

        self.layers: List[torch.nn.Module] = []
        for _ in range(num_layers):
            # For simplicity, no bias.
            self.layers.append(
                torch.nn.Linear(
                    layer_size, layer_size, device=device, dtype=dtype, bias=False
                )
            )
            self.layers.append(torch.nn.ReLU())
        self.layers: nn.ModuleList = nn.ModuleList(self.layers)
        self.inputs: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.lr: float = lr
        self.criterion = nn.MSELoss()
        # [TODO] Use a single optimizer for all layers.
        # [TODO] Performance comparison: 1 optim vs many?
        self.optimizers: List[optim.SGD] = [
            optim.SGD(self.layers[2 * i].parameters(), lr=self.lr)
            for i in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass for a single layer. Cache the input and activation.

        Args:
            x: Input for this layer.
            layer_idx: Index of the layer.
        """
        self.inputs.append(x)
        linear_layer: torch.nn.Linear = self.layers[2 * layer_idx]
        y: torch.Tensor = linear_layer(x)
        relu_activation: torch.nn.Module = self.layers[2 * layer_idx + 1]
        z: torch.Tensor = relu_activation(y)
        self.activations.append(z)
        return z

    def backward_layer(
        self, grad: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for a single layer. Return the gradient of the loss with
        respect to the input and that with respect to the weight.
        """
        z: torch.Tensor = self.activations[layer_idx]
        x: torch.Tensor = self.inputs[layer_idx]
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        W: torch.Tensor = layer.weight
        # Because the backward pass is done layer by layer, it is necessary to
        # retain the graph unless this is the first layer. Otherwise, the graph
        # is freed after use and cannot be backpropagated through a second time.
        retain_graph = layer_idx != 0
        z.backward(gradient=grad, retain_graph=retain_graph, inputs=[W, x])
        return x.grad, W.grad

    def update_layer(
        self, grad: torch.Tensor, layer_idx: int, check_correctness: bool
    ) -> Optional[torch.Tensor]:
        """
        Update the specified layer with the given gradient. Return the layer
        weight if check correctness, otherwise return None.

        Args:
            grad: Gradient for layer update.
            layer_idx: Index of the layer.
            check_correctness: Whether correctness is checked.
        """
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        [param for param in layer.parameters()][0].grad = grad
        optimizer = self.optimizers[layer_idx]
        optimizer.step()
        if check_correctness:
            return layer.weight
        else:
            return None
