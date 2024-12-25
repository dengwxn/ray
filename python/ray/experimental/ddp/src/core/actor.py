import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

import ray
from .common import secs_to_micros
from .model import LayeredModel
from ray.air._internal import torch_utils


@ray.remote
class RayDDPWorker:
    """
    An actor class wrapper around the pytorch model.

    Args:
        layer_size: Layer size. Each layer is a square (layer_size * layer_size).
        num_layers: Number of layers.
        world_size: Number of actors.
        dtype: Data type for the model.
        lr: Learning rate for the optimizer.
        check_correctness: Whether to check correctness.
        check_breakdown: Whether to check performance breakdown.
    """

    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        world_size: int,
        dtype: torch.dtype,
        lr: float,
        check_correctness: bool,
        check_tracing: bool,
    ):
        # Each device has a single GPU.
        self.device = torch_utils.get_devices()[0]
        self.model: LayeredModel = LayeredModel(
            layer_size, num_layers, self.device, dtype, lr
        )
        self.num_layers = num_layers
        self.world_size = world_size
        self.check_correctness = check_correctness

        self.x: torch.Tensor = None
        self.y: torch.Tensor = None

        self.check_tracing = check_tracing
        if check_tracing:
            self.it = 0
            self.time: Dict[str] = {}

    def init_tracing(self) -> None:
        if not self.check_tracing:
            return

        logger = logging.getLogger(__name__)
        logger.info(f"Start iteration {self.it}...")

        self.time = {
            "forward_starts": [],
            "forward_ends": [],
            "backward_starts": [],
            "backward_ends": [],
            "update_starts": [],
            "update_ends": [],
        }

    def finish_tracing(self) -> None:
        if not self.check_tracing:
            return

        logger = logging.getLogger(__name__)
        logger.info(f"Finish iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        def log(key: str, elapse: float):
            logger.info(
                f"{key} elapse: {secs_to_micros(elapse)} us, percent: {round(elapse / total * 100, 1)}%"
            )

        self.update_time("end")
        total = self.time["end"] - self.time["start"]
        logger.info("")
        log(
            "total",
            total,
        )
        log(
            "fw.total",
            self.time["forward_ends"][-1] - self.time["forward_starts"][0],
        )
        log(
            "loss.compute",
            self.time["backward_loss_start"] - self.time["forward_ends"][-1],
        )
        log(
            "loss.backward",
            self.time["backward_loss_end"] - self.time["backward_loss_start"],
        )
        log(
            "bw.total",
            self.time["update_ends"][-1] - self.time["backward_starts"][0],
        )
        log(
            "bw.backward",
            sum(
                [
                    self.time["backward_ends"][i] - self.time["backward_starts"][i]
                    for i in range(self.num_layers)
                ]
            ),
        )
        log(
            "bw.allreduce",
            sum(
                [
                    self.time["update_starts"][i] - self.time["backward_ends"][i]
                    for i in range(self.num_layers)
                ]
            ),
        )
        log(
            "bw.update",
            sum(
                [
                    self.time["update_ends"][i] - self.time["update_starts"][i]
                    for i in range(self.num_layers)
                ]
            ),
        )

        logger.info("")
        for i in range(self.num_layers):
            log(
                f"fw.{i}",
                self.time["forward_ends"][i] - self.time["forward_starts"][i],
            )

        logger.info("")
        for i in range(self.num_layers):
            log(
                f"bw.backward.{i}",
                self.time["backward_ends"][i] - self.time["backward_starts"][i],
            )
            log(
                f"bw.allreduce.{i}",
                self.time["update_starts"][i] - self.time["backward_ends"][i],
            )
            log(
                f"bw.update.{i}",
                self.time["update_ends"][i] - self.time["update_starts"][i],
            )
        logger.info("")

    def update_time(self, key: str) -> None:
        if not self.check_tracing:
            return
        timestamp = time.perf_counter()
        if key not in self.time:
            self.time[key] = timestamp
        else:
            assert isinstance(self.time[key], list)
            self.time[key].append(timestamp)

    def tensor_to_device(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Move the input and ground truth to the device. The input and ground truth
        were on the CPU so they need to be moved.

        Args:
            x: Input.
            y: Ground truth.
        """
        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def start_train(self) -> None:
        """
        Start the training process for one iteration. Clear the old gradients,
        stored inputs, and activations from last iteration.
        """
        self.init_tracing()
        self.update_time("start")
        self.model.zero_grad()
        self.model.inputs = []
        self.model.activations = []

    def forward(self, placeholder: Any) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for the model.
        1. Compute the prediction with the input.
        2. Compute the loss from the prediction and the ground truth.
        3. Compute and return the gradient of the loss with respect to the prediction.

        Args:
            placeholder: Unused placeholder for the input.

        Returns:
            Gradient of the loss with respect to the prediction.
        """
        self.start_train()
        x = self.x
        y = self.y

        for i in range(self.num_layers):
            self.update_time("forward_starts")
            x = self.model.forward_layer(x, i)
            self.update_time("forward_ends")

        pred = x
        loss: torch.Tensor = self.model.criterion(pred, y)

        self.update_time("backward_loss_start")
        # Compute the gradient of the loss with respect to the prediction. Retain the
        # graph (i.e., not free the graph) for subsequent back-propagation computations.
        loss.backward(retain_graph=True, inputs=[pred])
        self.update_time("backward_loss_end")

        return pred.grad, None

    def backward_layer(
        self, idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backward pass for the specified layer.

        Args:
            idx: Index of the layer.
            grad: `grad[0]` is the gradient of the loss with respect to this
                layer's output. This is a workaround for the issue:
                https://github.com/ray-project/ray/issues/48522

        Returns:
            Tuple of the gradients of the loss with respect to the input and the
            weight of this layer.
        """
        self.update_time("backward_starts")
        grad_to_bp, _ = grad
        result = self.model.backward_layer(grad_to_bp, idx)
        self.update_time("backward_ends")

        return result

    def update_layer(self, idx: int, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update the weights of the specified layer with the given allreduced gradient.

        Args:
            idx: Index of the layer.
            grad: Allreduced gradient for this layer.

        Returns:
            The updated weights of this layer if correctness is checked, otherwise
            None.
        """
        self.update_time("update_starts")
        # For mathematical equivalence, divide the allreduced gradient by the
        # world size (i.e., the number of actors).
        grad /= self.world_size
        result = self.model.update_layer(grad, idx, self.check_correctness)
        self.update_time("update_ends")

        return result

    def finish_train(
        self, *updates: Optional[torch.Tensor]
    ) -> List[Optional[torch.Tensor]]:
        """
        Finish the current iteration of training by gather all results from weight
        updates across different layers.

        Args:
            updates: A tuple of any number of results from weight updates.
                If correctness is checked, an update result is the updated weight of
                a layer. Otherwise, it is None.

        Returns:
            The list of all results from weight updates.
        """
        self.finish_tracing()
        if self.check_correctness:
            return updates
        return None

    def get_grad_to_reduce(
        self, grad: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extracts the gradient to be reduced.

        Args:
            grad: `grad[1]` is the gradient of the loss with respect to the weight.
                This gradient is used in the allreduce operation.

        Returns:
            The gradient to be reduced.

        When an allreduce binds a class method output with `num_returns > 1`,
        an error is thrown. This is a workaround.
        See: https://github.com/ray-project/ray/issues/48522
        """
        _, grad_to_weight = grad
        return grad_to_weight
