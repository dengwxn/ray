import logging
import time
from typing import Any, List, Optional, Tuple

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
        check_breakdown: bool,
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

        self.check_breakdown = check_breakdown
        if check_breakdown:
            self.it = 0
            self.start_time: Optional[float] = None
            self.pre_forward_time: Optional[float] = None
            self.forward_times: List[Tuple[float, Tuple]] = []
            self.loss_time: Optional[float] = None
            self.pre_backward_time: Optional[float] = None
            self.backward_times: List[Tuple[float, float]] = []
            self.update_times: List[Tuple[float, float]] = []
            self.end_time: Optional[float] = None

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
        if self.check_breakdown:
            self.start_time = time.perf_counter()

        self.model.zero_grad()
        self.model.inputs = []
        self.model.activations = []

    def forward(self, placeholder: Any) -> torch.Tensor:
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

        if self.check_breakdown:
            self.pre_forward_time = time.perf_counter()

        for i in range(self.num_layers):
            if self.check_breakdown:
                forward_start_time = time.perf_counter()

            x = self.model.forward_layer(x, i)

            if self.check_breakdown:
                forward_end_time = time.perf_counter()
                self.forward_times.append((forward_start_time, forward_end_time))

        pred = x
        loss: torch.Tensor = self.model.criterion(pred, y)

        if self.check_breakdown:
            self.loss_time = time.perf_counter()

        # Compute the gradient of the loss with respect to the prediction.
        # Retain the graph (i.e., not free the graph) for subsequent backprop
        # computations.
        loss.backward(retain_graph=True, inputs=[pred])

        if self.check_breakdown:
            self.pre_backward_time = time.perf_counter()

        return pred.grad, None

    def backward(
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
        if self.check_breakdown:
            backward_start_time = time.perf_counter()

        # No need to move the gradient because it is already on this device.
        bp_grad, _ = grad
        result = self.model.backward_layer(bp_grad, idx)

        if self.check_breakdown:
            backward_end_time = time.perf_counter()
            self.backward_times.append((backward_start_time, backward_end_time))

        return result

    def update(self, idx: int, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update the weights of the specified layer with the given allreduced gradient.

        Args:
            idx: Index of the layer.
            grad: Allreduced gradient for this layer.

        Returns:
            The updated weights of this layer if correctness is checked, otherwise
            None.
        """
        if self.check_breakdown:
            update_start_time = time.perf_counter()

        # No need to move the gradient because it is already on this device.
        # For mathematical equivalence, divide the allreduced gradient by the
        # world size (i.e., the number of actors).
        grad /= self.world_size
        result = self.model.update_layer(grad, idx, self.check_correctness)

        if self.check_breakdown:
            update_end_time = time.perf_counter()
            self.update_times.append((update_start_time, update_end_time))

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
        if self.check_breakdown:
            self.end_time = time.perf_counter()

            logger = logging.getLogger(__name__)
            logger.info(f"Iteration {self.it} finishes")
            self.it += 1

            total = self.end_time - self.start_time
            log("total", total)

            def log(key: str, elapse: float):
                logger.info(
                    f"{key} elapse: {secs_to_micros(elapse)} us, percentage: {round(elapse / total * 100)}%"
                )

            log("pre forward", self.pre_forward_time - self.start_time)
            for i, (start, end) in enumerate(self.forward_times):
                log(f"forward layer {i}", end - start)
            log("loss", self.loss_time - self.forward_times[-1][1])
            log("pre backward", self.pre_backward_time - self.loss_time)
            for i, (start, end) in enumerate(self.backward_times):
                log(f"backward layer {i}", end - start)
            for i, (start, end) in enumerate(self.update_times):
                log(f"allreduce layer {i}", start - self.backward_times[i][1])
                log(f"update layer {i}", end - start)

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
