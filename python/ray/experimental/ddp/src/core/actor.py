import time
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

import ray
from .common import secs_to_micros
from .model import LayeredModel
from ray.air._internal import torch_utils


@ray.remote
class RayDDPWorker:
    """
    An actor class wrapper around the pytorch model.

    Args:
        num_layers: Number of layers in the model.
        layer_size: Size of each layer. Each layer is a square (n * n).
        world_size: Number of actors.
        dtype: Data type of the parameters in the model.
        lr: Learning rate for the optimizer.
        check_correctness: Whether correctness is checked.
        breakdown_performance: Whether to print performance breakdown.
    """

    def __init__(
        self,
        num_layers: int,
        layer_size: int,
        world_size: int,
        dtype: torch.dtype,
        lr: float,
        check_correctness: bool,
        breakdown_performance: bool,
    ):
        self.num_layers = num_layers
        # Each device has a single GPU.
        self.device = torch_utils.get_devices()[0]

        self.model: LayeredModel = LayeredModel(
            layer_size, num_layers, self.device, dtype, lr
        )
        self.world_size: int = world_size

        self.check_correctness = check_correctness
        # [TODO] Remove manual timing and use profiler.
        self.breakdown_performance = breakdown_performance
        self.it = 0
        self.start_time: float = None
        self.pre_forward_time: float = None
        self.forward_times: List[Tuple[float, Tuple]] = None
        self.loss_time: float = None
        self.pre_backward_time: float = None
        self.backward_times: List[Tuple[float, float]] = None
        self.update_times: List[Tuple[float, float]] = None
        self.end_time: float = None

        self.x: torch.Tensor = None
        self.y: torch.Tensor = None

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
        if self.breakdown_performance:
            self.start_time = time.perf_counter()

        self.model.zero_grad()
        self.model.inputs = []
        self.model.activations = []

        if self.breakdown_performance:
            self.forward_times = []
            self.backward_times = []
            self.update_times = []

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
        if self.breakdown_performance:
            self.pre_forward_time = time.perf_counter()

        for i in range(self.num_layers):
            forward_start_time = None
            if self.breakdown_performance:
                forward_start_time = time.perf_counter()
            x = self.model.forward_layer(x, i)
            forward_end_time = None
            if self.breakdown_performance:
                forward_end_time = time.perf_counter()
                self.forward_times.append((forward_start_time, forward_end_time))

        pred = x
        loss: torch.Tensor = self.model.criterion(pred, y)
        if self.breakdown_performance:
            self.loss_time = time.perf_counter()

        # Compute the gradient of the loss with respect to the prediction.
        # Retain the graph (i.e., not free the graph) for subsequent backprop
        # computations.
        loss.backward(retain_graph=True, inputs=[pred])
        if self.breakdown_performance:
            self.pre_backward_time = time.perf_counter()
        return pred.grad, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backward pass for the specified layer.

        Args:
            layer_idx: Index of the layer.
            grad: `grad[0]` is the gradient of the loss with respect to this
                layer's output. This is a workaround for the issue:
                https://github.com/ray-project/ray/issues/48522

        Returns:
            Tuple of the gradients of the loss with respect to the input and the
            weight of this layer.
        """
        backward_start_time = None
        if self.breakdown_performance:
            backward_start_time = time.perf_counter()
        # No need to move the gradient because it is already on this device.
        bp_grad, _ = grad
        result = self.model.backward_layer(bp_grad, layer_idx)
        backward_end_time = None
        if self.breakdown_performance:
            backward_end_time = time.perf_counter()
            self.backward_times.append((backward_start_time, backward_end_time))
        return result

    def update(self, layer_idx: int, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update the weights of the specified layer with the given allreduced gradient.

        Args:
            layer_idx: Index of the layer.
            grad: Allreduced gradient for this layer.

        Returns:
            The updated weights of this layer if correctness is checked, otherwise
            None.
        """
        update_start_time = None
        if self.breakdown_performance:
            update_start_time = time.perf_counter()
        # No need to move the gradient because it is already on this device.
        # For mathematical equivalence, divide the allreduced gradient by the
        # world size (i.e., the number of actors).
        grad /= self.world_size
        result = self.model.update_layer(grad, layer_idx, self.check_correctness)
        update_end_time = None
        if self.breakdown_performance:
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
        if self.breakdown_performance:
            self.end_time = time.perf_counter()

            print(f"start time: {self.start_time}")
            print(f"pre forward time: {self.pre_forward_time}")
            for i, (start, end) in enumerate(self.forward_times):
                print(f"forward time layer {i}: start: {start}, end: {end}")
            print(f"loss time: {self.loss_time}")
            print(f"pre backward time: {self.pre_backward_time}")
            for i, (start, end) in enumerate(self.backward_times):
                print(f"backward time layer {i}: start: {start}, end: {end}")
            for i, (start, end) in enumerate(self.update_times):
                print(f"update time layer {i}: start: {start}, end: {end}")
            print(f"end time: {self.end_time}")
            print()

            self.start_time = secs_to_micros(self.start_time)
            self.pre_forward_time = secs_to_micros(self.pre_forward_time)
            for i, (start, end) in enumerate(self.forward_times):
                self.forward_times[i] = (secs_to_micros(start), secs_to_micros(end))
            self.loss_time = secs_to_micros(self.loss_time)
            self.pre_backward_time = secs_to_micros(self.pre_backward_time)
            for i, (start, end) in enumerate(self.backward_times):
                self.backward_times[i] = (secs_to_micros(start), secs_to_micros(end))
            for i, (start, end) in enumerate(self.update_times):
                self.update_times[i] = (secs_to_micros(start), secs_to_micros(end))
            self.end_time = secs_to_micros(self.end_time)

            print(f"=============== Iteration {self.it} Elapses =================")
            print(f"pre forward elapse: {self.pre_forward_time - self.start_time}")
            for i, (start, end) in enumerate(self.forward_times):
                print(f"forward layer {i} elapse: {end - start}")
            print(f"loss elapse: {self.loss_time - self.forward_times[-1][1]}")
            print(f"pre backward elapse: {self.pre_backward_time - self.loss_time}")
            for i, (start, end) in enumerate(self.backward_times):
                print(f"backward layer {i} elapse: {end - start}")
            for i, (start, end) in enumerate(self.update_times):
                print(
                    f"allreduce layer {i} elapse: {start - self.backward_times[i][1]}"
                )
                print(f"update layer {i} elapse: {end - start}")
            print(f"total elapse: {self.end_time - self.start_time}")
            print_time = secs_to_micros(time.perf_counter())
            print(f"print elapse: {print_time - self.end_time}")
            print(
                "====================================================================="
            )
            print()

            self.it += 1

        return updates if self.check_correctness else None

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
        _, reduce_grad = grad
        return reduce_grad
