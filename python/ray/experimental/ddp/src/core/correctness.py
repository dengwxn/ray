import logging
from typing import Dict, List

import torch


def get_torch_ddp_weights_per_device(
    weights_dict: Dict[int, List[List[torch.Tensor]]], world_size: int
) -> List[List[torch.Tensor]]:
    """
    Extract the per-device weights of all layers after each iteration.
    Check that the model is consistent across all devices after each iteration.

    Args:
        weights_dict: Dictionary that maps ranks (device ids) to its weights of
            all layers after each iteration.
        world_size: The total number of devices.

    Returns:
        Per-device weights of all layers after each iteration.
    """
    weights_across_devices = list(dict(weights_dict).values())
    assert len(weights_across_devices) == world_size
    weights_per_device = weights_across_devices[0]
    # Weights on each device.
    for i in range(1, world_size):
        cur_device_weights = weights_across_devices[i]
        assert len(weights_per_device) == len(cur_device_weights)
        # Weights per iteration on each device.
        for j in range(len(weights_per_device)):
            assert len(weights_per_device[j]) == len(cur_device_weights[j])
            # Weights per layer in each iteration.
            for k in range(len(weights_per_device[j])):
                assert torch.equal(weights_per_device[j][k], cur_device_weights[j][k])
    return weights_per_device


def get_ray_ddp_weights_per_device(
    weights: List[List[List[torch.Tensor]]], num_actors: int
) -> List[List[torch.Tensor]]:
    """
    Check that the model is consistent across all ranks after each iteration
    of training. Deduplicate the weights after the checks.

    Args:
        weights: Weights from running Ray DDP for a number of iterations.
            `weights[i]` is the weights across all actors for the ith iteration.

    Returns:
        Per-device weights of all layers after each iteration.
    """
    weights_per_device: List[List[torch.Tensor]] = []
    for single_iter_weights in weights:
        per_device_len = len(single_iter_weights[0])
        for i in range(1, num_actors):
            for j in range(per_device_len):
                assert torch.equal(
                    single_iter_weights[0][j], single_iter_weights[i][j]
                ), f"{single_iter_weights[0][j]} vs. {single_iter_weights[i][j]}"
        weights_per_device.append(single_iter_weights[0])
    return weights_per_device


def compare_weights(
    W1: List[List[torch.Tensor]],
    W2: List[List[torch.Tensor]],
    desc: str,
    allow_error: bool = False,
) -> None:
    """
    Compare the weights after each iteration across different training approaches.

    Args:
        W1: Weights after each iteration from one approach.
        W2: Weights after each iteration from the other approach.
        desc: Description of approaches
        allow_error: Whether small errors are allowed.
            Small errors are common if one of the approaches uses DDP and the
            other does not.
    """
    assert len(W1) == len(W2)
    # w1, w2 are weights after a single iteration for the 1st and 2nd approaches,
    # respectively.
    max_diff = 0
    for w1, w2 in zip(W1, W2):
        assert len(w1) == len(w2)
        # t1, t2 are weights of a single layer.
        for t1, t2 in zip(w1, w2):
            t1 = t1.to("cpu")
            t2 = t2.to("cpu")
            if not allow_error:
                assert torch.allclose(
                    t1, t2
                ), f"{desc} max diff: {torch.max(torch.abs(t1 - t2).flatten())}"
            elif not torch.allclose(t1, t2):
                max_diff = max(max_diff, torch.max(torch.abs(t1 - t2).flatten()))

    if max_diff != 0:
        logger = logging.getLogger()
        logger.info(f"{desc} max diff: {max_diff}")
