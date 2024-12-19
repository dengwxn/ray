import logging
from typing import Dict, List

import torch


def get_torch_ddp_weights(
    ranks_to_weights_dict: Dict[int, List[List[torch.Tensor]]], world_size: int
) -> List[List[torch.Tensor]]:
    """
    Get the weights after each iteration. Check that the weights are consistent
    across all ranks after each iteration.

    Args:
        ranks_to_weights_dict: Weights of all layers after each iteration for each rank.
        world_size: Number of ranks.

    Returns:
        Weights after each iteration for the first rank.
    """
    ranks_to_weights = list(ranks_to_weights_dict.values())
    assert len(ranks_to_weights) == world_size
    rank_weights_0 = ranks_to_weights[0]
    num_iters = len(rank_weights_0)
    # Weights per rank.
    for rank_weights in ranks_to_weights[1:]:
        assert len(rank_weights) == num_iters
        # Weights per iteration per rank.
        for i in range(num_iters):
            assert len(rank_weights[i]) == len(rank_weights_0[i])
            num_layers = len(rank_weights_0[i])
            # Weights per layer per iteration.
            for j in range(num_layers):
                assert torch.equal(rank_weights[i][j], rank_weights_0[i][j])
    return rank_weights_0


def get_ray_ddp_weights(
    iters_to_weights: List[List[List[torch.Tensor]]], num_actors: int
) -> List[List[torch.Tensor]]:
    """
    Get the weights after each iteration. Check that the weights are consistent
    across all ranks after each iteration.

    Args:
        iters_to_weights: Weights of all layers after each iteration for each actor.
        num_actors: Number of actors.

    Returns:
        Weights after each iteration for the first actor.
    """
    actor_weights = []
    # Weights per iteration.
    for actors_to_weights in iters_to_weights:
        actor_weights_0 = actors_to_weights[0]
        actor_weights.append(actor_weights_0)
        num_layers = len(actor_weights_0)
        # Weights per actor per iteration.
        for i in range(1, num_actors):
            actor_weights_i = actors_to_weights[i]
            assert len(actor_weights_i) == num_layers
            # Weights per layer per actor.
            for j in range(num_layers):
                assert torch.equal(actor_weights_i[j], actor_weights_0[j])
    return actor_weights


def compare_weights(
    weights1: List[List[torch.Tensor]],
    weights2: List[List[torch.Tensor]],
    cmp: str,
    check_diff: bool = False,
) -> None:
    """
    Compare the weights after each iteration for different approaches.
    """
    assert len(weights1) == len(weights2)
    max_diff = 0
    # weights1, weights2 are weights after a single iteration.
    for w1, w2 in zip(weights1, weights2):
        assert len(w1) == len(w2)
        # t1, t2 are weights of a single layer.
        for t1, t2 in zip(w1, w2):
            t1 = t1.to("cpu")
            t2 = t2.to("cpu")
            if not check_diff:
                assert torch.allclose(
                    t1, t2
                ), f"{cmp} max diff: {torch.max(torch.abs(t1 - t2).flatten())}"
            elif not torch.allclose(t1, t2):
                max_diff = max(max_diff, torch.max(torch.abs(t1 - t2).flatten()))

    if max_diff != 0:
        logger = logging.getLogger()
        logger.error(f"{cmp} max diff: {max_diff}, allow error: {check_diff}")
