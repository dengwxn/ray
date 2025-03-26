import logging
from typing import Any, Dict, List, Union

import ray
from ....core.actor import ActorTP2PP2DP as Actor
from ....core.common import get_end_time, get_start_time, log_elapses_to_csv
from ....core.config import parse_args
from ....core.model import LLAMA_DEBUG as LLAMA
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[Actor]:
    ray.init()
    model_args = LLAMA
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    num_tp = 2
    master_addr = "127.0.0.1"
    num_pp = 2
    num_pp_batches = 2
    num_actors_dp = 2
    tracing = args["tracing"]

    actor_cls = Actor.options(num_gpus=1)
    pp_to_tp_actors: List[List[Actor]] = []
    for i in range(num_pp):
        master_port = 12345 + i
        actors = []
        for j in range(num_tp):
            actors.append(
                actor_cls.remote(
                    model_args=model_args,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rank_tp=j,
                    num_tp=num_tp,
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_pp=i,
                    num_pp_batches=num_pp_batches,
                    rank_dp=0,  # [TODO]
                    num_actors_dp=num_actors_dp,
                    tracing=tracing,
                )
            )
        pp_to_tp_actors.append(actors)

    actors = [actor for tp_actors in pp_to_tp_actors for actor in tp_actors]
    return actors


def filter(items: List[Any], idxs: List[int]) -> List[Any]:
    assert isinstance(items, list)
    assert isinstance(idxs, list)
    return [items[idx] for idx in idxs]


def filter_map(
    items_inp: List[Any], idxs_inp: List[int], idxs_out: List[int]
) -> List[Any]:
    assert isinstance(items_inp, list)
    assert isinstance(idxs_inp, list)
    assert isinstance(idxs_out, list)
    items_out = []
    for idx_out in idxs_out:
        assert idx_out in idxs_inp
        items_out.append(items_inp[idxs_inp.index(idx_out)])
    return items_out


def train(
    actors: List[Actor],
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    tracing: bool,
) -> None:
    idxs_pp = [[0, 1], [2, 3]]
    idxs_pp_dp_tp = [[[0], [1]], [[2], [3]]]
    # idxs_pp = [[0, 1, 4, 5], [2, 3, 6, 7]]
    # idxs_pp_dp_tp = [[[0, 4], [1, 5]], [[2, 6], [3, 7]]]

    def get_pp_forwards(
        idx_batch: int,
        idxs_pp: List[int],
        inputs: Union[Any, List[Any]],
        with_nccl: bool = False,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs for _ in idxs_pp]
        fws = [
            actor.forward.bind(idx_batch, input)
            for actor, input in zip(
                filter(actors, idxs_pp),
                inputs,
            )
        ]
        if with_nccl:
            fws = [fw.with_tensor_transport(transport="nccl") for fw in fws]
        return fws

    def get_pp_backwards(
        idx_batch: int,
        idxs_pp: List[int],
        tensors: List[Any],
        with_nccl: bool = False,
    ):
        bws = [
            actor.backward.bind(idx_batch, tensor)
            for actor, tensor in zip(
                filter(actors, idxs_pp),
                tensors,
            )
        ]
        if with_nccl:
            bws = [bw.with_tensor_transport(transport="nccl") for bw in bws]
        return bws

    def get_pp_allreduces_updates(
        idx_batch: int,
        idxs_pp: List[int],
        idxs_dp_tp: List[List[int]],
        bwds: List[Any],
    ):
        upds = []
        for idxs_tp in idxs_dp_tp:
            grads_tp = [
                actor.get_flat_grad.bind(idx_batch, bwd)
                for actor, bwd in zip(
                    filter(actors, idxs_tp),
                    filter_map(bwds, idxs_pp, idxs_tp),
                )
            ]
            grads_tp = allreduce.bind(grads_tp)
            upds_tp = [
                actor.update.bind(idx_batch, grad, True)
                for actor, grad in zip(
                    filter(actors, idxs_tp),
                    grads_tp,
                )
            ]
            upds.extend(upds_tp)
        return upds

    def get_pp_updates(
        idx_batch: int,
        idxs_pp: List[int],
        grads: List[Any],
    ):
        upds = [
            actor.update.bind(idx_batch, grad, False)
            for actor, grad in zip(
                filter(actors, idxs_pp),
                grads,
            )
        ]
        return upds

    with InputNode() as inp:
        b1_fw1s = get_pp_forwards(0, idxs_pp[0], inp, True)

        b2_fw1s = get_pp_forwards(1, idxs_pp[0], inp, True)
        b1_fw2s = get_pp_forwards(0, idxs_pp[1], b1_fw1s)

        b1_bw1s = get_pp_backwards(0, idxs_pp[1], b1_fw2s, True)
        b1_upd1s = get_pp_allreduces_updates(0, idxs_pp[1], idxs_pp_dp_tp[1], b1_bw1s)

        b1_bw2s = get_pp_backwards(0, idxs_pp[0], b1_bw1s)
        b1_upd2s = get_pp_allreduces_updates(0, idxs_pp[0], idxs_pp_dp_tp[0], b1_bw2s)
        b2_fw2s = get_pp_forwards(1, idxs_pp[1], b2_fw1s)

        b2_bw1s = get_pp_backwards(1, idxs_pp[1], b2_fw2s, True)
        b2_upd1s = get_pp_allreduces_updates(1, idxs_pp[1], idxs_pp_dp_tp[1], b2_bw1s)

        b2_bw2s = get_pp_backwards(1, idxs_pp[0], b2_bw1s)
        b2_upd2s = get_pp_allreduces_updates(1, idxs_pp[0], idxs_pp_dp_tp[0], b2_bw2s)

        updates = b1_upd1s + b1_upd2s + b2_upd1s + b2_upd2s
        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile()

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    for actor in actors:
        ray.get(actor.clean.remote())

    actors_to_elapses = []
    for actor in actors:
        actors_to_elapses.append(ray.get(actor.fetch_traces.remote()))
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    metrics = Actor.get_metrics(tracing)
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
    )


if __name__ == "__main__":
    args = parse_args()
    actors = init_actors(args)

    train(
        actors,
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args["tracing"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()
