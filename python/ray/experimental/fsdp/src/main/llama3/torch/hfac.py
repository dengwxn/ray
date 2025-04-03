import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from src.core.common import get_timing_event_torch, millis_to_micros
from src.core.llama3.model import LLAMA_DEBUG as LLAMA
from src.core.llama3.model import TransformerWrapped


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_torch_fsdp(
    rank: int,
    args: Dict[str, Any],
) -> None:
    logger = logging.getLogger(__name__)

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    torch.manual_seed(998244353)

    model_args = LLAMA
    logger.info(f"model_args: {model_args}")
    model = TransformerWrapped(model_args).to("cuda").half()
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.warning(f"Model size: {size_bytes / 1024 / 1024} MiB")

    fsdp_model = model
    if rank == 0:
        logger.info(f"FSDP model: {fsdp_model}")

    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)
    elapses = defaultdict(list)

    for iter in range(args["num_iters"]):
        input_ids = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
            device=device,
        )
        target_ids = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
            requires_grad=True,
            device=device,
        )

        if rank == 0:
            logger.info(f"iter: {iter}")

        torch.cuda.synchronize()
        start = get_timing_event_torch()

        fw_start = get_timing_event_torch()
        pred = fsdp_model(input_ids)
        fw_end = get_timing_event_torch()

        bw_loss_comp_start = get_timing_event_torch()
        loss = criterion(pred, target_ids)
        bw_loss_comp_end = get_timing_event_torch()

        bw_grad_start = get_timing_event_torch()
        loss.backward()
        bw_grad_end = get_timing_event_torch()

        others_upd_start = get_timing_event_torch()
        optimizer.step()
        optimizer.zero_grad()
        others_upd_end = get_timing_event_torch()

        torch.cuda.synchronize()
        barrier_start = get_timing_event_torch()
        end = get_timing_event_torch()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)

        def log(key: str, elapse_ms: float):
            elapse_us = millis_to_micros(elapse_ms)
            elapses[key].append(elapse_us)
            logger.warning(
                f"rank: {rank}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
            )

        if iter > 0:
            log("total", total_ms)
            log("actor.total", total_ms)
            log("fw.total", fw_start.elapsed_time(fw_end))
            log("bw.total", bw_loss_comp_start.elapsed_time(bw_grad_end))
            log("bw.loss.comp", bw_loss_comp_start.elapsed_time(bw_loss_comp_end))
            log("bw.grad", bw_grad_start.elapsed_time(bw_grad_end))
            log("others.upd", others_upd_start.elapsed_time(others_upd_end))
            log("barrier", barrier_start.elapsed_time(end))
            logger.warning("")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    args = {
        "batch_size": 1,
        "seq_len": 1024,
        "num_iters": 20,
    }
    run_torch_fsdp(0, args)
