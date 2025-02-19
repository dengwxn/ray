import logging
import os
import time

import torch

from .....core.llama3.actor import _Actor_V4 as Actor
from .....core.llama3.model import LLAMA_1B

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(998244353)

    model_args = LLAMA_1B
    actor = Actor(model_args)
    logger.info(actor.model)

    n_iters = 3

    for _ in range(n_iters):
        actor.init_training()

        fw_start = time.perf_counter()
        logits = actor.forward(None)
        fw_end = time.perf_counter()
        logger.info(f"Forward time: {round((fw_end - fw_start) * 1e3)} ms")

        bw_start = time.perf_counter()
        actor.backward_aio(logits)
        bw_end = time.perf_counter()
        logger.info(f"Backward time: {round((bw_end - bw_start) * 1e3)} ms")

        upd_start = time.perf_counter()
        actor.copy_aio(None)
        upd_end = time.perf_counter()
        logger.info(f"Copy time: {round((upd_end - upd_start) * 1e3)} ms")

        upd_start = time.perf_counter()
        actor.step_aio(None)
        upd_end = time.perf_counter()
        logger.info(f"Step time: {round((upd_end - upd_start) * 1e3)} ms")

        logger.info("")


if __name__ == "__main__":
    main()
