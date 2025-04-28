import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import deepspeed
import fire
import torch
from deepspeed import comm
from src.core.common import get_timing_event_torch, millis_to_micros
from src.core.llama3.model import LLAMA_1B, LLAMA_8B
from src.core.llama3.model import (
    BucketParameterFirst,
    BucketParameterLast,
    BucketParameterTransformerBlock,
    TransformerWrapped,
)
from torch.utils.data import TensorDataset


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_torch_fsdp(
    output_path: str = "results/example",
    latency_prefix: str = "latency",
    local_rank: int = 0,
    model: str = "LLAMA_1B",
    num_actors: int = 2,
    num_iters: int = 50,
    batch_size: int = 1,
    seq_len: int = 1024,
) -> None:
    logger = logging.getLogger(__name__)
    set_seed(998244353)

    deepspeed_config_dict = {
        "train_batch_size": num_actors,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
        },
        "bf16": {"enabled": False},
        "fp16": {"enabled": True},
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
        "overlap_comm": True,
        # "use_all_reduce_for_fetch_params": True,
    }

    model_args = LLAMA_1B if model == "LLAMA_1B" else LLAMA_8B

    # Create model
    model = TransformerWrapped(model_args).half()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # Create a simple scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-7
    )

    # Creating random dataset
    def get_random_batch():
        input_ids = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
        )
        target_ids = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
        )
        return input_ids, target_ids

    # Create a small dataset of random tensors
    inputs = []
    targets = []
    for _ in range(1):
        input_batch, target_batch = get_random_batch()
        inputs.append(input_batch)
        targets.append(target_batch)

    train_dataset = TensorDataset(torch.cat(inputs), torch.cat(targets))
    # train_dataloader = DataLoader(train_dataset, shuffle=True)

    # Set up the criterion
    criterion = torch.nn.CrossEntropyLoss()

    deepspeed.utils.set_z3_leaf_modules(
        model,
        [BucketParameterFirst, BucketParameterTransformerBlock, BucketParameterLast],
    )

    # Prepare everything with accelerator
    model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        lr_scheduler=scheduler,
        config=deepspeed_config_dict,
    )
    device = model_engine.device

    size_bytes = sum(
        p.numel() * p.element_size() for p in model_engine.module.parameters()
    )
    logger.warning(f"Model size: {round(size_bytes / 1024 / 1024)} MiB")

    # For tracking time
    elapses = defaultdict(list)
    elapses_us = []

    # Training loop
    for i in range(num_iters):
        # Get batch from dataloader (it will cycle through if needed)
        try:
            input_ids, target_ids = next(train_iter)
        except Exception:
            train_iter = iter(train_dataloader)
            input_ids, target_ids = next(train_iter)

        if comm.get_rank() == 0:
            logger.info(f"iter: {i}")

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        torch.cuda.synchronize()
        start = get_timing_event_torch()

        # Forward pass
        fw_start = get_timing_event_torch()
        pred = model_engine(input_ids)
        fw_end = get_timing_event_torch()

        # Loss computation
        bw_loss_comp_start = get_timing_event_torch()
        loss = criterion(pred, target_ids)
        bw_loss_comp_end = get_timing_event_torch()

        # Backward pass
        bw_grad_start = get_timing_event_torch()
        model_engine.backward(loss)
        bw_grad_end = get_timing_event_torch()

        # Update weights
        others_upd_start = get_timing_event_torch()
        model_engine.step()
        others_upd_end = get_timing_event_torch()

        torch.cuda.synchronize()
        barrier_start = get_timing_event_torch()
        comm.barrier()
        end = get_timing_event_torch()

        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)

        def log(key: str, elapse_ms: float):
            elapse_us = millis_to_micros(elapse_ms)
            if key == "actor.total":
                elapses_us.append(elapse_us)
            elapses[key].append(elapse_us)
            logger.warning(
                f"rank: {comm.get_rank()}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
            )

        if i > 0:
            log("total", total_ms)
            log("actor.total", total_ms)
            log("fw.total", fw_start.elapsed_time(fw_end))
            log("bw.total", bw_loss_comp_start.elapsed_time(bw_grad_end))
            log("bw.loss.comp", bw_loss_comp_start.elapsed_time(bw_loss_comp_end))
            log("bw.grad", bw_grad_start.elapsed_time(bw_grad_end))
            log("others.upd", others_upd_start.elapsed_time(others_upd_end))
            log("barrier", barrier_start.elapsed_time(end))
            if comm.get_rank() == 0:
                logger.warning("")

    elapses_us = elapses_us[int(len(elapses) * 0.2) :]
    elapse_avg = round(sum(elapses_us) / len(elapses_us))
    logger.warning(f"Elapse avg: {elapse_avg}")


if __name__ == "__main__":
    fire.Fire(run_torch_fsdp)
