import torch
import ray
import os 
import csv
import numpy as np
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from model import Transformer, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_DEBUG
from typing import Any, Dict
from actor import LlamaActor
from ray.dag import InputNode

TEST_RANK = 0
model_args = LLAMA_3B
batch_size = 64
seq_len = 32
num_iters = 25
layers_per_rank = model_args.n_layers // 2
device = "cuda:0"

assert TEST_RANK == 0 or TEST_RANK == 1

def print_elapses(elapses, warmup=0.2):
    for key, vals in elapses.items():
        elapses[key] = vals[int(warmup * len(vals)):]
    total_mean = np.mean(elapses["total"])

    for key, vals in elapses.items():
        # print(f"Rank {TEST_RANK} {key} last elapse: {vals[-1]}")
        mean = np.mean(vals)
        std = np.std(vals)
        pct = (mean / total_mean * 100)
        print(f"Rank {TEST_RANK}", key, round(mean), round(std), round(pct))


if TEST_RANK == 0:
    input = torch.randint(
        0,
        model_args.vocab_size,
        (batch_size, seq_len),
        device=device,
    )
else:
    input = torch.randn(
        batch_size,
        seq_len,
        model_args.dim,
        device=device,
    )

actor_cls = LlamaActor.options(num_gpus=1)
actor = actor_cls.remote(model_args, TEST_RANK, layers_per_rank, batch_size, seq_len, device)

with ray.dag.InputNode() as inp:
    dag = actor.forward.bind(inp)
    dag = actor.backward.bind(dag)
dag = dag.experimental_compile()

for iter in range(num_iters):
    if iter % 5 == 0: print(f"iter {iter}/{num_iters}")

    ray.get(actor.init_tracing.remote())
    actor.update_tracing.remote("start")

    # pred = actor.forward.remote(input)
    # ray.get(actor.backward.remote(pred))
    ray.get(dag.execute(input))

    actor.update_tracing.remote("end")
    ray.get(actor.finish_tracing.remote())

elapses = ray.get(actor.fetch_traces.remote())
print_elapses(elapses)

# ray.kill(actor)
# ray.shutdown()