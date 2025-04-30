import torch
import ray
import os 
import csv
import numpy as np
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from model import Transformer, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_DEBUG
from typing import Any, Dict

@ray.remote
class LlamaActor:
    def __init__(
        self,
        model_args,
        test_rank,
        layers_per_rank,
        batch_size,
        seq_len,
        device,
    ):
        self.model_args = model_args
        self.test_rank = test_rank
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        model = Transformer(test_rank, layers_per_rank, device, model_args)
        if TEST_RANK == 0:
            model.norm = None
            model.output = None
        else:
            model.tok_embeddings = None
        model.to(device)
        model.train()
        self.model = model

    def run(self, num_iters):
        # generate data
        if self.test_rank == 0:
            input = torch.randint(
                0,
                self.model_args.vocab_size,
                (self.batch_size, self.seq_len),
                device=self.device,
            )
        else:
            input = torch.randn(
                self.batch_size,
                self.seq_len,
                self.model_args.dim,
                device=self.device,
            )
            target = torch.randn(
                self.batch_size,
                self.seq_len,
                self.model_args.vocab_size,
                device=self.device,
            )

        # train

        model = self.model
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        for iter in range(num_iters):
            print(f"iter {iter}/{num_iters}")
            model.init_tracing()
            model.update_tracing("start")
            
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    record_shapes=True, 
                    profile_memory=True,
                    with_stack=True) as prof:
                with record_function(f"forward"):

                    pred = model(input)

            prof.export_chrome_trace(f"forward_single_task.json")

            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            #         record_shapes=True, 
            #         profile_memory=True,
            #         with_stack=True) as prof:
            #     with record_function(f"backward"):

            model.update_tracing("bwd.starts")
            if TEST_RANK == 0:
                grad = torch.randn(
                    batch_size,
                    seq_len,
                    model_args.dim,
                    device=device,
                )
                pred.backward(grad)
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            model.update_tracing("bwd.ends")

            # prof.export_chrome_trace(f"backward_single_task.json")

            model.update_tracing("end")
            model.finish_tracing()

        elapses = model.fetch_traces()
        warmup = 0.2
        for key, vals in elapses.items():
            elapses[key] = vals[int(warmup * len(vals)):]
        total_mean = np.mean(elapses["total"])

        print("HERE")
        for key, vals in elapses.items():
            # print(f"Rank {TEST_RANK} {key} last elapse: {vals[-1]}")
            mean = np.mean(vals)
            std = np.std(vals)
            pct = (mean / total_mean * 100)
            print(f"Rank {TEST_RANK}", key, round(mean), round(std), round(pct))
        print("HERE")
        return total_mean



TEST_RANK = 0
model_args = LLAMA_3B
batch_size = 64
seq_len = 32
num_iters = 3

layers_per_rank = model_args.n_layers // 2
device = "cuda:0"

assert TEST_RANK == 0 or TEST_RANK == 1

actor_cls = LlamaActor.options(num_gpus=1)
actor = actor_cls.remote(model_args, TEST_RANK, layers_per_rank, batch_size, seq_len, device)

out = ray.get(actor.run.remote(num_iters))
print(out)

# ray.kill(actor)
# ray.shutdown()