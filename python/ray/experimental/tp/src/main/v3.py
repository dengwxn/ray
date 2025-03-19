import os

import torch
import torch.distributed as dist

import ray


@ray.remote(num_gpus=1)
class DistributedWorker:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank
        # Set environment variables needed for PyTorch Distributed
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        # Initialize process group
        dist.init_process_group(backend="nccl")

        # Verify GPU assignment
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Worker {rank} using device: {self.device}")

    def run_collective_op(self, tensor_data):
        # Create tensor on the assigned GPU
        tensor = torch.tensor(tensor_data, device=self.device)

        # Perform collective operation (e.g., all-reduce)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return tensor.cpu().numpy()

    def cleanup(self):
        dist.destroy_process_group()


# Initialize Ray
ray.init()

# Create four worker actors, each with one GPU
world_size = 2
master_addr = "127.0.0.1"
master_port = 22345

workers = [
    DistributedWorker.remote(i, world_size, master_addr, master_port)
    for i in range(world_size)
]

# Run a collective operation on each worker
results = ray.get(
    [workers[i].run_collective_op.remote([i + 1]) for i in range(world_size)]
)
print("Results after all-reduce:", results)

# Clean up
ray.get([worker.cleanup.remote() for worker in workers])
