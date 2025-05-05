import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group using NCCL backend
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        # Create a tensor on GPU 0
        tensor = torch.arange(10, dtype=torch.float, device=device)
        print(f"[Rank {rank}] Sending tensor: {tensor}")
        dist.send(tensor=tensor, dst=1)
    elif rank == 1:
        # Prepare empty tensor on GPU 1
        tensor = torch.empty(10, device=device)
        dist.recv(tensor=tensor, src=0)
        print(f"[Rank {rank}] Received tensor: {tensor}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
