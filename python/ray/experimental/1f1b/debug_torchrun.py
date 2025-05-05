import torch
import os
import torch.distributed as dist

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    print(f"rank: {rank}, world_size: {world_size}, device: {device}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


