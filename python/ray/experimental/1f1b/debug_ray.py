import ray
from ray.dag import InputNode
import torch
import torch.distributed as dist
import os
@ray.remote
class Actor:
    def __init__(self):
        print(torch.cuda.is_available())
        self.device = torch.device("cuda:0")
        print(f"initializing actor with device {self.device}")

    def plus_one(self, x):
        x = x.to(self.device)
        return x + 1.0

def main():
    ray.init()
    
    actor_cls = Actor.options(num_gpus=1)
    a = actor_cls.remote()
    b = actor_cls.remote()

    x = torch.tensor([0.0])
    out = b.plus_one.remote((a.plus_one.remote((x))))
    print(ray.get(out))
    
    with ray.dag.InputNode() as inp:
        dag = a.plus_one.bind(inp).with_tensor_transport(transport="nccl")
        dag = b.plus_one.bind(dag)
    dag = dag.experimental_compile()

    out = dag.execute(x)
    print(ray.get(out))

    ray.shutdown()

if __name__ == "__main__":
    main()
