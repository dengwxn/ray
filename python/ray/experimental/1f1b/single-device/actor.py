import ray
import torch 
from model import Transformer
from torch.profiler import profile, record_function, ProfilerActivity

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
        if test_rank == 0:
            model.norm = None
            model.output = None
        else:
            model.tok_embeddings = None
        model.to(device)
        model.train()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        if self.test_rank == 1:
            self.target = torch.randn(
                batch_size,
                seq_len,
                model_args.vocab_size,
                device=device,
            )

    def init_tracing(self):
        return self.model.init_tracing()
    
    def update_tracing(self, key):
        return self.model.update_tracing(key)

    def finish_tracing(self):
        return self.model.finish_tracing()

    def fetch_traces(self):
        return self.model.fetch_traces()

    def forward(self, tokens):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True, 
                profile_memory=True,
                with_stack=True) as prof:
            with record_function(f"forward"):

                ret = self.model.forward(tokens)

        prof.export_chrome_trace(f"forward_multi_task.json")
        return ret

    def backward(self, pred):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True, 
                profile_memory=True,
                with_stack=True) as prof:
            with record_function(f"backward"):

                self.model.update_tracing("bwd.starts")
                if self.test_rank == 0:
                    grad = torch.randn(
                        self.batch_size,
                        self.seq_len,
                        self.model_args.dim,
                        device=self.device,
                    )
                    pred.backward(grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    assert target is not None
                    loss = self.criterion(pred, self.target)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.model.update_tracing("bwd.ends")
        
        prof.export_chrome_trace(f"backward_multi_task.json")
