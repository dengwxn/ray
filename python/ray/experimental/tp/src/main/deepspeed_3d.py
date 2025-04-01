import os
import torch
import deepspeed
import numpy as np
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers import AutoTokenizer, LlamaConfig

# =========== Environment Setup ===========
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# =========== Model Configuration (Llama 3 small version) ===========
# Reduced size configuration based on Llama 3 architecture
hidden_size = 2048  # Reduced from 4096
intermediate_size = 5632  # Reduced from 11008
num_hidden_layers = 16  # Reduced from 32
num_attention_heads = 16  # Reduced from 32
num_key_value_heads = 16  # Same as attention heads for simplicity
hidden_act = "silu"  # Llama 3 uses SiLU/Swish
max_position_embeddings = 4096
vocab_size = 128256  # Using original Llama 3 vocabulary size
rope_theta = 10000.0
rope_scaling = None

# Create Llama 3 configuration
llama_config = LlamaConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
    hidden_act=hidden_act,
    max_position_embeddings=max_position_embeddings,
    rope_theta=rope_theta,
    rope_scaling=rope_scaling,
    rms_norm_eps=1e-5,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
)

# =========== Training Parameters ===========
batch_size = 8
seq_length = 2048
learning_rate = 2e-5
weight_decay = 0.01
warmup_steps = 100
total_steps = 10000

# =========== 3D Parallel Configuration ===========
# 8 GPUs: 2 (TP) × 2 (DP) × 2 (PP)
ds_config = {
    "train_batch_size": batch_size * 2,  # batch_size * dp_size
    "train_micro_batch_size_per_gpu": batch_size,
    "steps_per_print": 10,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": learning_rate,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": weight_decay,
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": learning_rate,
            "warmup_num_steps": warmup_steps,
            "total_num_steps": total_steps,
        },
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 12,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "zero_optimization": {"stage": 0},
    "gradient_clipping": 1.0,
    "prescale_gradients": False,
    "wall_clock_breakdown": False,
    "hybrid_engine": {
        "enabled": True,
        "max_out_tokens": seq_length,
        "inference_tp_size": 2,
        "release_inference_cache": False,
        "pin_parameters": True,
        "tp_gather_partition_size": 8,
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "profile": False,
    },
    "tensorboard": {
        "enabled": True,
        "output_path": "logs/llama3_3d",
        "job_name": "llama3_small_3d",
    },
    "parallelism": {"dp": 2, "tp": 2, "pp": 2},
}

# =========== Define Model Architecture ===========
# Import essential Llama components
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
)


class LlamaDecoderLayerPipe(torch.nn.Module):
    """Adapter for LlamaDecoderLayer for PipelineModule."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Self-attention
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # MLP
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def get_llama_layer(layer_id):
    """Return a Llama layer as a LayerSpec object."""
    return LayerSpec(LlamaDecoderLayerPipe, config=llama_config, layer_idx=layer_id)


class LlamaEmbeddings(torch.nn.Module):
    """Embedding layer for Llama model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LlamaOutputLayer(torch.nn.Module):
    """Final output layer for Llama model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


def create_llama_pipeline_model():
    """Create a pipeline parallel Llama model."""
    # Split layers across 2 pipeline stages
    num_stages = 2
    layers_per_stage = num_hidden_layers // num_stages

    # Stage 1: embeddings + first half of transformer layers
    stage1 = [LayerSpec(LlamaEmbeddings, config=llama_config)]

    for i in range(layers_per_stage):
        stage1.append(get_llama_layer(i))

    # Stage 2: second half of transformer layers + output layer
    stage2 = []
    for i in range(layers_per_stage, num_hidden_layers):
        stage2.append(get_llama_layer(i))

    # Add output layer (norm + lm_head)
    stage2.append(LayerSpec(LlamaOutputLayer, config=llama_config))

    # Combine all layers
    layers = stage1 + stage2

    # Create the pipeline model
    return PipelineModule(
        layers=layers, num_stages=num_stages, loss_fn=torch.nn.CrossEntropyLoss()
    )


# =========== Initialize DeepSpeed with 3D Parallelism ===========
def init_distributed():
    """Initialize distributed environment."""
    deepspeed.init_distributed(
        dist_backend="nccl", auto_mpi_discovery=True, verbose=True
    )


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right for causal language modeling."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = pad_token_id  # <pad> token (we'll mask this out)
    return shifted_input_ids


def main():
    # Initialize distributed environment
    init_distributed()

    # Get local and global batch size
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]
    global_batch_size = ds_config["train_batch_size"]

    # Load Llama tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
    except:
        # Fall back to a similar model if not available
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Create a synthetic dataset (replace with your actual dataset loading)
    # Here we're creating random token IDs
    dataset_size = 1000
    seq_len = seq_length

    # Create fake input data (random token IDs between 0 and vocab_size-1)
    # In a real scenario, you would load your tokenized dataset here
    input_ids = torch.randint(3, vocab_size - 1, (dataset_size, seq_len)).cuda()

    # Create labels by shifting input_ids right (standard causal LM approach)
    labels = input_ids.clone()

    # Create DataLoader
    train_data = torch.utils.data.TensorDataset(input_ids, labels)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=micro_batch_size, shuffle=True
    )

    # Create the Llama model with pipeline parallelism
    model = create_llama_pipeline_model()

    # Initialize DeepSpeed with 3D parallelism
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
        training_data=train_data,
    )

    # Training loop
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    for epoch in range(3):  # Just 3 epochs for demonstration
        model_engine.train()
        running_loss = 0.0

        for step, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(model_engine.device)
            targets = targets.to(model_engine.device)

            # Forward pass (loss is calculated within the pipeline)
            loss = model_engine(inputs, targets)

            # Backward pass
            model_engine.backward(loss)

            # Optimizer step
            model_engine.step()

            # Print stats
            running_loss += loss.item()
            if step % 10 == 0 and local_rank == 0:
                print(
                    f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Avg Loss: {running_loss / (step + 1):.4f}"
                )

    # Save the model after training
    if local_rank == 0:
        print("Training completed, saving model...")
    model_engine.save_checkpoint("checkpoints/llama3_small_3d")


if __name__ == "__main__":
    main()
