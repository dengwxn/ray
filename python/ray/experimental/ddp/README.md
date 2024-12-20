# Ray DDP Demo

A demonstration that implements and evaluates DDP in Ray compiled graphs.

## Scripts

- `run_correctness.sh`: Compare weights from Torch, Torch DDP, and Ray DDP.
- `run_layer_size.sh`: Compare the layer sizes and fix the number of layers.
- `run_num_layers.sh`: Compare the number of layers and fix the layer size.
- `clean_all.sh`: Clean up logs and csvs.

### Deprecated

- `run_all.sh`: Run all scripts.
- `run_mixed.sh`: Compare mixed number of layers and layer sizes.
- `run_breakdown.sh`: Get manual performance breakdown for Ray DDP.
- `run_nvidia_profiler.sh`: Run with NVIDIA profiler.

## Src

- `ddp.py`: Compare Torch, Torch DDP, and Ray DDP.
- `allreduce_loop.py`: Run all-reduce in a loop.
