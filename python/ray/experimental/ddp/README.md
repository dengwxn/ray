# Ray DDP Demo

A demonstration that implements and evaluates DDP in Ray compiled graphs.

## Scripts

- `run_correctness.sh`: Compare returned weights from Ray DDP, PyTorch, and PyTorch DDP.

- `run_all.sh`: Run all experiments.
- `run_breakdown.sh`: Print out the performance breakdown for Ray DDP.
- `run_layer_size.sh`: Fix number of layers and change the layer size.
- `run_num_layers.sh`: Fix the layer size and change the number of layers.
- `run_mixed.sh`: Run experiments for different combinations of (number of layers, layer size) pairs.
- `run_profile.sh`: Profile with NVIDIA profiler.
- `clean_all.sh`: Remove all logs and csvs.

## Src

- `ddp.py`: Compare Torch, Torch DDP, and Ray DDP.
- `allreduce_loop.py`: Run all-reduce in a loop.