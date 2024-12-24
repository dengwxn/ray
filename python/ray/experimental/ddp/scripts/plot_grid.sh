#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

export RAY_DEDUP_LOGS=0

dtype=float32
layer_size_values=(
	10 20 40 80 160 320 640 1280 2560
)
num_layers_values=(
	1 2 4 8 16 32 64
)

learning_rate=1e-5
num_actors=2
num_iters=10

output_path=results/grid
mkdir -p $output_path

python -m ray.experimental.ddp.src.scripts.plot_heatmap \
	--layer-size ${layer_size_values[@]} \
	--num-layers ${num_layers_values[@]} \
	--output-path $output_path
