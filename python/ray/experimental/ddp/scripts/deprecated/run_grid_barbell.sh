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
num_iters=30

output_path=results/grid/barbell
mkdir -p $output_path

for layer_size in ${layer_size_values[@]}; do
	for num_layers in ${num_layers_values[@]}; do
		echo "Running layer_size $layer_size, num_layers $num_layers..."
		output_file=$output_path/ls${layer_size}_nl${num_layers}
		python -m ray.experimental.ddp.src.main \
			--dtype $dtype \
			--layer-size $layer_size \
			--num-layers $num_layers \
			--learning-rate $learning_rate \
			--num-actors $num_actors \
			--num-iters $num_iters \
			--output-path $output_file.csv \
			>$output_file.log 2>&1
	done
done

python -m ray.experimental.ddp.src.scripts.plot_heatmap \
	--layer-size ${layer_size_values[@]} \
	--num-layers ${num_layers_values[@]} \
	--output-path $output_path
