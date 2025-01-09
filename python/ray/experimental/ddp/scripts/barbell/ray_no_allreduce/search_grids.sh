#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

debug=false
while getopts "d" opt; do
	case $opt in
	d) debug=true ;;
	*)
		echo "Usage: $0 [-d]" >&2
		echo "  -d    Enable debug mode"
		exit 1
		;;
	esac
done

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

export RAY_DEDUP_LOGS=0

output_path=results/barbell/ray_no_allreduce/grids
mkdir -p $output_path
rm -f $output_path/*.csv
rm -f $output_path/*.log

layer_size_values=(
	# 10 20 40 80 160 320 640 1280 2560
	80 320
)
num_layers_values=(
	# 1 2 4 8 16 32 64 128 256
	8 32
)

num_models=1
num_actors=2
num_epochs=20

for layer_size in ${layer_size_values[@]}; do
	for num_layers in ${num_layers_values[@]}; do
		latency_prefix=ls${layer_size}_nl${num_layers}
		model_prefix=${output_path}/${latency_prefix}_model
		log_file=${output_path}/${latency_prefix}.log

		echo "Running layer_size $layer_size, num_layers $num_layers..."
		python -m ray.experimental.ddp.src.main.ray_no_allreduce \
			--layer-size $layer_size \
			--num-layers $num_layers \
			--num-models $num_models \
			--num-actors $num_actors \
			--num-epochs $num_epochs \
			--output-path $output_path \
			--latency-prefix $latency_prefix \
			--model-prefix $model_prefix \
			>$log_file 2>&1
		status=$?
	done
done

# python -m ray.experimental.ddp.src.scripts.plot_heatmap \
# 	--layer-size ${layer_size_values[@]} \
# 	--num-layers ${num_layers_values[@]} \
# 	--output-path $output_path
