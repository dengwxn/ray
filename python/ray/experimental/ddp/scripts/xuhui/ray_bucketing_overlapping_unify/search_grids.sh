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

output_path=results/xuhui/ray_bucketing_overlapping_unify/grids/0111_s500_e50
mkdir -p $output_path
rm -f $output_path/*.csv
rm -f $output_path/*.log

layer_size_values=(
	2560 1280 640 512 320 160 80
)
num_layers_values=(
	# 10 40 160 250 640 2560 10240
	20 80 320 500 1280 5120 20480
)

# num_models=10
num_models=20
num_actors=4
num_epochs=50

for i in "${!layer_size_values[@]}"; do
	layer_size="${layer_size_values[$i]}"
	num_layers="${num_layers_values[$i]}"

	latency_prefix=ls${layer_size}_nl${num_layers}
	model_prefix=${output_path}/${latency_prefix}_model
	log_file=${output_path}/${latency_prefix}.log

	echo "Running layer_size $layer_size, num_layers $num_layers..."
	python -m ray.experimental.ddp.src.main.ray_bucketing_overlapping_unify \
		--layer-size $layer_size \
		--num-layers $num_layers \
		--num-models $num_models \
		--num-actors $num_actors \
		--num-epochs $num_epochs \
		--output-path $output_path \
		--latency-prefix $latency_prefix \
		--model-prefix $model_prefix \
		--check-tracing \
		>$log_file 2>&1
	status=$?
done
