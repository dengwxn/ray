#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

export RAY_DEDUP_LOGS=0

dtype=float32
layer_size=320
num_layers=32

learning_rate=1e-5
num_actors=2
num_iters=100

output_path=results/barbell/torch_ddp/unit
mkdir -p $output_path

output_prefix=ls${layer_size}_nl${num_layers}
output_file=$output_path/$output_prefix.log
code $output_file

echo "Running layer_size $layer_size, num_layers $num_layers..."
python -m ray.experimental.ddp.src.main_torch_ddp \
	--dtype $dtype \
	--layer-size $layer_size \
	--num-layers $num_layers \
	--learning-rate $learning_rate \
	--num-actors $num_actors \
	--num-iters $num_iters \
	--output-path $output_path \
	--output-prefix $output_prefix \
	--check-breakdown \
	>$output_file 2>&1
