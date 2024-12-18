#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run in the python/ray/experimental/ddp directory"
    exit 1
fi

export RAY_DEDUP_LOGS=0

dtype=float32
layer_sizes=(
    10 20 40 80 160 320 640 1280 1840 2560 3840 5120
)
num_layers=2

learning_rate=1e-5
num_actors=2
num_iters=10

output_path=results/layer-size
mkdir -p $output_path

for layer_size in ${layer_sizes[@]}; do
    python -m ray.experimental.ddp.src.main \
        --dtype $dtype \
        --layer-size $layer_size \
        --num-layers $num_layers \
        --learning-rate $learning_rate \
        --num-actors $num_actors \
        --num-iters $num_iters \
        --output-path $output_path/latency_$layer_size.csv \
        >$output_path/run_$layer_size.log 2>&1
done
