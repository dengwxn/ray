#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run in the python/ray/experimental/ddp directory"
    exit 1
fi

export RAY_DEDUP_LOGS=0

dtype=float32
layer_size=1280
num_layers=(
    1 2 3 4 6 8 12 16 24 32 48 64
)

learning_rate=1e-5
num_actors=2
num_iters=10

output_path=results/num-layers
mkdir -p $output_path

for num_layer in ${num_layers[@]}; do
    python3 -m ray.experimental.ddp.src.main \
        --dtype $dtype \
        --layer-size $layer_size \
        --num-layers $num_layer \
        --learning-rate $learning_rate \
        --num-actors $num_actors \
        --num-iters $num_iters \
        --output-path $output_path/latency_$num_layer.csv \
        >$output_path/run_$num_layer.log 2>&1
done
