#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run this script from the python/ray/experimental/ddp directory"
    exit 1
fi

dtype=float32
layer_size=1024
num_layers=2

learning_rate=5e-4
num_iters=10
num_actors=2

output_path=results/correctness
mkdir -p $output_path

RAY_DEDUP_LOGS=0 \
    python -m ray.experimental.ddp.src.ddp \
    --dtype $dtype \
    --layer-size $layer_size \
    --num-layers $num_layers \
    --learning-rate $learning_rate \
    --num-iters $num_iters \
    --num-actors $num_actors \
    --output-path $output_path/latency.csv \
    --check-correctness \
    >$output_path/run.log 2>&1
