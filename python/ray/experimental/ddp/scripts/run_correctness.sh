#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run in the python/ray/experimental/ddp directory"
    exit 1
fi

export RAY_DEDUP_LOGS=0

dtype=float32
layer_size=1024
num_layers=2

learning_rate=1e-5
num_actors=2
num_iters=10

output_path=results/correctness
mkdir -p $output_path

python -m ray.experimental.ddp.src.main \
    --dtype $dtype \
    --layer-size $layer_size \
    --num-layers $num_layers \
    --learning-rate $learning_rate \
    --num-actors $num_actors \
    --num-iters $num_iters \
    --output-path $output_path/latency.csv \
    --check-correctness \
    >$output_path/run.log 2>&1
