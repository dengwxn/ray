#!/bin/bash

layer_size=1280
dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2

rm results/debug/num-layers/*.log
rm results/debug/num-layers/*.csv

num_layers=(1 2 3 4 6 8 12 16 24 32 48 64)
for num_layer in ${num_layers[@]}; do
    RAY_DEDUP_LOGS=0 \
        python3 ddp_debug.py \
        --num-layers $num_layer \
        --layer-size $layer_size \
        --dtype $dtype \
        --num-iters $num_iters \
        --learning-rate $learning_rate \
        --num-actors $num_actors \
        --check-correctness false \
        --output-file results/debug/num-layers/lat_$num_layer.csv \
        >results/debug/num-layers/run_$num_layer.log 2>&1
done
