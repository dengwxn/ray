#!/bin/bash

dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2

num_layers=(2 4 8 16 32 64)
layer_sizes=(10 20 40 80 160 320 640 1280 1840 2560 3840 5120)
for num_layer in ${num_layers[@]}; do
    for layer_size in ${layer_sizes[@]}; do
        RAY_DEDUP_LOGS=0 \
            python3 ddp_breakdown.py \
            --num-layers $num_layer \
            --layer-size $layer_size \
            --dtype $dtype \
            --num-iters $num_iters \
            --learning-rate $learning_rate \
            --num-actors $num_actors \
            --output-file results/mixed/$num_layer-layers/lat_$layer_size.csv \
            >results/mixed/$num_layer-layers/run_$layer_size.log 2>&1
    done
done
