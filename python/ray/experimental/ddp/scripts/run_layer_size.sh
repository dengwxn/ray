#!/bin/bash

mkdir -p results/layer-size

dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2
num_layers=2

layer_sizes=(10 20 40 80 160 320 640 1280 1840 2560 3840 5120)
for layer_size in ${layer_sizes[@]}; do
    RAY_DEDUP_LOGS=0 \
        python3 ddp.py \
        --num-layers $num_layers \
        --layer-size $layer_size \
        --dtype $dtype \
        --num-iters $num_iters \
        --learning-rate $learning_rate \
        --num-actors $num_actors \
        --output-file results/layer-size/lat_$layer_size.csv \
        >results/layer-size/run_$layer_size.log 2>&1
done