#!/bin/bash

layer_size=2560
dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2
num_layer=64

RAY_DEDUP_LOGS=0 \
    nsys profile \
    python3 ddp_breakdown.py \
    --num-layers $num_layer \
    --layer-size $layer_size \
    --dtype $dtype \
    --num-iters $num_iters \
    --learning-rate $learning_rate \
    --num-actors $num_actors \
    --output-file results/profile/lat_$num_layer.csv \
    2>results/profile/run_$num_layer.log \
    >results/profile/out_$num_layer.log
