#!/bin/bash

layer_size=1280
dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2

num_layer=64
RAY_DEDUP_LOGS=0 \
    python3 ddp_breakdown.py \
    --num-layers $num_layer \
    --layer-size $layer_size \
    --dtype $dtype \
    --num-iters $num_iters \
    --learning-rate $learning_rate \
    --num-actors $num_actors \
    --check-correctness false \
    --output-file results/breakdown/lat_$num_layer.csv \
    2>results/breakdown/run_$num_layer.log \
    >results/breakdown/out_$num_layer.log
