#!/bin/bash

num_layers=2
layer_size=1024
dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2

# Correctness check
RAY_DEDUP_LOGS=0 \
    python3 ddp_breakdown.py \
    --num-layers $num_layers \
    --layer-size $layer_size \
    --dtype $dtype \
    --num-iters $num_iters \
    --learning-rate $learning_rate \
    --num-actors $num_actors \
    --check-correctness \
    --output-file results/correctness/lat.csv \
    >results/correctness/run.log 2>&1
