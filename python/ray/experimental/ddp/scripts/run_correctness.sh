#!/bin/bash

result=results/correctness
num_layers=2
layer_size=1024
dtype=float32
num_iters=10
learning_rate=5e-4
num_actors=2

mkdir -p $result

RAY_DEDUP_LOGS=0 \
    python3 src/ddp.py \
    --num-layers $num_layers \
    --layer-size $layer_size \
    --dtype $dtype \
    --num-iters $num_iters \
    --learning-rate $learning_rate \
    --num-actors $num_actors \
    --check-correctness \
    --output-file $result/latency.csv \
    >$result/run.log 2>&1
