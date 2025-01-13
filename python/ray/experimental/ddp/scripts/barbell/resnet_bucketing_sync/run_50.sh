#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run in the python/ray/experimental/ddp directory"
    exit 1
fi

rm ./results/barbell/resnet_bucketing_sync/drys/*.csv

num_trials=50
skip_trials=10
for i in $(seq 1 $num_trials); do
    ./scripts/barbell/resnet_bucketing_sync/dry_units.sh
done

python -m ray.experimental.ddp.scripts.barbell.avg \
    --input-folder ./results/barbell/resnet_bucketing_sync/drys \
    --output-path ./results/barbell/resnet_bucketing_sync \
    --num-trials $num_trials \
    --skip-trials $skip_trials
