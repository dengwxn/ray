#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
    echo "Please run in the python/ray/experimental/ddp directory"
    exit 1
fi

rm ./results/barbell/resnet_bucketing_overlapping/drys/*.csv

for i in {1..50}; do
    ./scripts/barbell/resnet_bucketing_overlapping/dry_units.sh
done

python -m ray.experimental.ddp.scripts.barbell.avg \
    --input-folder ./results/barbell/resnet_bucketing_overlapping/drys \
    --output-path ./results/barbell/resnet_bucketing_overlapping
