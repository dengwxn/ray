#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

debug=false
while getopts "d" opt; do
	case $opt in
	d) debug=true ;;
	*)
		echo "Usage: $0 [-d]" >&2
		echo "  -d    Enable debug mode"
		exit 1
		;;
	esac
done

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

output_path=results/barbell/ray_mp/drys
mkdir -p $output_path
rm -f $output_path/*.csv

num_actors=2
output_file=$output_path/${timestamp}.log
model_prefix=$output_path/${timestamp}_model

python -m ray.experimental.ddp.src.main.ray_mp \
	--layer-size 1024 \
	--num-layers 8 \
	--num-models 4 \
	--num-actors $num_actors \
	--num-epochs 10 \
	--model-prefix $model_prefix \
	>$output_file 2>&1
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -eq 0 ]; then
	echo -e "${GREEN}AC${NC}"
	exit 0
else
	echo -e "${RED}ER${NC}"
	exit 1
fi