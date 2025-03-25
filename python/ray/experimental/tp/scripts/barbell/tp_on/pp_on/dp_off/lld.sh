#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/tp ]]; then
	echo "Please run in the python/ray/experimental/tp directory"
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

export RAY_DEDUP_LOGS=0

output_path=results/barbell/tp_on/pp_on/dp_off/lld
mkdir -p $output_path
rm -f ${output_path}/*.csv
rm -f ${output_path}/*.log
echo "Running $output_path..."

batch_size=1
seq_len=1024
num_iters=20
latency_prefix=${timestamp}
model_prefix=$output_path/${timestamp}_model
log_file=$output_path/${timestamp}.log

RANK=0 \
	WORLD_SIZE=1 \
	MASTER_ADDR=localhost \
	MASTER_PORT=12345 \
	python -m ray.experimental.tp.src.main.tp_on.pp_on.dp_off \
	--batch-size $batch_size \
	--seq-len $seq_len \
	--num-iters $num_iters \
	--output-path $output_path \
	--latency-prefix $latency_prefix \
	--model-prefix $model_prefix \
	--tracing \
	>$log_file 2>&1
# --save-model \
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

echo -e "${GREEN}AC${NC}"
