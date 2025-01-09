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

export RAY_DEDUP_LOGS=0

output_path=results/barbell/ray_mp/drys
mkdir -p $output_path
rm -f $output_path/*.csv

layer_size=1024
num_layers=8
num_models=1
num_actors=2
num_epochs=10
latency_prefix=ls${layer_size}_nl${num_layers}
model_prefix=$output_path/${timestamp}_model
log_file=$output_path/${timestamp}.log

python -m ray.experimental.ddp.src.main.ray_mp \
	--layer-size $layer_size \
	--num-layers $num_layers \
	--num-models $num_models \
	--num-actors $num_actors \
	--num-epochs $num_epochs \
	--output-path $output_path \
	--latency-prefix $latency_prefix \
	--model-prefix $model_prefix \
	--check-tracing \
	>$log_file 2>&1
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

compare_files() {
	local file1="$1"
	local file2="$2"

	if [ ! -f "$file1" ]; then
		echo -e "${RED}Error: File '$file1' does not exist${NC}"
		exit 1
	fi
	if [ ! -f "$file2" ]; then
		echo -e "${RED}Error: File '$file2' does not exist${NC}"
		exit 1
	fi

	if ! diff "$file1" "$file2"; then
		echo -e "${RED}ER${NC}"
		if $debug; then
			code "$file1"
			code "$file2"
		fi
		exit 1
	fi
}

file1="${output_path}/${timestamp}_model_0.log"
file2="${output_path}/${timestamp}_model_1.log"
compare_files "$file1" "$file2"

echo -e "${GREEN}AC${NC}"
