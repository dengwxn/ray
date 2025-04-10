#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/vit ]]; then
	echo "Please run in the python/ray/experimental/vit directory"
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

output_path=results/xuhui_n4/v1/lld/$timestamp
mkdir -p $output_path
echo "Running $output_path..."

num_iters=20
log_file=$output_path/actors.log

python src/core/v1.py \
	>$log_file 2>&1
status=$?

if $debug; then
	code $output_path/${timestamp}.log
fi

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

echo -e "${GREEN}AC${NC}"
