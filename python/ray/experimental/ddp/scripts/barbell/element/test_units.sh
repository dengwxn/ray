#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

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

output_path=results/barbell/element/tests
mkdir -p $output_path
rm -f $output_path/*.csv

modes=(
	sequential
	cot
)

for mode in ${modes[@]}; do
	output_file=$output_path/${timestamp}_${mode}.log
	model_file=$output_path/${timestamp}_${mode}_model.log
	python -m ray.experimental.ddp.src.main_element \
		--mode $mode \
		--model-file $model_file \
		>$output_file 2>&1
done

if $debug; then
	code $output_path/${timestamp}_${modes[0]}.log
	code $output_path/${timestamp}_${modes[1]}.log
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if ! diff \
	"$output_path/${timestamp}_${modes[0]}_model.log" \
	"$output_path/${timestamp}_${modes[1]}_model.log"; then
	case $? in
	1) echo -e "${YELLOW}WA${NC}" ;;
	*) echo -e "${RED}ER${NC}" ;;
	esac
	exit 1
else
	echo -e "${GREEN}AC${NC}"
fi
