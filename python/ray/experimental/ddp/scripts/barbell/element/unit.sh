#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

output_path=results/barbell/element/unit
mkdir -p $output_path
rm -f $output_path/*.csv

modes=(
	sequential
	# checkpoint
	cot
)
for mode in ${modes[@]}; do
	output_file=$output_path/unit_${timestamp}_${mode}.log
	model_file=$output_path/unit_${timestamp}_${mode}_model.log

	python -m ray.experimental.ddp.src.main_element \
		--mode $mode \
		--model-file $model_file \
		>$output_file 2>&1
done

code $output_path/unit_${timestamp}_${modes[0]}.log
code $output_path/unit_${timestamp}_${modes[1]}.log

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

if ! diff \
	"$output_path/unit_${timestamp}_${modes[0]}_model.log" \
	"$output_path/unit_${timestamp}_${modes[1]}_model.log"; then
	case $? in
	1) echo -e "${YELLOW}WA${NC}" ;;
	*) echo -e "${RED}ER${NC}" ;;
	esac
	code $output_path/unit_${timestamp}_${modes[0]}_model.log
	code $output_path/unit_${timestamp}_${modes[1]}_model.log
	exit 1
else
	echo -e "${GREEN}AC${NC}"
fi
