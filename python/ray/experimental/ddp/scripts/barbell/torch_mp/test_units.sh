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

output_path=results/barbell/torch_mp/tests
mkdir -p $output_path
rm -f $output_path/*.csv

num_models_values=(
	1
	4
)

for num_models in ${num_models_values[@]}; do
	output_file=$output_path/${timestamp}_nm${num_models}.log
	model_file=$output_path/${timestamp}_nm${num_models}_model.log
	python -m ray.experimental.ddp.src.main.torch_mp \
		--layer-size 1024 \
		--num-layers 32 \
		--num-models $num_models \
		--num-epochs 20 \
		--model-file $model_file \
		>$output_file 2>&1
	status=$?

	if [ $status -ne 0 ]; then
		echo -e "${RED}ER${NC}"
		if $debug; then
			code $output_file
		fi
		exit 1
	fi
done

if ! diff \
	"$output_path/${timestamp}_nm${num_models_values[0]}_model.log" \
	"$output_path/${timestamp}_nm${num_models_values[1]}_model.log"; then
	echo -e "${RED}ER${NC}"
	if $debug; then
		code "$output_path/${timestamp}_nm${num_models_values[0]}_model.log"
		code "$output_path/${timestamp}_nm${num_models_values[1]}_model.log"
	fi
	exit 1
else
	echo -e "${GREEN}AC${NC}"
fi
