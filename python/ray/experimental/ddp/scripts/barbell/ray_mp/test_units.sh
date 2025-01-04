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

modes=(
	torch_mp
	ray_mp
)

output_path=results/barbell/ray_mp/tests
mkdir -p $output_path
rm -f $output_path/*.csv

num_actors=1
model_prefix=$output_path/${timestamp}_${modes[1]}_model

for mode in ${modes[@]}; do
	output_file=$output_path/${timestamp}_${mode}.log
	model_file=$output_path/${timestamp}_${mode}_model.log
	python -m ray.experimental.ddp.src.main.${mode} \
		--layer-size 1024 \
		--num-layers 32 \
		--num-models 4 \
		--num-actors $num_actors \
		--num-epochs 20 \
		--model-file $model_file \
		--model-prefix $model_prefix \
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

compare_files() {
	local file1="$1"
	local file2="$2"

	if [ -f "$file1" ] && [ -f "$file2" ]; then
		if ! diff "$file1" "$file2"; then
			echo -e "${RED}ER${NC}"
			if $debug; then
				code "$file1"
				code "$file2"
			fi
			exit 1
		fi
	fi
}

file1="$output_path/${timestamp}_${modes[0]}_model.log"
file2="$output_path/${timestamp}_${modes[1]}_model.log"
compare_files "$file1" "$file2"

echo -e "${GREEN}AC${NC}"
