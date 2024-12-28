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

output_file=$output_path/unit_${timestamp}.log
code $output_file

# python -m ray.experimental.ddp.src.main_element \
# 	--mode sequential \
# 	--output-path $output_path \
# 	>$output_file 2>&1

python -m ray.experimental.ddp.src.main_element \
	--mode checkpoint \
	--output-path $output_path \
	>$output_file 2>&1
