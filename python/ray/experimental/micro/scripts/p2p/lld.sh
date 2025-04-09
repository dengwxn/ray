#!/bin/bash

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

# Iterate size from 2^10 to 2^30
for power in {10..30}; do
	size=$((2 ** $power))
	echo "Running with size: $size..."

	# Run compiled version
	output_path=results/titan_n2/p2p/compiled/$timestamp
	mkdir -p $output_path
	log_file=$output_path/actors_s$size.log

	python src/p2p/compiled.py \
		--size $size \
		>$log_file 2>&1

	# Run interpreted version
	output_path=results/titan_n2/p2p/interpreted/$timestamp
	mkdir -p $output_path
	log_file=$output_path/actors_s$size.log

	python src/p2p/interpreted.py \
		--size $size \
		>$log_file 2>&1
done
