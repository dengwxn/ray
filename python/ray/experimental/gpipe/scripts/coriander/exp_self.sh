#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/gpipe ]]; then
	echo "Please run in the python/ray/experimental/gpipe directory"
	exit 1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

export TZ="America/Los_Angeles"
timestamp=$(date '+%Y%m%d_%H%M%S')

export RAY_DEDUP_LOGS=0
# export RAY_CGRAPH_VISUALIZE_SCHEDULE=1

output_path=results/coriander/exp_self/${timestamp}
mkdir -p $output_path

stage=(16 32 64)
nbatch=(1 2 4 8 16 32 64 128 256)

batch=(16 64 128)
nstage=(1 2 4 8 16 32 64)

# nbatch_nstage=(
# 	"1 1"
# "1 2"
# "1 4"
# "1 8"
# "1 16"
# "1 32"
# "1 64"
# "2 2"
# # "2 4"
# # "2 8"
# "4 4"
# # "4 8"
# "8 8"
# "16 16"
# "16 32"
# "32 32"
# "32 64"
# "64 64"
# "128 64"
# "256 64"
# )
# nbatch_nstage=(
# "2 2"
# "3 3"
# )
num_trials=10
tensor_size=8192

# for pair in "${nbatch_nstage[@]}"; do
# 	read -r num_batches num_stages <<<"$pair"
# 	log_file=$output_path/${num_batches}batch_${num_stages}stage.log
# 	echo "Running $log_file..."

# 	python -m ray.experimental.gpipe.src.main \
# 		--num_microbatches $num_batches \
# 		--num_stages $num_stages \
# 		--num_trials $num_trials \
# 		--tensor_size $tensor_size \
# 		--verbose >$log_file 2>&1
# done

for num_stages in "${stage[@]}"; do
	stage_dir=$output_path/${num_stages}stage
	mkdir -p $stage_dir
	for num_batches in "${nbatch[@]}"; do
		log_file=$stage_dir/${num_batches}batch_${num_stages}stage.log
		echo "Running $log_file..."

		python -m ray.experimental.gpipe.src.main \
			--num_microbatches $num_batches \
			--num_stages $num_stages \
			--num_trials $num_trials \
			--tensor_size $tensor_size \
			--verbose >$log_file 2>&1
	done
done

for num_batches in "${batch[@]}"; do
	batch_dir=$output_path/${num_batches}batch
	mkdir -p $batch_dir
	for num_stages in "${nstage[@]}"; do
		log_file=$batch_dir/${num_batches}batch_${num_stages}stage.log
		echo "Running $log_file..."

		python -m ray.experimental.gpipe.src.main \
			--num_microbatches $num_batches \
			--num_stages $num_stages \
			--num_trials $num_trials \
			--tensor_size $tensor_size \
			--verbose >$log_file 2>&1
	done
done

if [ $status -ne 0 ]; then
	echo -e "${RED}ER${NC}"
	exit 1
fi

echo -e "${GREEN}AC${NC}"
