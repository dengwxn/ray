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

modes=(sequential checkpoint)
for mode in ${modes[@]}; do
	output_file=$output_path/unit_${timestamp}_${mode}.log
	model_file=$output_path/unit_${timestamp}_${mode}_model.log

	python -m ray.experimental.ddp.src.main_element \
		--mode $mode \
		--model-file $model_file \
		>$output_file 2>&1
done

if ! diff \
	"$output_path/unit_${timestamp}_${modes[0]}_model.log" \
	"$output_path/unit_${timestamp}_${modes[1]}_model.log"; then
	case $? in
	1) echo "WA" ;;
	*) echo "ER" ;;
	esac
	exit 1
else
	echo "AC"
fi
