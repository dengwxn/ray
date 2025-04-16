#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/gpipe ]]; then
    echo "Please run in the python/ray/experimental/gpipe directory"
    exit 1
fi

# Parent directory containing the subfolders
results_dir=results/barbell/exp_self
highest_timestamp_folder=$(ls -d "$results_dir"/*/ | sort -r | head -n 1)
echo "Folder with highest timestamp: $highest_timestamp_folder"

# Process each subfolder in the parent directory
for subfolder in "$highest_timestamp_folder"/*/; do
    # Check if it's a directory
    if [ -d "$subfolder" ]; then
        echo "Processing subfolder: $subfolder"
        # The file to append all last lines to
        output_file=$subfolder/result.csv
        echo "nbatch,nstage,tput" >$output_file

        # Process each file in the subfolder
        for file in $subfolder/*.log; do
            # Check if it's a regular file
            if [ -f "$file" ]; then
                # Get the last line of the file and append it to the output file
                tail -n 1 "$file" >>"$output_file"
            fi
        done
    fi
done
