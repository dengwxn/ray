#!/bin/bash

if [[ "$(pwd)" != */python/ray/experimental/ddp ]]; then
	echo "Please run in the python/ray/experimental/ddp directory"
	exit 1
fi

find results -name "*.log" -type f -delete
find results -name "*.csv" -type f -delete
