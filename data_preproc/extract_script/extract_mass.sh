#!/bin/bash

size=512

sizex=$size
sizey=$size
count=0

if [ "$1" == "" ]; then
	echo "Usage: extract_mass.sh <base dirname>"
	exit 1
fi

base_dirname=$1

for y in {1..5}; do
	for x in {1..5}; do
		topy=$((y * 10 * size))
		leftx=$((x * 10 * size))
		curr_dirname="$base_dirname$count"
		python3 sonoma_extract.py $leftx $topy $sizex $sizey $curr_dirname
		count=$((count + 1))
	done
done
