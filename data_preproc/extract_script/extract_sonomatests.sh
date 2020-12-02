#!/bin/bash

size=128

sizex=$size
sizey=$size
count=0

if [ "$1" == "" ]; then
	echo "Usage: extract_mass.sh <base dirname>"
	exit 1
fi

base_dirname=$1

for y in {0..1}; do
	for x in {0..1}; do
		topy=$((3000 + y * size))
		leftx=$((3000 + x * size))
		curr_dirname="$base_dirname$((y + 1))$((x + 1))"
		python3 sonoma_extract.py $leftx $topy $sizex $sizey $curr_dirname
		count=$((count + 1))
	done
done
