#!/bin/bash

count=0

if [ "$1" == "" ] || [ "$2" == "" ]; then
	echo "Usage: create_circ_count_txts.sh <dirname> <ndirs>"
	exit 1
fi

base_dirname=$1
ndirs=$2

dbname="/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db"

for count in $(eval echo "{0..$(($ndirs - 1))}"); do
	dirname="$base_dirname$count"
	datasetname="$(basename $dirname)"
	/home/konrad/PhDStuff/prototypes/repo/code/cpp/build-match_densities-Qt_5_10_0_GCC_64bit-Default/create_circ_counts $dirname
done
