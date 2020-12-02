#!/bin/bash

datadir=$1
niters=$2

if [[ $datadir == "" ]] || [[ $niters == "" ]]; then
	echo "Usage: test_canopy_placement.sh <data directory> <number of iterations>"
	exit 1
fi

for i in $(eval echo {1..$niters});
do
	echo "iteration $i"
	../build-default/canopy_placement/canopy_placement $datadir ../common_data/sonoma.db > /dev/null
	if [ $? -ne 0 ]; then
		exit 1
	fi
done
