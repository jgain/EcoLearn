#!/bin/bash

if [[ $1 == "" ]]; then
	echo "Usage: test_extract.sh <number of iterations>"
	exit 1
fi

if [ -d ~/PhDStuff/data/test_extract_script ]; then
	echo "Deleting test_extract_script directory to make a new one..."
	rm -r ~/PhDStuff/data/test_extract_script --interactive=never
fi

for i in $(eval echo {1..$1})
do
	echo "----------------------------"
	echo "Iteration $i"
	echo "----------------------------"

	randx=$(($RANDOM % 10000))
	randy=$(($RANDOM % 10000))

	if [ $randx -gt 10000 ] || [ $randy -gt 10000 ]; then
		echo "Invalid random number generated"
		exit 1
	fi

	echo "Creating 128 x 128 for the top-left corner at $randx, $randy"

	python3 sonoma_extract.py $randx $randy 128 128 ~/PhDStuff/data/test_extract_script > /dev/null
	if [ $? -ne 0 ]; then
		echo "Non-zero exit code encountered in sonoma_extract.py script"
		exit 1
	fi
done
