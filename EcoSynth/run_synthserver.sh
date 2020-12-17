#!/bin/bash

if [ "$1" == "" ]; then
	echo "Parameter needed: image size"
	exit 1
else
	cd build/viewer
	python3 ../../../py_scripts/pix2pix_server.py --checkpoint $HOME/ecolearn-models/cfpampsynth_trained/ --checkpoint2 $HOME/ecolearn-models/chm_guess_trained/ --crop_size $1 --scale_size $1
fi
