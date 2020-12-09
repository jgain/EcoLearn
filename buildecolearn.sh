#!/bin/sh

rm -r build
mkdir build
rm CMakeCache.txt
cd build
make
