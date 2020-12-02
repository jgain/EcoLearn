#!/bin/sh

rm -r build
mkdir build
rm CMakeCache.txt
cd build
cmake -DCMAKE_CXX_COMPILER=g++ ..
cmake -DCMAKE_BUILD_TYPE=Release ..
make
