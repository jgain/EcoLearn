cmake_minimum_required(VERSION 3.2)
project(cluster_distribs)

find_package(Qt5Widgets 5.5 REQUIRED)

set(HEADERS
    src/common.h
    src/dice.h
    src/distribution.h
    src/generic_rng.h)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(src)
