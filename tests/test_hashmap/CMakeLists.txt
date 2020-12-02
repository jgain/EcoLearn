cmake_minimum_required(VERSION 3.5)
project(test_hashmap)

set(BASEDIR ${PROJECT_SOURCE_DIR}/../..)
set(SRCDIR ${BASEDIR}/cluster_distribs/src)

set(CMAKE_BUILD_TYPE Debug)

add_executable(test_add1 src/test_add1.cpp ${SRCDIR}/PlantSpatialHashmap.cpp)
add_executable(test_add2 src/test_add2.cpp ${SRCDIR}/PlantSpatialHashmap.cpp)

include_directories(${BASEDIR} ${SRCDIR})
