cmake_minimum_required(VERSION 3.10)
project(gpusample LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(BASE_ALL_DIR ${PROJECT_SOURCE_DIR}/..)
set(COMMON_DIR ${BASE_ALL_DIR}/common)
set(CLUSTERDIR ${BASE_ALL_DIR}/cluster_distribs/src)

set(SOURCES
    src/kmeans_assign.cu
    src/cudatypes.h
    ${COMMON_DIR}/basic_types.h
    ${BASE_ALL_DIR}/data_importer/data_importer.cpp
    ${BASE_ALL_DIR}/data_importer/data_importer.h
    ${CLUSTERDIR}/AllClusterInfo.cpp
    ${CLUSTERDIR}/ClusterMatrices.cpp
    ${CLUSTERDIR}/dice.cpp
    ${CLUSTERDIR}/ClusterMaps.cpp
    ${CLUSTERDIR}/ClusterAssign.cpp
    ${CLUSTERDIR}/ClusterData.cpp
    ${CLUSTERDIR}/HistogramMatrix.cpp
    ${CLUSTERDIR}/HistogramDistrib.cpp
    ${CLUSTERDIR}/common.cpp
    ${CLUSTERDIR}/generic_rng.cpp
    ${CLUSTERDIR}/PlantSpatialHashmap.cpp
    ${BASE_ALL_DIR}/data_importer/AbioticMapper.cpp
    ${BASE_ALL_DIR}/EcoSynth/kmeans/src/kmeans.cpp
    ${COMMON_DIR}/constants.cpp
    ${BASE_ALL_DIR}/data_importer/map_procs.cu)

add_executable(gpusample src/gpusample.cu ${SOURCES})

target_include_directories(gpusample PRIVATE ${BASE_ALL_DIR} ${COMMON_DIR} ${BASE_ALL_DIR}/EcoSynth/kmeans/src ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(gpusample PRIVATE sqlite3)

target_compile_definitions(gpusample PRIVATE GPUSAMPLE)

get_target_property(GPUSAMPLE_INCLUDES gpusample INCLUDE_DIRECTORIES)

foreach(dir ${GPUSAMPLE_INCLUDES})
    message(${dir})
endforeach()
