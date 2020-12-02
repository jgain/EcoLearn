cmake_minimum_required(VERSION 3.5)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(canopy_placement_tests LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)

find_package(Boost COMPONENTS system filesystem REQUIRED)

get_filename_component(PROJECT_BASEDIR ../../ ABSOLUTE)
set(VIEWER_DIR ${PROJECT_BASEDIR}/viewer)
set(CANOPY_DIR ${PROJECT_BASEDIR}/viewer/canopy_placement)
set(SPECIES_DIR ${PROJECT_BASEDIR}/viewer/species_optim)
set(UNDERSIM_DIR ${PROJECT_BASEDIR}/viewer/UnderSim)
set(INCLUDE_DIRS ${VIEWER_DIR} ${CANOPY_DIR} ${SPECIES_DIR})
set(COMMON_LINK_LIBRARIES sqlite3)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_50")

#find_package(Boost COMPONENTS system REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(SDL2 REQUIRED)
find_package(assimp REQUIRED)

set(CUDA_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIR ${CUDA_DIR}/include)

set(PACKAGE_INCLUDE_DIRS "${SDL2_INCLUDE_DIRS}" "${GLM_INCLUDE_DIRS}" "${Boost_INCLUDE_DIRS}" "${OPENGL_INCLUDE_DIR}" "${ASSIMP_INCLUDE_DIRS}" "${CUDA_INCLUDE_DIR}")

add_definitions(-DPRJ_SRC_DIR="${PROJECT_BASEDIR}")
add_definitions(-DSHADER_BASEDIR="${CANOPY_DIR}")

set(SOURCES
        test_canopy_placement.cpp
        ${CANOPY_DIR}/gpu_procs.cu
        ${CANOPY_DIR}/canopy_placer.cpp
        ${VIEWER_DIR}/data_importer.cpp
        ${CANOPY_DIR}/gl_wrapper.cpp
        ${CANOPY_DIR}/misc.cpp
        ${CANOPY_DIR}/extract_png.cpp)

message("Boost libraries: ")
message("${Boost_LIBRARIES}")

add_executable(canopy_placement_test1 ${SOURCES})
target_include_directories(canopy_placement_test1 PRIVATE "${PACKAGE_INCLUDE_DIRS}" "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
target_link_libraries(canopy_placement_test1 ${Boost_LIBRARIES} GL GLEW SDL2 SDL2main assimp png ${COMMON_LINK_LIBRARIES})

