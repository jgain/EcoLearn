cmake_minimum_required(VERSION 3.6)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(undersample LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Assimp REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)

get_filename_component(PROJECT_BASEDIR ../ ABSOLUTE)

set(VIEWER_DIR ${PROJECT_BASEDIR}/viewer)
set(CANOPY_DIR ${VIEWER_DIR}/canopy_placement)

add_definitions(-DPRJ_SRC_DIR="${PROJECT_BASEDIR}")
add_definitions(-DSHADER_BASEDIR="${CANOPY_DIR}")

set(SOURCES 
	main.cpp
        ${CANOPY_DIR}/gl_wrapper.cpp
        ${CANOPY_DIR}/extract_png.cpp
        ${VIEWER_DIR}/data_importer.cpp
        ${CANOPY_DIR}/gpu_procs.cu
        )

set(SHADERS
        ${CANOPY_DIR}/sphere_fast.frag
        ${CANOPY_DIR}/sphere_fast.vert)

add_executable(undersample "${SOURCES}" "${SHADERS}")

message("${SDL2_INCLUDE_DIRS}")

target_link_libraries(undersample GL GLEW GLU ${SDL2_LIBRARIES} ${ASSIMP_LIBRARIES} ${Boost_LIBRARIES} png sqlite3)
target_include_directories(undersample PRIVATE ${VIEWER_DIR} ${CUDA_INCLUDE_DIRS} ${SDL2_INCLUDE_DIRS} ${PROJECT_BASEDIR})
