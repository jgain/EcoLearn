cmake_minimum_required(VERSION 3.5)
project(canopy_placement LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_50 -g -Xcompiler -rdynamic")
#string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -g")

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
add_definitions(-DSPHEREMODEL_BASEDIR="${CANOPY_DIR}")

set(SOURCES
        main.cpp
        ${CANOPY_DIR}/gpu_procs.cu
        ${CANOPY_DIR}/canopy_placer.cpp
        ${IMPORTER_DIR}/data_importer.cpp
        ${CANOPY_DIR}/gl_wrapper.cpp
        #${CANOPY_DIR}/misc.cpp
        ${IMPORTER_DIR}/extract_png.cpp)

message("Boost libraries: ")
message("${Boost_LIBRARIES}")

add_executable(canopy_placement ${SOURCES})
target_include_directories(canopy_placement PRIVATE "${PACKAGE_INCLUDE_DIRS}" "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
target_link_libraries(canopy_placement ${Boost_LIBRARIES} GL GLEW SDL2 SDL2main assimp png ${COMMON_LINK_LIBRARIES})
