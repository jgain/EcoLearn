cmake_minimum_required(VERSION 3.5)

project(species_assign LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Release)

find_package(SDL2)
find_package(glm)
find_package(assimp)

set(CUDA_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIR ${CUDA_DIR}/include)

set(PACKAGE_INCLUDE_DIRS "${SDL2_INCLUDE_DIRS}" "${GLM_INCLUDE_DIRS}" "${OPENGL_INCLUDE_DIR}" "${ASSIMP_INCLUDE_DIRS}" "${CUDA_INCLUDE_DIR}")


#include_directories(../../viewer ../../viewer/species_optim ../../viewer/canopy_placement)

set(SOURCES
        ${SPECIES_DIR}/gpu_eval.cu
        ${SPECIES_DIR}/species_assign_exp.cpp
        ${IMPORTER_DIR}/data_importer.cpp
	${VIEWER_DIR}/MapFloat.h
        ${COMMON_DIR}/basic_types.h
        )

#add_executable(assign_species main.cpp "${SOURCES}" ${CANOPY_DIR}/gpu_procs.cu)
#target_link_libraries(assign_species ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES} GL GLEW GLU )
#target_include_directories(assign_species PRIVATE "${PACKAGE_INCLUDE_DIRS}" "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})

add_executable(assign_species_single main_single.cpp "${SOURCES}" ${CANOPY_DIR}/gpu_procs.cu)
target_link_libraries(assign_species_single ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES} GL GLEW GLU )
target_include_directories(assign_species_single PRIVATE "${PACKAGE_INCLUDE_DIRS}" "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})


