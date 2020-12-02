cmake_minimum_required(VERSION 3.5)
project(sunlight_sim LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
#set(PROJECT_SOURCEDIR ../..)

#include_directories(${PROJECT_SOURCEDIR}/viewer ${PROJECT_SOURCEDIR}/viewer/canopy_placement ${PROJECT_SOURCEDIR}/viewer/species_optim)

set(SUPPORT_FILES
    terrain.cpp
    sunsim.cpp
    ${IMPORTER_DIR}/extract_png.cpp
    ${IMPORTER_DIR}/lodepng.cpp
    count_pixels.cu)
set(EXTRA_FILES
    sunlight.vert
    sunlight.frag
    quad.vert
    quad.frag)
set(HEADER_FILES
    terrain.h
    sunsim.h
    extract_png.h)

add_executable(main main.cpp ${SUPPORT_FILES} ${EXTRA_FILES} ${IMPORTER_DIR}/data_importer.cpp)
target_link_libraries(main SDL2 SDL2main GL GLEW png ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_include_directories(main PRIVATE "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
