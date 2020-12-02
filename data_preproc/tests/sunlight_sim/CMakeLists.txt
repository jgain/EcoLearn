cmake_minimum_required(VERSION 3.6)

project(test_sunlight_sim LANGUAGES CXX CUDA)

set(SUN_DIR ${PROJECT_BASEDIR}/data_preproc/sunlight_sim)

set(CMAKE_BUILD_TYPE Debug)
string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_50")

set(SUPPORT_FILES ${SUN_DIR}/terrain.cpp ${SUN_DIR}/sunsim.cpp ${SUN_DIR}/extract_png.cpp ${SUN_DIR}/lodepng.cpp)
set(EXTRA_FILES ${SUN_DIR}/sunlight.vert ${SUN_DIR}/sunlight.frag ${SUN_DIR}/quad.vert ${SUN_DIR}/quad.frag)
set(HEADER_FILES
    ${SUN_DIR}/terrain.h
    ${SUN_DIR}/sunsim.h
    ${SUN_DIR}/extract_png.h
    ${SUN_DIR}/count_pixels.h)

set (SOURCES
        main.cpp
        ${SUN_DIR}/count_pixels.cu)

add_executable(test_sunlight_sim ${SOURCES} ${SUPPORT_FILES} ${EXTRA_FILES} ${IMPORTER_DIR}/data_importer.cpp)
target_link_libraries(test_sunlight_sim SDL2 SDL2main GL GLEW png ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_include_directories(test_sunlight_sim PRIVATE ${SUN_DIR} "${INCLUDE_DIRS}" ${PROJECT_BASEDIR} )
