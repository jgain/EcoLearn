cmake_minimum_required(VERSION 3.5)
set(CMAKE_AUTOMOC TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
set(CMAKE_BUILD_TYPE Release)

set(SOURCES
        ${IMPORTER_DIR}/extract_png.cpp
        ${COMMON_DIR}/basic_types.h
        ${IMPORTER_DIR}/data_importer.cpp
        compute_normals.cpp
)

add_executable(compute_slope "${SOURCES}")
target_link_libraries(compute_slope GLEW GL SDL2 SDL2main png ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_include_directories(compute_slope PRIVATE "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
