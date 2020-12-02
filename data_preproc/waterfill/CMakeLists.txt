cmake_minimum_required(VERSION 3.5)

set(CMAKE_BUILD_TYPE Release)
set(SOURCES
        ${IMPORTER_DIR}/extract_png.cpp
	main.cpp
	moisture.cpp
    waterfill.cpp
        ${IMPORTER_DIR}/data_importer.cpp
	)

add_executable(waterfill "${SOURCES}")
target_link_libraries(waterfill png ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_include_directories(waterfill PRIVATE "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
