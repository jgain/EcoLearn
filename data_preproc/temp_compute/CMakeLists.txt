cmake_minimum_required(VERSION 3.5)

#set(PROJECT_BASEDIR ../../)

#include_directories(${PROJECT_BASEDIR}/viewer ${PROJECT_BASEDIR}/viewer/canopy_placement ${PROJECT_BASEDIR}/viewer/species_optim)

set(CMAKE_BUILD_TYPE Release)

message("CMAKE_SOURCE_DIR:")
message(${CMAKE_SOURCE_DIR})

set(SOURCES
	main.cpp
        ${IMPORTER_DIR}/data_importer.cpp
	)

add_executable(temp_compute main.cpp ${SOURCES})
target_link_libraries(temp_compute ${Boost_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_include_directories(temp_compute PRIVATE "${INCLUDE_DIRS}" ${PROJECT_BASEDIR})
