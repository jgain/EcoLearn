cmake_minimum_required(VERSION 3.10)
project(grass_sim)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(SYSTEM /home/konrad/EcoSynth)
include_directories(SYSTEM /home/konrad/EcoSynth/viewer)

set(PROJECT_SOURCES
        main.cpp
        grass_sim.cpp
        grass_sim.h
        /home/konrad/EcoSynth/viewer/data_importer.cpp
        /home/konrad/EcoSynth/viewer/data_importer.h
        /home/konrad/EcoSynth/viewer/canopy_placement/basic_types.h)

add_executable(grass_sim ${PROJECT_SOURCES})
target_link_libraries(grass_sim sqlite3)
#target_include_directories(grass_sim PRIVATE /home/konrad/EcoSynth)
