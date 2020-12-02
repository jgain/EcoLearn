cmake_minimum_required(VERSION 3.10)
project(grass_sim)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

set(PROJECT_SOURCES
        main.cpp
        grass_sim.cpp
        grass_sim.h
        data_importer.cpp
        data_importer.h
        basic_types.h
        MapFloat.h)

add_executable(grass_sim ${PROJECT_SOURCES})
target_link_libraries(grass_sim sqlite3 stdc++fs)

