cmake_minimum_required(VERSION 2.8)

project(kmeans)

find_package(Qt5 COMPONENTS Core Widgets REQUIRED)
#set(CMAKE_AUTOMOC ON)

# Note : better let your IDE switch between configs.
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set (CMAKE_CXX_STANDARD 17)


# Note : using GLOB requires to re-run CMake for every new file added to the project.
FILE(GLOB_RECURSE HEADER_FILES src/*.h)
FILE(GLOB_RECURSE SOURCE_FILES src/*.cpp)
#FILE(GLOB_RECURSE QML_FILES qml/*.qml)
#FILE(GLOB SHADER_FILES shaders/*.glsl)


#set(QRC_FILES
#    resources/resources.qrc
#)

#qt5_add_resources(RESOURCE_FILES ${QRC_FILES})

#include_directories(src/)

add_executable(main ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(main Qt5::Core Qt5::Widgets)
