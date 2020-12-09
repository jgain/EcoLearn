# This is needed for automoc
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(EcoLearn LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.5)
include(CheckLibraryExists)
#set(CMAKE_PREFIX_PATH $ENV{HOME}/Qt/5.7/gcc_64)


option(SMOKE_TEST "run a smoke test on every build" 1)
option(COVERAGE "compile for use with gcov" 0)
option(ASAN "compile with the address sanitiser" 0)
option(TSAN "compile with the thread sanitiser" 0)
option(SYNTHESIS_STATS "collect extra statistics about synthesis" 0)
enable_testing()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

message("g++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")


#if (${CMAKE_CXX_COMPILER_VERSION} LESS 5.0)
#    message(FATAL_ERROR "You need a version of gcc > 5.0")
#endif (${CMAKE_CXX_COMPILER_VERSION} LESS 5.0)

### Check for dependencies
set(Boost_USE_MULTITHREADED ON)
add_definitions(-DBOOST_FILESYSTEM_VERSION=3)
find_package(Boost REQUIRED COMPONENTS filesystem program_options serialization system)
find_package(Doxygen)
find_package(PkgConfig REQUIRED)
find_package(PythonInterp)
find_package(OpenCL)
find_package(Qt5Widgets 5.5 REQUIRED)
find_package(Qt5OpenGL 5.5 REQUIRED)
find_package(OpenGL)
find_package(GLUT)
## use dpkg -L libzmq3-dev to find library location
find_library(ZMQ_LIBRARY NAMES zmq zmq3 HINTS "/usr/lib/x86_64-linux-gnu/")
pkg_check_modules(CppUnit cppunit)
pkg_check_modules(GLEW glew)
pkg_check_modules(OpenEXR REQUIRED OpenEXR)
pkg_check_modules(Eigen REQUIRED eigen3)
pkg_check_modules(ImageMagick REQUIRED Magick++)

### Set include and link directories and definitions
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/khronos_headers
    ${OpenEXR_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${ImageMagick_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/viewer/
)
link_directories(
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/clh
    ${PROJECT_SOURCE_DIR}/viewer
    ${CppUnit_LIBRARY_DIRS}
    ${OpenEXR_LIBRARY_DIRS}
    ${Eigen_LIBRARY_DIRS}
    ${ImageMagick_LIBRARY_DIRS}
    ${Boost_LIB_DIRS}
)

if (OPENCL_FOUND)
    add_definitions(-D__CL_ENABLE_EXCEPTIONS)
    # Needed to get OpenCL 1.1 symbols back
    add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

    include_directories(${OPENCL_INCLUDE_DIRS})
    link_directories(${OPENCL_LIB_DIR})
endif()

if (GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIR})
endif()
if (LIBSVM_FOUND)
    include_directories(${LIBSVM_INCLUDE_DIRS})
endif()
if (Qt5Widgets_FOUND)
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    add_definitions(${Qt5Widgets_DEFINITIONS})
endif()
if (GLUT_FOUND)
    include_directories(${GLUT_INCLUDE_DIR})
endif()
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_directories(${GLEW_LIBRARY_DIRS})
endif()
if (Qt5OpenGL_FOUND)
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    add_definitions(${Qt5OpenGL_DEFINITIONS})
endif()
if (OPENGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR})
endif()
if (CPPUNIT_FOUND)
    include_directories(${CPPUNIT_INCLUDE_DIRS})
    link_directories(${CppUnit_LIBRARY_DIRS})
endif()

### Check which components to build
if (PYTHONINTERP_FOUND AND (OPENCL_FOUND OR OPENGL_FOUND))
    set(BUILD_SOURCE2CPP TRUE)
endif()
if (OPENCL_FOUND AND PYTHONINTERP_FOUND)
    set(BUILD_SYNTHESIZER TRUE)
    set(BUILD_CLH TRUE)
endif()
if (Qt5Widgets_FOUND
        AND Qt5OpenGL_FOUND
        AND OPENGL_FOUND
        AND GLUT_FOUND
        AND GLEW_FOUND
        AND BUILD_SYNTHESIZER)
    set(BUILD_GUI TRUE)
endif()
if (DOXYGEN_FOUND)
    set(BUILD_DOCS TRUE)
endif()
if (LIBSVM_FOUND)
    set(BUILD_UTSCLASSIFY TRUE)
endif()
if (GDAL_FOUND)
    set(BUILD_GDAL2EXR TRUE)
endif()
if (CPPUNIT_FOUND)
    set(BUILD_TESTS TRUE)
endif()

### Configure documentation build
if (BUILD_DOCS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        MAIN_DEPENDENCY ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/clh/*
            ${CMAKE_CURRENT_SOURCE_DIR}/common/*
            ${CMAKE_CURRENT_SOURCE_DIR}/viewer/*
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation" VERBATIM)
    add_custom_target(doc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doc/html/index.html)
    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES doc/html)
endif()

### Set flags specific to the compiler
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
    # Eigen gives lots of warnings for this
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare")
    # On Apple the framework headers are found ahead of the khronos_headers
    # directory, and they give deprecation warnings about OpenCL 1.1 functions.
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    endif()
    if (NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_CXX_LINKER_FLAGS_RELEASE} -O1 -s")
    endif()
    # Link time optimization gives much better performance
    if (APPLE)
    	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
    	set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -flto")
    	set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -flto")
    endif()
endif()
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wno-unused-local-typedefs")
    if (${COVERAGE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    endif()
    if (${ASAN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libasan")
    endif()
    if (${TSAN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fpie -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -static-libtsan")
    endif()
endif()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Eigen generates lots of warnings due to the register keyword being deprecated
    add_definitions(-Dregister=)
endif()
if (SYNTHESIS_STATS)
    add_definitions(-DSYNTHESIS_STATS)
endif()
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DUTS_DEBUG_CONTAINERS")
# commented out because it causes an issue with later version of BOOST

add_subdirectory(common)
add_subdirectory(clh)
add_subdirectory(viewer)
