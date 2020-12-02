set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenEXR_CFLAGS_OTHER}")

set(COMMON_SOURCES
    initialize.cpp
    map.cpp
    maputils.cpp
    mathutils.cpp
    obj.cpp
    progress.cpp
    region.cpp
    stats.cpp
    terragen.cpp
    timer.cpp)

if (BUILD_SOURCE2CPP)
    set(KERNELS
        ../sim/shaders/basic.vert
        ../sim/shaders/basic.frag
        ../sim/shaders/genNormal.vert
        ../sim/shaders/genNormal.frag
        ../sim/shaders/simple.vert
        ../sim/shaders/simple.frag
        ../sim/shaders/phong.vert
        ../sim/shaders/phong.frag
        ../sim/shaders/phongRS.vert
        ../sim/shaders/phongRS.frag
        ../sim/shaders/phongRSmanip.vert
        ../sim/shaders/phongRSmanip.frag
        ../sim/shaders/rad_scaling_pass1.vert
        ../sim/shaders/rad_scaling_pass1.frag
        ../sim/shaders/rad_scaling_pass2.vert
        ../sim/shaders/rad_scaling_pass2.frag
        ../sim/shaders/sun.vert
        ../sim/shaders/sun.frag
        ../sim/shaders/canopy.vert
        ../sim/shaders/canopy.frag
        ../clh/texmark.cl)
    add_custom_command(
        OUTPUT source2cpp.cpp
        COMMAND ${PYTHON_EXECUTABLE} source2cpp.py ${KERNELS} ${CMAKE_CURRENT_BINARY_DIR}/source2cpp.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        MAIN_DEPENDENCY source2cpp.py
        DEPENDS ${KERNELS}
        COMMENT "Creating C++ source for shaders and kernels" VERBATIM)
    list(APPEND COMMON_SOURCES source2cpp.cpp)
endif()

add_library(common ${COMMON_SOURCES})
target_link_libraries(common ${OpenEXR_LIBRARIES} ${Boost_SERIALIZATION_LIBRARY})
add_executable(map2exr map2exr.cpp)
target_link_libraries(map2exr common)
#add_executable(exrrewrite exrrewrite.cpp)
#target_link_libraries(exrrewrite common ${OpenEXR_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY})
# if (BUILD_GDAL2EXR)
#    add_executable(gdal2exr gdal2exr.cpp)
#    target_link_libraries(gdal2exr common ${Boost_PROGRAM_OPTIONS_LIBRARY} ${GDAL_LIBRARY})
# endif()
# if (BUILD_UTSCLASSIFY)
#    add_executable(utsclassify utsclassify.cpp)
#    target_link_libraries(utsclassify common
#        ${Boost_PROGRAM_OPTIONS_LIBRARY} ${LIBSVM_LIBRARIES}
#    )
# endif()

# cmake sets ImageMagick_CFLAGS_OTHER as a list, but we can't set the
# COMPILE_FLAGS property from a list because it puts semi-colons into the
# command line. Paste the items back into a string.
set(ImageMagick_CFLAGS_STR "")
foreach (FLAG ${ImageMagick_CFLAGS_OTHER})
    set(ImageMagick_CFLAGS_STR "${ImageMagick_CFLAGS_STR} ${FLAG}")
endforeach()
set_source_files_properties(map_rgba.cpp PROPERTIES COMPILE_FLAGS "${ImageMagick_CFLAGS_STR}")
add_library(common_rgba map_rgba.cpp)
target_link_libraries(common_rgba common ${ImageMagick_LIBRARIES})

add_executable(color2gray color2gray.cpp)
target_link_libraries(color2gray common common_rgba)
#add_executable(heighttool heighttool.cpp)
#target_link_libraries(heighttool common common_rgba ${Boost_PROGRAM_OPTIONS_LIBRARY})
