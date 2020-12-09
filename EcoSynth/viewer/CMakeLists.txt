    cmake_minimum_required(VERSION 3.5)
    #set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_AUTOMOC TRUE)
    SET(CMAKE_AUTOUIC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR TRUE)  # MOC files generated in the binary dir

    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -03 -gencode arch=sm_40)
    message("CUDAFLAGS:")
    message("${CUDA_NVCC_FLAGS}")
    #set(CUDA_NVCC_FLAGS -03 -gencode arch=sm_40 --compiler-options -fPIC)
    set(CUDA_NVCC_FLAGS -03 -gencode arch=sm_40)
    message("CUDAFLAGS:")
    message("${CUDA_NVCC_FLAGS}")

    # set some useful directories
    set(CUDA_DIR /usr/local/cuda)
    set(CUDA_INCLUDE_DIR ${CUDA_DIR}/include)
    set(BASE_ALL_DIR ${PROJECT_SOURCE_DIR}/..)
    set(VIEWER_DIR ${PROJECT_SOURCE_DIR}/viewer)
    set(CANOPY_DIR ${BASE_ALL_DIR}/canopy_placement)
    set(SPECIES_DIR ${BASE_ALL_DIR}/species_optim)
    set(GPUSYNTH_DIR ${BASE_ALL_DIR}/gpusynth/src)
    set(ANALYSIS_DIR ${BASE_ALL_DIR}/cluster_distribs/src)
    set(KMEANS_DIR ${PROJECT_SOURCE_DIR}/kmeans/src)
    set(COLMAP_DIR ${PROJECT_SOURCE_DIR}/colourmaps)
    set(DATAIMPORT_DIR ${BASE_ALL_DIR}/data_importer)
    set(COMMON_DIR ${BASE_ALL_DIR}/common)
    set(GPUSAMPLE_DIR ${BASE_ALL_DIR}/gpusample/src)

    set(SPECIES_SOURCES
       ${SPECIES_DIR}/species_assign_exp.cpp
       ${SPECIES_DIR}/species_assign_exp.h
       ${SPECIES_DIR}/gpu_eval.cu
        )

    set(CANOPY_SOURCES
       ${CANOPY_DIR}/gl_wrapper.cpp
       ${CANOPY_DIR}/gpu_procs.cu
       ${CANOPY_DIR}/sphere_obj_string.cpp
       ${CANOPY_DIR}/canopy_placer.cpp
        )

    set(DATAIMPORT_SOURCES
       ${DATAIMPORT_DIR}/data_importer.cpp
       ${DATAIMPORT_DIR}/AbioticMapper.cpp
       ${DATAIMPORT_DIR}/map_procs.cu
       ${DATAIMPORT_DIR}/extract_png.cpp
        )

    set(ANALYSIS_SOURCES
       ${ANALYSIS_DIR}/ClusterDistribDerivator.cpp
       ${ANALYSIS_DIR}/ClusterDistribDerivator.h
       ${ANALYSIS_DIR}/ClusterMatrices.cpp
       ${ANALYSIS_DIR}/ClusterMatrices.h
       ${ANALYSIS_DIR}/HistogramMatrix.cpp
       ${ANALYSIS_DIR}/HistogramMatrix.h
       ${ANALYSIS_DIR}/HistogramDistrib.cpp
       ${ANALYSIS_DIR}/HistogramDistrib.h
       ${ANALYSIS_DIR}/UndergrowthRefiner.cpp
       ${ANALYSIS_DIR}/UndergrowthRefiner.h
       ${ANALYSIS_DIR}/UndergrowthSampler.cpp
       ${ANALYSIS_DIR}/ClusterData.cpp
       ${ANALYSIS_DIR}/ClusterAssign.cpp
       ${ANALYSIS_DIR}/ClusterMaps.cpp
       ${ANALYSIS_DIR}/PlantSpatialHashmap.cpp
       ${ANALYSIS_DIR}/AllClusterInfo.cpp
       ${ANALYSIS_DIR}/common.h
       ${ANALYSIS_DIR}/common.cpp
       ${ANALYSIS_DIR}/dice.h
       ${ANALYSIS_DIR}/dice.cpp
       ${ANALYSIS_DIR}/generic_rng.h
       ${ANALYSIS_DIR}/generic_rng.cpp
        )

    set(COMMON_SOURCES
       ${COMMON_DIR}/constants.cpp
       ${COMMON_DIR}/custom_exceptions.cpp
        )

    set(GPUSAMPLE_SOURCES
       ${GPUSAMPLE_DIR}/gpusample.h
       ${GPUSAMPLE_DIR}/gpusample.cu
        )

    set(GUI_SOURCES
       glwidget.cpp
       timer.cpp
       pft.cpp
       sim.cpp
       sun.cpp
       eco.cpp
       grass.cpp
       descriptor.cpp
       dice_roller.cpp
       palette.cpp
       fill.cpp
       ipc.cpp
       moisture.cpp
       stroke.cpp
       terrain.cpp
       shape.cpp
       typemap.cpp
       outimage.cpp
       vecpnt.cpp
       view.cpp
       waterfill.cpp
       window.cpp
       shaderProgram.cpp
       trenderer.cpp
       specselect_window.cpp
       specpalette_window.cpp
       palette_base.cpp
       histcomp_window.cpp
       histcomp_window.h
       ConfigReader.cpp
       convertpaintingdialog.cpp
       convertpaintingdialog.ui
       speciesColoursWindow.cpp
       speciesColoursWindow.h

       ${SPECIES_SOURCES}
       ${CANOPY_SOURCES}
       ${DATAIMPORT_SOURCES}
       ${ANALYSIS_SOURCES}
       ${COMMON_SOURCES}
       ${GPUSAMPLE_SOURCES}

       ${KMEANS_DIR}/kmeans.cpp
)

    set(SHADER_SOURCES
        ${CANOPY_DIR}/sphere_fast.vert
        ${CANOPY_DIR}/sphere_fast.frag)

	set(INCLUDE_DIRS "${SDL2_INCLUDE_DIRS}" "${GLM_INCLUDE_DIRS}" "${OPENGL_INCLUDE_DIR}" "${ASSIMP_INCLUDE_DIRS}" "${CUDA_INCLUDE_DIR}" ${BASE_ALL_DIR})
    set(LINK_LIBS "${SDL2_LIBRARIES}" "${GLM_LIBRARIES}" "${OPENGL_LIBRARIES}" "${ASSIMP_LIBRARIES}" pthread X11 png z)
    set(added_libs png SDL2 SDL2main opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs assimp boost_system ${CUDART_LIBRARIES} "${LINK_LIBS}")

    message("PROJECT_SOURCE_DIR: ")
    message("${PROJECT_SOURCE_DIR}")


    add_library(vgui ${GUI_SOURCES})
    target_link_libraries(vgui common 	#add clh here if compilation fails
        ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${Qt5Widgets_LIBRARIES} ${Qt5OpenGL_LIBRARIES}
        ${OPENGL_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} ${ZMQ_LIBRARY} ${added_libs})
    target_include_directories(vgui PRIVATE
        "${INCLUDE_DIRS}"
        ${VIEWER_DIR}
        ${CANOPY_DIR}
        ${SPECIES_DIR}
        ${GPUSYNTH_DIR}
        ${ANALYSIS_DIR}
        ${KMEANS_DIR}
        ${GPUSAMPLE_DIR})
    set_property(TARGET vgui PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    #convert the fpic option in Qt5::Core over to INTERFACE_POSITION_INDEPENDENT_CODE
    get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
    string(REPLACE "-fPIC" "" new_core_options ${core_options})
    set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
    set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
    if ( TARGET Qt5::Core )
            get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
            string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
            set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
            set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
            set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
    endif()

    message("g++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

    include_directories(${VIEWER_DIR} ${CANOPY_DIR} ${SPECIES_DIR})

    message("CMAKE_CXX_FLAGS:")
    message("${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
    set(basedir ${BASE_ALL_DIR}/EcoSynth)

    add_definitions(-DPRJ_SRC_DIR="${PROJECT_SOURCE_DIR}")
    #add_definitions(-DCC_GL_DEBUG_)	TODO: shouldnt be necessary to uncomment this. We have too many GL error messages and it slows the program down. Fix them?
    add_definitions(-DSHADER_BASEDIR="${CANOPY_DIR}")
    add_definitions(-DSPHEREMODEL_BASEDIR="${CANOPY_DIR}")
    add_definitions(-DSRC_BASEDIR="${basedir}")
    add_definitions(-DCOLMAP_DIR="${COLMAP_DIR}")
    add_definitions(-Dcimg_use_png)
    add_executable(viewer main.cpp)

    target_link_libraries(viewer vgui sqlite3)
    set_property(TARGET viewer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    target_include_directories(viewer PRIVATE
        "${INCLUDE_DIRS}"
        ${BASE_ALL_DIR}/species_interact
        ${GPUSYNTH_DIR}
        ${ANALYSIS_DIR}
        ${KMEANS_DIR})
