cmake_minimum_required(VERSION 3.2)
project(cluster_distribs LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)

set(INCLUDE_DERIVATOR YES)
set(INCLUDE_SAMPLER YES)
set(INCLUDE_OPTIMISER YES)

set(BASEALL_DIR /home/konrad/PhDStuff/prototypes/repo/code/cpp)
set(VIEWERDIR ${BASEALL_DIR}/EcoSynth/viewer)
set(ECODIR ${BASEALL_DIR}/EcoSynth)
set(DATAIMPORT_DIR ${BASEALL_DIR}/data_importer)

set(SUPPORT_FILES
    #GridDistribs.cpp
    #SandboxDistrib.cpp
    #PlantTypeDistrib.cpp
    HistogramDistrib.cpp
    dice.cpp
    common.cpp
    generic_rng.cpp
    ${DATAIMPORT_DIR}/extract_png.cpp
    HistogramMatrix.cpp
    AllClusterInfo.cpp
    ClusterMatrices.cpp
    ClusterDistribDerivator.cpp
    ClusterMaps.cpp
    ClusterAssign.cpp
    ClusterData.cpp
    PlantSpatialHashmap.cpp
    UndergrowthSampler.cpp
    UndergrowthRefiner.cpp
    ${BASEALL_DIR}/gpusample/src/gpusample.cu
    ${BASEALL_DIR}/data_importer/map_procs.cu
    ${BASEALL_DIR}/data_importer/AbioticMapper.cpp
    ${BASEALL_DIR}/data_importer/AbioticMapper.h
        ClusterAssign.cpp
	${ECODIR}/kmeans/src/kmeans.cpp
        ${BASEALL_DIR}/data_importer/data_importer.cpp
	${VIEWERDIR}/histcomp_window.cpp
        ${VIEWERDIR}/timer.cpp
        ${BASEALL_DIR}/common/constants.cpp)

if (INCLUDE_DERIVATOR)
    add_executable(main main.cpp ${SUPPORT_FILES})
    target_link_libraries(main png sqlite3 stdc++fs ${Qt5Widgets_LIBRARIES})
    target_include_directories(main PRIVATE ${VIEWERDIR} ${ECODIR} ${ECODIR}/kmeans/src ${Qt5Widgets_INCLUDE_DIRS} ${BASEALL_DIR})
endif()

if (INCLUDE_SAMPLER)
    add_executable(main_sample main_sample.cpp ${SUPPORT_FILES})
    target_link_libraries(main_sample png sqlite3 ${Qt5Widgets_LIBRARIES})
    target_include_directories(main_sample PRIVATE ${VIEWERDIR} ${ECODIR} ${ECODIR}/kmeans/src ${Qt5Widgets_INCLUDE_DIRS} ${BASEALL_DIR})
endif()

set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET main_sample PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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

#add_executable(main_basic_pdb main_basic_pdb.cpp ${SUPPORT_FILES})
#add_executable(main_read main_read.cpp ${SUPPORT_FILES})
#add_executable(test_synth test_synth.cpp ${SUPPORT_FILES})
#add_executable(tests tests.cpp ${SUPPORT_FILES})

#target_link_libraries(main_basic_pdb png)
#target_link_libraries(main_read png)
#target_link_libraries(test_synth png)
#target_link_libraries(tests png)


#target_include_directories(main_basic_pdb PRIVATE /home/konrad/EcoSynth/viewer/canopy_placement /home/konrad/EcoSynth/viewer)
#target_include_directories(main_read PRIVATE /home/konrad/EcoSynth/viewer/canopy_placement /home/konrad/EcoSynth/viewer)
#target_include_directories(test_synth PRIVATE /home/konrad/EcoSynth/viewer/canopy_placement /home/konrad/EcoSynth/viewer)
#target_include_directories(tests PRIVATE /home/konrad/EcoSynth/viewer/canopy_placement /home/konrad/EcoSynth/viewer)
