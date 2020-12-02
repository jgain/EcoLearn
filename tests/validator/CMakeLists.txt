cmake_minimum_required(VERSION 3.10)

project(validator)
find_package(Qt5Widgets 5.5 REQUIRED)

if (Qt5Widgets_FOUND)
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    add_definitions(${Qt5Widgets_DEFINITIONS})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)

set(ECOSYNTH_DIR /home/konrad/EcoSynth/viewer)
set(CLUSTERDIR /home/konrad/PhDStuff/prototypes/repo/code/cpp/cluster_distribs/src)

set(INCLUDE_DIRS
        /home/konrad/EcoSynth
        /home/konrad/EcoSynth/kmeans/src
        ${ECOSYNTH_DIR}
        ${ECOSYNTH_DIR}/canopy_placement
        ${CLUSTERDIR}
        )

set(COMMON_COMPILE_SOURCES
    ${ECOSYNTH_DIR}/data_importer.cpp
    ${ECOSYNTH_DIR}/data_importer.h)

set(PDB_VALIDATE_SOURCES
	${COMMON_COMPILE_SOURCES}
        main.cpp
        validator.cpp
        validator.h
	)

set(CLUSTERFILE_VALIDATE_SOURCES
        ${COMMON_COMPILE_SOURCES}
        ${CLUSTERDIR}/ClusterMatrices.cpp
        ${CLUSTERDIR}/ClusterMatrices.h
        ${CLUSTERDIR}/HistogramDistrib.cpp
        ${CLUSTERDIR}/HistogramDistrib.h
        ${CLUSTERDIR}/HistogramMatrix.cpp
        ${CLUSTERDIR}/HistogramMatrix.h
        ${CLUSTERDIR}/dice.cpp
        ${CLUSTERDIR}/dice.h
        ${CLUSTERDIR}/common.cpp
        ${CLUSTERDIR}/common.h
        ${CLUSTERDIR}/generic_rng.cpp
        ${CLUSTERDIR}/generic_rng.h
        ${ECOSYNTH_DIR}/histcomp_window.cpp
        ${ECOSYNTH_DIR}/timer.cpp
        /home/konrad/EcoSynth/kmeans/src/kmeans.cpp
        )
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

include_directories(${INCLUDE_DIRS})

#add_executable(main ${PDB_VALIDATE_SOURCES})
#add_executable(validate_clusterfile validate_clusterfile.cpp ${CLUSTERFILE_VALIDATE_SOURCES})
add_executable(validate_commondata validate_commondata.cpp ${COMMON_COMPILE_SOURCES})
#target_link_libraries(main sqlite3 stdc++fs)
#target_link_libraries(validate_clusterfile sqlite3 stdc++fs)
target_link_libraries(validate_commondata sqlite3 stdc++fs)
target_compile_options(validate_commondata PRIVATE -fPIC)

add_executable(validate_underdensity validate_underdensity.cpp ${CLUSTERFILE_VALIDATE_SOURCES})
target_link_libraries(validate_underdensity sqlite3 ${Qt5Widgets_LIBRARIES})
target_compile_options(validate_underdensity PRIVATE -fPIC)

add_executable(validate_addremove validate_addremove.cpp ${CLUSTERFILE_VALIDATE_SOURCES})
target_link_libraries(validate_addremove sqlite3 stdc++fs ${Qt5Widgets_LIBRARIES})
target_compile_options(validate_addremove PRIVATE -fPIC)

add_executable(validate_canopytree_mod validate_canopytree_mod.cpp ${CLUSTERFILE_VALIDATE_SOURCES})
target_link_libraries(validate_canopytree_mod sqlite3 stdc++fs ${Qt5Widgets_LIBRARIES})
target_compile_options(validate_canopytree_mod PRIVATE -fPIC)

add_executable(rw_sampler rw_sampler.cpp)
