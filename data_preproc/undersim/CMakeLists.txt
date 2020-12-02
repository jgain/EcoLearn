cmake_minimum_required(VERSION 3.6)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenEXR REQUIRED OpenEXR)
pkg_check_modules(ImageMagick REQUIRED Magick++)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_BUILD_TYPE Release)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets OpenGL)
#find_package(Qt5Widgets 5.5 CONFIG REQUIRED)
#find_package(Qt5OpenGL 5.5 CONFIG REQUIRED)
#find_package(Qt5Core 5.5 CONFIG REQUIRED)

add_definitions(-DSONOMA_DB_FILEPATH "/home/konrad/PhDStuff/prototypes/repo/code/cpp/EcoSynth/ecodata/sonoma.db")

set(UNDERSIM_VIEWDIR "${UNDERSIM_DIR}/sim")

file(GLOB viewer_files
    ${UNDERSIM_VIEWDIR}/*.cpp
    )

set(common_files
    ${UNDERSIM_DIR}/common/map.cpp
    ${UNDERSIM_DIR}/common/region.cpp
    ${UNDERSIM_DIR}/common/terragen.cpp
    ${UNDERSIM_DIR}/common/source2cpp.cpp
    ${UNDERSIM_DIR}/common/obj.cpp)

list(REMOVE_ITEM viewer_files ${UNDERSIM_VIEWDIR}/clusters.cpp)
list(REMOVE_ITEM viewer_files ${UNDERSIM_VIEWDIR}/interp.cpp)
list(REMOVE_ITEM viewer_files ${UNDERSIM_VIEWDIR}/synth.cpp)
list(REMOVE_ITEM viewer_files ${UNDERSIM_VIEWDIR}/main.cpp)
list(REMOVE_ITEM viewer_files ${UNDERSIM_VIEWDIR}/main_sunsim.cpp)
list(FILTER viewer_files EXCLUDE REGEX "${UNDERSIM_VIEWDIR}/moc_.+")
list(FILTER viewer_files EXCLUDE REGEX "${UNDERSIM_VIEWDIR}/.+automoc.cpp$")

set (SOURCES 
        main.cpp
        ${BASE_ALL}/data_importer/data_importer.cpp
        ${viewer_files}
        ${common_files}
        )

add_executable(undersim "${SOURCES}")

message("OpenEXR:")
message(${OpenEXR_LIBRARIES})

target_include_directories(undersim PRIVATE
    "${UNDERSIM_DIR}/sim"
    "${UNDERSIM_DIR}"
    "${UNDERSIM_DIR}/common"
    ${PROJECT_BASEDIR}
    ${UNDERSIM_DIR}/khronos_headers
    ${OpenEXR_INCLUDE_DIRS}
    ${PROJECT_BASEDIR}/viewer/canopy_placement/
    ${ImageMagick_INCLUDE_DIRS}
    ${BASE_ALL})
link_directories(${OpenEXR_LIBRARY_DIRS})
target_link_libraries(undersim Qt5::Core Qt5::Widgets Qt5::OpenGL GL GLEW sqlite3 GLU ${OpenEXR_LIBRARIES})
