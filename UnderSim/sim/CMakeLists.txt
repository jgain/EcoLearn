set (BUILD_GUI 1)
if (BUILD_GUI)

    set(CMAKE_AUTOMOC TRUE)
    set(CMAKE_INCLUDE_CURRENT_DIR TRUE)  # MOC files generated in the binary dir

	#set(BASE_DIR ${PROJECT_SOURCE_DIR}/../..)
        #set(BASE_DIR /home/konrad/EcoSynth)
        set(DATA_IMPORT_DIR ${PROJECT_SOURCE_DIR}/../data_importer)
        set(BASE_ALL_DIR ${PROJECT_SOURCE_DIR}/..)

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
       moisture.cpp
       stroke.cpp
       terrain.cpp
       shape.cpp
       typemap.cpp
       vecpnt.cpp
       view.cpp
       waterfill.cpp
       window.cpp
       shaderProgram.cpp
       trenderer.cpp
       ${DATA_IMPORT_DIR}/data_importer.cpp
       ${BASE_ALL_DIR}/common/basic_types.h
)

   #set(CMAKE_BUILD_TYPE Debug)
   set(CMAKE_BUILD_TYPE Release)
    add_library(vgui ${GUI_SOURCES})
    target_include_directories(vgui PRIVATE ${PROJECT_SOURCE_DIR} ${BASE_ALL_DIR})
    target_link_libraries(vgui common
        ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} ${Qt5Widgets_LIBRARIES} ${Qt5OpenGL_LIBRARIES}
        ${OPENGL_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
    add_executable(sim main.cpp)
    target_include_directories(sim PRIVATE ${PROJECT_SOURCE_DIR} ${BASE_ALL_DIR})
    target_link_libraries(sim vgui sqlite3)

    add_executable(sunsim main_sunsim.cpp)
    target_include_directories(sunsim PRIVATE ${PROJECT_SOURCE_DIR} ${BASE_ALL_DIR})
    target_link_libraries(sunsim vgui sqlite3)

    ADD_DEFINITIONS(-DSONOMA_DB_FILEPATH="${PROJECT_SOURCE_DIR}/../EcoSynth/ecodata/sonoma.db")

endif()
