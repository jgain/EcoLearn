VPATH += ../shared
INCLUDEPATH += ../shared

#LIBS += -lglut32
#LIBS += /Developer/SDKs/MacOSX10.7.sdk/System/Library/Frameworks/GLUT.framework/glut
LIBS += /System/Library/Frameworks/GLUT.framework/glut
HEADERS       = glwidget.h \
                window.h \
                view.h \
                terrain.h \
                vecpnt.h \
                qtglut.h

SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                view.cpp \
                terrain.cpp \
                vecpnt.cpp \
                qtglut.cpp
QT           += opengl widgets

# install
target.path = $$[QT_INSTALL_EXAMPLES]/qtbase/opengl/hellogl
sources.files = $$SOURCES $$HEADERS $$RESOURCES $$FORMS hellogl.pro
sources.path = $$[QT_INSTALL_EXAMPLES]/qtbase/opengl/hellogl
INSTALLS += target sources


simulator: warning(This example might not fully work on Simulator platform)
