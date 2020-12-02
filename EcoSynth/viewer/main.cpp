/****************************************************************************
**
** Copyright (C) 2012 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/
#include <GL/glew.h>

#include "glheaders.h"
#include <QApplication>
#include <QGLFormat>
#include <QCoreApplication>
#include <QDesktopWidget>
#include <string>
#include <stdexcept>
#include <utility>
#include <memory>

#include "window.h"



int main(int argc, char *argv[])
{
    std::cout << "Shader base directory: " << SHADER_BASEDIR << std::endl;
    try
    {
        QApplication app(argc, argv);
        Window * window = new Window();

        QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedKingdom));

        window->resize(window->sizeHint());
        window->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
        window->getView().setForcedFocus(window->getTerrain().getFocus());
        //window->setAttribute(Qt::WA_DeleteOnClose);

#ifndef PAINTCONTROL
        // sunwindow
        SunWindow * sunwindow = new SunWindow();
        sunwindow->resize(window->sizeHint());
        sunwindow->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
        sunwindow->setOrthoView(window->getGLWidget());
        sunwindow->getView().setForcedFocus(window->getTerrain().getFocus());
        sunwindow->show();
#endif
        int desktopArea = QApplication::desktop()->width() *
            QApplication::desktop()->height();
        int widgetArea = window->width() * window->height();
        if (((float)widgetArea / (float)desktopArea) < 0.75f)
            window->show();
        else
            window->showMaximized();

#ifndef PAINTCONTROL
        window->setSunWindow(sunwindow);
#endif

        if (argc == 3)
        {
            std::string cfgfilename;
            if (strcmp(argv[1], "--config"))
            {
                throw std::invalid_argument(std::string("unknown switch: ") + std::string(argv[1]));
            }
            else
            {
                cfgfilename = argv[2];
                window->loadConfig(cfgfilename);
            }
            //window->loadConfig();		// just testing config load for now, still have to write parser
        }

        int status = app.exec();

        //cudaDeviceReset();

        return status;
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception caught in main" << std::endl;
        //cudaDeviceReset();
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
