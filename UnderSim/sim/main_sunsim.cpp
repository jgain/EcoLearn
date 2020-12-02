/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/



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


#include "glheaders.h"
#include <QApplication>
#include <QGLFormat>
#include <QCoreApplication>
#include <QDesktopWidget>
#include <string>
#include <stdexcept>
#include <utility>
#include <memory>
#include <QTimer>

#include "window.h"

struct functor_callback
{
    Window *window;
    int nruns;		// for now, this is obsolete: we will re-run the entire program for successive placements for different canopy placements
    int run_id;
    int nyears;
    std::string chmfile;
    std::string output_file;

    void operator() ()
    {
        // window->run_sunsim_only(chmfile, output_file);
        QCoreApplication::exit();
    }
};

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: sunsim <data directory> <chm file> <output filename>" << std::endl;
        return 1;
    }

    try
    {
        QApplication app(argc, argv);

        std::string datadir = argv[1];
        while (datadir.back() == '/')
            datadir.pop_back();

        std::string chmfile = argv[2];
        std::string output_file = argv[3];

        Window * window = new Window(datadir);

        QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedKingdom));

        window->resize(window->sizeHint());
        window->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
        window->getView().setForcedFocus(window->getTerrain().getFocus());

        // sunwindow

        SunWindow * sunwindow = new SunWindow();
        sunwindow->resize(window->sizeHint());
        sunwindow->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
        sunwindow->setOrthoView(window->getGLWidget());
        sunwindow->getView().setForcedFocus(window->getTerrain().getFocus());
        //sunwindow->show();

        int desktopArea = QApplication::desktop()->width() *
            QApplication::desktop()->height();
        int widgetArea = window->width() * window->height();
        if (((float)widgetArea / (float)desktopArea) < 0.75f)
            window->show();
        else
            window->showMaximized();

        window->setSunWindow(sunwindow);

        functor_callback func;
        func.window = window;
        func.nruns = 1;
        func.chmfile = chmfile;
        func.output_file = output_file;

        QTimer::singleShot(1000, func);

        /*
        window->run_undersim_foolproof();
        return 0;
        */

        int status = app.exec();
        return status;
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
