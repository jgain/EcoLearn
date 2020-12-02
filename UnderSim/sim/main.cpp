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

int main(int argc, char *argv[])
{
    int run_id, nyears;
    SunWindow * sunwindow;

    if (argc > 5 || argc < 3)
    {
        std::cerr << "Usage: sim -sim <data directory> <run id> <number of years>" << std::endl;
        std::cerr << "or sim -view <data directory>" << std::endl;
        std::cerr << "or sim -sun <data directory> [--ignore-canopy]" << std::endl;
        return 1;
    }

    try
    {
        QApplication app(argc, argv);

        std::string usagestr = argv[1];

        std::string datadir = argv[2];
        while (datadir.back() == '/')
            datadir.pop_back();

        Window * window = new Window(datadir);

        QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedKingdom));

        window->resize(window->sizeHint());
        window->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
        window->getView().setForcedFocus(window->getTerrain().getFocus());

        if(usagestr == "-sim" || usagestr == "-sun")
        {
            if(argc == 5)
            {
                run_id = std::stoi(argv[3]);
                nyears = std::stoi(argv[4]);
            }

            // sunwindow
            sunwindow = new SunWindow();
            sunwindow->setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);
            sunwindow->resize(window->sizeHint());
            sunwindow->setOrthoView(window->getGLWidget());
            sunwindow->getView().setForcedFocus(window->getTerrain().getFocus());
        }

        int desktopArea = QApplication::desktop()->width() *
            QApplication::desktop()->height();
        int widgetArea = window->width() * window->height();
        if (((float)widgetArea / (float)desktopArea) < 0.75f)
            window->show();
        else
            window->showMaximized();

        if(usagestr == "-sim") // run simulation
        {
            window->setSunWindow(sunwindow);
            window->run_undersim_foolproof(run_id, nyears);
            window->close();
            QCoreApplication::exit();
        }
        else
        {
            if(usagestr == "-sun") // run abiotic component of simulation. This actually includes moisture if not already computed.
            {
                bool include_canopy = true;
                if (argc >= 4)	// if there is another argument, we check if it's the ignore-canopy switch and quit with informative message if switch is wrong
                {
                    std::string canopyignore_string = argv[3];
                    if (canopyignore_string == "--ignore-canopy")
                    {
                        include_canopy = false;
                    }
                    else
                    {
                        std::cout << "Unknown switch " << canopyignore_string << " for option -sun. Known switches are: --ignore-canopy" << std::endl;
                        QCoreApplication::exit(1);
                    }
                }
                window->setSunWindow(sunwindow);
                window->run_abiotics_only(include_canopy);
                window->close();
                QCoreApplication::exit();
            }
            else
                if(usagestr == "-view") // viewing only
                {
                    window->run_undersim_viewer();
                    int status = app.exec();
                    return status;
                }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
