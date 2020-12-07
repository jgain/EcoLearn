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

/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020 J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp  (konrad.p.kapp@gmail.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/

#ifndef WINDOW_H
#define WINDOW_H

#include "glwidget.h"
#include "specpalette_window.h"
#include "speciesColoursWindow.h"
#include <QWidget>
#include <QtWidgets>
#include <string>

class specselect_window;

class QAction;
class QMenu;
class QLineEdit;
class UndergrowthSpacingWindow;

struct configparams;

class SunWindow : public QMainWindow
{
    Q_OBJECT

public:
    SunWindow(){}

    ~SunWindow(){}

    QSize sizeHint() const;

    /// Various getters for rendering and scene context
    View &getView() { return *orthoView->getView(); }
    Terrain &getTerrain() { return *orthoView->getTerrain(); }
    void setOrthoView(GLWidget * ortho);

public slots:
    void repaintAllGL();

private:
    GLWidget * orthoView;
};

class Window : public QMainWindow
{
    Q_OBJECT


private:

public:
    Window(int scale_size);

    ~Window(){ std::cerr << "Entering window dtor" << std::endl; delete mainWidget; delete perspectiveView; }

    QSize sizeHint() const;

    /// Various getters for rendering and scene context
    View &getView() { return *perspectiveView->getView(); }
    Terrain &getTerrain() { return *perspectiveView->getTerrain(); }
    GLWidget * getGLWidget(){ return perspectiveView; }
    void setSunWindow(SunWindow * sun){ sunwindow = sun; }
    BrushPalette * getPalette() { return perspectiveView->getPalette(); }

    /// Adjust rendering parameters, grid and contours, to accommodate current scale
    void scaleRenderParams(float scale);

    specpalette_window *get_species_palette_window() { return species_palette_window; }



public slots:
    void repaintAllGL();


    void cleanup();
    // menu items
    /*
    void openEco();
    void openState();
    void open();
    void saveState();
    void saveGrass();
    void exportPlants();
    void exportFile();
    void exportAs();
    */
    void openScene();
    void openTerrain();
    void saveScene();
    void saveAsScene();
    void saveAsPaint();
    void showRenderOptions();
    void showContours(int show);
    void showGridLines(int show);

    // render panel
    void lineEditChange();

    // paint panel
    void treeEditChange();

    void saveAsCHM();
    QSlider *addSpeciesSlider(QVBoxLayout *layout, const QString &label, float startValue, int scale, float low, float high);

    void speciesSliderCallback();

    void species_added(int id);
    void species_removed(int id);
    void speciesSliderCallbackChangeAll();
    void changeSpeciesSliders(std::vector<QSlider *> other_sliders, QSlider *sender_slider);
    void changeAllSpeciesSliders(QSlider *sender_slider);
    void addSpecRadSlider(QVBoxLayout *layout, const QString &label, float startValue, int scale, float low, float high);
    void showSpeciesCompareDialog();
    void showImportCanopy();
    void showImportUndergrowth();
    void showSpeciesColours();
    void showClusterCounts();
    void compareUnderUnderDialog();
    void toSynthesisMode();
    void importDrawing();
    void showImportClusterFiles();
    void toFirstPersonMode();
    void toOverviewMode();
    void showImportCanopyshading();
    void toCanopyTreeAddMode();
    void toCanopyTreeRemoveMode();

    void set_canopy_label();
    void set_species_optim_label();
    void set_undersample_label();
    void set_underrefine_label();
    void loadConfig();
    void loadConfig(std::string configfilename);
    void openScene(std::string dirName, bool import_cluster_dialog);
protected:
    void keyPressEvent(QKeyEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void optionsChanged();

    void addPercSlider(QVBoxLayout *layout, const QString &label, int startval);
protected slots:
    void convertPainting();
    void hide_all_ctrlwindows();
private:
    GLWidget * perspectiveView; ///< Main OpenGL window
    QWidget * renderPanel;      ///< Side panel to adjust various rendering style parameters
    QVBoxLayout * renderLayout;
    SunWindow * sunwindow;      ///< Link to renderer for sunshine
    specselect_window *specwindow;

    // high level windows, widgets and layouts
    QWidget *mainWidget;
    QGridLayout *mainLayout;
    QVBoxLayout *palLayout;
    QVBoxLayout *specpalLayout;

    QPushButton *procdrawingButton;

    // rendering parameters
    float gridSepX, numGridX, gridSepZ, numGridZ, gridWidth, gridIntensity; // grid params
    float contourSep, numContours, contourWidth, contourIntensity; // contour params
    float radianceTransition, radianceEnhance; // radiance scaling params
    float minTree, maxTree; // canopy height model texture display params
    
    // render panel widgets
    QLineEdit * gridSepXEdit, * gridSepZEdit, * gridWidthEdit, * gridIntensityEdit, * contourSepEdit, * contourWidthEdit, * contourIntensityEdit, * radianceEnhanceEdit;


    QWidget *progress_bar_window;
    QProgressBar *canopy_placement_progress;
    QProgressBar *quick_undergrowth_progress;
    QProgressBar *undersynth_progress;
    QProgressBar *synth_progress;
    QLabel *currsynth_label;

    // palette panel widgets
    QSlider * radslider;
    QSlider * specradslider;
    QSlider * percslider;
    QLabel * radlabel;
    QLabel * perclabel;
    QLineEdit * minTreeEdit, * maxTreeEdit;



    // menu widgets and actions
    QMenu *fileMenu;
    QMenu *viewMenu;
    QMenu *importMenu;
    QMenu *actionMenu;
    QMenu *cmodeMenu;
    QMenu *viewmodeMenu;
    /*
    QAction *newAct;
    QAction *openAct;
    QAction *openEcoAct;
    QAction *openStateAct;
    QAction *openSamplingAct;
    QAction *saveStateAct;
    QAction *saveGrassAct;
    QAction *exportPlantsAct;
    QAction *exportAct;
    QAction *exportAsAct;
    */
    QAction *openTerrainAct;
    QAction *openSceneAct;
    QAction *saveSceneAct;
    QAction *saveSceneAsAct;
    QAction *savePaintAsAct;
    QAction *saveCHMAsAct;
    QAction *showRenderAct;
    QAction *compareSpeciesAct;

    QAction *importCanopyAct;
    QAction *importUndergrowthAct;
    QAction *importClusterfilesAct;
    QAction *importCanopyshadingAct;

    QAction *sampleUndergrowthAct;

    QAction *toUnderUnderCmodeAct;
    QAction *showClusterCountsAct;
    QAction *compareUnderUnderDialogAct;
    QAction *toCanopyTreeAddAct;
    QAction *toCanopyTreeRemoveAct;

    QAction *toFirstPersonModeAct;
    QAction *toOverviewModeAct;

    QAction *doUndergrowthSynthesisAct;
    QAction *doCompleteUndergrowthSynthesisAct;

    QAction *importDrawingAct;

    QAction *convertPaintingAct;

    QAction *processDrawingAct;

    QAction *viewSpeciesColoursAct;

    // file management
    std::string scenedirname;

    std::vector<QSlider *> speciesSliders;
    std::vector<QCheckBox *> speciesCheckBoxes;

    // For creating a slider widgets
    void addLearnRadSlider(QVBoxLayout *layout, const QString &label, float defaultValue, int scale, float low, float high);

    // init menu
    void createActions();
    void createMenus();

    specpalette_window *species_palette_window;

    SpeciesColoursWindow *specColoursWindow = nullptr;
    void loadConfig(configparams params);
    void importClusterFiles(std::vector<std::string> fname_list);
};

#endif
