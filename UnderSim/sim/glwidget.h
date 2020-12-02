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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include "glheaders.h" // Must be included before QT opengl headers
#include <QGLWidget>
#include <QLabel>
#include <QTimer>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPushButton>
#include <list>
#include <common/debug_vector.h>
#include <common/debug_list.h>

#include "view.h"
#include "sim.h"
#include "typemap.h"
// #include "palette.h"
#include "stroke.h"
#include "typemap.h"
#include "shape.h"
/*
#include <QWidget>
#include <QImage>
#include <common/map.h>
*/


//! [0]

const float manipradius = 75.0f;
const float manipheight = 750.0f;
const float armradius = manipradius / 2.5f;
const float tolzero = 0.01f;

const float seaval = 2000.0f;
const float initmint = 0.0f;
const float initmaxt = 40.0f;
const float mtoft = 3.28084f;

enum class ControlMode
{
    VIEW,   // free viewing of scene
    PAINTLEARN,  // painting for training
    PAINTECO, // painting ecosystems
    CMEND
};

class Scene
{
public:

    View * view;
    Terrain * terrain;
    TypeMap * maps[(int) TypeMapType::TMTEND];
    MapFloat * moisture, * illumination, * temperature, * chm, * cdm; //< condition maps
    EcoSystem * eco;
    Biome * biome;
    Simulation * sim;
    TypeMapType overlay; //< currently active overlay texture: CATEGORY, WATER, SUNLIGHT, TEMPERATURE, etc

    Scene();

    ~Scene();
};

class Window;

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    GLWidget(const QGLFormat& format, std::string datadir, QWidget *parent = 0);
    ~GLWidget();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    /**
     * capture the framebuffer as an image
     * @param capImg    framebuffer is written to this image
     * @param capSize   the image is scaled by linear interpolation to this size
     */
    void screenCapture(QImage * capImg, QSize capSize);

    /// getters for currently active view, terrain, typemaps, renderer, ecosystem
    View * getView();
    Terrain * getTerrain();
    TypeMap * getTypeMap(TypeMapType purpose);
    PMrender::TRenderer * getRenderer();
    EcoSystem * getEcoSys();
    Simulation * getSim();
    MapFloat * getSunlight(int month);
    MapFloat * getSlope();
    MapFloat * getMoisture(int month);
    MapFloat * getTemperature();
    MapFloat * getCanopyHeightModel();
    MapFloat * getCanopyDensityModel();
    Biome * getBiome();
    GLSun * getGLSun();

    // initialize simulator for current scene
    void initSceneSim();

    /// getter and setter for brush radii
    float getRadius();
    void setRadius(float rad);

    /// getter, setter, refresher for overlay texture being displayed
    void refreshOverlay();
    void setOverlay(TypeMapType purpose);
    TypeMapType getOverlay();
    void setMap(TypeMapType type, int mth);

    /**
     * @brief setIncludeCanopy Toggle switch on exclusion or exclusion of canopy in sunlight calculations
     * @param canopyon true if canopy is to be included, false otherwise.
     */
    void setIncludeCanopy(bool canopyon){ inclcanopy = canopyon; }

    /**
     * @brief bandCanopyHeightTexture   Recolour the canopy height texture according to a band of min and max tree heights
     * @param mint  Minimum tree height (below which heights are coloured black)
     * @param maxt  Maximum tree height (above which heights are coloured red)
     */
    void bandCanopyHeightTexture(float mint, float maxt);

    /**
     * Load scene attributes that are located in the directory specified
     * @param dirprefix     directory path and file name prefix combined for loading a scene
     */
    void loadScene(int curr_canopy);
    void loadScene(std::string dirprefix, int curr_canopy);
    void loadFinScene(int curr_canopy);
    void loadFinScene(std::string dirprefix, int curr_canopy);

     /**
      * Save scene attributes to the directory specified
      * @param dirprefix     directory path and file name prefix combined for saving a scene, directory is assumed to exist
      */
     void saveScene(std::string dirprefix);

    /**
     * @brief writePaintMap Output image file encoding the paint texture layer. Paint codes are converted to greyscale values
     * @param paintfile image file name
     */
    void writePaintMap(std::string paintfile);

    /**
     * @brief writeGrass Output terragen image files related to the grass layer
     * @param grassrootfile  name of root image file, all images use this as the prefix
     */
    void writeGrass(std::string grassrootfile);

    /// Add an extra scene with placeholder view, terrain and typemap onto the end of the scene list
    void addScene();

    /// change the scene being displayed
    void setScene(int s);

    /// Prepare decal texture
    void loadDecals();

    /// Load from file to appropriate TypeMap depending on purpose
    int loadTypeMap(MapFloat * map, TypeMapType purpose);

    /// Respond to key press events
    void keyPressEvent(QKeyEvent *event);

    /// set scaling value for all terrains
    void setScales(float sc);

    /// Make all plants visible
    void setAllPlantsVis();

    /// Toggle canopy plant visibility
    void setCanopyVis(bool vis);

    /// Toggle undergrowth plant visibility
    void setUndergrowthVis(bool vis);

    /// Turn all species either visible or invisible
    void setAllSpecies(bool vis);

    /// Turn on visibility for a single plant species only (all others off)
    void setSinglePlantVis(int p);

    /// Toggle visibility of an individual species on or off
    void toggleSpecies(int p, bool vis);

    void sim_canopy_sunlight(std::string sunfile);
    void loadAndSimSunlightMoistureOnly(string dirprefix);
    void loadAndSimSunlightMoistureOnly();
signals:
    void signalRepaintAllGL();
    
public slots:
    void animUpdate(); // animation step for change of focus
    void rotateUpdate(); // animation step for rotating around terrain center

    void run_undersim_only(int curr_run, int nyears);
    std::string get_dirprefix();
    void loadSceneWithoutSims(std::string dirprefix, int curr_canopy, string chmfile, std::string output_sunfile);
    void loadSceneWithoutSims(int curr_canopy, string chmfile, string output_sunfile);
protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);

    void mousePressEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent * wheel);

private:

     QGLFormat glformat; //< format for OpenGL
    // scene control
    uts::vector<Scene *> scenes;
    GLSun * glsun;
    bool simvalid; //< whether or not a simulation can be run
    bool firstsim; //< is this the first simulation run
    bool inclcanopy; //< is the canopy included in the sunlight simulation

    std::string datadir;

    int currscene;
    bool dbloaded, ecoloaded; // set to true once the user has opened a database and ecosystem

    // render variables
    PMrender::TRenderer * renderer;
    bool decalsbound;
    GLuint decalTexture;

    // gui variables
    bool viewing;
    bool viewlock;
    bool focuschange;
    bool focusviz;
    bool timeron;
    bool active; //< scene only rendered if this is true
    std::vector<bool> plantvis;
    bool canopyvis; //< display the canopy plants if true
    bool undervis; //< display the understorey plants if true
    float scf;
    ControlMode cmode;
    int sun_mth; // which month to display in the sunlight texture
    int wet_mth; // which month to display in the moisture texture

    QPoint lastPos;
    QColor qtWhite;
    QTimer * atimer, * rtimer; // timers to control different types of animation
    QLabel * vizpopup;  //< for debug visualisation

    /**
     * @brief pickInfo  write information about a terrain cell to the console
     * @param x         x-coord on terrain grid
     * @param y         y-coord on terrain grid
     */
    void pickInfo(int x, int y);
};

#endif
