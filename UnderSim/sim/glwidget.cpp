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

#include "glwidget.h"
#include "eco.h"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <QGridLayout>
#include <QGLFramebufferObject>
#include <QImage>
#include <QCoreApplication>
#include <QMessageBox>
#include <QInputDialog>

#include <fstream>

using namespace std;

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

////
// Scene
////

Scene::Scene()
{
    view = new View();
    terrain = new Terrain();
    terrain->initGrid(1024, 1024, 10000.0f, 10000.0f);
    view->setForcedFocus(terrain->getFocus());
    view->setViewScale(terrain->longEdgeDist());

    sim = new Simulation();
    eco = new EcoSystem();
    biome = new Biome();

    int dx, dy;
    terrain->getGridDim(dx, dy);

    for(TypeMapType t: all_typemaps)
        maps[(int) t] = new TypeMap(dx, dy, t);
    maps[2]->setRegion(terrain->coverRegion());
    moisture = new MapFloat();
    illumination = new MapFloat();
    temperature = new MapFloat();
    chm = new MapFloat();
    cdm = new MapFloat();
    overlay = TypeMapType::EMPTY;
}

Scene::~Scene()
{
    delete view;
    delete terrain;
    for(TypeMapType t: all_typemaps)
        if(maps[(int) t] != nullptr)
        {
            delete maps[(int) t];
            maps[(int) t] = nullptr;
        }
    delete sim;
    delete eco;
    delete biome;
    delete illumination;
    delete moisture;
    delete temperature;
    delete chm;
    delete cdm;
}


////
// GLWidget
////

GLWidget::GLWidget(const QGLFormat& format, string datadir, QWidget *parent)
    : QGLWidget(format, parent)
{
    this->datadir = datadir;

    qtWhite = QColor::fromCmykF(0.0, 0.0, 0.0, 0.0);
    vizpopup = new QLabel();
    atimer = new QTimer(this);
    connect(atimer, SIGNAL(timeout()), this, SLOT(animUpdate()));

    rtimer = new QTimer(this);
    connect(rtimer, SIGNAL(timeout()), this, SLOT(rotateUpdate()));
    glformat = format;

    // main design scene
    addScene();

    // database display and picking scene
    addScene();

    currscene = 0;

    renderer = new PMrender::TRenderer(NULL, "../sim/shaders/");
    cmode = ControlMode::VIEW;
    viewing = false;
    viewlock = false;
    decalsbound = false;
    focuschange = false;
    focusviz = false;
    timeron = false;
    dbloaded = false;
    ecoloaded = false;
    inclcanopy = true;
    active = true;
    scf = 10000.0f;
    decalTexture = 0;

    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);

    resize(sizeHint());
    setSizePolicy (QSizePolicy::Ignored, QSizePolicy::Ignored);

    simvalid = false;
    firstsim = false;
    glsun = new GLSun(glformat);
}

GLWidget::~GLWidget()
{
    delete atimer;
    delete rtimer;
    if(vizpopup) delete vizpopup;

    if (renderer) delete renderer;
    if(glsun) delete glsun;

    // delete views
    for(int i = 0; i < (int) scenes.size(); i++)
        delete scenes[i];

    if (decalTexture != 0)	glDeleteTextures(1, &decalTexture);
}

QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLWidget::sizeHint() const
{
    return QSize(1000, 800);
}


void GLWidget::screenCapture(QImage * capImg, QSize capSize)
{
    paintGL();
    glFlush();

    (* capImg) = grabFrameBuffer();
    (* capImg) = capImg->scaled(capSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

View * GLWidget::getView()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->view;
    else
        return NULL;
}

Terrain * GLWidget::getTerrain()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->terrain;
    else
        return NULL;
}

TypeMap * GLWidget::getTypeMap(TypeMapType purpose)
{
    if((int) scenes.size() > 0)
    {
        int idx = (int)purpose;
        return scenes[currscene]->maps[(int) purpose];
    }
    else
        return NULL;
}

MapFloat * GLWidget::getSunlight(int month)
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->sim->getSunlightMap(month);
    else
        return NULL;
}

MapFloat * GLWidget::getSlope()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->sim->getSlopeMap();
    else
        return NULL;
}

MapFloat * GLWidget::getMoisture(int month)
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->sim->getMoistureMap(month);
    else
        return NULL;
}

MapFloat * GLWidget::getCanopyHeightModel()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->chm;
    else
        return NULL;
}

MapFloat * GLWidget::getCanopyDensityModel()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->cdm;
    else
        return NULL;
}

PMrender::TRenderer * GLWidget::getRenderer()
{
    return renderer;
}

EcoSystem * GLWidget::getEcoSys()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->eco;
    else
        return NULL;
}

Simulation * GLWidget::getSim()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->sim;
    else
        return NULL;
}

void GLWidget::initSceneSim()
{
    // assumes terrain and biome already loaded
    if((int) scenes.size() > 0)
    {
        scenes[currscene]->sim = new Simulation(getTerrain(), getBiome(), 5);
    }
}

Biome * GLWidget::getBiome()
{
    if((int) scenes.size() > 0)
        return scenes[currscene]->biome;
    else
        return NULL;
}

GLSun * GLWidget::getGLSun()
{
    return glsun;
}

void GLWidget::refreshOverlay()
{
    renderer->updateTypeMapTexture(getTypeMap(scenes[currscene]->overlay), PMrender::TRenderer::typeMapInfo::PAINT, false);
    update();
}

void GLWidget::setOverlay(TypeMapType purpose)
{
    scenes[currscene]->overlay = purpose;
    renderer->updateTypeMapTexture(getTypeMap(scenes[currscene]->overlay), PMrender::TRenderer::typeMapInfo::PAINT, true);
    update();
}

TypeMapType GLWidget::getOverlay()
{
    return scenes[currscene]->overlay;
}

void GLWidget::bandCanopyHeightTexture(float mint, float maxt)
{
    getTypeMap(TypeMapType::CHM)->bandCHMMap(getCanopyHeightModel(), mint*mtoft, maxt*mtoft);
    focuschange = true;
}

std::string GLWidget::get_dirprefix()
{
    std::cout << "Datadir before fixing: " << datadir << std::endl;
    while (datadir.back() == '/')
        datadir.pop_back();

    std::cout << "Datadir after fixing: " << datadir << std::endl;

    int slash_idx = datadir.find_last_of("/");
    std::string setname = datadir.substr(slash_idx + 1);
    std::string dirprefix = datadir + "/" + setname;
    return dirprefix;
}

void GLWidget::loadScene(int curr_canopy)
{
    std::cout << "Datadir before fixing: " << datadir << std::endl;
    while (datadir.back() == '/')
        datadir.pop_back();

    std::cout << "Datadir after fixing: " << datadir << std::endl;

    int slash_idx = datadir.find_last_of("/");
    std::string setname = datadir.substr(slash_idx + 1);
    std::string dirprefix = get_dirprefix();

    loadScene(dirprefix, curr_canopy);
}

void GLWidget::loadFinScene(int curr_canopy)
{
    std::cout << "Datadir before fixing: " << datadir << std::endl;
    while (datadir.back() == '/')
        datadir.pop_back();

    std::cout << "Datadir after fixing: " << datadir << std::endl;

    int slash_idx = datadir.find_last_of("/");
    std::string setname = datadir.substr(slash_idx + 1);
    std::string dirprefix = get_dirprefix();

    loadFinScene(dirprefix, curr_canopy);
}

void GLWidget::sim_canopy_sunlight(std::string sunfile)
{
    getGLSun()->setScene(getTerrain(), getCanopyHeightModel(), getCanopyDensityModel());
    getGLSun()->deriveAlpha(getEcoSys(), getBiome());
    cerr << "About to report Alpha Map Stats" << endl;
    getGLSun()->alphaMapStats();
    auto bt = std::chrono::steady_clock::now().time_since_epoch();
    getSim()->calcSunlight(getGLSun(), 15, 50, inclcanopy);
    auto et = std::chrono::steady_clock::now().time_since_epoch();
    auto t = std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count();
    std::cout << "Time ms: " << t << std::endl;
    getSim()->reportSunAverages();
    getSim()->writeSunCopy(sunfile);
}

void GLWidget::loadAndSimSunlightMoistureOnly()
{
    std::cout << "Datadir before fixing: " << datadir << std::endl;
    while (datadir.back() == '/')
        datadir.pop_back();

    std::cout << "Datadir after fixing: " << datadir << std::endl;

    int slash_idx = datadir.find_last_of("/");
    std::string setname = datadir.substr(slash_idx + 1);
    std::string dirprefix = get_dirprefix();
    loadAndSimSunlightMoistureOnly(dirprefix);
}

void GLWidget::loadAndSimSunlightMoistureOnly(std::string dirprefix)
{
    simvalid = true;
    std::string terfile = dirprefix+".elv";
    //std::string pdbfile = dirprefix+".pdb";
    std::string pdbfile = dirprefix + "_canopy";
    pdbfile += std::to_string(0) + ".pdb";
    std::string chmfile = dirprefix+".chm";
    std::string cdmfile = dirprefix+".cdm";
    std::string sunfile = dirprefix+"_sun.txt";
    std::string wetfile = dirprefix+"_wet.txt";
    std::string climfile = dirprefix+"_clim.txt";
    std::string bmefile = dirprefix+"_biome.txt";
    std::string catfile = dirprefix+"_plt.png";

    // load terrain
    currscene = 0;
    getTerrain()->loadElv(terfile);
    cerr << "Elevation file loaded" << endl;
    scf = getTerrain()->getMaxExtent();
    getView()->setForcedFocus(getTerrain()->getFocus());
    getView()->setViewScale(getTerrain()->longEdgeDist());
    getView()->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());
    getTerrain()->calcMeanHeight();

    // match dimensions for empty overlay
    int dx, dy;
    getTerrain()->getGridDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->matchDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->fill(0);
    getTypeMap(TypeMapType::EMPTY)->matchDim(dx, dy);
    getTypeMap(TypeMapType::EMPTY)->clear();

    if(getCanopyHeightModel()->read(chmfile))
    {
        cerr << "CHM file loaded" << endl;
        loadTypeMap(getCanopyHeightModel(), TypeMapType::CHM);
    }
    else
    {
        cerr << "No Canopy Height Model found. Simulation invalidated." << endl;
        simvalid = false;
    }

    if(getCanopyDensityModel()->read(cdmfile))
    {
        loadTypeMap(getCanopyDensityModel(), TypeMapType::CDM);
        cerr << "CDM file loaded" << endl;
    }
    else
    {
        cerr << "No Canopy Density Model found. Simulation invalidated." << endl;
        simvalid = false;
    }

    if(getBiome()->read(bmefile))
    {
        cerr << "Biome file loaded" << endl;
        for(int t = 0; t < getBiome()->numPFTypes(); t++)
            plantvis.push_back(true);
        canopyvis = true;
        undervis = true;

        /*
        // report max plant size
        for(int t = 0; t < getBiome()->numPFTypes(); t++)
        {
            cerr << getBiome()->getPFType(t)->code << " max hght = " << getBiome()->maxGrowthTest(t) << endl;
        }
        */

        // initialize simulation
        initSceneSim();

        // read climate parameters
        if(!getSim()->readClimate(climfile))
        {
            simvalid = false;
            cerr << "No climate file " << climfile << " found. Simulation invalidated" << endl;
        }
        else
        {
            cerr << "Climate file loaded" << endl;
        }


        // read sunlight, and if that fails simulate it and store results
        if(!getSim()->readSun(sunfile))
        {
            cerr << "No Sunlight file " << sunfile << " found, so simulating sunlight" << endl;
            sim_canopy_sunlight(sunfile);
        }
        else
        {
            cerr << "Sunlight file loaded" << endl;

        }

        sun_mth = 0;
        loadTypeMap(getSunlight(sun_mth), TypeMapType::SUNLIGHT);
        loadTypeMap(getSlope(), TypeMapType::SLOPE);

        // read soil moisture, and if that fails simulate it and store results
        if(!getSim()->readMoisture(wetfile))
        {
            cerr << "No Soil moisture file " << wetfile << " found, so simulating soil moisture" << endl;
            getSim()->calcMoisture();
            getSim()->writeMoisture(wetfile);
        }
        else
        {
            cerr << "Soil moisture file loaded" << endl;
        }
    }
}

void GLWidget::loadSceneWithoutSims(int curr_canopy, std::string chmfile, string output_sunfile)
{
    std::cout << "Datadir before fixing: " << datadir << std::endl;
    while (datadir.back() == '/')
        datadir.pop_back();

    std::cout << "Datadir after fixing: " << datadir << std::endl;

    int slash_idx = datadir.find_last_of("/");
    std::string setname = datadir.substr(slash_idx + 1);
    std::string dirprefix = get_dirprefix();

    loadSceneWithoutSims(dirprefix, curr_canopy, chmfile, output_sunfile);
}

void GLWidget::loadSceneWithoutSims(std::string dirprefix, int curr_canopy, std::string chmfile, std::string output_sunfile)
{
    simvalid = true;
    std::string terfile = dirprefix+".elv";
    std::string pdbfile = dirprefix + "_canopy";
    pdbfile += std::to_string(curr_canopy) + ".pdb";
    std::string cdmfile = dirprefix+".cdm";
    std::string sunfile = dirprefix+"_sun.txt";
    std::string wetfile = dirprefix+"_wet.txt";
    std::string climfile = dirprefix+"_clim.txt";
    std::string bmefile = dirprefix+"_biome.txt";
    std::string catfile = dirprefix+"_plt.png";

    // load terrain
    currscene = 0;
    getTerrain()->loadElv(terfile);
    cerr << "Elevation file loaded" << endl;
    scf = getTerrain()->getMaxExtent();
    getView()->setForcedFocus(getTerrain()->getFocus());
    getView()->setViewScale(getTerrain()->longEdgeDist());
    getView()->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());
    getTerrain()->calcMeanHeight();

    // match dimensions for empty overlay
    int dx, dy;
    getTerrain()->getGridDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->matchDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->fill(0);
    getTypeMap(TypeMapType::EMPTY)->matchDim(dx, dy);
    getTypeMap(TypeMapType::EMPTY)->clear();

    if(getCanopyHeightModel()->read(chmfile))
    {
        cerr << "CHM file loaded" << endl;
        loadTypeMap(getCanopyHeightModel(), TypeMapType::CHM);
    }
    else
    {
        cerr << "No Canopy Height Model found. Simulation invalidated." << endl;
        simvalid = false;
    }

    if(getCanopyDensityModel()->read(cdmfile))
    {
        loadTypeMap(getCanopyDensityModel(), TypeMapType::CDM);
        cerr << "CDM file loaded" << endl;
    }
    else
    {
        cerr << "No Canopy Density Model found. Simulation invalidated." << endl;
        simvalid = false;
    }

    if (getBiome()->read_dataimporter(SONOMA_DB_FILEPATH))
    {
        if (plantvis.size() < getBiome()->numPFTypes())
            plantvis.resize(getBiome()->numPFTypes());
        cerr << "Biome file load" << endl;
        for(int t = 0; t < getBiome()->numPFTypes(); t++)
            plantvis[t] = true;

        // initialize simulation
        initSceneSim();

        // read climate parameters
        if(!getSim()->readClimate(climfile))
        {
            simvalid = false;
            cerr << "No climate file " << climfile << " found. Simulation invalidated" << endl;
        }
        sim_canopy_sunlight(output_sunfile);

    }
    else
    {
        std::cerr << "Biome file " << bmefile << "does not exist. Simulation invalidated." << endl;
    }

    focuschange = false;
    if(simvalid)
        firstsim = true;
}

void GLWidget::loadFinScene(std::string dirprefix, int curr_canopy)
{
    simvalid = true;
    std::string terfile = dirprefix+".elv";
    std::string cpdbfile = dirprefix + "_canopy";
    std::string updbfile = dirprefix + "_undergrowth";
    cpdbfile += std::to_string(curr_canopy) + ".pdb";
    updbfile += std::to_string(curr_canopy) + ".pdb";
    std::string chmfile = dirprefix+".chm";
    std::string cdmfile = dirprefix+".cdm";
    std::string sunfile = dirprefix+"_sun.txt";
    std::string wetfile = dirprefix+"_wet.txt";
    std::string climfile = dirprefix+"_clim.txt";
    std::string bmefile = dirprefix+"_biome.txt";
    std::string catfile = dirprefix+"_plt.png";

    simvalid = false;
    // load terrain
    currscene = 0;
    getTerrain()->loadElv(terfile);
    cerr << "Elevation file loaded" << endl;
    scf = getTerrain()->getMaxExtent();
    getView()->setForcedFocus(getTerrain()->getFocus());
    getView()->setViewScale(getTerrain()->longEdgeDist());
    getView()->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());
    getTerrain()->calcMeanHeight();

    // match dimensions for empty overlay
    int dx, dy;
    getTerrain()->getGridDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->matchDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->fill(0);
    getTypeMap(TypeMapType::EMPTY)->matchDim(dx, dy);
    getTypeMap(TypeMapType::EMPTY)->clear();

    if(getCanopyHeightModel()->read(chmfile))
    {
        cerr << "CHM file loaded" << endl;
        loadTypeMap(getCanopyHeightModel(), TypeMapType::CHM);
    }
    else
    {
        cerr << "No Canopy Height Model found. Simulation invalidated." << endl;
    }

    if(getCanopyDensityModel()->read(cdmfile))
    {
        loadTypeMap(getCanopyDensityModel(), TypeMapType::CDM);
        cerr << "CDM file loaded" << endl;
    }
    else
    {
        cerr << "No Canopy Density Model found. One will be calculated from the imported canopy trees." << endl;
    }

    if (getBiome()->read_dataimporter(SONOMA_DB_FILEPATH))
    {
        if (plantvis.size() < getBiome()->numPFTypes())
            plantvis.resize(getBiome()->numPFTypes());
        cerr << "Biome file load" << endl;
        for(int t = 0; t < getBiome()->numPFTypes(); t++)
            plantvis[t] = true;

        canopyvis = true;
        undervis = true;

        initSceneSim();

        // read climate parameters
        if(!getSim()->readClimate(climfile))
            cerr << "No climate file " << climfile << " found." << endl;

        // read soil moisture, and if that fails simulate it and store results
        if(!getSim()->readMoisture(wetfile))
        {
            cerr << "No Soil moisture file " << wetfile << " found" << endl;
        }
        else
        {
            cerr << "Soil moisture file loaded" << endl;
            wet_mth = 0;
            loadTypeMap(getMoisture(wet_mth), TypeMapType::WATER);
        }

        // loading plant distribution
        getEcoSys()->setBiome(getBiome());
        if(!getEcoSys()->loadNichePDB(cpdbfile, getTerrain()))
             std::cerr << "Plant distribution file " << cpdbfile << "does not exist" << endl; // just report but not really an issue
        else
            std::cerr << "Plant canopy distribution file loaded" << std::endl;

        if(!getEcoSys()->loadNichePDB(updbfile, getTerrain(), 1))
             std::cerr << "Undergrowth distribution file " << updbfile << " does not exist" << endl; // just report but not really an issue
        else
            std::cerr << "Plant undergrowth distribution file loaded" << std::endl;

        setAllPlantsVis();
        focuschange = !focuschange;
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        getEcoSys()->redrawPlants();
        update();

        // read sunlight, and if that fails simulate it and store results
        if(!getSim()->readSun(sunfile))
        {
            cerr << "No Sunlight file " << sunfile << " found" << endl;
        }
        else
        {
            cerr << "Sunlight file loaded" << endl;
            sun_mth = 0;
            loadTypeMap(getSunlight(sun_mth), TypeMapType::SUNLIGHT);
        }

        loadTypeMap(getSlope(), TypeMapType::SLOPE);
    }
    else
    {
        std::cerr << "Biome file " << bmefile << "does not exist." << endl;
        focuschange = false;
    }
}

void GLWidget::loadScene(std::string dirprefix, int curr_canopy)
{
    simvalid = true;
    std::string terfile = dirprefix+".elv";
    std::string pdbfile = dirprefix + "_canopy";
    pdbfile += std::to_string(curr_canopy) + ".pdb";
    std::string chmfile = dirprefix+".chm";
    std::string cdmfile = dirprefix+".cdm";
    std::string sunfile = dirprefix+"_sun.txt";
    std::string sunlandfile = dirprefix+"_sun_landscape.txt";
    std::string wetfile = dirprefix+"_wet.txt";
    std::string climfile = dirprefix+"_clim.txt";
    std::string bmefile = dirprefix+"_biome.txt";
    std::string catfile = dirprefix+"_plt.png";

    // load terrain
    currscene = 0;
    getTerrain()->loadElv(terfile);
    cerr << "Elevation file loaded" << endl;
    scf = getTerrain()->getMaxExtent();
    getView()->setForcedFocus(getTerrain()->getFocus());
    getView()->setViewScale(getTerrain()->longEdgeDist());
    getView()->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());
    getTerrain()->calcMeanHeight();

    // match dimensions for empty overlay
    int dx, dy;
    getTerrain()->getGridDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->matchDim(dx, dy);
    getTypeMap(TypeMapType::PAINT)->fill(0);
    getTypeMap(TypeMapType::EMPTY)->matchDim(dx, dy);
    getTypeMap(TypeMapType::EMPTY)->clear();

    if(getCanopyHeightModel()->read(chmfile))
    {
        cerr << "CHM file loaded" << endl;
        loadTypeMap(getCanopyHeightModel(), TypeMapType::CHM);
    }
    else
    {
        cerr << "No Canopy Height Model found. Simulation invalidated." << endl;
        simvalid = false;
    }

    bool sim_densitymodel = false;

    if(getCanopyDensityModel()->read(cdmfile))
    {
        loadTypeMap(getCanopyDensityModel(), TypeMapType::CDM);
        cerr << "CDM file loaded" << endl;
    }
    else
    {
        cerr << "No Canopy Density Model found. One will be calculated from the imported canopy trees." << endl;
        sim_densitymodel = true;
    }

    if (getBiome()->read_dataimporter(SONOMA_DB_FILEPATH))
    {
        if (plantvis.size() < getBiome()->numPFTypes())
            plantvis.resize(getBiome()->numPFTypes());
        cerr << "Biome file load" << endl;
        for(int t = 0; t < getBiome()->numPFTypes(); t++)
            plantvis[t] = true;

        canopyvis = true;
        undervis = true;

        // initialize simulation
        initSceneSim();

        // read climate parameters
        if(!getSim()->readClimate(climfile))
        {
            simvalid = false;
            cerr << "No climate file " << climfile << " found. Simulation invalidated" << endl;
        }

        // read soil moisture, and if that fails simulate it and store results
        if(!getSim()->readMoisture(wetfile))
        {
            cerr << "No Soil moisture file " << wetfile << " found, so simulating soil moisture" << endl;
            getSim()->calcMoisture();
            getSim()->writeMoisture(wetfile);
        }
        else
        {
            cerr << "Soil moisture file loaded" << endl;
        }
        wet_mth = 0;
        loadTypeMap(getMoisture(wet_mth), TypeMapType::WATER);

        // loading plant distribution
        getEcoSys()->setBiome(getBiome());
        if(!getEcoSys()->loadNichePDB(pdbfile, getTerrain()))
        {
             std::cerr << "Plant distribution file " << pdbfile << "does not exist" << endl; // just report but not really an issue
        }
        else
        {
            getEcoSys()->pickAllPlants(getTerrain());
            std::cerr << "Plant canopy distribution file loaded" << std::endl;
        }

        getEcoSys()->redrawPlants();

        if (sim_densitymodel)
        {
            std::cerr << "Computing canopy density model from imported canopy..." << std::endl;
            getSim()->calcCanopyDensity(getEcoSys(), getCanopyDensityModel(), cdmfile);
            std::cerr << "Done computing canopy density model" << std::endl;
        }

        std::string simsun_file = inclcanopy ? sunfile : sunlandfile;		// if we do not include canopy, we only simulate sunlight based on landscape shadowing
                                                                            // This is only applicable when running viewer with the -sun option, for sunlight sim only

        std::cerr << "Looking for sunfile " << simsun_file << " (include canopy = " << inclcanopy << ")" << std::endl;
        // read sunlight, and if that fails simulate it and store results
        if(!getSim()->readSun(simsun_file))
        {
            cerr << "No Sunlight file " << simsun_file << " found, so simulating sunlight" << endl;
            sim_canopy_sunlight(simsun_file);
        }
        else
        {
            cerr << "Sunlight file loaded" << endl;
        }

        sun_mth = 0;
        loadTypeMap(getSunlight(sun_mth), TypeMapType::SUNLIGHT);
        loadTypeMap(getSlope(), TypeMapType::SLOPE);
    }
    else
    {
        std::cerr << "Biome file " << bmefile << "does not exist. Simulation invalidated." << endl;
    }

    focuschange = false;
    if(simvalid)
        firstsim = true;
}

void GLWidget::saveScene(std::string dirprefix)
{
    std::string terfile = dirprefix+".elv";
    std::string pdbfile = dirprefix+".pdb";

    // load terrain
    getTerrain()->saveElv(terfile);

    if(!getEcoSys()->saveNichePDB(pdbfile))
        cerr << "Error GLWidget::saveScene: saving plane file " << pdbfile << " failed" << endl;
}

void GLWidget::writePaintMap(std::string paintfile)
{
    getTypeMap(TypeMapType::PAINT)->saveToPaintImage(paintfile);
}

void GLWidget::addScene()
{
    Scene * scene = new Scene();
    scene->view->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());

    plantvis.clear();
    scenes.push_back(scene);
    currscene = (int) scenes.size() - 1;
}

void GLWidget::setScene(int s)
{
    if(s >= 0 && s < (int) scenes.size())
    {
        currscene = s;
        getTerrain()->setBufferToDirty();
        refreshOverlay();
        update();
    }
}


void GLWidget::loadDecals()
{
    QImage decalImg, t;

    // load image
    if(!decalImg.load(QCoreApplication::applicationDirPath() + "/../../sim/Icons/manipDecals.png"))
        cerr << QCoreApplication::applicationDirPath().toUtf8().constData() << "/../../sim/Icons/manipDecals.png" << " not found" << endl;

    // Qt prep image for OpenGL
    QImage fixedImage(decalImg.width(), decalImg.height(), QImage::Format_ARGB32);
    QPainter painter(&fixedImage);
    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(fixedImage.rect(), Qt::transparent);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    painter.drawImage( 0, 0, decalImg);
    painter.end();

    t = QGLWidget::convertToGLFormat( fixedImage );

    renderer->bindDecals(t.width(), t.height(), t.bits());
    decalsbound = true;
}

int GLWidget::loadTypeMap(MapFloat * map, TypeMapType purpose)
{
    int numClusters = 0;

    switch(purpose)
    {
        case TypeMapType::EMPTY:
            break;
        case TypeMapType::PAINT:
            break;
        case TypeMapType::CATEGORY:
            break;
        case TypeMapType::SLOPE:
            numClusters = getTypeMap(purpose)->convert(map, purpose, 90.0f);
            break;
        case TypeMapType::WATER:
            numClusters = getTypeMap(purpose)->convert(map, purpose, 100.0); // 1000.0f);
            break;
        case TypeMapType::SUNLIGHT:
             numClusters = getTypeMap(purpose)->convert(map, purpose, 13.0f);
             break;
        case TypeMapType::TEMPERATURE:
            numClusters = getTypeMap(purpose)->convert(map, purpose, 20.0f);
            break;
        case TypeMapType::CHM:
            numClusters = getTypeMap(purpose)->convert(map, purpose, mtoft*initmaxt);
            break;
        case TypeMapType::CDM:
            numClusters = getTypeMap(purpose)->convert(map, purpose, 1.0f);
            break;
        default:
            break;
    }
    return numClusters;
}

void GLWidget::setMap(TypeMapType type, int mth)
{
    if(type == TypeMapType::SUNLIGHT)
        loadTypeMap(getSunlight(mth), type);
    if(type == TypeMapType::WATER)
        loadTypeMap(getMoisture(mth), type);
    setOverlay(type);
}

void GLWidget::initializeGL()
{
    // get context opengl-version
    qDebug() << "Widget OpenGl: " << format().majorVersion() << "." << format().minorVersion();
    qDebug() << "Context valid: " << context()->isValid();
    qDebug() << "Really used OpenGl: " << context()->format().majorVersion() << "." <<
              context()->format().minorVersion();
    qDebug() << "OpenGl information: VENDOR:       " << (const char*)glGetString(GL_VENDOR);
    qDebug() << "                    RENDERDER:    " << (const char*)glGetString(GL_RENDERER);
    qDebug() << "                    VERSION:      " << (const char*)glGetString(GL_VERSION);
    qDebug() << "                    GLSL VERSION: " << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);

    QGLFormat glFormat = QGLWidget::format();
    if ( !glFormat.sampleBuffers() )
        qWarning() << "Could not enable sample buffers";

    qglClearColor(qtWhite.light());

    int mu;
    glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &mu);
    cerr << "max texture units = " << mu << endl;

    // *** PM Render code - start ***

    // To use basic shading: PMrender::TRenderer::BASIC
    // To use radianvce scaling: PMrender::TRenderer::RADIANCE_SCALING

    PMrender::TRenderer::terrainShadingModel sMod = PMrender::TRenderer::RADIANCE_SCALING;

    // set terrain shading model
    renderer->setTerrShadeModel(sMod);

    // set up light
    Vector dl = Vector(0.6f, 1.0f, 0.6f);
    dl.normalize();

    GLfloat pointLight[3] = { 0.5, 5.0, 7.0}; // side panel + BASIC lighting
    GLfloat dirLight0[3] = { dl.i, dl.j, dl.k}; // for radiance lighting
    GLfloat dirLight1[3] = { -dl.i, dl.j, -dl.k}; // for radiance lighting

    renderer->setPointLight(pointLight[0],pointLight[1],pointLight[2]);
    renderer->setDirectionalLight(0, dirLight0[0], dirLight0[1], dirLight0[2]);
    renderer->setDirectionalLight(1, dirLight1[0], dirLight1[1], dirLight1[2]);

    // initialise renderer/compile shaders
    renderer->initShaders();

    // set other render parameters
    // can set terrain colour for radiance scaling etc - check trenderer.h

    // terrain contours
    renderer->drawContours(false);
    renderer->drawGridlines(false);

    // turn on terrain type overlay (off by default); NB you can stil call methods to update terrain type,
    renderer->useTerrainTypeTexture(true);
    renderer->useConstraintTypeTexture(false);

    // use manipulator textures (decal'd)
    renderer->textureManipulators(true);

    // *** PM Render code - end ***

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_CLAMP);
    glEnable(GL_TEXTURE_2D);

    loadDecals();
    paintGL();
}

void GLWidget::paintGL()
{
    vpPoint mo;
    glm::mat4 tfm, idt;
    glm::vec3 trs, rot;
    uts::vector<ShapeDrawData> drawParams; // to be passed to terrain renderer
    Shape shape;  // geometry for focus indicator
    std::vector<Shape>::iterator sit;
    std::vector<glm::mat4> sinst;
    std::vector<glm::vec4> cinst;

    Timer t;

    if(active)
    {
        t.start();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // note: bindinstances will not work on the first render pass because setup time is needed

        if(focuschange && focusviz)
        {
            ShapeDrawData sdd;
            float scale;

            GLfloat manipCol[] = {0.325f, 0.235f, 1.0f, 1.0f};

            // create shape
            shape.clear();
            shape.setColour(manipCol);

            // place vertical cylinder at view focus
            mo = getView()->getFocus();
            scale = getView()->getScaleFactor();
            idt = glm::mat4(1.0f);
            trs = glm::vec3(mo.x, mo.y, mo.z);
            rot = glm::vec3(1.0f, 0.0f, 0.0f);
            tfm = glm::translate(idt, trs);
            tfm = glm::rotate(tfm, glm::radians(-90.0f), rot);
            shape.genCappedCylinder(scale*armradius, 1.5f*scale*armradius, scale*(manipheight-manipradius), 40, 10, tfm, false);
            if(shape.bindInstances(getView(), &sinst, &cinst)) // passing in an empty instance will lead to one being created at the origin
            {
                sdd = shape.getDrawParameters();
                sdd.current = false;
                drawParams.push_back(sdd);
            }
        }

        // prepare plants for rendering
        if(focuschange)
            getEcoSys()->bindPlantsSimplified(getTerrain(), drawParams, &plantvis);

        // pass in draw params for objects
        renderer->setConstraintDrawParams(drawParams);

        // draw terrain and plants
        getTerrain()->updateBuffers(renderer); 

        if(focuschange)
            renderer->updateTypeMapTexture(getTypeMap(getOverlay())); // only necessary if the texture is changing dynamically
        renderer->draw(getView());

        t.stop();

        if(timeron)
            cerr << "rendering = " << t.peek() << " fps = " << 1.0f / t.peek() << endl;
    }
}

void GLWidget::resizeGL(int width, int height)
{
    // TO DO: fix resizing
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, width, height);

    // apply to all views
    for(int i = 0; i < (int) scenes.size(); i++)
    {
        scenes[i]->view->setDim(0.0f, 0.0f, (float) this->width(), (float) this->height());
        scenes[i]->view->apply();
    }
}


void GLWidget::keyPressEvent(QKeyEvent *event)
{
    if(event->key() == Qt::Key_A) // 'A' for animated spin around center point of terrain
    {
        getView()->startSpin();
        rtimer->start(20);
    }
    if(event->key() == Qt::Key_C) // 'C' to show canopy height model texture overlay
    {
        setOverlay(TypeMapType::CHM);
    }
    if(event->key() == Qt::Key_E) // 'E' to remove all texture overlays
    {
        setOverlay(TypeMapType::EMPTY);
    }
    if(event->key() == Qt::Key_F) // 'F' to toggle focus stick visibility
    {
        if(focusviz)
            focusviz = false;
        else
            focusviz = true;
        update();
    }
    if(event->key() == Qt::Key_N) // 'N' to toggle display of canopy trees on or off
    {
        cerr << "canopy visibility toggled" << endl;
        setAllPlantsVis();
        canopyvis = !canopyvis; // toggle canopy visibility
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        update();
    }
    if(event->key() == Qt::Key_P) // 'P' to toggle plant visibility
    {
        cerr << "plant visibility toggled" << endl;
        setAllPlantsVis();
        focuschange = !focuschange;
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        update();
    }
    if(event->key() == Qt::Key_R) // 'R' to show temperature texture overlay
    {
        setOverlay(TypeMapType::TEMPERATURE);
    }
    if(event->key() == Qt::Key_S) // 'S' to show sunlight texture overlay
    {
        sun_mth++;
        if(sun_mth >= 12)
            sun_mth = 0;
        loadTypeMap(getSunlight(sun_mth), TypeMapType::SUNLIGHT);
        setOverlay(TypeMapType::SUNLIGHT);
    }
    if(event->key() == Qt::Key_T) // 'T' to show slope texture overlay
    {
        loadTypeMap(getSlope(), TypeMapType::SLOPE);
        setOverlay(TypeMapType::SLOPE);
    }
    if(event->key() == Qt::Key_U) // 'U' toggle undergrowth display on/off
    {
        setAllPlantsVis();
        undervis = !undervis; // toggle canopy visibility
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        update();
    }
    if(event->key() == Qt::Key_V) // 'V' for top-down view
    {
        getTerrain()->setMidFocus();
        getView()->setForcedFocus(getTerrain()->getFocus());
        getView()->topdown();
        update();
    }
    if(event->key() == Qt::Key_W) // 'W' to show water texture overlay
    {
        wet_mth++;

        if(wet_mth >= 12)
            wet_mth = 0;
        loadTypeMap(getMoisture(wet_mth), TypeMapType::WATER);
        setOverlay(TypeMapType::WATER);
    }
    // '1'-'9' make it so that only plants of that functional type are visible
    if(event->key() == Qt::Key_0)
    {
        cerr << "KEY 0" << endl;
        int p = 0;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() >= Qt::Key_1 && event->key() <= Qt::Key_9)
    {
        int p = (int) event->key() - (int) Qt::Key_1 + 1;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_ParenRight)
    {
         cerr << "KEY )" << endl;
        int p = 10;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_Exclam)
    {
         cerr << "KEY !" << endl;
        int p = 11;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_At)
    {
         cerr << "KEY @" << endl;
        int p = 12;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_NumberSign)
    {
         cerr << "KEY #" << endl;
        int p = 13;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_Dollar)
    {
         cerr << "KEY $" << endl;
        int p = 14;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_Percent)
    {
         cerr << "KEY %" << endl;
        int p = 15;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
    if(event->key() == Qt::Key_Ampersand)
    {
         cerr << "KEY &" << endl;
        int p = 16;
        setSinglePlantVis(p);
        cerr << "single species visibility " << p << endl;
    }
}

void GLWidget::setAllPlantsVis()
{
    for(int i = 0; i < (int) plantvis.size(); i++)
        plantvis[i] = true;
}

void GLWidget::setCanopyVis(bool vis)
{
    setAllPlantsVis();
    canopyvis = vis; // toggle canopy visibility
    getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
    update();
}

void GLWidget::setUndergrowthVis(bool vis)
{
    setAllPlantsVis();
    undervis = vis;
    getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
    update();
}

void GLWidget::setAllSpecies(bool vis)
{
    for(int i = 0; i < (int) plantvis.size(); i++)
        plantvis[i] = vis;
    getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
    update();
}

void GLWidget::setSinglePlantVis(int p)
{
    if(p < (int) plantvis.size())
    {
        for(int i = 0; i < (int) plantvis.size(); i++)
            plantvis[i] = false;
        plantvis[p] = true;
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        update();
    }
    else
    {
        cerr << "non-valid pft and so unable to toggle visibility" << endl;
    }
}

void GLWidget::toggleSpecies(int p, bool vis)
{
    if(p < (int) plantvis.size())
    {
        plantvis[p] = vis;
        getEcoSys()->pickAllPlants(getTerrain(), canopyvis, undervis);
        update();
    }
    else
    {
        cerr << "non-valid pft and so unable to toggle visibility" << endl;
    }
}

void GLWidget::run_undersim_only(int curr_run, int nyears)
{
    while (datadir.back() == '/')
        datadir.pop_back();
    std::string out_filename = get_dirprefix() + "_undergrowth";
    out_filename += std::to_string(curr_run); // + ".pdb";
    std::string out_filename_over = get_dirprefix() + "_overgrowth";
    out_filename_over += std::to_string(curr_run) + ".pdb";
    std::string seedbank_file = get_dirprefix() + "_seedbank" + std::to_string(curr_run) + ".sdb";
    std::string seedchance_file = get_dirprefix() + "_seedchance" + std::to_string(curr_run) + ".txt";
    out_filename += ".pdb";
    if (simvalid)
    {
        if(firstsim)
        {
            std::cout << "Importing canopy..." << std::endl;
            getSim()->importCanopy(getEcoSys(), seedbank_file, seedchance_file); // transfer plants to simulation
            std::cout << "Done importing canopy" << std::endl;
            firstsim = false;
        }
        // strictly only needs to be done on the first simulation run
        if (nyears > 0)
        {
            getSim()->simulate(getEcoSys(), seedbank_file, seedchance_file, nyears);
            getSim()->exportUnderstory(getEcoSys()); // transfer plants from simulation
            this->getEcoSys()->saveNichePDB(out_filename, 1);
        }
        else
        {
            std::cout << "No simulation done because specified number of years is not greater than zero" << std::endl;
        }
    }

}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
    float nx, ny;
    vpPoint pnt;
    
    int x = event->x(); int y = event->y();
    float W = (float) width(); float H = (float) height();

    update(); // ensure this viewport is current for unproject

    // control view orientation with right mouse button or ctrl/alt modifier key and left mouse
    if(!viewlock && (event->modifiers() == Qt::MetaModifier || event->modifiers() == Qt::AltModifier || event->buttons() == Qt::RightButton))
    {
        // arc rotate in perspective mode
  
        // convert to [0,1] X [0,1] domain
        nx = (2.0f * (float) x - W) / W;
        ny = (H - 2.0f * (float) y) / H;
        lastPos = event->pos();
        getView()->startArcRotate(nx, ny);
        viewing = true;
    }

    lastPos = event->pos();
}

void GLWidget::mouseDoubleClickEvent(QMouseEvent *event)
{
    // set the focus for arcball rotation
    // pick point on terrain or zero plane if outside the terrain bounds
    vpPoint pnt;
    int sx, sy;
    
    sx = event->x(); sy = event->y();
    if(!viewlock && ((event->modifiers() == Qt::MetaModifier && event->buttons() == Qt::LeftButton) || (event->modifiers() == Qt::AltModifier && event->buttons() == Qt::LeftButton) || event->buttons() == Qt::RightButton))
    {
        getView()->apply();
        if(getTerrain()->pick(sx, sy, getView(), pnt))
        {
            if(!decalsbound)
                loadDecals();
            vpPoint pickpnt = pnt;
            getView()->setAnimFocus(pickpnt);
            getTerrain()->setFocus(pickpnt);
            cerr << "Pick Point = " << pickpnt.x << ", " << pickpnt.y << ", " << pickpnt.z << endl;
            focuschange = true; focusviz = true;
            atimer->start(10);
        }
        // ignores pick if terrain not intersected, should possibly provide error message to user
    }
}

void GLWidget::pickInfo(int x, int y)
{
   std::string catName;

   cerr << endl;
   cerr << "*** PICK INFO ***" << endl;
   cerr << "location: " << x << ", " << y << endl;
   // getSim()->pickInfo(x, y);
   cerr << "Canopy Height (m): " << getCanopyHeightModel()->get(x, y) * 0.3048f  << endl;
   cerr << "Canopy Density: " << getCanopyDensityModel()->get(x, y) << endl;
   cerr << "Sunlight: " << getSunlight(sun_mth)->get(x,y) << endl;
   cerr << "Moisture: " << getMoisture(wet_mth)->get(x,y) << endl;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    viewing = false;

    if(event->button() == Qt::LeftButton && cmode == ControlMode::VIEW) // info on terrain cell
    {
        vpPoint pnt;
        int sx, sy;

        sx = event->x(); sy = event->y();

        if(getTerrain()->pick(sx, sy, getView(), pnt))
        {
            int x, y;
            getTerrain()->toGrid(pnt, x, y);
            pickInfo(x, y);
        }
    }
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    float nx, ny, W, H;

    int x = event->x();
    int y = event->y();

    W = (float) width();
    H = (float) height();

    // control view orientation with right mouse button or ctrl modifier key and left mouse
    if(!viewlock && ((event->modifiers() == Qt::MetaModifier && event->buttons() == Qt::LeftButton) || (event->modifiers() == Qt::AltModifier && event->buttons() == Qt::LeftButton) || event->buttons() == Qt::RightButton))
    {
        // convert to [0,1] X [0,1] domain
        nx = (2.0f * (float) x - W) / W;
        ny = (H - 2.0f * (float) y) / H;
        getView()->arcRotate(nx, ny);
        update();
        lastPos = event->pos();
    }
}

void GLWidget::wheelEvent(QWheelEvent * wheel)
{
    float del;
 
    QPoint pix = wheel->pixelDelta();
    QPoint deg = wheel->angleDelta();

    if(!viewlock)
    {
        if(!pix.isNull()) // screen resolution tracking, e.g., from magic mouse
        {
            del = (float) pix.y() * 10.0f;
            getView()->incrZoom(del);
            update();

        }
        else if(!deg.isNull()) // mouse wheel instead
        {
            del = (float) deg.y() * 2.5f;
            getView()->incrZoom(del);
            update();
        }
    }
}

void GLWidget::animUpdate()
{
    if(getView()->animate())
        update();
}

void GLWidget::rotateUpdate()
{
    if(getView()->spin())
        update();
}
