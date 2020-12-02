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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sun.h"
#include "dice_roller.h"

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#define AXIS_TILT 0.408407f
const float SunLight::_axis_tilt = AXIS_TILT;
const float SunLight::_monthly_axis_tilt = (AXIS_TILT)/3.f;
const float SunLight::_half_day_in_minutes = 720.0f;
const float SunLight::_quarter_day_in_minutes = SunLight::_half_day_in_minutes/2.0f;
const float SunLight::_3_quarters_day_in_minutes = SunLight::_half_day_in_minutes + SunLight::_quarter_day_in_minutes;
const float avgtransmit = 0.45f; // proportion of light blocked by leaves

////
// SunScene
////

SunScene::SunScene()
{
    view = new View();
    terrain = new Terrain();
    terrain->initGrid(1024, 1024, 10000.0f, 10000.0f);
    view->setForcedFocus(terrain->getFocus());
    view->setViewScale(terrain->longEdgeDist());
    view->setViewType(ViewState::ORTHOGONAL);

    float tx, ty;
    terrain->getTerrainDim(tx, ty);
    view->setOrthoViewExtent(tx,ty);
    alpha = nullptr;
}

SunScene::~SunScene()
{
    delete view;
    if(alpha)
        delete alpha;
}

////
// GLSun
////

GLSun::GLSun(const QGLFormat& format, QWidget *parent)
    : QGLWidget(format, parent)
{
    qtWhite = QColor::fromCmykF(0.0, 0.0, 0.0, 0.0);
    renderer = new PMrender::TRenderer(NULL, "../sim/shaders/");
    scene = new SunScene();
    setFocusPolicy(Qt::StrongFocus);
    resize(sizeHint());
    renderPass = 1;
    sun = new CanopyShape(0.75f, 1.0f);
}

GLSun::~GLSun()
{
    if(renderer) delete renderer;
    if(scene) delete scene;
    if(sun) delete sun;
}

QSize GLSun::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize GLSun::sizeHint() const
{
    return QSize(2000, 2000);
}

void GLSun::colToCoord(QColor col, int &gx, int &gy)
{
    int r, g, b, idx, dx, dy;

    getTerrain()->getGridDim(dx, dy);
    col.getRgb(&r, &g, &b);

    // assumes 8-bits per colour channel
    idx = (r * 65536) + (g * 256) + b;

    // then derive grid coordinates
    gx = (int) (idx / dy);
    gy = idx - gx * dy;
}

void GLSun::calcVisibility(MapFloat * sunvis, float timestep)
{
    QImage baseImg, canImg;
    int dx, dy, gx, gy;
    MapFloat mask; // check off wether a gridpoint is visible

    // north = (0,0,-1), west = (-1,0,0), east = (1,0,0), south = (0, 0, 1)
    sunvis->getDim(dx, dy);
    mask.setDim(dx, dy);
    mask.fill(0.0f);

    for(int qx = 0; qx < orthodiv; qx++)
        for(int qy = 0; qy < orthodiv; qy++)
        {
            getView()->setOrthoQuadrant(qx, qy);

            // first pass: terrain indices
            renderPass = 1;
            paintGL();
            glFlush();
            baseImg = grabFrameBuffer();

            // second pass: // delete terrain; alpha-blended canopies
            renderPass = 2;
            paintGL();
            glFlush();
            canImg = grabFrameBuffer();

            // first use exact location for incrementing
            for(int x = 0; x < baseImg.width(); x++)
                for(int y = 0; y < baseImg.height(); y++)
                {
                    QColor col = baseImg.pixelColor(x, y);
                    colToCoord(col, gx, gy);

                    if(gx < dx && gy < dy) // not the background
                    {
                        if(mask.get(gx, gy) < 0.5f) // not already incremented
                        {
                            qreal r, g, b;
                            QColor viscol = canImg.pixelColor(x, y);
                            viscol.getRgbF(&r, &g, &b); // all channels store the same info so just use red

                            sunvis->set(gx, gy, sunvis->get(gx, gy) + (float) r * timestep);
                            mask.set(gx, gy, 1.0f);

                        }
                    }
                }

            // now do a pass on the neighbours for hole filling
            for(int x = 0; x < baseImg.width(); x++)
                for(int y = 0; y < baseImg.height(); y++)
                {
                    QColor col = baseImg.pixelColor(x, y);
                    colToCoord(col, gx, gy);

                    if(gx < dx && gy < dy) // not the background
                    {
                        qreal r, g, b;
                        QColor viscol = canImg.pixelColor(x, y);
                        viscol.getRgbF(&r, &g, &b); // all channels store the same info so just use red

                        for(int i = std::max(0, gx-2); i <= std::min(dx-1, gx+2); i++)
                           for(int j = std::max(0, gy-2); j <= std::min(dy-1, gy+2); j++)
                           {
                               if(mask.get(i, j) < 0.5f)
                               {
                                   sunvis->set(i, j, sunvis->get(i, j) + (float) r * timestep);
                                   mask.set(i, j, 1.0f);
                               }
                           }
                    }
                }

        }
}

View * GLSun::getView()
{
    return scene->view;
}

Terrain * GLSun::getTerrain()
{
    return scene->terrain;
}

MapFloat * GLSun::getCanopyHeight()
{
    return scene->chght;
}

MapFloat * GLSun::getCanopyDensity()
{
    return scene->cdense;
}

MapFloat * GLSun::getAlpha()
{
    return scene->alpha;
}

PMrender::TRenderer * GLSun::getRenderer()
{
    return renderer;
}

void GLSun::setScene(Terrain * ter, MapFloat * ch, MapFloat * cd)
{
    float tx, ty;
    int dx, dy;

    scene->terrain = ter;
    scene->chght = ch;
    scene->cdense = cd;

    getView()->setForcedFocus(getTerrain()->getFocus());
    getView()->setViewScale(getTerrain()->longEdgeDist());
    getTerrain()->calcMeanHeight();
    getTerrain()->setBufferToDirty();
    getTerrain()->getTerrainDim(tx, ty);
    getView()->setOrthoViewExtent(tx,ty);
    getTerrain()->getGridDim(dx, dy);
    if(scene->alpha == nullptr)
        scene->alpha = new MapFloat();
    scene->alpha->setDim(dx, dy);

    update();
}

void GLSun::deriveAlpha(EcoSystem *eco, Biome * biome)
{
    getAlpha()->fill(0.0f);
    eco->sunSeeding(getTerrain(), biome, getAlpha());
}

void GLSun::alphaMapStats()
{
    int dx, dy;
    float avgalpha = 0.0f, a;
    int cntalpha = 0;

    cerr << "**** ALPHA MAP STATS ***" << endl;
    if(getAlpha() != nullptr)
    {
        getAlpha()->getDim(dx, dy);
        cerr << "Alpha dim = " << dx << ", " << dy << endl;
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
            {
                a = getAlpha()->get(x, y);
                if(a > 0.0f)
                {
                    avgalpha += a;
                    cntalpha++;
                }
            }
        cerr << "Average alpha = " << avgalpha / (float) cntalpha << " with nonzero of " << cntalpha << " from " << dx*dy << endl;
        cerr << "with percentage coverage = " << (float) cntalpha / (float) (dx*dy) << endl;
    }
    else
    {
        cerr << "alpha map does not exist" << endl;
    }
}

void GLSun::bind()
{
    if(sun)
        delete sun;
    sun = new CanopyShape(0.75, getTerrain()->getCellExtent());
}

void GLSun::initializeGL()
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

    // *** PM REnder code - start ***
    PMrender::TRenderer::terrainShadingModel sMod = PMrender::TRenderer::SUN;

    // set terrain shading model
    renderer->setTerrShadeModel(sMod);

    // set up light
    Vector dl = Vector(0.6f, 1.0f, 0.6f);
    dl.normalize();

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
    renderer->textureManipulators(false);

    // *** PM REnder code - end ***

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glDisable(GL_MULTISAMPLE);
    glDisable(GL_DEPTH_CLAMP);
    glEnable(GL_TEXTURE_2D);

    paintGL(); // complete initialization
}

void GLSun::paintGL()
{
    uts::vector<ShapeDrawData> drawParams; // to be passed to terrain renderer
    drawParams.clear();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // no longer use CHM
    // UNDO
    // if(sun)
    //     sun->drawCanopy(drawParams);


    // pass in draw params and render
    renderer->setConstraintDrawParams(drawParams);

    getTerrain()->updateBuffers(renderer);
    renderer->drawSun(getView(), renderPass);
}

void GLSun::resizeGL(int width, int height)
{
    // TO DO: fix resizing
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, width, height);
}

////
// CanopyShape
///

void CanopyShape::genCanopyBox(float trunkratio, float cellscale)
{

    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkratio;

    GLfloat basecol[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    canopybox->setColour(basecol);
    // canopy - tapered box
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, trunkratio, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::rotate(tfm, glm::radians(-90.0f), rotx);
    canopybox->genPyramid(cellscale * 1.0f, cellscale * 1.0f, canopyheight, tfm);
}

void CanopyShape::bindCanopy(Terrain * ter, View * view, MapFloat * hght, MapFloat *dnsty)
{
    int dx, dy, bndplants = 0;
    std::vector<glm::mat4> xform; // transformation to be applied to each instance
    std::vector<glm::vec4> colvar; // colour variation to be applied to each instance
    float rndoff;

    xform.clear();
    colvar.clear();

    ter->getGridDim(dx, dy);
    DiceRoller * dice = new DiceRoller(-400,400);

    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
        {
            float h = hght->get(x, y);
            h = h * 0.3048f; // convert feet to metres
            // float d = dnsty->get(x, y);
            // stop using density because it is not available from pipeline
            // mean density is 0.91, add random variation of +- 0.4
            // rndoff = (float) dice->generate() / 10000.0f;
            float d = (0.91f + rndoff) * dnsty->get(x, y);

            // previously used #define avgtransmit

            if(h > 1.0f)
            {
                // setup transformation for individual plant, including scaling and translation
                glm::mat4 idt, tfm;
                glm::vec3 trs, sc;
                vpPoint loc = ter->toWorld(y, x, ter->getHeight(x, y)); // center of cell
                idt = glm::mat4(1.0f);
                trs = glm::vec3(loc.x, loc.y, loc.z);
                tfm = glm::translate(idt, trs); // translate to correct position
                sc = glm::vec3(1.0f, h, 1.0f); // scale to correct tree height
                tfm = glm::scale(tfm, sc);
                xform.push_back(tfm);

                colvar.push_back(glm::vec4(0.0f, 0.0f, 0.0f, d));
                // d is the blocking density of the canopy, adjusted to allow some light through beyond what can be seen from the ground
                // avgtransmit is calculated as the average transmission factor for sonoma tree species.
                bndplants++;
            }
        }

    std::cout << "binding " << bndplants << " instances..." << std::endl;
    if(!canopybox->bindInstances(nullptr, &xform, &colvar))
        cerr << "CanopyShape::bindCanopies: binding failed" << endl;
    std::cout << "finished binding instances" << std::endl;
    delete dice;
}

void CanopyShape::drawCanopy(std::vector<ShapeDrawData> &drawParams)
{
    ShapeDrawData sdd;

    sdd = canopybox->getDrawParameters();
    sdd.current = false;
    drawParams.push_back(sdd);
}

////
// HemSample
///

// Efficient sampling of the hemisphere based on the Hammersley Point Set
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html

float HemSample::radicalInverse_VdC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

void HemSample::hammersley2d(uint i, uint N, float &u, float &v)
{
    u = (float) i / (float) N;
    v = radicalInverse_VdC(i);
}

void HemSample::convertUniform(float u, float v, Vector &dirn)
{
    float phi = v * 2.0 * PI;
    float cosTheta = 1.0 - u;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    dirn = Vector(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
}

void HemSample::getSample(int s, int totSamples, Vector &dirn)
{
    float u, v;
    hammersley2d((uint) s, (uint) totSamples, u, v);
    convertUniform(u, v, dirn);
}


////
// SunLight
////

SunLight::SunLight()
{
    tx = 0; ty = 0;
    month = 0; time = 0;
    latitude = 0;
    setNorthOrientation(Vector(0.0, 0.0, -1.0f));
    refreshSun();
    genSphereSun();
}

SunLight::~SunLight()
{
}

// Various conversion helper routines
float SunLight::minutesToAngle(float minutes)
{
    return ((_half_day_in_minutes - minutes) / _half_day_in_minutes) * M_PI;
}

/*
float SunLight::getAxisTiltAngle(int mnth)
{
    return -_axis_tilt + ((float) std::abs(6 - mnth) * _monthly_axis_tilt);
}*/

float SunLight::getAxisTiltAngle(float mnth)
{
    return -_axis_tilt + (std::fabs(6.0 - mnth) * _monthly_axis_tilt);
}

float SunLight::latitudeToAngle(float lat)
{
    return -lat / (180.0f / M_PI);
}

void SunLight::splitTilt(int time_of_day, float & pitch, float & roll)
{
    float f_time_of_day ((float) time_of_day);
    // Pitch
    {
        if(time_of_day <= _half_day_in_minutes) // Before midday
            pitch = 1.0f - ((f_time_of_day/_half_day_in_minutes) * 2);
        else // After midday
            pitch = -1 + (((f_time_of_day-_half_day_in_minutes)/_half_day_in_minutes) * 2);
    }

    // Roll
    {
        if(time_of_day < (_quarter_day_in_minutes))
            roll = (f_time_of_day/_quarter_day_in_minutes) * 1.0f;
        else if(f_time_of_day >= _quarter_day_in_minutes && f_time_of_day <= _3_quarters_day_in_minutes)
            roll = 1 - (((f_time_of_day-_quarter_day_in_minutes)/_half_day_in_minutes)*2.0f);
        else // 6 pm -> midnight
            roll = -1 + ((f_time_of_day - _3_quarters_day_in_minutes) / _quarter_day_in_minutes);
    }
}

void SunLight::refreshSun()
{   
    // First calculate some orientations we need values we need
    glm::vec3 east_orientation = glm::rotateY(north, (float)M_PI_2);
    glm::vec3 true_north_orientation = glm::rotate(north, glm::radians((float) (-latitude)), east_orientation); // is sign correct here?

    int sun_trajectory_radius(500000);
    float max_axis_tilt(SunLight::getAxisTiltAngle(month));
    float day_angle(SunLight::minutesToAngle((float) time));
    glm::vec3 cp_tn_and_east (glm::normalize(glm::cross(true_north_orientation, east_orientation)));

    // First calculate the sun position at midday during the equinox
    glm::vec3 sun_position ( ((float)sun_trajectory_radius) * cp_tn_and_east );

    // Now take into consideration axis tilt based on the month
    sun_position = glm::rotate( sun_position, -max_axis_tilt, east_orientation ); // PITCH

    // Now rotate around true north for the day
    sun_position = glm::rotate(sun_position, day_angle, true_north_orientation);

    // Now align to the center of the terrain (i.e the center of the terrain is at the latitude specified)
    // sun_position += glm::vec3(center.x, center.y, center.z);

    sunpos = vpPoint(sun_position[0], sun_position[1], sun_position[2]);
}

void SunLight::genSphereSun()
{
    glm::mat4 idt;

    // simple unit diameter sphere
    idt = glm::mat4(1.0f);
    GLfloat suncol[] = {1.0f, 1.0f, 0.0f, 1.0f};
    sunRender.genSphere(20.0f, 20, 20, idt);
    sunRender.setColour(suncol);
}

void SunLight::bindDiffuseSun(View * view, Terrain * ter)
{
    std::vector<glm::mat4> xform; // transformation to be applied to each instance
    std::vector<glm::vec4> colvar; // colour variation to be applied to each instance
    HemSample hem;

    xform.clear();
    colvar.clear();

    for(int t = 0; t < 100; t++)
    {
        // setup transformation for individual sun
        glm::mat4 idt, tfm;
        glm::vec3 trs, sc;
        Vector sundir;
        vpPoint termid, loc;

        hem.getSample(t, 100, sundir);
        sundir.mult(400.0f);
        ter->getMidPoint(termid);
        loc = vpPoint(termid.x+sundir.i, termid.y+sundir.j, termid.z+sundir.k);
        //cerr << "sundir = " << sundir.i << " " << sundir.j << " " << sundir.k << endl;

        idt = glm::mat4(1.0f);
        trs = glm::vec3(loc.x+center.x, loc.y+center.y, loc.z+center.z);
        tfm = glm::translate(idt, trs);
        xform.push_back(tfm);
        glm::vec4 col = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // no colour variation
        colvar.push_back(col); // colour variation
     }
    if(!sunRender.bindInstances(view, &xform, &colvar))
        cerr << "SUN BINDING FAILED" << endl;
}

void SunLight::bindSun(View * view)
{
    std::vector<glm::mat4> xform; // transformation to be applied to each instance
    std::vector<glm::vec4> colvar; // colour variation to be applied to each instance

    xform.clear();
    colvar.clear();
    float tothours = 0.0f;

    for(int t = 0; t < 1440; t+= 1)
    {
        // setup transformation for individual sun
        glm::mat4 idt, tfm;
        glm::vec3 trs, sc;

        setTime(t);
        Vector sundir = Vector(sunpos.x, sunpos.y, sunpos.z);
        sundir.normalize();
        sundir.mult(400.0f);
        vpPoint loc = vpPoint(sundir.i, sundir.j, sundir.k);
        //cerr << "sundir = " << sundir.i << " " << sundir.j << " " << sundir.k << endl;
        if(loc.y > 0.0f)
            tothours += (1.0 / 60.0f);

        idt = glm::mat4(1.0f);
        trs = glm::vec3(loc.x+center.x, loc.y+center.y, loc.z+center.z);
        tfm = glm::translate(idt, trs);
        xform.push_back(tfm);
        glm::vec4 col = glm::vec4(abs((float) t - 720.0f) / -1400.0f, abs((float) t - 720.0f) / -1400.0f, abs((float) t - 720.0f) / -1400.0f, 0.0f); // midday is peak colour, all others are darker
        colvar.push_back(col); // colour variation
     }
    if(!sunRender.bindInstances(view, &xform, &colvar))
        cerr << "SUN BINDING FAILED" << endl;
    cerr << "TOTAL HOURS OF SUNLIGHT = " << tothours << endl;
}

void SunLight::drawSun(std::vector<ShapeDrawData> &drawParams)
{
    ShapeDrawData sdd;

    sdd = sunRender.getDrawParameters();
    sdd.current = false;
    drawParams.push_back(sdd);
}

void SunLight::diffuseSun(Terrain * ter, MapFloat * diffusemap, GLSun * glsun, int numSamples, bool enable)
{
    //int numSamples = 100;
    int dx, dy;
    Timer tmm;
    HemSample hem;

    ter->getGridDim(dx, dy);
    diffusemap->setDim(dx, dy);
    diffusemap->fill(0.0f);
    tmm.start();
    if (enable)
    {
        for(int s = 0; s < numSamples; s++) // hemisphere samples
        {
            Vector sunvec;
            hem.getSample(s, numSamples, sunvec);
            if(sunvec.j > 0.0f) // above the horizon
            {
                glsun->getView()->sundir(sunvec);
                glsun->calcVisibility(diffusemap, 1.0f);
            }
        }

        // normalization
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
            {
                diffusemap->set(x, y, diffusemap->get(x, y) / (float) numSamples);
            }
    }
    tmm.stop();
    cerr << "Diffuse sampling complete in " << tmm.peek() << " seconds" << endl;
}

void SunLight::projectSun(Terrain * ter, std::vector<MapFloat> &sunmaps, GLSun * glsun, std::vector<float> &sunhours, int minutestep, int startm, int endm, int mincr, bool enable)
{
    float tothours;
    int dx, dy;
    vpPoint gridpos, xsectpos;
    Vector sunvec;
    float hourstep = (float) minutestep / 60.0f;
    Timer tmm;
    float tstep, tx, ty;

    ter->getGridDim(dx, dy);
    ter->getTerrainDim(tx, ty);
    tstep = 1.0f * tx / (float) dx;
    sunhours.clear();

    int month_incr = mincr;
    int start_month = startm;
    int end_month = endm;

    for(int m = start_month; m <= end_month; m += month_incr) // months of the year
    {
        tothours = 0.0f;

        sunmaps[m-1].fill(0.0f);

        // for(int d = 0; d < 2; d++)
        // {
        int d = 1;
        setMonth((float) m + ((float) (d-1) / 30.0f));

        tmm.start();
        for(int t = 0; t < 1440; t+= minutestep)
        {
            setTime(t);

            // cerr << "zoom dist = " << glsun->getView()->getZoom() << endl;
            // cerr << "sunpos " << sunpos.x << ", " << sunpos.y << ", " << sunpos.z << endl;
            if(sunpos.y > 0.0f) // above the horizon
            {
                if (enable)
                {
                    sunvec = Vector(sunpos.x, sunpos.y, sunpos.z);
                    sunvec.normalize();
                    glsun->getView()->sundir(sunvec);
                    glsun->calcVisibility(&sunmaps[m-1], hourstep);
                }
                tothours += hourstep; // sun above horizon
            }
        }
        // }
        tmm.stop();
        cerr << "Month " << m << " Sunlight Pass Complete in " << tmm.peek() << " seconds" << endl;
        cerr << "with " << tothours << " of sunlight" << endl;
        sunhours.push_back(tothours);
    }
}

void SunLight::mergeSun(std::vector<MapFloat> &sunmaps, MapFloat * diffusemap, std::vector<float> cloudiness, std::vector<float> sunhours)
{
    int dx, dy;
    float direct, diffuse;

    diffusemap->getDim(dx, dy);
    for(int m = 0; m < 12; m++) // months of the year
    {
        blurSun(sunmaps[m], 2, 2);
        blurSun((* diffusemap), 2, 2);
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
            {
                // direct sunlight portion
                direct = sunmaps[m].get(x, y) * (1.0f - cloudiness[m]);
                // diffuse sunlight portion
                diffuse = diffusemap->get(x, y) * cloudiness[m] * sunhours[m];

                sunmaps[m].set(x, y, direct+diffuse);
            }
        // cerr << "Cloudiness for month " << m << " is " << cloudiness[m] << " and sun hours are " << sunhours[m] << endl;
    }

}

void SunLight::applyAlpha(GLSun * sun, std::vector<MapFloat> &sunmaps)
{
    int dx, dy;

    // apply previously derived alpha map
    sun->getAlpha()->getDim(dx, dy);

    // do radial smoothing. Note this changes the alpha map
    radialBlur((* sun->getAlpha()), 15);

    for(int m = 0; m < 12; m++) // months of the year
    {
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
                sunmaps[m].set(x, y, sunmaps[m].get(x, y) * (1.0f - sun->getAlpha()->get(x, y)));
 //               if(sun->getAlpha()->get(x, y) > 0.1f)
 //                   sunmaps[m].set(x, y, 0.0f);
    }
}

void SunLight::radialBlur(MapFloat &map, int radius)
{
    MapFloat newmap;
    int filterwidth = radius+1;
    int sqrrad = radius * radius;

    int dx, dy;
    map.getDim(dx, dy);
    newmap.setDim(dx, dy);

    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
        {
            float avg = 0.0f;
            int cnt = 0;

            for(int cx = x-filterwidth; cx <= x+filterwidth; cx++)
                for(int cy = y-filterwidth; cy <= y+filterwidth; cy++)
                {
                    int r = (cx-x) * (cx-x) + (cy-y) * (cy-y);
                    // within radius and within bounds
                    if(r <= sqrrad && cx >= 0 && cx < dx && cy >= 0 && cy < dy)
                    {
                        avg += map.get(cx, cy);
                        cnt++;
                    }
                }
            newmap.set(x,y, avg / (float) cnt);
        }

    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
            map.set(x, y, newmap.get(x, y));
}

void SunLight::blurSun(MapFloat &map, int filterwidth, int passes)
{
    float filterarea;
    MapFloat newmap;

    filterarea = (float) ((filterwidth*2+1)*(filterwidth*2+1));

    int dx, dy;
    map.getDim(dx, dy);
    newmap.setDim(dx, dy);

    for(int i = 0; i < passes; i++)
    {
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
            {
                float avg = 0.0f;

                for(int cx = x-filterwidth; cx <= x+filterwidth; cx++)
                    for(int cy = y-filterwidth; cy <= y+filterwidth; cy++)
                    {
                            if(cx < 0 || cx >= dx || cy < 0 || cy >= dy)
                                avg += map.get(x, y);
                            else
                                avg += map.get(cx, cy);
                    }
                    newmap.set(x,y, avg / filterarea);
            }

        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
                map.set(x, y, newmap.get(x, y));
    }
}
