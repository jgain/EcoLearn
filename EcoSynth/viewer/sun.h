/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
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

#ifndef Sun
#define Sun
/* file: sun.h
   author: (c) James Gain, 2018
   notes: sun direction simulation over the course of a year
*/

#include "eco.h"

class CanopyShape
{
private:
    Shape * canopybox;    //< shape template for each CHM cell to represent the canopy

    /**
      * Create geometry for simple canopy model ready for scaling according to cell instance
      * @param trunkratio  proportion of height devoted to bare trunk
      * @param cellscale   side length of a grid cell in metres
      */
    void genCanopyBox(float trunkratio, float cellscale);

public:

    CanopyShape(){ canopybox = new Shape(); genCanopyBox(0.75f, 1.0f); }

    /**
      * Constructor
      * @param trunkratio  proportion of height devoted to bare trunk
      * @param cellscale   side length of a grid cell in metres
      */
    CanopyShape(float trunkratio, float cellscale){ canopybox = new Shape(); genCanopyBox(trunkratio, cellscale); }

    ~CanopyShape(){ delete canopybox; }

    /**
     * @brief bindSun       Update positioning information for sun by varying time of day based on latitude and month
     * @param ter           heightfield terrain on which plants are located
     * @param view          the current view state
     * @param hght          canopy height model
     * @param dnsty         canopy density model
     */
    void bindCanopy(Terrain * ter, View * view, MapFloat * hght, MapFloat *dnsty);

    /**
     * @brief drawSun    Bundle rendering parameters for instancing lists
     * @param drawParams    Rendering parameters for the sun
     */
    void drawCanopy(std::vector<ShapeDrawData> &drawParams);
};

class SunScene
{
public:

    View * view;
    Terrain * terrain;
    MapFloat * chght;   //< canopy height model
    MapFloat * cdense;  //< canopy density model

    SunScene();
    ~SunScene();
};

class GLSun : public QGLWidget
{
    Q_OBJECT

public:

    GLSun(const QGLFormat& format, QWidget *parent = 0);
    ~GLSun();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    /**
     * @brief ColToCoord    Convert colour to grid coordinates
     * @param col           grid coordinates encoded as colour
     * @param gx            output x-coord
     * @param gy            output y-coord
     */
    void colToCoord(QColor col, int &gx, int &gy);

    /// bind canopy geometry
    void bind();

    /**
     * @brief calcVisibility    Determine if points on the terrain are visible from the sun's perspective and
     *                          if so increment by the timestep
     * @param sunmap            map storing sun exposure values
     * @param timestep          time quantum for sun exposure
     */
    void calcVisibility(MapFloat * sunvis, float timestep);

    /// getters for currently active view, terrain, renderer
    View * getView();
    Terrain * getTerrain();
    MapFloat * getCanopyHeight();
    MapFloat * getCanopyDensity();
    PMrender::TRenderer * getRenderer();

    /**
     * @brief setScene  Configure a scene for sunlight rendering
     * @param ter   Terrain onto which plants are placed
     * @param ch    Canopy height model
     * @param cd    Canopy density model
     */
    void setScene(Terrain * ter, MapFloat * ch, MapFloat * cd);

    void calcVisibilitySelfShadowOnly(MapFloat *sunvis, float timestep);
    void nullifySun();
    void init_gl();
signals:
    void signalRepaintAllGL();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);

private:
    PMrender::TRenderer * renderer;
    SunScene * scene;
    CanopyShape * sun;
    QColor qtWhite;
    int renderPass; // base visibility or canopy render pass
};



class HemSample
{
private:

    float radicalInverse_VdC(uint bits);

    void hammersley2d(uint i, uint N, float &u, float &v);

    void convertUniform(float u, float v, Vector &dirn);

public:

    HemSample(){}
    ~HemSample(){}

    /**
     * @brief getSample get the sth sample from the surface of a hemisphere that is y-oriented
     * @param s             the sample number in the sequence
     * @param totSamples    total number of samples
     * @param dirn          vector pointing to sample position on unit-hemisphere
     */
    void getSample(int s, int totSamples, Vector &dirn);
};


class SunLight
{
private:
    int month;          //< month of the year [1=january,12=december]
    int time;           //< time of day in minutes
    float latitude;     //< latitude
    float tx, ty;       //< terrain dimensions
    vpPoint center;     //< center of the terrain in grid units
    vpPoint sunpos;     //< sun position
    glm::vec3 north;    //< direction of true north in local coordinate system of the terrain
    Shape sunRender;    //< for rendering a sun sphere

    /// various conversion constants
    static const float _axis_tilt;
    static const float _monthly_axis_tilt;
    static const float _half_day_in_minutes;
    static const float _quarter_day_in_minutes;
    static const float _3_quarters_day_in_minutes;

    /// reclalculate the sun's position in the sky
    void refreshSun();

    /// minutesToAngle: the rotation relative to noon, in radians
    static float minutesToAngle(float minutes);
    /// getAxisTiltAngle: axial tilt based on month of the year
    static float getAxisTiltAngle(int mnth);
    static float latitudeToAngle(float lat);
    static void splitTilt(int time_of_day, float & pitch, float & roll);

    /**
      * Create geometry for Sun
      * @param shape        geometry for Sun instance
      */
    void genSphereSun();

public:
    SunLight();
    ~SunLight();

    /// getters and setters
    void setNorthOrientation(Vector nth){ north = glm::normalize(glm::vec3(nth.i, nth.j, nth.k)); }
    void setMonth(int mnth){ month = mnth; refreshSun(); }
    void setTime(int min){ time = min; refreshSun(); }
    void setLatitude(float lat){ latitude = lat; refreshSun(); }
    void setTerrainDimensions(Terrain * ter)
    {
        ter->getTerrainDim(tx, ty);
        center = vpPoint((float) tx/2.0f, 0.0f, (float) ty/2.0);
        refreshSun();
    }
    vpPoint &getSun(){ refreshSun(); return sunpos; }

    /**
     * @brief bindDiffuseSun       Generate sun positions based on hemispheric sampling
     * @param view          the current view state
     * @param ter           terrain on which sun shines
     */
    void bindDiffuseSun(View * view, Terrain * ter);

    /**
     * @brief bindSun       Update positioning information for sun by varying time of day based on latitude and month
     * @param view          the current view state
     */
    void bindSun(View * view);

    /**
     * @brief drawSun    Bundle rendering parameters for instancing lists
     * @param drawParams    Rendering parameters for the sun
     */
    void drawSun(std::vector<ShapeDrawData> &drawParams);

    /**
     * @brief diffuseSun    Sample the hemisphere of the sky to build a diffuse sunlight contribution for cloudy days
     * @param ter           The heightfield terrain that provides self-shadowing
     * @param diffusemap    The output diffuse lighting map
     * @param glsun         OpenGL widget for efficient calculation of sun visibility, also includes the canopy height and density
     */
    void diffuseSun(Terrain * ter, MapFloat * diffusemap, GLSun * glsun, int numSamples);

    /**
     * @brief projectSun    Calculate collection of sun exposure maps for the middle of each month.
     *                      This takes into account self-shadowing by the terrain.
     * @param ter           The heightfield terrain that provides self-shadowing
     * @param sunmaps       12 grid maps with the same dimensions as the terrain, 0 = January, 11 = December.
     *                      Populated with the number of hours of direct sunlight per grid cell.
     * @param glsun         OpenGL widget for efficient calculation of sun visibility by rendering
     * @param sunhours      To store the average number of hours of sunlight per day for each month
     * @param minutestep    The increment in minutes for sun position sampling.
     *                      Lower numbers take longer to calculate but are more accurate.
     */
    void projectSun(Terrain * ter, std::vector<MapFloat> &sunmaps, GLSun * glsun, std::vector<float> &sunhours, int minutestep=5);

    /**
     * @brief mergeSun      Combine direct and diffuse sunlight according to the proportions
     * @param sunmaps       Hours of direct sunlight per cell of the terrain
     * @param diffusemap    The proportion of diffuse sunlight per cell of the terrain
     * @param cloudiness    Average cloud cover per day for each month
     * @param sunhours      Average hours of sunlight per day for each month
     */
    void mergeSun(std::vector<MapFloat> &sunmaps, MapFloat * diffusemap, std::vector<float> cloudiness, std::vector<float> sunhours);
    void projectSunSelfShadowOnly(Terrain *ter, std::vector<MapFloat> &sunmaps, GLSun *glsun, std::vector<float> &sunhours, int minutestep);
};


#endif
