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


#ifndef TRENDERER_H
#define TRENDERER_H

#include "glheaders.h"

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "glheaders.h"

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <common/map.h>
#include <common/debug_string.h>
#include "shaderProgram.h"
#include <QGLWidget>
#include "shape.h"
#include "typemap.h"

namespace PMrender {

class TRenderer
{
 public:
    // select which render methos to use for terrain
    enum terrainShadingModel {BASIC, RADIANCE_SCALING, SUN};
    // select type map info to update - PAINT upfdated paint type map,
    // CONSTRAINT updates the overlay data (additional constraints etc)
    enum typeMapInfo {PAINT, CONSTRAINT};

 private:

    QGLWidget *canvas;

    std::string shaderDir; // location of all shaders

    glm::vec4 terMatDiffuse; // colour of terrain - Phong
    glm::vec4 terMatSpec;
    glm::vec4 terMatAmbient;
    glm::vec4 terSurfColour; // radiance scaling surface colour
    glm::vec4 pointLight; // position of light source in world space
    glm::vec4 directionalLight[2]; // directional lights  (vector) for radiance scaling
    glm::vec4 lightDiffuseColour; // colour of light
    glm::vec4 lightSpecColour;
    glm::vec4 lightAmbientColour;
    GLfloat shinySpec;

    // features to enable
    bool terrainTypeTexture; // use paint overlay texture
    bool constraintTypeTexture; // use constraint overlay texture
    bool contours; // draw countours on terrain
    bool contoursWall; // if this is false wall contours will be turned off/overrides contours setting
    bool gridlines; // draw grid lines on terrain
    bool gridlinesWall; // draw grid lines on side walls (assumes gridlines are ON)
    bool shadersReady; // have shaders been compiled/linked?
    bool manipulatorTextures; // use texture on manipulators
    bool drawOutOfBounds;// shade heights if they are out of bounds
    bool drawHiddenManipulators; // do extra render pass to show hidden manipulators

    // need model and view matrices

    GLuint vaoTerrain;
    GLuint vboTerrain;
    GLuint iboTerrain;

    GLuint vaoWalls[5];
    GLuint vboWalls[5];
    GLuint iboWalls[5];
    glm::vec3 normalWalls[5];
    int wallDrawEls[5];

    unsigned int indexSize; // number of elements in index buffer.

    GLfloat *typeBuffer;// local storage used to avoid multiple allocatiosn when managing type painting updates
    GLuint normalTexture; //  normal texture identifier - data generated from shader pass
    GLuint fboNormalMap; // normal map FBO
    GLuint heightmapTexture; // id of heightmap texture; generated outside class
    GLuint typeMapTexture; // texure used to store type map
    GLuint fboRadScaling; // FBO for radiance scaling renders
    GLuint fboRSOutput; // FBO for final composited Rad scaling output
    GLuint fboManipLayer; // FBO for manipulator transparency fix
    GLuint constraintTexture; // texture to store additional terrain vis data (freezng, overlay etc)

    GLenum htmapTexUnit; // height/normal map - texture units reserved
    GLenum normalMapTexUnit;
    GLenum typemapTexUnit;// for region type overlay
    GLenum rsNormTexUnit; // radiance scaling - texture units reserved
    GLenum rsGradTexUnit;
    GLenum rsColTexUnit;
    GLenum rsDestTexUnit;
    GLenum constraintTexUnit; // used for terrain freezing and other overlay data
    GLenum manipTranspTexUnit; // used for transparency affect on manipulators

    GLuint depthTexture; // radiance scaling: FBO depth texture
    GLuint normTexture; // radiance scaling: FBO normal texture
    GLuint colTexture; // radiance scaling: FBO colour texture
    GLuint gradTexture; // radiance scaling: FBO gradient texture
    GLuint destTexture; // radiance scaling: FBO final tesxture composit, to be written back to framebuffer

    GLuint manipDepthTexture; // manipulator transparency fix: depth texture for FBO
    GLuint manipTranspTexture; // manipulator transparency fix: colour texture for FBO

    GLenum decalTexUnit; // for texturing manipulator
    GLuint decalTexture;

    // OpenGL transformation/view matrices
    glm::mat4x4 viewMx;
    glm::mat4x4 MVmx;
    glm::mat4x4 projMx;
    glm::mat4x4 MVP;
    //glm::mat4x4 modelMx;  // should be identity (?)
    glm::mat3x3 normalMatrix;

    // values for drawing contours/gridlines
    float gridColFactor;
    float gridXsep;
    float gridZsep;
    float gridThickness;
    float contourSep;
    float contourThickness;
    float contourColFactor;

    // radiance scaling parameters
    bool RSinvertCurvature;
    float RStransition;
    float RSenhance;

    // misc FBO rendering params

    float scalex, scaley; // extent of terrain in metres
    int width, height; // dimensions of height map
    int _w, _h;       // dimensions of FrameBuffer for radiance scaling
    terrainShadingModel shadModel;
    float terrainBase; // lowest point on terrain - based moved to this height
    float terrainBasePad; // extra space used to avoid have 'thin' terrains

    glm::vec4 outOfBoundsColour; // used to indicate when the user is dragging terrain heights out of allowable range
    float outOfBoundsWeight; // alpha blend to use for outOfBoundsColour
    float outOfBoundsMean; // mean value for offset used to flag disallowed heights
    float outOfBoundsOffset; // this offset around mean will be used to flag disallowed heights

    float manipAlpha; // transparency blend factor for hidden manipulators (current only)

   std::map<std::string, shaderProgram*> shaders;

   // screen (z-plane) aligned quad: (X,Y, s, t); tex coords are centred on pixels, hence +0.25 contribution
   GLuint vaoScreenQuad;
   GLuint vboScreenQuad;
   GLfloat screenQuad[16] = {-1.0f, -1.0f,   0.0f, 0.0f,
                             1.0f, -1.0f,    1.0f, 0.0f,
                             1.0f, 1.0f,     1.0f, 1.0f,
                             -1.0f, 1.0f,    0.0f, 1.0f
                            };                        // for postprocessing: screen aligned quad

    // constraint/manipulator drawing

    std::vector<ShapeDrawData> manipDrawCallData;

    // private methods:

    void deleteTerrainOpenGLbuffers(void);
    void deleteFBOrscalingBuffers(void);
    GLuint addShader(const std::string& shadName, const char *frag, const char *vert);
    bool prepareTerrainGeometry(void);
    void makeXwall(int atY, GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices, bool reverse);
    void makeYwall(int atX, GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices, bool reverse);
    void makeBase(GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices);
    bool prepareWalls(void);
    void generateNormalTexture(void);
    void drawManipulators(GLuint program, bool drawTO_FB=false);
    bool initRadianceScalingBuffers(int vWd, int vHt);
    // manage creation/destruction of new models
    void initInstanceData(void);
    void destroyInstanceData(void);
public:

    TRenderer(QGLWidget *drawTo = NULL, const std::string &dir="."); // the QGLwidget is created by the GUI manager
    ~TRenderer();

    // load in new terrain data; this will come from an grid structure. paintMap is the associated
    // terrain type map, and constrainTmap is the map with freeze constrainst etc. Both of these can be NULL.
    void loadTerrainData(const float* data, int wd, int ht, float scx, float scy,
                         TypeMap* paintMap = NULL, TypeMap* constraintMap = NULL);

    void useTerrainTypeTexture(bool v)
    {
      terrainTypeTexture = v;
    }

    void useConstraintTypeTexture(bool v)
    {
      constraintTypeTexture = v;
    }

    void setTerrShadeModel(TRenderer::terrainShadingModel m)
    {
      shadModel = m;
    }

    // set terrain colour for radiance scaling:
   void setTerrainColourRS(float r, float g, float b, float a)
    {
      terSurfColour = glm::vec4(r, g, b, a);
    }

    // update radiance scaling FBO textures/attachments if viewport has changed size
    // vwd and vht are current viewport width and height, resp.
    void updateRadianceScalingBuffers(int vwd, int vht);

    // assumes modelling matrix is Identity for terrain
    void setCamera(glm::mat4x4& mx)
    {
        viewMx = mx;
        MVmx = mx;
        normalMatrix = glm::transpose(glm::inverse(glm::mat3(MVmx)));
        MVP = projMx * MVmx;
    }

    // show HiddenManipulators
    void showHiddenManipulators(bool v)
    {
      drawHiddenManipulators = v;
    }

    // blend factor for hidden manipulators
    void setManipAlphaFactor(float v)
    {
      manipAlpha = v;
    }

    // texture manipulators
    void textureManipulators(bool v)
    {
      manipulatorTextures = v;
    }

    // wall contours on/off
    void drawWallContours(bool v)
    {
      contoursWall = v;
    }

    // draw wall grid lines
    void drawWallGridlines(bool v)
    {
      gridlinesWall = v;
    }

    // activate/deactivate contours
    void drawContours(bool v)
    {
      contours = v;
    }

    // activate/deactivate gridlines
    void drawGridlines(bool v)
    {
      gridlines = v;
    }
    
    // setters for various rendering parameters
    void setRadianceScalingParams(float enhance, bool invert = false)
    {
        RSenhance = enhance; RSinvertCurvature = invert;
    }
    
    void setContourParams(float sep, float thickness, float intensityFactor)
    {
        contourSep = sep; contourThickness = thickness; contourColFactor = intensityFactor;
    }
    
    void setGridParams(float sepX, float sepZ, float thickness, float intensityFactor)
    {
        gridXsep = sepX; gridZsep = sepZ; gridThickness = thickness; gridColFactor = intensityFactor;
    }

    void getGridParams(float &sepX, float &sepZ, float &thickness, float &intensityFactor)
    {
        sepX = gridXsep; sepZ = gridZsep; thickness = gridThickness; intensityFactor = gridColFactor;
    }

    float getTerrainBase(void) const
    {
      return terrainBase;
    }

    float getTerrainBasePadding(void) const
    {
      return terrainBasePad;
    }

    void setTerrainBasePadding(float f)
    {
      terrainBasePad = f;
    }

    // draw outOfBounds shade
    void setDrawOutOfBounds(bool v)
    {
      drawOutOfBounds = v;
    }

    // set colour and blending weight
    void setOutOfBoundsParams(float r, float g, float b, float blendF)
    {
      outOfBoundsColour[0] = r;
      outOfBoundsColour[1] = g;
      outOfBoundsColour[2] = b;
      outOfBoundsColour[3] = 1.0f;
      outOfBoundsWeight = blendF;
    }

    // set out of bounds mean height
    void setoutOfBoundsMean(float v)
    {
      outOfBoundsMean = v;
    }

    // set out of bounds offset from mean
    void setoutOfBoundsOffset(float v)
    {
      outOfBoundsOffset = v;
    }

    void setModelViewProjection(glm::mat4x4& mv, glm::mat4x4& proj)
    {
      MVmx = mv;
      projMx = proj;
      normalMatrix = glm::transpose(glm::inverse(glm::mat3(MVmx)));
      MVP = projMx * MVmx;

    }
    void setModelView(glm::mat4x4& mx)
    {
      MVmx = mx;
      normalMatrix = glm::transpose(glm::inverse(glm::mat3(MVmx)));
      MVP = projMx * MVmx;
    }

    void setProjection(glm::mat4x4 &mx)
    {
        projMx = mx;
        MVP = projMx * MVmx;
    }

    // world space light position (transformed by MV matrix)
    void setPointLight(float x, float y, float z)
    {
      pointLight = glm::vec4(x, y, z, 1.0);
      std::cout << "point Light position = (" << x << "," << y << ","<< z << ")\n";
    }

   // set directional light N {0,1}(radiance scaling)
    void setDirectionalLight(int n, float x, float y, float z)
    {
      if (n < 0 || n > 1)
        {
        std::cerr << "Directional light index must be 0 or 1!\n";
        return;
        }
      directionalLight[n] = glm::vec4(x, y, z, 0.0);
      std::cout << "Directional light " << n << " = (" << x << "," << y << ","<< z << ")\n";
    }
    // load heightmap buffer (wd x ht) into texture (this does not set any internal state, only sets up OpenGL texture)
    // static GLuint setHeightField(GLenum texUnit, int wd, int ht, const float* data);

    // load a test RAW file into texture (this does not set any internal state, only sets up OpenGL texture)
    static GLuint loadTest(const std::string &filename, GLenum texUnit, int &wd, int& ht);

    // set directory for shader sources (deprecated)
    void setShaderDirectory(const std::string& dir) { shaderDir = dir; }

    // get a pointer to a compiled shader program object; this can be queried for program ID etc
    PMrender::shaderProgram* getShaderProgramObject(const std::string& name) const
    {
      std::map<std::string, PMrender::shaderProgram*>::const_iterator it;
      if ((it = shaders.find(name)) == shaders.end())
       return NULL;
      else
       return it->second;
    }

    // utility function to write out RGB colour byte buffer as PPM image
    static void savePPMImage(const std::string filename, unsigned char *buffer, int w, int h);

    // copy across manipulator/constraint draw data
    void setConstraintDrawParams(const std::vector<ShapeDrawData>& indata)
    {
        manipDrawCallData = indata;
    }

    // init render object - call before any other operations! - just sets up and compiles shaders
    void initShaders(void);

    // call when height data has been changed (including whenit is first generated)
    // force==true will cause everything to be rebuilt, rather than simply updated
    void updateHeightMap(int wd, int ht, float scx, float scy, const float* data, bool force = false);

    /// load Decal texture map given an image stored in a suitable buffer (of width * height dimensions)
    void bindDecals(int width, int height, unsigned char * buffer);

    // draw call -takes the View object; if you want manipulators drawn the drawParams must be ready when
    // this method is disptached.i.e. call setConstraintDrawParams() first
    void draw(View * view);

    // draw call for sunlight visibility rendering. Pass 1 = for base indexed terrain, pass 2 = for canopy
    // light intersection
    void drawSun(View * view, int renderPass);

    // use tmap to generate/update the internal texture overay representation for terrain. This texture
    // will usually be created only once and dirty regions in tmap then be sub'd into the internal texture
    // for performamce reasons.
    void updateTypeMapTexture(TypeMap* tmap, typeMapInfo tinfo = typeMapInfo::PAINT, bool force = false);
};

}
#endif // TRENDERER_H
