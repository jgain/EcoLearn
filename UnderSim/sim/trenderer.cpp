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


#include "trenderer.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace PMrender {


  // create and update heightmap texture

// this sets the heightmap texture; if dimensions have changed, mesh+normals are rebuilt
// if force == draw, then we a rebuild will take place - use to manage changed data from synthesis

  void TRenderer::updateHeightMap(int wd, int ht, float scx, float scy, const float* data, bool force)
  {
    if (data == NULL)
      {
        std::cerr << "TRenderer::updateHeightMapTexture - NULL pointer for heightmap?";
        return;
      }

    assert(wd != 0 && ht != 0);

    if (width == wd && height == ht && !force) // nothing to do - make sure binding is intact
      {
        //std::cerr << "rebind heightmap texture\n";
        if (heightmapTexture == 0)
        {
             std::cerr << "Error! Heighmap texture undefined!\n";
          }
        glActiveTexture(htmapTexUnit); CE();
        glBindTexture(GL_TEXTURE_2D, heightmapTexture); CE();
        return;
      }

    // if grid dimensions have changed:
    if(heightmapTexture != 0 && (width != wd || height != ht))
      {
        // std::cerr << "- Delete texture\n";
        glDeleteTextures(1, &heightmapTexture);  CE();
        heightmapTexture = 0;
      }

    if (heightmapTexture == 0) // create texture if it does not exist
      {
        // std::cerr << "- Create texture\n";
        glGenTextures(1, &heightmapTexture); CE();
        glActiveTexture(htmapTexUnit); CE();
        glBindTexture(GL_TEXTURE_2D, heightmapTexture ); CE();

        glTexImage2D(GL_TEXTURE_2D, 0,GL_R32F, wd, ht, 0,GL_RED, GL_FLOAT,  (GLfloat*)data); CE();
        // no filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
        // deal with out of array access
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
      }
    else // otherwise sub in new texture data
      {
        // std::cerr << " - sub texture\n";
        glActiveTexture(htmapTexUnit); CE();
        glBindTexture(GL_TEXTURE_2D, heightmapTexture ); CE();
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, wd, ht, GL_RED, GL_FLOAT, (GLfloat*)data); CE();
      }

    // test all values for lowest terrain height:
    terrainBase = 1000000.0; // +infinity
    for (int i = 0; i < wd*ht; i++)
        terrainBase = std::min(terrainBase, data[i]);


    // std::cout << "Terrain BAse: " << terrainBase << std::endl;

    width = wd;
    height = ht;
    scalex = scx;
    scaley = scy;

    // rebuild VAO and everything else if this was first image or new
    // dimensions or data in heightmap has changed

    deleteTerrainOpenGLbuffers();
    prepareTerrainGeometry(); // set up VBO, IBO, FBO etc
    prepareWalls(); // build capping walls for terrain
    generateNormalTexture(); // generate new normals and set up texture unit
  }

// utility function to write out colour (RGB)  buffer as PPM image
  void TRenderer::savePPMImage(const std::string filename, unsigned char *buffer, int w, int h)
{
  ofstream ofs;

  ofs.open(filename.c_str(), std::ios::binary);
  if (!ofs)
    {
      std::cerr<< "TRenderer::savePPMImage - unable to open file for writing: " << filename << std::endl;
      return;
    }
  // write header:
  ofs << "P6\n" << w << " " << h << "\n255\n";

  // write data:
  ofs.write((char*)buffer, w*h*3);
  if (!ofs)
     std::cerr<< "TRenderer::savePPMImage - error occured when writing file " << filename << std::endl;

  ofs.close();
}

// load test data set; returns width and height of terrain and
// the texture id of the single channel texture that stores it
// the texture unit to which this should be bound is also passed in

GLuint TRenderer::loadTest(const std::string &filename, GLenum texUnit, int &wd, int &ht)
{
 int i;
  GLfloat *buffer = NULL;
  float biggest = 0.0f;

  std::ifstream infile;
  infile.open(filename.c_str(), std::ios_base::in);
  if(infile.is_open() && !infile.eof())
    {
      infile >> wd;
      infile >> ht;

      buffer  = new GLfloat [wd*ht];

      GLfloat *ptr = buffer;

      for (i = 0; i < wd*ht; i++)
      {
         infile >> *ptr;
         biggest = std::max(*ptr, biggest);
         ptr++;
      }
      infile.close();
    }
  else
    {
      std::cerr << "TRenderer::loadTest: file not found" << filename << std::endl;
      infile.close();
      return 0;
    }

    std::cerr << "Maxx Z value = " << biggest << std::endl;
    // create texture for heightmap

    GLuint htId;
    glGenTextures(1, &htId); CE();
    glActiveTexture(texUnit); CE();
    glBindTexture(GL_TEXTURE_2D,  htId ); CE();

    glTexImage2D(GL_TEXTURE_2D, 0,GL_R32F, wd, ht, 0,GL_RED, GL_FLOAT,  buffer); CE();
    // no filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();

    // deal with out of array access

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    if (buffer) delete [] buffer;


    return htId;
}

GLuint TRenderer::addShader(const std::string& shadName, const char *frag, const char *vert) // add a new shader to the collection
{
    bool ok = false;
    PMrender::shaderProgram *ns = new PMrender::shaderProgram(frag, vert);

    ns->compileAndLink();
    if (!ok)
    {
        std::cerr << "addShader failed for shader = " << shadName << std::endl;
        return 0;
    }
    else
    {
        shaders[shadName] = ns;
        return ns->getProgramID();
    }
}


void TRenderer::makeXwall(int atY, GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices, bool reverse)
{
    int vidx = 0, x, y;

    y = atY;  // create wall at y = atY;

    for (x = 0; x < width; x++, vidx++)
    {
        // positions: z wil be offset from height texture in the shader
        verts[5*vidx] = (float) x / (float) (width - 1) * scalex;
        //verts[5*vidx+1] = 1.0f - ((float) y / (float) (height - 1)) - 0.5f;
        //verts[5*vidx+2] = 0.0f;
        verts[5*vidx+1] =  0.0f;
        verts[5*vidx+2] = /* 1.0f - */ ((float) y / (float) (height - 1)) * scaley;
        // texture coordinates
        verts[5*vidx+3] = (float) (x+0.5f) / (float) (width);
        verts[5*vidx+4] = (float) (y+0.5f) / (float) (height);

        // positions: z will remain 0  - bottom edge of bottom wall, fixed to base plane
        // use negative text coords to signal this in shader
        verts[5*vidx + 5*width] = verts[5*vidx];
        verts[5*vidx+1 + 5*width] = verts[5*vidx+1];
        verts[5*vidx+2 + 5*width] = verts[5*vidx+2];
        // texture coordinates
        verts[5*vidx+3 + 5*width] = -1.0f;
        verts[5*vidx+4 + 5*width] = -1.0f;

        indices[2*vidx] = (reverse ? width-1-x: x);
        indices[2*vidx+1] = (reverse ? 2*width-1-x: x+width);

    }

    glGenVertexArrays(1, &vao); CE();
    glBindVertexArray(vao); CE();

    // set up vertex buffer an copy in data
    glGenBuffers(1, &vbo); CE();
    glBindBuffer(GL_ARRAY_BUFFER, vbo); CE();
    glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*width*2, verts, GL_STATIC_DRAW); CE();

    // enable position attribute
    glEnableVertexAttribArray(0); CE();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(0)); CE();
    // enable texture coord attribute
    const int sz = 3*sizeof(GLfloat);
    glEnableVertexAttribArray(1); CE();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(sz) ); CE();

    // set up index buffer
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); CE();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*2*width, indices, GL_STATIC_DRAW); CE();

    // unbind everything and clean up

    glBindBuffer(GL_ARRAY_BUFFER, 0); CE();
    glBindVertexArray(0); CE();
}

  void TRenderer::makeYwall(int atX, GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices, bool reverse)
{
    int vidx = 0, x, y;

    x = atX;  // create vertical wall at x = atX;

    for (y = 0; y < height; y++, vidx++)
    {
        // positions: z  wil be offset from height texture in the shader
        verts[5*vidx] = (float) x / (float) (width - 1) * scalex;
        //verts[5*vidx+1] = 1.0f - ((float) y / (float) (height - 1)) - 0.5f;
        //verts[5*vidx+2] = 0.0f;
        verts[5*vidx+1] = 0.0f;
        verts[5*vidx+2] = /*1.0f -*/ ((float) y / (float) (height - 1)) * scaley;
        // texture coordinates
        verts[5*vidx+3] = (float) (x+0.5f) / (float) (width);
        verts[5*vidx+4] = (float) (y+0.5f) / (float) (height);

        // positions: z will remain 0  - bottome edge of bottom wall, fixed to base plane
        // use negative text coords to signal this in shader
        verts[5*vidx + 5*height] = verts[5*vidx];
        verts[5*vidx+1 + 5*height] = verts[5*vidx+1];
        verts[5*vidx+2 + 5*height] = verts[5*vidx+2];
        // texture coordinates
        verts[5*vidx+3 + 5*width] = -1.0f;
        verts[5*vidx+4 + 5*width] = -1.0f;

        indices[2*vidx] = (reverse ? height-1-y: y);
        indices[2*vidx+1] = (reverse ? 2*height-1-y: y+height);

    }

    glGenVertexArrays(1, &vao); CE();
    glBindVertexArray(vao); CE();

    // set up vertex buffer an copy in data
    glGenBuffers(1, &vbo); CE();
    glBindBuffer(GL_ARRAY_BUFFER, vbo); CE();
    glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*height*2, verts, GL_STATIC_DRAW); CE();

    // enable position attribute
    glEnableVertexAttribArray(0); CE();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(0)); CE();
    // enable texture coord attribute
    const int sz = 3*sizeof(GLfloat);
    glEnableVertexAttribArray(1); CE();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(sz) ); CE();

    // set up index buffer
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); CE();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*2*height, indices, GL_STATIC_DRAW); CE();

    // unbind everything and clean up

    glBindBuffer(GL_ARRAY_BUFFER, 0); CE();
    glBindVertexArray(0); CE();
}

void TRenderer::makeBase(GLuint &vao, GLuint&vbo, GLuint& ibo, GLfloat *verts, GLuint *indices)
 {
   float coords[4][2] = { {0.0f, 0.0f}, {scalex, 0.0f}, {0.0f, scaley}, {scalex, scaley}};
    for (int vidx = 0; vidx < 4; vidx++)
      {
        verts[5*vidx] = coords[vidx][0];
        verts[5*vidx+1] = 0.0f; // y=0, ground plane
        verts[5*vidx+2] = coords[vidx][1];

        verts[5*vidx+3] = -1.0f; // tex coords: these verts are NOT displaced in shader...
        verts[5*vidx+4] = -1.0f;
      }
    // ensure correct strip winding: normal must point down into ground plane, y=0: N=(0,-1,0)
    indices[0] = 1; indices[1] = 3; indices[2] = 0; indices[3] = 2;

    // standard VAO/VBO setup
    glGenVertexArrays(1, &vao); CE();
    glBindVertexArray(vao); CE();

    // set up vertex buffer an copy in data
    glGenBuffers(1, &vbo); CE();
    glBindBuffer(GL_ARRAY_BUFFER, vbo); CE();
    glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*4, verts, GL_STATIC_DRAW); CE();

    // enable position attribute
    glEnableVertexAttribArray(0); CE();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(0)); CE();
    // enable texture coord attribute
    const int sz = 3*sizeof(GLfloat);
    glEnableVertexAttribArray(1); CE();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(sz) ); CE();

    // set up index buffer
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); CE();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*4, indices, GL_STATIC_DRAW); CE();

    // unbind everything and clean up

    glBindBuffer(GL_ARRAY_BUFFER, 0); CE();
    glBindVertexArray(0); CE();
 }

bool TRenderer::prepareWalls(void)
{
    GLfloat *vertexStorage;
    GLuint *indexStorage;

    // max space required across both vert/horiz walls - we'll re-use this array

    int vertexStorageSize = std::max(width,height)*2*5; // 5 float attribs per vertex
    int indexStorageSize = std::max(width,height)*2;

    vertexStorage = new GLfloat [vertexStorageSize]; // 4 walls: top, bottom, left, right
    if (vertexStorage == NULL)
    {
        std::cerr <<   "prepareWalls: vertex allocation failed for " << vertexStorageSize <<
                       " vertices\n";
        return false;
    }

    indexStorage = new GLuint [indexStorageSize];
    if (indexStorage == NULL)
    {
        std::cerr <<   "prepareWalls: index buffer  allocation failed\n";
        return false;
    }

    makeXwall(height-1, vaoWalls[0], vboWalls[0], iboWalls[0], vertexStorage, indexStorage, false);
    normalWalls[0] = glm::vec3(0.0f, 0.0f, 1.0f); wallDrawEls[0] = 2*width;
    makeXwall(0, vaoWalls[1], vboWalls[1], iboWalls[1], vertexStorage, indexStorage, true);
    normalWalls[1] = glm::vec3(0.0f, 0.0f, -1.0f); wallDrawEls[1] = 2*width;
    makeYwall(0, vaoWalls[2], vboWalls[2], iboWalls[2], vertexStorage, indexStorage, false);
    normalWalls[2] = glm::vec3(-1.0f, 0.0f, 0.0f); wallDrawEls[2] = 2*height;
    makeYwall(width-1, vaoWalls[3], vboWalls[3], iboWalls[3], vertexStorage, indexStorage, true);
    normalWalls[3] = glm::vec3(1.0f, 0.0f, 0.0f); wallDrawEls[3] = 2*height;
    makeBase(vaoWalls[4], vboWalls[4], iboWalls[4], vertexStorage, indexStorage);
    normalWalls[4] = glm::vec3(0.0f, -1.0f, 0.0f); wallDrawEls[4] = 4;

    delete [] vertexStorage;
    delete [] indexStorage;

    return true;
}


// creates a VBO, IBO and an encapsulating VAO to represent terrain geometry, set up
// texture for normal map computation

bool TRenderer::prepareTerrainGeometry(void)
{
    GLfloat *vertexStorage;
    GLuint *indexStorage;

    vertexStorage = new GLfloat [width*height*5];
    if (vertexStorage == NULL)
    {
        std::cerr <<   "prepareTerrainGeometry: vertex allocation failed for " << (width*height) <<
                       " vertices\n";
        return false;
    }

    int vidx = 0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++, vidx++)
        {
            // positions: z wil be offset from height texture in the shader
            vertexStorage[5*vidx] = (float) x / (float) (width - 1) * scalex;
            //vertexStorage[5*vidx+1] = 1.0f - ((float) y / (float) (height - 1)) - 0.5f;
            //vertexStorage[5*vidx+2] = 0.0f;
            vertexStorage[5*vidx+1] = 0.0f;
            vertexStorage[5*vidx+2] = /* 1.0f - */ (float) y / (float) (height - 1) * scaley;

             // texture coordinates
             vertexStorage[5*vidx+3] = (float) x / (float) (width-1);
             vertexStorage[5*vidx+4] = (float) y / (float) (height-1);
        }
    }


    int numStripsRequired = height - 1;
    int numDegensRequired = 2 * (numStripsRequired - 1);
    int verticesPerStrip = 2 * width;

    indexSize = verticesPerStrip * numStripsRequired + numDegensRequired;

    indexStorage = new GLuint [indexSize];
    if (indexStorage == NULL)
    {
        std::cerr <<   "prepareTerrainGeometry: index buffer  allocation failed\n";
        return false;
    }


    int offset = 0;
    for (int y = 0; y < height - 1; y++)
    {
        if (y > 0) // Degenerate begin: repeat first vertex
            indexStorage[offset++] = (GLuint) (y * width);

        for (int x = 0; x < width; x++)
        {
            // One part of the strip
            indexStorage[offset++] = (GLuint) ((y * width) + x);
            indexStorage[offset++] = (GLuint) (((y + 1) * width) + x);
        }

        if (y <  height - 2)   // Degenerate end: repeat last vertex
            indexStorage[offset++] = (GLuint) (((y + 1) * width) + (width - 1));

    }

    // generate index array: set up for triangle strips with degeneraret tris linking strips

    glGenVertexArrays(1, &vaoTerrain); CE();
    glBindVertexArray(vaoTerrain); CE();

    // set up vertex buffer an copy in data
    glGenBuffers(1, &vboTerrain); CE();
    glBindBuffer(GL_ARRAY_BUFFER, vboTerrain); CE();
    glBufferData(GL_ARRAY_BUFFER, 5*sizeof(GLfloat)*width*height, vertexStorage, GL_STATIC_DRAW); CE();

    // enable position attribute
    glEnableVertexAttribArray(0); CE();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(0)); CE();
    // enable texture coord attribute
    const int sz = 3*sizeof(GLfloat);
    glEnableVertexAttribArray(1); CE();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(sz) ); CE();

    // set up index buffer
    glGenBuffers(1, &iboTerrain);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboTerrain); CE();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*indexSize, indexStorage, GL_STATIC_DRAW); CE();

    // unbind everything and clean up

    delete [] vertexStorage;
    delete [] indexStorage;

    glBindBuffer(GL_ARRAY_BUFFER, 0); CE();
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); CE();
    glBindVertexArray(0); CE();

    //set up screen quad for screen rendering
    glGenVertexArrays(1, &vaoScreenQuad); CE();
    glBindVertexArray(vaoScreenQuad); CE();

    glGenBuffers(1, &vboScreenQuad); CE();
    glBindBuffer(GL_ARRAY_BUFFER, vboScreenQuad); CE();
    glBufferData(GL_ARRAY_BUFFER, sizeof(screenQuad), screenQuad, GL_STATIC_DRAW); CE();

    // enable position attribute
    //glEnableVertexAttribArray(0); CE();
    //glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)(0)); CE();
    // enable position attribute
    glEnableVertexAttribArray(0); CE();
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(0)); CE();
    // enable texture coord attribute
    const int sz2 = 2*sizeof(GLfloat);
    glEnableVertexAttribArray(1); CE();
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), (void*)(sz2) ); CE();

    glBindBuffer(GL_ARRAY_BUFFER, 0); CE();
    glBindVertexArray(0); CE();


    // create an FBO for normal map rendering

    glGenFramebuffers(1, &fboNormalMap); CE();
    glBindFramebuffer(GL_FRAMEBUFFER, fboNormalMap); CE();
    // create texture target for normal map computation


    glActiveTexture(normalMapTexUnit); // normal map is bound to this TIU
    glGenTextures(1, &normalTexture); CE();
    glBindTexture(GL_TEXTURE_2D, normalTexture); CE();
    // set up texture state.

    // Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F, width, height, 0,GL_RGBA, GL_FLOAT, 0); CE();

    // no filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    // configure FBO

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, normalTexture, 0); CE();

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DrawBuffers);  CE(); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "Normal map FBO initialisation failed\n";
        return false;
    }

    // unbind FBO

    glBindFramebuffer(GL_FRAMEBUFFER,  0); CE();
    return true;
}

void TRenderer::generateNormalTexture(void)
{
    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f ); CE();

    glDisable(GL_DEPTH_TEST); CE();
    glDisable(GL_CULL_FACE); CE();


    glClear(GL_COLOR_BUFFER_BIT); CE();

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport); // save current viewport

    // reset current viewport
    glViewport(0,0,width, height); CE();

    std::string shaderName = "normalShader";

    GLuint programID = (*shaders[shaderName]).getProgramID();
    glUseProgram(programID); CE();


    GLfloat imgDims[2] = {float(width), float(height)};
    GLuint locDims = glGetUniformLocation(programID, "imgSize");  CE();
    glUniform2fv(locDims, 1, imgDims); CE();

    GLuint textur = glGetUniformLocation(programID, "htMap");  CE();
    glUniform1i(textur, (GLint)(htmapTexUnit - GL_TEXTURE0)); CE(); // assumes heightmap texture is bound to this TIU

    // pass in scale
    GLfloat terDims[2] = {float(scalex), float(scaley)};
    GLuint scDims = glGetUniformLocation(programID, "scale");  CE();
    glUniform2fv(scDims, 1, terDims); CE();


    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fboNormalMap); CE();

    // set shader program to normal map gen

    glBindVertexArray(vaoScreenQuad); CE();

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);  CE();

    // unbind everthing

    glBindFramebuffer(GL_FRAMEBUFFER, 0);  CE();
    glBindVertexArray(0);  CE();
    glUseProgram(0);  CE();

    // reset viewport
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
}

 // ******* Radiance Scaling setup **************
  // rebuild buffer for viewport of vWd X vHt size

  bool TRenderer::initRadianceScalingBuffers(int vWd, int vHt)
  {
    // get viewport size - every time this changes we'll have to rebuild the textures
    _w = vWd;
    _h = vHt;

    // Create & Configure FBO: final composite

    glGenFramebuffers(1, &fboRSOutput); CE();
    // std::cout << "Framebuffer2 ID = " << fboRSOutput << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, fboRSOutput); CE();

    // set up final compositon buffer for radiance scaling

    glActiveTexture(rsDestTexUnit); CE();
    glGenTextures(1, &destTexture); CE();
    glBindTexture(GL_TEXTURE_2D, destTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, _w, _h, 0,GL_RGBA, GL_UNSIGNED_BYTE, 0); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); CE();

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, destTexture, 0); CE();

    GLenum DestBuffer[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, DestBuffer);  CE();

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "RS FBO (output) - initialisation: failed\n";
        return false;
    }

    // unbind FBO

    glBindFramebuffer(GL_FRAMEBUFFER,  0); CE();

    // ****************************************************************************************************

    // create an FBO for rendering output
    glGenFramebuffers(1, &fboRadScaling); CE();
    glBindFramebuffer(GL_FRAMEBUFFER, fboRadScaling); CE();
    // std::cout << "Framebuffer1 ID = " << fboRadScaling << std::endl;

    // ****** depth texture:: ********************

    glGenTextures(1, &depthTexture); CE();
    glBindTexture(GL_TEXTURE_2D, depthTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT24, _w, _h, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // ****** gradient texture:: *******************

    glActiveTexture(rsGradTexUnit);
    glGenTextures(1, &gradTexture); CE();
    glBindTexture(GL_TEXTURE_2D, gradTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F, _w, _h, 0,GL_RGBA, GL_FLOAT, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // ****** screenspace normal texture:: ********************

    glActiveTexture(rsNormTexUnit);
    glGenTextures(1, &normTexture); CE();
    glBindTexture(GL_TEXTURE_2D, normTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F, _w, _h, 0,GL_RGBA, GL_FLOAT, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   // ****** colour texture:: ********************

    glActiveTexture(rsColTexUnit);
    glGenTextures(1, &colTexture); CE();
    glBindTexture(GL_TEXTURE_2D, colTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, _w, _h, 0,GL_RGBA, GL_UNSIGNED_BYTE, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // configure FBO: intermediate

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0); CE();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, gradTexture, 0); CE();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, normTexture, 0); CE();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, colTexture, 0); CE();

    // Set the list of draw buffers.
    GLenum DrawBuffers[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
    glDrawBuffers(3, DrawBuffers);  CE(); // "3" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "RS FBO initialisation: failed\n";
        return false;
    }

    // unbind FBO

    glBindFramebuffer(GL_FRAMEBUFFER,  0); CE();

    // ******************************************************************************************************

    // create FBO for manipulator transparency overlay; requires depth buffer and colour attachment

    glGenFramebuffers(1, &fboManipLayer); CE();
    glBindFramebuffer(GL_FRAMEBUFFER, fboManipLayer); CE();
    // std::cout << "FramebufferN ID = " << fboManipLayer << std::endl;

    // ****** manipulator depth texture:: ********************


    glActiveTexture(manipTranspTexUnit);

    glGenTextures(1, &manipDepthTexture); CE();
    glBindTexture(GL_TEXTURE_2D, manipDepthTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT24, _w, _h, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

   // ****** manipulator colour texture:: ********************

    glGenTextures(1, &manipTranspTexture); CE();
    glBindTexture(GL_TEXTURE_2D, manipTranspTexture); CE();
    // - Give an empty image to OpenGL ( the last "0" )
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, _w, _h, 0,GL_RGBA, GL_UNSIGNED_BYTE, 0); CE();
    // - linear filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); CE();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); CE();
    // - deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, manipDepthTexture, 0); CE();
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, manipTranspTexture, 0); CE();

    // Set the list of draw buffers.
    GLenum mDrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, mDrawBuffers);  CE(); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "manip FBO initialisation: failed\n";
        return false;
    }

    // unbind FBO
    glBindFramebuffer(GL_FRAMEBUFFER,  0); CE();

    return true;
  }

  // update radiance scaling FBO textures/attachments if viewport has changed size
  // vwd and vht are current viewport width and height, resp.
  // render conext

  void TRenderer::updateRadianceScalingBuffers(int vwd, int vht)
  {
    if (shadModel != RADIANCE_SCALING)
      return;

    if (_w == 0 || _h == 0 || vwd != _w || vht != _h)
      {
        // std::cerr << "Delete old rad scaling buffer\n";

        // clean old Radiance scaling FBO data if required
        deleteFBOrscalingBuffers();

        //std::cerr << "Calling Rad scaling...\n";
        initRadianceScalingBuffers(vwd, vht);
        //std::cerr << "Done\n";
      }
  }

  // create data for instance
void TRenderer::initInstanceData(void)
{
      vboTerrain = iboTerrain = vaoTerrain = 0;
      for (int i = 0; i < 5; i++)
      {
          vboWalls[i] = 0;
          iboWalls[i] = 0;
          vaoWalls[i] = 0;
      }

      normalTexture = fboNormalMap = vaoScreenQuad = 0;
      depthTexture = gradTexture = normTexture = colTexture = destTexture = 0;
      manipDepthTexture = manipTranspTexture = 0;
      typeMapTexture = 0;
      constraintTexture = 0;
      heightmapTexture = 0;
      decalTexture = 0;
      vboScreenQuad = 0;
      fboRadScaling = 0;
      fboRSOutput = 0;
      fboManipLayer = 0;
      typeBuffer = NULL;

      // texture units reserved for rendering:
      htmapTexUnit = GL_TEXTURE0;
      normalMapTexUnit = GL_TEXTURE1;
      rsGradTexUnit = GL_TEXTURE2;
      rsNormTexUnit = GL_TEXTURE3;
      rsColTexUnit = GL_TEXTURE4;
      rsDestTexUnit = GL_TEXTURE5;
      typemapTexUnit = GL_TEXTURE6;
      decalTexUnit = GL_TEXTURE7;
      constraintTexUnit = GL_TEXTURE8;
      manipTranspTexUnit = GL_TEXTURE9;

      width = height = 0;
      scalex = scaley = 0.0f;
      indexSize = 0;
      _w = _h = 0;
      terrainBase = 1000000.0; // +infinity (well kind of ...)

     // default colours

      terMatDiffuse = glm::vec4(0.7f, 0.6f, 0.5f, 1.0f); // colour of terrain
      terMatSpec = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
      terMatAmbient = glm::vec4(0.3f, 0.3f, 0.3f, 1.0f);
      lightDiffuseColour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f); // colour of light
      lightSpecColour = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
      lightAmbientColour = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
      shinySpec = 5.0f; // specular power

      // default colour: radiance scaling
      terSurfColour = glm::vec4(0.7f, 0.6f, 0.5f, 1.0f);

      // default camera

      viewMx = glm::lookAt(glm::vec3(2.0f, 3.0f, 3.0f), glm::vec3(0.0f,0.0f,0.0f), glm::vec3(0.0, 1.0f, 0.0f));

      MVmx = viewMx; //  glm::mat4(1.0f); Model mx = Id

      normalMatrix = glm::transpose(glm::inverse(glm::mat3(MVmx))); // glm::mat3(1.0f);
      //normalMatrix = glm::mat3(1.0f);
      if(shadModel == SUN)
         projMx = glm::ortho(-500.0f, 500.0f, -500.0f, 500.0f, -10000.0f, 10000.0f);
      else
         projMx = glm::frustum(-8.0f*ACTUAL_ASPECT, 8.0f*ACTUAL_ASPECT, -8.0f, 8.0f, 50.0f, 100000.0f);

      MVP = projMx  * MVmx;
}

TRenderer::TRenderer(QGLWidget *drawTo, const std::string& dir)
{
  canvas = drawTo;
  shaderDir = dir;

  shadersReady = false;
  contours = true; // contours on by default
  gridlines = true; //draw gridlines by default
  contoursWall = true; //draw contours on side walls
  gridlinesWall = true; // draw gridlines on side walls
  terrainTypeTexture = false; // turn off terrain type texture by default
  constraintTypeTexture = false; // turn off constraint type texture by default
  shadModel = RADIANCE_SCALING; // RADIANCE_SCALING; // BASIC
  manipulatorTextures = false; // do not texture manipulators by default
  drawOutOfBounds = false; // shade out of bound heights
  drawHiddenManipulators = false; // do/do not draw hidden manipulators

  terrainBasePad = 50.0; // 50 metre padding from lowest point to avoid flat terrains having tiny side walls

  // lights
  pointLight = glm::vec4(0.0f, 0.0f, -1.0f, 1.0f);
  directionalLight[0] = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
  directionalLight[1] = glm::vec4(0.5f, 0.5f, 0.0f, 0.0f);

  // values for drawing contours/gridlines
  gridColFactor = 0.7f; // 70% of base colour
  gridXsep = 2.0f; // separation of griod lines, depends on how input data is scaled
  gridZsep = 2.0f;
  gridThickness = 1.0f;
  contourSep = 70.0f; // separation (Y direction) depends on how input data is normalized
  contourThickness = 0.5f;
  contourColFactor = 1.3f; // 130% of base colour

  outOfBoundsColour = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // red shading for our of bounds colour
  outOfBoundsWeight = 0.2f;
  outOfBoundsMean = 0.0f;
  outOfBoundsOffset = 0.0f;

  manipAlpha = 0.4f; // manipulator blend factor for hidden manipulators

  // radiance scaling parameters
  RSinvertCurvature = false;
  RStransition = 0.2f;
  RSenhance = 0.5f;
  initInstanceData();
}

void TRenderer::deleteTerrainOpenGLbuffers(void)
{
    // delete old VAO etc
    if (vboTerrain != 0) glDeleteBuffers(1, &vboTerrain);  CE();
    if (iboTerrain != 0) glDeleteBuffers(1, &iboTerrain);  CE();
    if (vaoTerrain != 0) glDeleteVertexArrays(1, &vaoTerrain);  CE();
    if (normalTexture != 0) glDeleteTextures(1, &normalTexture);  CE();
    if (fboNormalMap != 0) glDeleteFramebuffers(1, &fboNormalMap);  CE();
    if (vaoScreenQuad != 0) glDeleteVertexArrays(1, &vaoScreenQuad);  CE();
    if (vboScreenQuad != 0) glDeleteBuffers(1, &vboScreenQuad);  CE();

    for (int i = 0; i < 5; i++)
    {
        if (vboWalls[i] != 0) glDeleteBuffers(1, &vboWalls[i]); CE();
        if (iboWalls[i] != 0) glDeleteBuffers(1, &iboWalls[i]); CE();
        if (vaoWalls[i] != 0) glDeleteVertexArrays(1, &vaoWalls[i]); CE();
    }
}

  void TRenderer::deleteFBOrscalingBuffers(void)
  {
   // Radiance scaling FBO data

    if (depthTexture !=0) glDeleteTextures(1, &depthTexture);  CE();
    if (gradTexture !=0) glDeleteTextures(1, &gradTexture);  CE();
    if (normTexture !=0) glDeleteTextures(1, &normTexture);  CE();
    if (colTexture !=0) glDeleteTextures(1, &colTexture);  CE();
    if (destTexture != 0)  glDeleteTextures(1, &destTexture);  CE();
    if (manipTranspTexture != 0)  glDeleteTextures(1, &manipTranspTexture);  CE();
    if (manipDepthTexture != 0)  glDeleteTextures(1, &manipDepthTexture);  CE();

    if (fboRadScaling != 0) glDeleteFramebuffers(1, &fboRadScaling);  CE();
    if (fboRSOutput != 0) glDeleteFramebuffers(1, &fboRSOutput); CE();
    if (fboManipLayer != 0) glDeleteFramebuffers(1, &fboManipLayer); CE();
  }

void TRenderer::destroyInstanceData(void)
{
    if (typeBuffer != NULL) delete [] typeBuffer;

    if (typeMapTexture != 0)  glDeleteTextures(1, &typeMapTexture);  CE();
    if (heightmapTexture != 0)	glDeleteTextures(1, &heightmapTexture);  CE();
    if (constraintTexture != 0) glDeleteTextures(1, &constraintTexture);  CE();

    deleteTerrainOpenGLbuffers();
    deleteFBOrscalingBuffers();
}

TRenderer::~TRenderer()
{
  if (decalTexture != 0) glDeleteTextures(1, &decalTexture);
  // delete shaders
  std::map<std::string, shaderProgram*>::iterator it = shaders.begin();
  while (it != shaders.end() )
  {       delete (*it).second; it++; }

  destroyInstanceData();
}

// load in a new heightfield with accompanying terrain type map.
void TRenderer::loadTerrainData(const float* data, int wd, int ht, float scx, float scy,
                               TypeMap* paintMap, TypeMap* constraintMap)
{
    if (!shadersReady)
        std::cerr << "Shaders not yet compiled!!!!\n";

    // clean up existing  data:
    destroyInstanceData();

    // create new data for this instance
    initInstanceData();

    // create height map textures and geometry
    updateHeightMap(wd, ht, scx, scy, data, true);

    // set up terrain type colour map (if provided)
    if (paintMap) updateTypeMapTexture(paintMap, PAINT);
    if (constraintMap) updateTypeMapTexture(constraintMap,  CONSTRAINT);
}

  // the typ emap can be either a PAINT map (for painting on types) or an CONSTRAINT map
  // for painting on freezing or other constraints.
  void TRenderer::updateTypeMapTexture(TypeMap* tmap, typeMapInfo tinfo, bool force)
{
  if (tmap == NULL)
    {
      // std::cerr << "TRenderer::updateTypeMapTexture - tmap is NULL\n";
      return;
    }

  MemMap<int> *tm =  tmap->getMap();
  if (tm == NULL)
    {
      // std::cerr << "TRenderer::updateTypeMapTexture - Map buffer is NULL, map ignored\n";
      return;
    }

  Region R = tmap->getRegion();
  int    wd = tm->width();
  int    ht = tm->height();
  const int* ptr = (int*)tm->get();
  std::vector<GLfloat *> *ct = tmap->getColourTable();
  int Rwidth  = R.x1 - R.x0; // region bounds from [R.x0, R.x1)
  int Rheight = R.y1 - R.y0;
  int xoff = R.x0;
  int yoff = R.y0;
  int index;

  bool initTexture = false;

  // width and height of terrain must match the width and height of type map buffer

  // assert(wd == width && ht == height);

  // texturing ops affect this texture unit

  GLenum texUnit = GL_TEXTURE0;
  GLuint *texId = NULL;

  switch (tinfo)
    {
    case PAINT:
      texUnit = typemapTexUnit;
      texId = &typeMapTexture;
     break;
    case CONSTRAINT:
      texUnit = constraintTexUnit;
      texId = &constraintTexture;
     break;
    default: std::cerr << "TRenderer::updateTypeMapTexture - Illegal type map mode!\n"; return;
    }

  glActiveTexture(texUnit); CE();

    // if grid dimensions have changed:
    if(*texId != 0 && force)
    {
        /*
        cerr << "tmap = " << wd << ", " << ht << endl;
        cerr << "terrain = " << width << ", " << height << endl;
        cerr << "updating texture width and height" << endl;*/
        glDeleteTextures(1, texId);  CE();
        *texId = 0;
    }

  if (*texId == 0) // internal typeMap texture not allocated yet
    {
      // allocate subbuffer: this is used to avoid mem allocation each time this method is called (frequently)

      //std::cout << "Region: " << R.x0 << "," << R.y0 << "," << R.x1 << "," << R.y1 << "\n";
      if(typeBuffer != NULL)
          delete [] typeBuffer;
      typeBuffer = new GLfloat [width*height*4];
      if (typeBuffer == NULL)
        {
          std::cerr << "TRenderer::updateTypeMapTexture - typeBuffer could not be allocated\n";
          return;
        }

      glGenTextures(1, texId); CE();
      glBindTexture(GL_TEXTURE_2D, *texId); CE();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); CE();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); CE();
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      initTexture = true;
      xoff = yoff = 0;
      Rwidth = wd;
      Rheight = ht; // copy in whole buffer region, not sub-window
    }
  else // need to sub in region - bind texture first
    {
      glBindTexture(GL_TEXTURE_2D, *texId); CE();
    }

  // build colour buffer for texture:

  int cnt = 0;
  int colIdx;
  //std::cout << "Width, Height = " << Rwidth << "," << Rheight << std::endl;
  for (int i = 0; i < Rheight; i++)
    for (int j = 0; j < Rwidth; j++)
    {
      index = (yoff + i)*wd + (xoff + j);
        colIdx = ptr[index];
      if (colIdx == 0 && tinfo != CONSTRAINT) // background should be used - use terMatDiffuse; for CONSTRAINT, ignore
        {
          //std::cout << "BG col at [" << i << "," << j << "]\n";
          //std::cout << "Ter Colour=[" << terMatDiffuse[0] << "," <<
          //  terMatDiffuse[1] << "," << terMatDiffuse[2] << "," << terMatDiffuse[3] << "]\n";
          typeBuffer[4*cnt]   = terMatDiffuse[0];
          typeBuffer[4*cnt+1] = terMatDiffuse[1];
          typeBuffer[4*cnt+2] = terMatDiffuse[2];
          typeBuffer[4*cnt+3] = terMatDiffuse[3];
        }
      else
        {
          typeBuffer[4*cnt]   = (*ct)[ colIdx ][0];
          typeBuffer[4*cnt+1] = (*ct)[ colIdx ][1];
          typeBuffer[4*cnt+2] = (*ct)[ colIdx ][2];
          typeBuffer[4*cnt+3] = (*ct)[ colIdx ][3];
          //std::cout << "Map Colour=[" << typeBuffer[4*cnt] << "," <<
          // typeBuffer[4*cnt + 1] << "," << typeBuffer[4*cnt + 2] << "," << typeBuffer[4*cnt + 3] << "]\n";
        }
      cnt++;
    }
  //std::cout << "Buffer built...\n";
  if (initTexture)
    {
      //std::cout << "Overlay texture created at full resolution\n";
       glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA8, Rwidth, Rheight, 0,GL_RGBA, GL_FLOAT, typeBuffer); CE();
    }
  else
    {
      //std::cout << "Overlay textured sub'd\n";
      glTexSubImage2D(GL_TEXTURE_2D, 0, xoff, yoff, Rwidth, Rheight, GL_RGBA, GL_FLOAT, typeBuffer); CE();
    }
  //std::cout << "Overlay created\n";
}

// initialise renderer - compile shaders etc This must be called before any other operations on the
// renderer.

void TRenderer::initShaders(void)
{
   // set up shaders for loading and compilation

  if (shadersReady) return; // already compiled

      shaderProgram *s;

      s = new shaderProgram();
      s->setShaderSources(std::string("basic.frag"), std::string("basic.vert"));
      shaders["basicShader"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("genNormal.frag"), std::string("genNormal.vert"));
      shaders["normalShader"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("phong.frag"), std::string("phong.vert"));
      shaders["phong"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("rad_scaling_pass1.frag"), std::string("rad_scaling_pass1.vert"));
      shaders["rscale1"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("rad_scaling_pass2.frag"), std::string("rad_scaling_pass2.vert"));
      shaders["rscale2"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("phongRS.frag"), std::string("phongRS.vert"));
      shaders["phongRS"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("phongRSmanip.frag"), std::string("phongRSmanip.vert"));
      shaders["phongRSmanip"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("sun.frag"), std::string("sun.vert"));
      shaders["sunShader"] = s;

      s = new shaderProgram();
      s->setShaderSources(std::string("canopy.frag"), std::string("canopy.vert"));
      shaders["canopyShader"] = s;

    std::cout << "Compiling shaders...\n";
    std::map<std::string, shaderProgram*>::iterator it = shaders.begin();
    while (it != shaders.end() )
    {
        std::cout << " -- shader: " << (*it).first << " -- ";
        (void)((*it).second)->compileAndLink();
        std::cout << "ID = " << ((*it).second)->getProgramID() << std::endl;
        it++;
    }

    shadersReady = true;

    std::cout << "done!\n";
}

void TRenderer::bindDecals(int width, int height, unsigned char * buffer)
{
    GLint maxtexunits;
    GLint boundtexunit;
    glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxtexunits);
    std::cout << "maximum texture units: " << maxtexunits << std::endl;

    // bind texture
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(decalTexUnit); CE();
    glGenTextures( 1, &decalTexture ); CE();
    glBindTexture( GL_TEXTURE_2D, decalTexture ); CE();
    // int off = 56 * width * 4 + 767 * 4;
    // cerr << "width = " << width << ", height = " << height << endl;
    // cerr << "decal buffer [0][0] = " << (int) buffer[0] << ", " << (int) buffer[1] << ", " << (int) buffer[2] << ", " << (int) buffer[3] << endl;
    // cerr << "decal buffer [767][56] = " << (int) buffer[off] << ", " << (int) buffer[off+1] << ", " << (int) buffer[off+2] << ", " << (int) buffer[off+3] << endl;

    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer); CE(); // ? is this actually working
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ); CE();
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ); CE();

    // deal with out of array access
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
}

void TRenderer::draw(View * view)
{
    if (!shadersReady) // not compiled!
    {
        std::cerr << "Shaders not built before draw() call - compiling...\n";
        initShaders();
    }

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    //std::cout << "Viewport chk - [" << viewport[0] << "," << viewport[1] << "," << viewport[3] << "," << viewport[3] << "]\n";

    // render at 2X resolution for later linear downsampling (basic anti-aliasing)
    updateRadianceScalingBuffers(2*viewport[2], 2*viewport[3]);

    // Set the clear color to white
    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f ); CE();

    glEnable(GL_DEPTH_TEST); CE();
    glDepthMask(GL_TRUE); CE();
    glDepthFunc(GL_LEQUAL); CE();
    glDepthRange(0.0f, 1.0f); CE();
    glEnable(GL_CULL_FACE); CE();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CE();

    //glEnable(GL_DEPTH_CLAMP);

    // configure shading params.


    // ************************* Terrain Setup & render code *****************************************
    normalMatrix = view->getNormalMtx();
    MVP = view->getMatrix();
    MVmx = view->getViewMtx();
    projMx = view->getProjMtx();

    std::string shaderName;
    if (shadModel == BASIC) // basic Phong
      shaderName = "basicShader";
    else if(shadModel == RADIANCE_SCALING)
      shaderName = "rscale1";
    else // sun shading
      shaderName = "sunShader";

    GLuint programID = (*shaders[shaderName]).getProgramID();
    glUseProgram(programID); CE();

    glUniformMatrix3fv(glGetUniformLocation(programID, "normMx"), 1, GL_FALSE, glm::value_ptr(normalMatrix)); CE();
    glUniformMatrix4fv(glGetUniformLocation(programID, "MV"), 1, GL_FALSE, glm::value_ptr(MVmx)); CE();
    glUniformMatrix4fv(glGetUniformLocation(programID, "MVproj"), 1, GL_FALSE, glm::value_ptr(MVP)); CE();

    glm::vec4 lightPos = MVmx * pointLight; // map light pos into camera space

    // configure texturing: region texture overlay
    if (terrainTypeTexture == false)
     {
        glUniform1i(glGetUniformLocation(programID, "useRegionTexture"), 0); CE(); //  turn texture off
     }
    else if (typeMapTexture != 0)
      {
        glUniform1i(glGetUniformLocation(programID, "useRegionTexture"), 1); CE(); // draw region texture
        GLuint rtex = glGetUniformLocation(programID, "overlayTexture"); CE();
        glUniform1i(rtex, (GLint)(typemapTexUnit - GL_TEXTURE0));  CE(); // texture unit for region overlay
      }
    else
      {
        glUniform1i(glGetUniformLocation(programID, "useRegionTexture"), 0); CE(); // region texture not yet defined
      }

    // configure texturing: constraint texture overlay
    if (constraintTypeTexture == false)
     {
        glUniform1i(glGetUniformLocation(programID, "useConstraintTexture"), 0); CE(); //  turn texture off
     }
    else if (constraintTexture != 0)
      {
        glUniform1i(glGetUniformLocation(programID, "useConstraintTexture"), 1); CE(); // draw constraint texture
        GLuint rtex = glGetUniformLocation(programID, "constraintTexture"); CE();
        glUniform1i(rtex, (GLint)(constraintTexUnit - GL_TEXTURE0));  CE(); // texture unit for constraint overlay
      }
    else
      {
        glUniform1i(glGetUniformLocation(programID, "useConstraintTexture"), 0); CE(); // region texture not yet defined
      }

    // contouring and grids
    glUniform1i(glGetUniformLocation(programID, "drawContours"), (int)(contours ? 1 : 0) ); CE(); // draw contours
    glUniform1i(glGetUniformLocation(programID, "drawGridLines"), (int)(gridlines ? 1 : 0) ); CE(); // draw grdlines
    glUniform1i(glGetUniformLocation(programID, "drawWallContours"), (int)(contoursWall ? 1 : 0) ); CE(); // draw wall contours
    glUniform1i(glGetUniformLocation(programID, "drawWallGridLines"), (int)(gridlinesWall ? 1 : 0) ); CE(); // draw wall contours

    // change shading for out of bounds heights
    glUniform1i(glGetUniformLocation(programID, "drawOutOfBounds"), (int)(drawOutOfBounds ? 1 : 0) ); CE(); // draw wall contours
    glUniform1f(glGetUniformLocation(programID, "outBoundsBlend"), outOfBoundsWeight); CE();
    glUniform1f(glGetUniformLocation(programID, "outBoundsMax"), outOfBoundsMean + outOfBoundsOffset); CE();
    glUniform1f(glGetUniformLocation(programID, "outBoundsMin"), outOfBoundsMean - outOfBoundsOffset); CE();
    glUniform4fv(glGetUniformLocation(programID, "outBoundsCol"), 1, glm::value_ptr(outOfBoundsColour) ); CE();

    // set contouring params (not implemented in BASIC at the moment...)
    glUniform1f(glGetUniformLocation(programID, "gridX"),  gridXsep); CE();
    glUniform1f(glGetUniformLocation(programID, "gridZ"),  gridZsep); CE();
    glUniform1f(glGetUniformLocation(programID, "gridColFactor"),  gridColFactor); CE();
    glUniform1f(glGetUniformLocation(programID, "gridThickness"),  gridThickness); CE();
    glUniform1f(glGetUniformLocation(programID, "contourSep"),  contourSep); CE();
    glUniform1f(glGetUniformLocation(programID, "contourColFactor"),  contourColFactor); CE();
    glUniform1f(glGetUniformLocation(programID, "contourThickness"),  contourThickness); CE();

    // pass light and colours to shader
    if (shadModel == BASIC)
      {
        glUniform4fv(glGetUniformLocation(programID, "lightpos"), 1, glm::value_ptr(lightPos)); CE();

        // set colours
        glUniform4fv(glGetUniformLocation(programID, "matDiffuse"), 1, glm::value_ptr(terMatDiffuse) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "matAmbient"), 1, glm::value_ptr(terMatAmbient) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "matSpec"), 1, glm::value_ptr(terMatSpec) ); CE();

        glUniform4fv(glGetUniformLocation(programID, "diffuseCol"), 1, glm::value_ptr(lightDiffuseColour) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "ambientCol"), 1, glm::value_ptr(lightAmbientColour) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "specularCol"), 1, glm::value_ptr(lightSpecColour) ); CE();
        glUniform1f(glGetUniformLocation(programID, "shiny"), shinySpec); CE();
      }
    else if(shadModel == RADIANCE_SCALING) // radiance scaling
      {
        // map side wall lights into camera space; lights at corners of terrain, moved along diagonal
        glm::vec4 LP1 = MVmx * glm::vec4(2.0*scalex, 0.5, 2.0*scaley, 1.0);
        glm::vec4 LP2 = MVmx * glm::vec4(-scalex, 0.5, -scaley, 1.0);
        glUniform4fv(glGetUniformLocation(programID, "ptLightPos1"), 1, glm::value_ptr(LP1) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "ptLightPos2"), 1, glm::value_ptr(LP2) ); CE();
        glUniform4fv(glGetUniformLocation(programID, "surfColour"), 1,  glm::value_ptr(terSurfColour) ); CE();
      }
    else if(shadModel == SUN) // sun simulator
    {
        glUniform1f(glGetUniformLocation(programID, "terdim"), (float) (width)); CE();
    }

    //pass height and normal map to shader
    GLuint textur = glGetUniformLocation(programID, "htMap"); CE();
    glUniform1i(textur, (GLint)(htmapTexUnit - GL_TEXTURE0));  CE(); // assumes texture unit 0 is bound to heightmap texture

    GLuint textur2 = glGetUniformLocation(programID, "normalMap"); CE();
    glUniform1i(textur2, (GLint)(normalMapTexUnit - GL_TEXTURE0)); CE(); // assumes texture unit 1 is bound to normal map texture

    // draw terrain:

    if (shadModel == RADIANCE_SCALING)
      {
        GLfloat depthClear = 1.0f;
        GLfloat colClear[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        GLfloat manipLayerClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        GLfloat gradClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        GLfloat normClear[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // clear manipulator transparency FBO
        glBindFramebuffer(GL_FRAMEBUFFER, fboManipLayer); CE();
        glViewport(0,0,_w, _h); CE(); // draw into entire frame
        // clear frame buffer depth/textures
        glClearBufferfv(GL_DEPTH, 0, &depthClear);
        glClearBufferfv(GL_COLOR, 0, manipLayerClear);

        // Render to RS FBO; clear buffers first
        glBindFramebuffer(GL_FRAMEBUFFER, fboRadScaling); CE();
        glViewport(0,0,_w, _h); CE(); // draw into entire frame
        // clear frame buffer depth/textures
        glClearBufferfv(GL_DEPTH, 0, &depthClear);
        glClearBufferfv(GL_COLOR, 0, gradClear); // may be able to ignore this one - gradTexture?
        glClearBufferfv(GL_COLOR, 1, normClear); // this has to be set to (0,0,0) for RS shader to work
        glClearBufferfv(GL_COLOR, 2, colClear);
      }
    else
      {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CE();
      }

    glUniform1i(glGetUniformLocation(programID, "drawWalls"), 0); CE(); // do NOT draw walls
    glBindVertexArray(vaoTerrain); CE();

    glDrawElements(GL_TRIANGLE_STRIP, indexSize, GL_UNSIGNED_INT, 0); CE();

    // *** draw capping walls ***
    // --- move base up/down to avoid edits to surface which 'punch through' base

    glUniform1f(glGetUniformLocation(programID, "terrainBase"), terrainBase); CE();
    glUniform1f(glGetUniformLocation(programID, "terrainBasePad"), terrainBasePad); CE();
    glUniform1i(glGetUniformLocation(programID, "drawWalls"), 1); // draw walls - ignore normal map lookup
    GLuint loc = glGetUniformLocation(programID, "normalWall");

    // turn off region layer texturing for walls:
    glUniform1i(glGetUniformLocation(programID, "useRegionTexture"), 0); CE(); // always off for walls!

    for (int i = 0; i < 5; i++)
    {
        glUniform3fv(loc, 1, glm::value_ptr(normalWalls[i])); CE();
        glBindVertexArray(vaoWalls[i]); CE();
        glDrawElements(GL_TRIANGLE_STRIP, wallDrawEls[i], GL_UNSIGNED_INT, 0); CE();
    }

    // **************************** draw manipulators/constraints with phong **********************************

    if (shadModel == RADIANCE_SCALING)
    {
      programID = (*shaders["phongRS"]).getProgramID();
    }
    else if (shadModel == SUN)
    {
        programID = (*shaders["canopyShader"]).getProgramID();
    }
    else
    {
        programID = (*shaders["phong"]).getProgramID();
    }

    drawManipulators(programID);

    // draw second pass for manipulator transparency (not supported in BASIC mode)

    if (shadModel == RADIANCE_SCALING)
      {
        programID  = (*shaders["phongRSmanip"]).getProgramID();
        glBindFramebuffer(GL_FRAMEBUFFER, fboManipLayer); CE();
        glViewport(0,0,_w, _h); CE();
        drawManipulators(programID, true);
      }

    // reset frame buffer buffer etc

    if (shadModel == RADIANCE_SCALING)
      {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);  CE();
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]); // reset viewport to system setting
      }

// **************************** radiance scaling pass 2 ****************************************************

if (shadModel == RADIANCE_SCALING)
  {
    glDisable(GL_DEPTH_TEST); CE(); // not required for screen  aligned quad

    // Render to our composition framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fboRSOutput); CE();
    glViewport(0,0,_w, _h); // draw into entire frame

    shaderName = "rscale2";
    programID = (*shaders[shaderName]).getProgramID();
    //std::cout << "Shader ID (RScaling) = " << programID << std::endl;
    glUseProgram(programID); CE();

    //glClearColor( 0.5f, 0.0f, 0.0f, 1.0f ); CE();

    glClear(GL_COLOR_BUFFER_BIT); CE();

    // set image size to allow tex coordinate gen per rasterized quad fragment
    GLfloat imgDims[2] = {float(_w), float(_h)};
    GLuint locDims = glGetUniformLocation(programID, "imgSize");  CE();
    glUniform2fv(locDims, 1, imgDims); CE();

    // light is directional - specified in  ***** WORLD  space ******
    // map DIRECTIONAL light(s)  into camera space
    for (int i = 0; i < 2; i++)
      {
        std::ostringstream oss;
        oss << "lightPos" << i;
        glm::vec3 dlight = normalMatrix * glm::vec3(directionalLight[i][0], directionalLight[i][1], directionalLight[i][2]);
        lightPos = glm::vec4(dlight[0], dlight[1], dlight[2], 0.0f);
        glUniform4fv(glGetUniformLocation(programID, oss.str().c_str()), 1, glm::value_ptr(lightPos)); CE();
      }

    // use lambertian lighting and params for RS calculation
    GLuint dispmode = glGetUniformLocation(programID, "display"); CE();
    glUniform1i(dispmode, 0);  CE();
    GLuint RSenabled = glGetUniformLocation(programID, "enabled"); CE();
    glUniform1i(RSenabled, 1);  CE();
    GLuint invertCurvature = glGetUniformLocation(programID, "invert"); CE();
    glUniform1i(invertCurvature, (RSinvertCurvature ?  1: 0));  CE();
    GLuint enhancement = glGetUniformLocation(programID, "enhancement"); CE();
    glUniform1f(enhancement, RSenhance);  CE();
    GLuint transition = glGetUniformLocation(programID, "transition"); CE();
    glUniform1f(transition, RStransition);  CE();
    GLuint swidth = glGetUniformLocation(programID, "sw"); CE();
    glUniform1f(swidth, (float)(1.0/_w) );  CE();
    GLuint sheight = glGetUniformLocation(programID, "sh"); CE();
    glUniform1f(sheight, (float)(1.0/_h) );  CE();

    // configure FBO textures for render ops
    GLuint textur = glGetUniformLocation(programID, "grad"); CE();
    glUniform1i(textur, (GLuint)(rsGradTexUnit - GL_TEXTURE0) );  CE();
    textur = glGetUniformLocation(programID, "norm"); CE();
    glUniform1i(textur, (GLuint)(rsNormTexUnit - GL_TEXTURE0) );  CE();
    textur = glGetUniformLocation(programID, "colormap"); CE();
    glUniform1i(textur, (GLuint)(rsColTexUnit - GL_TEXTURE0) );  CE();
    textur = glGetUniformLocation(programID, "manipTTexture"); CE(); // manipulator transparency
    glUniform1i(textur, (GLuint)(manipTranspTexUnit - GL_TEXTURE0) );  CE();

    // draw screeen aligned quad to compose RS calculations
    glBindVertexArray(vaoScreenQuad); CE();
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);  CE();

    // unbind everthing

    glBindVertexArray(0);  CE();

    // bind draw buffer (system
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);  CE();
    // bind read buffer (fbo)
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fboRSOutput);  CE();
    // set draw buffer
    glDrawBuffer(GL_BACK);
    // blit to the default framebuffer/back_buffer
    glBlitFramebuffer(0, 0, _w, _h,
                viewport[0], viewport[1], viewport[0] + viewport[2], viewport[1] + viewport[3], GL_COLOR_BUFFER_BIT, GL_LINEAR);

    //std::cout << "Blit target: [" << viewport[0] << "," << viewport[1] << "," <<
    //viewport[0] + viewport[2] << "," << viewport[1] + viewport[3] << "]\n";

   // reset viewport to system setting
   glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }

  glUseProgram(0);  CE();
}

void TRenderer::drawSun(View * view, int renderPass)
{
    if (!shadersReady) // not compiled!
    {
        std::cerr << "Shaders not built before draw() call - compiling...\n";
        initShaders();
    }

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // render at 2X resolution for later linear downsampling (basic anti-aliasing)
    updateRadianceScalingBuffers(2*viewport[2], 2*viewport[3]);

    // Set the clear color to white
    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f ); CE();


    glEnable(GL_DEPTH_TEST); CE();
    glDepthMask(GL_TRUE); CE();
    glDepthFunc(GL_LEQUAL); CE();
    glDepthRange(0.0f, 1.0f); CE();
    glEnable(GL_CULL_FACE); CE();

    if(renderPass == 2) // enable blending
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CE();

    // configure shading params.


    // ************************* Terrain Setup & render code *****************************************
    normalMatrix = view->getNormalMtx();
    MVP = view->getMatrix();
    MVmx = view->getViewMtx();
    projMx = view->getProjMtx();

    std::string shaderName;
    shaderName = "sunShader";
    GLuint programID = (*shaders[shaderName]).getProgramID();
    glUseProgram(programID); CE();

    glUniform1i(glGetUniformLocation(programID, "drawCanopies"), renderPass-1); CE(); // whether or not to use indexed colours

    // PHASE 1: Draw Terrain
    glUniformMatrix3fv(glGetUniformLocation(programID, "normMx"), 1, GL_FALSE, glm::value_ptr(normalMatrix)); CE();
    glUniformMatrix4fv(glGetUniformLocation(programID, "MV"), 1, GL_FALSE, glm::value_ptr(MVmx)); CE();
    glUniformMatrix4fv(glGetUniformLocation(programID, "MVproj"), 1, GL_FALSE, glm::value_ptr(MVP)); CE();

    glUniform1f(glGetUniformLocation(programID, "terdim"), (float) (width)); CE();

    // pass height and normal map to shader
    GLuint textur = glGetUniformLocation(programID, "htMap"); CE();
    glUniform1i(textur, (GLint)(htmapTexUnit - GL_TEXTURE0));  CE(); // assumes texture unit 0 is bound to heightmap texture

    GLuint textur2 = glGetUniformLocation(programID, "normalMap"); CE();
    glUniform1i(textur2, (GLint)(normalMapTexUnit - GL_TEXTURE0)); CE(); // assumes texture unit 1 is bound to normal map texture

    // draw terrain:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); CE();
    glUniform1i(glGetUniformLocation(programID, "drawWalls"), 0); CE(); // do NOT draw walls


    glBindVertexArray(vaoTerrain); CE();
    glDrawElements(GL_TRIANGLE_STRIP, indexSize, GL_UNSIGNED_INT, 0); CE();

    // PHASE 2: Draw capping walls
    // --- move base up/down to avoid edits to surface which 'punch through' base

    glUniform1f(glGetUniformLocation(programID, "terrainBase"), terrainBase); CE();
    glUniform1f(glGetUniformLocation(programID, "terrainBasePad"), terrainBasePad); CE();
    glUniform1i(glGetUniformLocation(programID, "drawWalls"), 1); // draw walls - ignore normal map lookup
    GLuint loc = glGetUniformLocation(programID, "normalWall");

    // turn off region layer texturing for walls:
    glUniform1i(glGetUniformLocation(programID, "useRegionTexture"), 0); CE(); // always off for walls!

    for (int i = 0; i < 5; i++)
    {
        glUniform3fv(loc, 1, glm::value_ptr(normalWalls[i])); CE();
        glBindVertexArray(vaoWalls[i]); CE();
        glDrawElements(GL_TRIANGLE_STRIP, wallDrawEls[i], GL_UNSIGNED_INT, 0); CE();
    }

    // PHASE 3: Draw canopies
    // draw alpha blended canopies
    if(renderPass == 2)
    {
        // disable depth writes so that more than one canopy can block sunlight
        glDepthMask(GL_FALSE); CE();

        programID  = (*shaders["canopyShader"]).getProgramID();
        drawManipulators(programID);
        // revert to previous settings
        glDepthMask(GL_TRUE); CE();
    }

    glUseProgram(0);  CE();
}


// **************************** draw manipulators/constraints with phong **********************************

void TRenderer::drawManipulators(GLuint programID, bool drawToFB)
{
    glUseProgram(programID); CE();

    // use textured manipulators (decals)?
    if (shadModel == RADIANCE_SCALING)
    {
        glUniform1i(glGetUniformLocation(programID, "useTexturing"), (manipulatorTextures ? 1:0) ); CE();
        if (manipulatorTextures)
        {
            GLuint mtex = glGetUniformLocation(programID, "decalTexture"); CE();
            glUniform1i(mtex, (GLint)(decalTexUnit - GL_TEXTURE0));  CE(); // texture unit for manipulators
        }
    }

    for (int i = 0; i < (int)manipDrawCallData.size(); i++)
    {
        float alpha = 1.0f;
        if (drawToFB) // used when drawing frame for manipulator transparanecy blending
        {
            // draw transparent manipulator - current only
            if (shadModel == RADIANCE_SCALING)
            {
                if (manipDrawCallData[i].current && drawHiddenManipulators == true)
                    alpha = manipAlpha;
                else
                    alpha = 0.0f;
                glUniform1f(glGetUniformLocation(programID, "manipAlpha"), alpha ); CE();
            }
        }
      // we have something to draw...

      glUniformMatrix4fv(glGetUniformLocation(programID, "MV"), 1, GL_FALSE, glm::value_ptr(MVmx) ); CE();
      glUniformMatrix4fv(glGetUniformLocation(programID, "MVproj"), 1, GL_FALSE, glm::value_ptr(MVP) ); CE();
      glUniformMatrix3fv(glGetUniformLocation(programID, "normMx"), 1, GL_FALSE, glm::value_ptr(normalMatrix)); CE();
      //glUniformMatrix3fv(glGetUniformLocation(programID, "normMx"), 1, GL_FALSE, glm::value_ptr(glm::mat3(1.0f))); CE();

      glm::vec4 MatDiffuse = glm::vec4(manipDrawCallData[i].diffuse[0], manipDrawCallData[i].diffuse[1],
                manipDrawCallData[i].diffuse[2], manipDrawCallData[i].diffuse[3]); // diffuse colour of manipulator
      //cerr << "BASE COLOUR: " << manipDrawCallData[i].diffuse[0] << " " << manipDrawCallData[i].diffuse[1] << " " <<
      //                manipDrawCallData[i].diffuse[2] << " " << manipDrawCallData[i].diffuse[3] << endl;
      glm::vec4 MatAmbient = glm::vec4(manipDrawCallData[i].ambient[0], manipDrawCallData[i].ambient[1],
                manipDrawCallData[i].ambient[2], manipDrawCallData[i].ambient[3]); // ambient colour of manipulator
      glm::vec4 MatSpecular = glm::vec4(manipDrawCallData[i].specular[0], manipDrawCallData[i].specular[1],
                manipDrawCallData[i].specular[2], manipDrawCallData[i].specular[3]); // JG: ambient colour of manipulator
      glm::vec4 lightDiffuseColour = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f); // colour of light
      glm::vec4 lightAmbientColour = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);

      // set colours and light
      glUniform4fv(glGetUniformLocation(programID, "matDiffuse"), 1, glm::value_ptr(MatDiffuse) ); CE();
      glUniform4fv(glGetUniformLocation(programID, "matAmbient"), 1, glm::value_ptr(MatAmbient) ); CE();
      glUniform4fv(glGetUniformLocation(programID, "matSpec"), 1, glm::value_ptr(MatSpecular) ); CE();
      glUniform4fv(glGetUniformLocation(programID, "lightPos"), 1, glm::value_ptr(pointLight)); CE();
      glUniform4fv(glGetUniformLocation(programID, "diffuseCol"), 1, glm::value_ptr(lightDiffuseColour) ); CE();
      glUniform4fv(glGetUniformLocation(programID, "ambientCol"), 1, glm::value_ptr(lightAmbientColour) ); CE();
      glUniform4fv(glGetUniformLocation(programID, "specularCol"), 1, glm::value_ptr(lightSpecColour) ); CE();
      glUniform1f(glGetUniformLocation(programID, "shiny"), shinySpec); CE();

      glBindVertexArray(manipDrawCallData[i].VAO); CE();

      glDrawElementsInstanced(GL_TRIANGLES, manipDrawCallData[i].indexBufSize, GL_UNSIGNED_INT, (void*)(0), manipDrawCallData[i].numInstances); CE();
      //glDrawElements(GL_TRIANGLES, manipDrawCallData[i].indexBufSize, GL_UNSIGNED_INT, (void*)(0)); CE();
      glBindVertexArray(0); CE();
    }

    // unbind vao
    glBindVertexArray(0); CE();
}


} // end of namespace PMrender
