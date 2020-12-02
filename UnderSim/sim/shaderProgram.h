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



#ifndef SHADER_PROGRAM_H
#define SHADER_PROGRAM_H

#include "glheaders.h"

#include <string>
#include <common/source2cpp.h>

namespace PMrender
{
class shaderProgram
{
private:
        GLuint program_ID;
        GLuint frag_ID;
        GLuint vert_ID;
        bool shaderReady;
        bool fileInput; // input comes from file rather than strings
        std::string fragSrc, vertSrc;

    // private mehods
    GLenum compileProgram(GLenum target, GLchar* sourcecode, GLuint & shader);
    GLenum linkProgram(GLuint program);
public:
        // construct from string source
        shaderProgram(const std::string& frageSource, const std::string& vertSource);
        // construct from file names
        shaderProgram(const char * fragSourceFile, const char * vertSourceFile);
    shaderProgram(void)
    {
        program_ID = frag_ID = vert_ID = 0;
        shaderReady = false;
        fileInput = false;
        fragSrc ="";
        vertSrc = "";

    }
        ~shaderProgram(){}

    void setShaderSources(const std::string& frageSource, const std::string& vertSource);
    void setShaderSources(const char * fragSourceFile, const char * vertSourceFile);

        bool  compileAndLink(void); //compile and link shaders

        GLuint getProgramID(void) const { return program_ID; }
        bool initialised(void) const {return shaderReady; }
};

}


#endif
