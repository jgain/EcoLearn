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


#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <fstream>
#include <string>
//#include "glwidget.h"
//#include <QGLWidget>
//#include <QTextStream>
//#include <QResource>
//#include <QtGui>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include "shaderProgram.h"

#define FailGLError(X) {int err = (int)glGetError(); \
        if (err != GL_NO_ERROR) \
                {std::cerr << (X); std::cerr << " error " << err << std::endl; \
        const char* err_str = reinterpret_cast<const char *>(gluErrorString(err));\
        std::cerr << " - GL Error message: " << err_str << std::endl;\
        return err;} }

namespace PMrender
{

GLenum shaderProgram::compileProgram(GLenum target, GLchar* sourcecode, GLuint & shader)
{
        GLint   logLength = 0;
        GLint   compiled  = 0;

        if (sourcecode != 0)
        {            
        shader = glCreateShader(target);
                FailGLError("Failed to create fragment shader");
                glShaderSource(shader,1,(const GLchar **)&sourcecode,0);
                FailGLError("Failed glShaderSource")
                glCompileShader(shader);
                FailGLError("Failed glCompileShader")


                glGetShaderiv(shader,GL_COMPILE_STATUS,&compiled);
                glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&logLength);

        if (logLength > 1)
                {
            GLint       charsWritten;
                        GLchar *log = new char [logLength+128];
            glGetShaderInfoLog(shader, logLength, &charsWritten, log);
            std::cerr << "Compilation log: nchars=(" << logLength << "): "<< (char*)log << std::endl;
                        delete [] log;
                }

                if (compiled == 0)
                        FailGLError("shader could not compile")

        }
        return GL_NO_ERROR;
}

GLenum shaderProgram::linkProgram(GLuint program)
{
        GLint   logLength = 0;
        GLint linked = 0;

        glLinkProgram(program);
        FailGLError("Failed glLinkProgram")
        glGetProgramiv(program,GL_LINK_STATUS ,&linked);
        glGetProgramiv(program,GL_INFO_LOG_LENGTH,&logLength);

    if (logLength > 1)
        {
                GLint   charsWritten;
        GLchar *log = new char [logLength+128];

                glGetProgramInfoLog(program, logLength, &charsWritten, log);
        std::cerr << "Link GetProgramInfoLog: nchars=(" << charsWritten << "): " << (char*)log << std::endl;
                delete [] log;
        }

        if (linked == 0)
                FailGLError("shader did not link")

        return GL_NO_ERROR;
}

shaderProgram::shaderProgram(const std::string& fragSource, const std::string& vertSource)
{
        fragSrc = fragSource;
        vertSrc = vertSource;
        frag_ID = vert_ID = program_ID = 0;

        fileInput = false;
        shaderReady = false;

}

shaderProgram::shaderProgram(const char * fragSourceFile, const char * vertSourceFile)
{
        fragSrc = fragSourceFile;
        vertSrc = vertSourceFile;
        fileInput = true;

        frag_ID = vert_ID = program_ID = 0;

        shaderReady = false;
}

  // use Bruce's shader/kernel source code bake...
    void shaderProgram::setShaderSources(const std::string& fragSource, const std::string& vertSource)
    {
        if (shaderReady)
        {
            std::cerr << "setShaderSources: shader sources already set\n";
            return ;
        }

        fragSrc = getSource(fragSource); // fragSource;
        vertSrc = getSource(vertSource); //vertSource;
        frag_ID = vert_ID = program_ID = 0;

        fileInput = false;
        shaderReady = false;
    }


    void shaderProgram::setShaderSources(const char * fragSourceFile, const char * vertSourceFile)
    {
        if (shaderReady)
        {
            std::cerr << "setShaderSources: shader sources already set\n";
            return ;
        }

        fragSrc = fragSourceFile;
        vertSrc = vertSourceFile;
        fileInput = true;

        frag_ID = vert_ID = program_ID = 0;

        shaderReady = false;
    }

bool shaderProgram::compileAndLink(void)
{
        if (shaderReady) return true;

        if (fileInput)
        {
        std::ifstream vshader(vertSrc.c_str(), std::ios::binary),
                fshader(fragSrc.c_str(), std::ios::binary);

                if (!vshader)
                {
                        std::cerr << "could not open shader source: " << vertSrc << std::endl;
                        return false;
                }
                if (!fshader)
                {
            std::cerr << "could not open shader source: " << fragSrc << std::endl;
                        return false;
                }

        vertSrc.clear(); fragSrc.clear();

        std::string tempstr;

        while (std::getline(vshader, tempstr))
        {
            vertSrc += tempstr;
            vertSrc += "\n";
        }


        //std::cout << "vertSrc=\n" << vertSrc << std::endl;

        while (std::getline(fshader, tempstr))
        {
            fragSrc += tempstr;
            fragSrc += "\n";
        }


        // std::cout << "fragSrc=\n" << fragSrc << std::endl;
                vshader.close();
                fshader.close();
        }
    GLuint err = compileProgram(GL_VERTEX_SHADER, const_cast<GLchar*>(vertSrc.c_str()), vert_ID);
        if (0 != err)
        {
                std::cerr << "Vertex Shader could not compile\n";
                return false;
        }

    err = compileProgram(GL_FRAGMENT_SHADER, const_cast<GLchar*>(fragSrc.c_str()), frag_ID);
        if (0 != err)
        {
                std::cerr <<"Fragment Shader could not compile\n";
                return false;
        }

        program_ID = glCreateProgram();
        //std::cout << "Shader ID = " << program_ID << std::endl;
        glAttachShader(program_ID, vert_ID);
        glAttachShader(program_ID, frag_ID);

        err = linkProgram(program_ID);
        if (GL_NO_ERROR != err)
        {
                std::cerr << "Program could not link\n";
                return false;
        }

        // detach and delete shader objects, we don't need them anymore (NB: may cause issues with dodgy drivers)
        if (frag_ID != 0)
        {
                glDetachShader(program_ID, frag_ID);
                glDeleteShader(frag_ID);
                frag_ID = 0;
        }
        if (vert_ID != 0)
        {
                glDetachShader(program_ID, vert_ID);
                glDeleteShader(vert_ID);
                vert_ID = 0;
        }
    shaderReady = true;
    return true;
}
    // end namespace PMrender
}
