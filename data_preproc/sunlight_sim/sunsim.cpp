/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com)
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


#include "count_pixels.h"
#include "sunsim.h"
#include "extract_png.h"
#include "data_importer/lodepng.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#define AXIS_TILT 0.408407f

const float sunsim::_half_day_in_minutes = 720.0f;
const float sunsim::_axis_tilt = AXIS_TILT;
const float sunsim::_month_axis_tilt = AXIS_TILT / 3.0f;
const float sunsim::_quarter_day_in_minutes = sunsim::_half_day_in_minutes / 2.0f;

void gl_errcheck(GLenum code, int line, const char *file)
{
    if (code)
    {
        std::string str;
        switch (code)
        {
            case GL_INVALID_ENUM:
                str = "INVALID_ENUM";
                break;
            case (GL_INVALID_VALUE):
                str = "INVALID_VALUE";
                break;
            case (GL_INVALID_OPERATION):
                str = "INVALID_OPERATION";
                break;
            case (GL_INVALID_FRAMEBUFFER_OPERATION):
                str = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            case (GL_OUT_OF_MEMORY):
                str = "GL_OUT_OF_MEMORY";
                break;
            case (GL_STACK_OVERFLOW):
                str = "GL_STACK_OVERFLOW";
                break;
            case (GL_STACK_UNDERFLOW):
                str = "GL_STACK_UNDERFLOW";
                break;
            default:
                str = "UNKNOWN ERROR";
                break;
        }
        std::cout << "GL error " << str <<" at line " << line << ", file " << file << std::endl;
    }
}

#define GL_ERRCHECK(code) \
    gl_errcheck(code, __LINE__, __FILE__)

sunsim::sunsim(terrain &ter, bool render_to_framebuf, int render_width, int render_height)
    : ter(ter),
      visitmap(ter.get_height() * ter.get_width(), false),
      render_to_framebuf(render_to_framebuf),
      sunmap_monthly(12),
      ortho_width(3),
      ortho_height(3)
{
    int terh, terw;
    terh = ter.get_width();
    terw = ter.get_height();
    float hr, wr;
    hr = ((float)render_height) / terh;
    wr = ((float)render_width) / terw;
    float multratio = std::min(hr, wr);
    window_width = multratio * terw + 0.01f;
    window_height = multratio * terh + 0.01f;
    sunmap.setDim(ter.get_width(), ter.get_height());
    base_color[0] = 15;
    base_color[1] = base_color[2] = 0;
    assign_shader_sources();
    generate_rendering_data();
    init();
    cudamem = alloc_gpu_memory(window_width, window_height, ter.get_width(), ter.get_height());
}

sunsim::~sunsim()
{
    free_gpu_memory(cudamem);
    close_sdl();
}

int sunsim::get_index_from_color(std::array<short, 3> color)
{
    color[0] -= base_color[0];
    if (color[0] < 0)
    {
        color[0] += 256;
        color[1]--;
    }
    color[1] -= base_color[1];
    if (color[1] < 0)
    {
        color[1] += 256;
        color[2]--;
    }
    color[2] -= base_color[2];

    return color[0] + color[1] * 256 + color[2] * 256 * 256;
}

std::array<short, 3> sunsim::get_color_from_index(int idx)
{
    int red = idx % 256;
    idx -= red;
    idx /= 256;
    int green = idx % 256;
    idx -= green;
    idx /= 256;
    int blue = idx % 256;

    if (red + base_color[0] >= 256)
    {
        green++;
        red = red + base_color[0] - 256;
    }
    else
        red += base_color[0];
    if (green + base_color[1] >= 256)
    {
        blue++;
        green = green + base_color[1] - 256;
    }
    else
        green = green + base_color[1];
    blue += base_color[2];

    return {(short)red, (short)green, (short)blue};
}

void sunsim::init_sdl()
{
    SDL_Init(SDL_INIT_VIDEO);
    uint32_t sdl_hidden_flag;
    if (render_to_framebuf)
        sdl_hidden_flag = SDL_WINDOW_HIDDEN;
    else
        sdl_hidden_flag = 0;

    window = SDL_CreateWindow("Sunlight calculator", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, window_width, window_height, SDL_WINDOW_OPENGL | sdl_hidden_flag);
}

void sunsim::init_gl()
{
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_CreateContext(window);

    glewExperimental = GL_TRUE;

    GLenum glewInitResult = glewInit();
    if (glewInitResult != GLEW_OK)
    {
        std::cerr << "Coult not initialize glew" << std::endl;
    }
}

void sunsim::init()
{
    init_sdl();
    init_gl();
    compile_shaders();
    init_gl_rendering_objects();
    setup_test_window();
}

void sunsim::close_sdl()
{
    SDL_Quit();
}

void sunsim::assign_sunlight_shader_source()
{
    sunlight_vert_shader_source =
        "#version 430 \n \
         \
        layout(location = 0) in vec4 vertex; \
        layout(location = 1) in vec3 normal; \
        layout(location = 2) in vec3 incolor; \
        uniform mat4 mvp; \
        uniform vec3 eye; \
        uniform vec3 lightsource; \
        \
        out vec4 color; \
        \
        void main(void) \
        { \
            vec3 lightray = vec3(vertex) - lightsource; \
            gl_Position = mvp * vertex; \
            color = vec4(incolor, 1.0f); \
        } \
        ";

   sunlight_frag_shader_source =
        "#version 430 \n \
        \
        in vec4 color; \
        layout (location = 0)out vec4 outcolor; \
        \
        void main(void) \
        { \
            if (gl_FrontFacing) \
                outcolor = color; \
            else \
                outcolor = vec4(0.0f, 0.0f, 0.0f, 1.0f); \
        }";
}

void sunsim::assign_quad_shader_source()
{
    quad_vert_shader_source =
        "#version 430 \n \
        \
        layout (location = 0) in vec3 pos; \
        uniform sampler2D tex_in; \
        \
        out vec2 texpos; \
        \
        void main() \
        { \
                texpos = vec2((pos + 1) / 2); \
                gl_Position = vec4(pos, 1.0f); \
        } \
        ";

    quad_frag_shader_source =
        "#version 430 \n \
        \
        in vec2 texpos;\
        uniform sampler2D tex_in;\
        \
        out vec4 color;\
        \
        void main()\
        {\
                color = vec4(texture2D(tex_in, texpos).rgb, 1.0f);\
        }\
        ";
}

void sunsim::assign_shader_sources()
{
    assign_sunlight_shader_source();
    assign_quad_shader_source();
}

GLint sunsim::check_compile_status(GLuint shader)
{
    GLint compiled;
    glGetObjectParameterivARB(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE)
    {
        int blen = 0;
        GLsizei slen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &blen);
        if (blen > 1)
        {
            GLchar * compiler_log = (GLchar *)malloc(sizeof(GLchar) * blen);
            glGetInfoLogARB(shader, blen, &slen, compiler_log);
            std::cout << "compiler log: " << compiler_log << std::endl;
            free(compiler_log);
        }
    }
    return compiled;
}

void sunsim::init_gl_rendering_objects()
{
    generate_gl_objects();
    allocate_gl_buffers();
    send_gl_bufferdata();
    setup_vao();
    setup_framebuffer();
    setup_quadrenderer();
}

void sunsim::setup_vao()
{
    glBindVertexArray(terrain_vao);

    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, terrain_normals_vbo);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, terrain_colors_vbo);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void sunsim::setup_framebuffer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, framebuf);
    GL_ERRCHECK(glGetError());

    glBindTexture(GL_TEXTURE_2D, targettex);
    GL_ERRCHECK(glGetError());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GL_ERRCHECK(glGetError());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    GL_ERRCHECK(glGetError());

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, NULL);
    GL_ERRCHECK(glGetError());

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, targettex, 0);
    GL_ERRCHECK(glGetError());

    glBindRenderbuffer(GL_RENDERBUFFER, framebuf_depthbuf);
    GL_ERRCHECK(glGetError());
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH32F_STENCIL8, window_width, window_height);
    GL_ERRCHECK(glGetError());
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, framebuf_depthbuf);
    GL_ERRCHECK(glGetError());

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::runtime_error("Framebuffer not complete. Aborting");
    }

    GLenum bufs [] = {GL_COLOR_ATTACHMENT0};
    glDrawBuffers(1, bufs);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL_ERRCHECK(glGetError());
    glBindTexture(GL_TEXTURE_2D, 0);
    GL_ERRCHECK(glGetError());
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    GL_ERRCHECK(glGetError());
}

void sunsim::generate_gl_objects()
{
    glGenVertexArrays(1, &terrain_vao);
    GL_ERRCHECK(glGetError());
    glGenBuffers(1, &terrain_vbo);
    GL_ERRCHECK(glGetError());
    glGenBuffers(1, &terrain_normals_vbo);
    GL_ERRCHECK(glGetError());
    glGenBuffers(1, &terrain_colors_vbo);
    GL_ERRCHECK(glGetError());

    glGenFramebuffers(1, &framebuf);
    GL_ERRCHECK(glGetError());
    glGenTextures(1, &targettex);
    GL_ERRCHECK(glGetError());
    glGenRenderbuffers(1, &framebuf_depthbuf);
    GL_ERRCHECK(glGetError());
}

void sunsim::allocate_gl_buffers()
{
    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo);
    GL_ERRCHECK(glGetError());
    glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_square) * landscape_triangles.size(), NULL, GL_STATIC_DRAW);
    GL_ERRCHECK(glGetError());
    glBindBuffer(GL_ARRAY_BUFFER, terrain_normals_vbo);
    GL_ERRCHECK(glGetError());
    glBufferData(GL_ARRAY_BUFFER, sizeof(normal_square) * landscape_normals.size(), NULL, GL_STATIC_DRAW);
    GL_ERRCHECK(glGetError());
    glBindBuffer(GL_ARRAY_BUFFER, terrain_colors_vbo);
    GL_ERRCHECK(glGetError());
    glBufferData(GL_ARRAY_BUFFER, sizeof(square_color) * landscape_colors.size(), NULL, GL_STATIC_DRAW);
    GL_ERRCHECK(glGetError());
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_ERRCHECK(glGetError());
}

void sunsim::setup_quadrenderer()
{
    GLfloat verts [] = {-1.0f, -1.0f, 0.0f,
                       1.0, 1.0f, 0.0f,
                       -1.0f, 1.0f, 0.0f,

                       -1.0f, -1.0f, 0.0f,
                       1.0f, -1.0f, 0.0f,
                       1.0f, 1.0f, 0.0f};

    glGenBuffers(1, &quadrender_vbo);
    glGenVertexArrays(1, &quadrender_vao);

    glBindBuffer(GL_ARRAY_BUFFER, quadrender_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glBindVertexArray(quadrender_vao);
    glBindBuffer(GL_ARRAY_BUFFER, quadrender_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //quadrender_program = compile_shaders("/home/konrad/PhDStuff/prototypes/code/cpp/sunlight_sim/src/quad.vert",
    //                "/home/konrad/PhDStuff/prototypes/code/cpp/sunlight_sim/src/quad.frag");
    //quadrender_program = compile_shaders("../quad.vert",
    //                "../quad.frag");
}

void sunsim::render_to_quad()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_DEPTH_TEST);		GL_ERRCHECK(glGetError());
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);		GL_ERRCHECK(glGetError());
    glClear(GL_COLOR_BUFFER_BIT);		GL_ERRCHECK(glGetError());

    glUseProgram(quadrender_program);		GL_ERRCHECK(glGetError());

    GLint texloc = glGetUniformLocation(quadrender_program, "tex_in");		GL_ERRCHECK(glGetError());
    glUniform1i(texloc, 0);		GL_ERRCHECK(glGetError());

    glBindVertexArray(quadrender_vao);		GL_ERRCHECK(glGetError());
    glBindTexture(GL_TEXTURE_2D, targettex);
    glDrawArrays(GL_TRIANGLES, 0, 6);		GL_ERRCHECK(glGetError());
    glBindVertexArray(0);		GL_ERRCHECK(glGetError());
}

float sunsim::get_axis_tilt_angle(int month)
{
    return -_axis_tilt + ((float) std::abs(6 - month) * _month_axis_tilt);
}

float sunsim::minutes_to_angle(float minutes)
{
    return ((_half_day_in_minutes - minutes) / _half_day_in_minutes) * M_PI;
}

void sunsim::calculate_sunpos(glm::vec3 &sunpos, glm::vec3 &up, glm::vec3 &center, float sun_radius)
{
    glm::vec3 base_pos = glm::vec3(ter.get_width() / 2, 0.0f, ter.get_height() / 2);


    glm::vec3 east_orient = glm::rotateY(ter.get_north(), -(float)M_PI_2);
    glm::vec3 true_north_orient = glm::rotate(ter.get_north(), glm::radians((float) (ter.get_latitude())), east_orient);
    float max_axis_tilt = sunsim::get_axis_tilt_angle(curr_month);
    float day_angle = minutes_to_angle(curr_time);
    glm::vec3 cp_tn_and_east(glm::normalize(glm::cross(east_orient, true_north_orient)));

    //std::cout << "true north orient: " << true_north_orient[0] << ", " << true_north_orient[1] << ", " << true_north_orient[2] << std::endl;
    //std::cout << "East orient: " << east_orient[0] << ", " << east_orient[1] << ", " << east_orient[2] << std::endl;

    center = glm::vec3(ter.get_width() / 2.0f, 0.0f, ter.get_height() / 2.0f);

    sunpos = sun_radius * cp_tn_and_east;
    sunpos = glm::rotate(sunpos, max_axis_tilt, east_orient);
    sunpos = glm::rotate(sunpos, day_angle, true_north_orient);

    if (curr_time >= 11.9 * 60 && curr_time <= 12.1 * 60 || curr_time >= 14.9 * 60 && curr_time <= 15.1 * 60 || curr_time >= 8.9 * 60 && curr_time <= 9.1 * 60)
    {
        std::cout << "Curr month: " << curr_month << std::endl;
        std::cout << "Angle sun and north: " << acos(glm::dot(sunpos, ter.get_north()) / (glm::length(sunpos) * glm::length(ter.get_north()))) / (2 * M_PI) * 360.0f << std::endl;
        glm::vec3 flat_sunpos(glm::normalize(glm::vec3(sunpos.x, 0.0f, sunpos.z)));
        float det = flat_sunpos.x * ter.get_north().z - flat_sunpos.z * ter.get_north().x;
        float dot = flat_sunpos.x * ter.get_north().x + flat_sunpos.z * ter.get_north().z;
        float azim = atan2(det, dot);
        //std::cout << "north: " << ter.get_north().x << ", " << ter.get_north().z << std::endl;
        //std::cout << "sunpos: " << sunpos.x << ", " << sunpos.z << std::endl;
        std::cout << "Flat sunpos: " << flat_sunpos.x << ", " << flat_sunpos.z << std::endl;
        std::cout << "Azimuth: " << azim << std::endl;
        std::cout << "Current time: " << int(curr_time) / 60 << ":" << int(curr_time) % 60 << std::endl;
    }

    up = glm::normalize(glm::cross(center - sunpos, true_north_orient));

    sunpos += base_pos;
}

bool sunsim::render(int curr_ortho_x, int curr_ortho_y)
{
    float sun_radius = ter.get_extent() / 2.1f * 100.0f;
    glm::vec3 sunpos, up, center;
    calculate_sunpos(sunpos, up, center, sun_radius);

    if (sunpos.y < 0.0f)
    {
        return false;
    }
    //std::cout << "sunpos x, y, z: " << sunpos.x << ", " << sunpos.y << ", " << sunpos.z << std::endl;

    GL_ERRCHECK(glGetError());
    if (render_to_framebuf)
        glBindFramebuffer(GL_FRAMEBUFFER, framebuf);
    else
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GL_ERRCHECK(glGetError());

    glEnable(GL_DEPTH_TEST);
    GL_ERRCHECK(glGetError());
    //glEnable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);
    GL_ERRCHECK(glGetError());

    //glCullFace(GL_BACK);
    //


    /*
    float sunheight = ter.get_width() * 2.0f;

    glm::vec3 eye = glm::vec3(-sunheight, 0.0f, ter.get_height() / 2.0f);
    eye = glm::vec3(glm::vec4(eye, 1.0f) * glm::rotate(glm::mat4(1.0f), rotate_angle, glm::vec3(0.0f, 0.1f, 1.0f)));
    eye += glm::vec3(ter.get_width() / 2.0f, 0.0f, 0.0f);
    eye = sunpos;
    */

    glm::vec3 eye = sunpos;
    glm::vec3 lookdir = eye - center;		// this is wrong? but it does not get used anyway...

    //glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    //up = glm::vec3(glm::vec4(up, 1.0f) * glm::rotate(glm::mat4(1.0f), rotate_angle, glm::vec3(0.0f, 0.1f, 1.0f)));

    int ostep_x = ter.get_extent() / ortho_width;
    int ostep_y = ter.get_extent() / ortho_height;

    float curr_ox = ostep_x * curr_ortho_x;
    float curr_oy = ostep_y * curr_ortho_y;

    float znear = sun_radius - ter.get_extent() / 2.0f;
    float zfar = znear + ter.get_extent() * 4.0f;

    glm::mat4 mvp(1.0f);
    mvp = mvp * glm::ortho(-0.5f * ter.get_extent(), 0.5f * ter.get_extent(), -0.5f * ter.get_extent(), 0.5f * ter.get_extent(), znear, zfar);
    //mvp = mvp * glm::lookAt(glm::vec3(4.0f, 50.0f, 4.0f), glm::vec3(250.0f, 0.0f, 250.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    mvp = mvp * glm::lookAt(eye, center, up);
    mvp = glm::translate(mvp, glm::vec3(0.0f, 0.0f, 0.0f));
    mvp = glm::scale(mvp, glm::vec3(1.0f, 1.0f, 1.0f));
    //mvp = glm::rotate(mvp, rotate_angle, glm::vec3(0.0f, 1.0f, 0.0f));

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    GL_ERRCHECK(glGetError());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GL_ERRCHECK(glGetError());

    glUseProgram(shader_program);
    GL_ERRCHECK(glGetError());

    GLint unifloc = glGetUniformLocation(shader_program, "mvp"); GL_ERRCHECK(glGetError());
    glUniformMatrix4fv(unifloc, 1, GL_FALSE, glm::value_ptr(mvp)); GL_ERRCHECK(glGetError());
    GLint eyeloc = glGetUniformLocation(shader_program, "eye"); GL_ERRCHECK(glGetError());
    glUniform3fv(eyeloc, 1, glm::value_ptr(eye)); GL_ERRCHECK(glGetError());
    GLint lightsource_loc = glGetUniformLocation(shader_program, "lightsource"); GL_ERRCHECK(glGetError());
    glUniform3fv(lightsource_loc, 1, glm::value_ptr(glm::vec3(10.0f, 1000.0f, 10.0f))); GL_ERRCHECK(glGetError());

    glBindVertexArray(terrain_vao); GL_ERRCHECK(glGetError());

    glDrawArrays(GL_TRIANGLES, 0, 6 * landscape_triangles.size()); GL_ERRCHECK(glGetError());

    glBindVertexArray(0); GL_ERRCHECK(glGetError());

    rotate_angle += 0.03f;

    glBindFramebuffer(GL_FRAMEBUFFER, 0); GL_ERRCHECK(glGetError());

    glTextureBarrier();

    return true;
}

void sunsim::send_gl_bufferdata()
{
    float testarr [] = {-1.0f, -1.0f, 0.0f,
                       1.0f, -1.0f, 0.0f,
                       0.0f, 1.0f, 0.0f};
    float testarr_colors [] = {1.0f, 1.0f, 0.0f, 1.0f,
                       1.0f, 1.0f, 0.0f, 1.0f,
                       0.0f, 1.0f, 0.0f, 1.0f};
    glBindBuffer(GL_ARRAY_BUFFER, terrain_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(triangle_square) * landscape_triangles.size(), landscape_triangles.data());
    glBindBuffer(GL_ARRAY_BUFFER, terrain_normals_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(normal_square) * landscape_normals.size(), landscape_normals.data());
    glBindBuffer(GL_ARRAY_BUFFER, terrain_colors_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(square_color) * landscape_colors.size(), landscape_colors.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void sunsim::setup_test_window()
{
}

void sunsim::add_sunlight(float hourstep)
{
    int count = 0;
    std::vector<uint32_t> pixels(window_width * window_height);
    uint32_t *colors = pixels.data();
    glBindTexture(GL_TEXTURE_2D, targettex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, pixels.data());

    //count_pixels_gpu(pixels, sunmap.data(), hourstep, window_width, window_height, ter.get_width(), ter.get_height(), base_color);
    count_pixels_gpu(pixels, sunmap.data(), hourstep, base_color, cudamem);

    /*
    for (int y = 0; y < window_height; y++)
    {
        for (int x = 0; x < window_width; x++)
        {

            uint32_t col = *(colors + y * window_width + x);
            short r = col >> 24;
            short g = (col << 8) >> 24;
            short b = (col << 16) >> 24;
            int idx = get_index_from_color({r, g, b});
            int actual_x = idx % ter.get_width();
            int actual_y = idx / ter.get_width();
            if (idx < 0)
                continue;
            if (!visitmap[idx])
            {
                visitmap[idx] = true;
                //sunmap[idx] += 30;
                sunmap.set(actual_x, actual_y, sunmap.get(actual_x, actual_y) + hourstep);
                if (col != 0x000000FF)
                {
                    count++;
                }
            }
        }
    }
    */
    //std::cout << "Number of colors not equal to opaque black: " << count << std::endl;
}

/*
void sunsim::compile_shaders()
{
    //shader_program = compile_shaders("/home/konrad/PhDStuff/prototypes/code/cpp/sunlight_sim/src/sunlight.vert",
    //                                 "/home/konrad/PhDStuff/prototypes/code/cpp/sunlight_sim/src/sunlight.frag");
    shader_program = compile_shaders("../sunlight.vert",
                                     "../sunlight.frag");
}
*/

void sunsim::compile_shaders()
{
    shader_program = compile_shaders_source(sunlight_vert_shader_source, sunlight_frag_shader_source);
    quadrender_program = compile_shaders_source(quad_vert_shader_source, quad_frag_shader_source);
}

void sunsim::create_landscape_triangles()
{
    landscape_triangles = std::vector<triangle_square>((ter.get_width() - 1) * (ter.get_height() - 1));
    for (int y = 0; y < ter.get_height() - 1; y++)
    {
        for (int x = 0; x < ter.get_width() - 1; x++)
        {
            int idx = y * (ter.get_width() - 1) + x;
            landscape_triangles[idx].botleft = landscape_triangles[idx].botleft2 = glm::vec3(x, ter.at(x, y + 1), y + 1);
            landscape_triangles[idx].topleft = glm::vec3(x, ter.at(x, y), y);
            landscape_triangles[idx].topright = landscape_triangles[idx].topright2 = glm::vec3(x + 1, ter.at(x + 1, y), y);
            landscape_triangles[idx].botright2 = glm::vec3(x + 1, ter.at(x + 1, y + 1), y + 1);
        }
    }
}

void sunsim::create_landscape_normals()
{
    landscape_normals = std::vector<normal_square>((ter.get_width() - 1) * (ter.get_height() - 1));
    for (int y = 0; y < ter.get_height() - 1; y++)
    {
        for (int x = 0; x < ter.get_width() - 1; x++)
        {
            int idx = y * (ter.get_width() - 1) + x;
            glm::vec3 vec1, vec2;

            vec1 = landscape_triangles[idx].botleft - landscape_triangles[idx].topright;
            vec2 = landscape_triangles[idx].topright - landscape_triangles[idx].topleft;
            glm::vec3 normal = glm::normalize(glm::cross(vec1, vec2));
            glm::vec3 *p = landscape_normals[idx].normal1;
            p[0] = p[1] = p[2] = normal;

            vec1 = landscape_triangles[idx].botleft2 - landscape_triangles[idx].botright2;
            vec2 = landscape_triangles[idx].botright2 - landscape_triangles[idx].topright2;
            normal = glm::normalize(glm::cross(vec1, vec2));
            p = landscape_normals[idx].normal2;
            p[0] = p[1] = p[2] = normal;
        }
    }
}

void sunsim::create_landscape_colors()
{
    landscape_colors = std::vector<square_color>((ter.get_width() - 1) * (ter.get_height() - 1));
    for (int y = 0; y < ter.get_height() - 1; y++)
    {
        for (int x = 0; x < ter.get_width() - 1; x++)
        {
            int idx = y * (ter.get_width() - 1) + x;
            int idxcol = y * ter.get_width() + x;
            std::array<short, 3> intcol = get_color_from_index(idxcol);
            float r = intcol[0] / 255.0f;
            float g = intcol[1] / 255.0f;
            float b = intcol[2] / 255.0f;
            for (int i = 0; i < 6; i++)
            {
                landscape_colors[idx].color[i] = glm::vec3(r, g, b);
            }
        }
    }
}

void sunsim::generate_rendering_data()
{
    create_landscape_triangles();
    create_landscape_normals();
    create_landscape_colors();
}

GLuint sunsim::compile_shaders_source(std::string vert_src, std::string frag_src)
{
    std::string vert_str = vert_src;
    std::string frag_str = frag_src;

    GLuint shader_vert = glCreateShader(GL_VERTEX_SHADER);
    GLuint shader_frag = glCreateShader(GL_FRAGMENT_SHADER);

    const char * vert_c_str = vert_str.c_str();
    const char * frag_c_str = frag_str.c_str();

    glShaderSource(shader_vert, 1, &vert_c_str, NULL);
    glShaderSource(shader_frag, 1, &frag_c_str, NULL);

    glCompileShader(shader_vert);
    glCompileShader(shader_frag);

    if (check_compile_status(shader_vert) != GL_TRUE)
    {
        std::cout << "Vertex shader failed to compile" << std::endl;
        return 0;
    }
    if (check_compile_status(shader_frag) != GL_TRUE)
    {
        std::cout << "Fragment shader failed to compile" << std::endl;
        return 0;
    }

    GLuint shader_program = glCreateProgram();

    glAttachShader(shader_program, shader_frag);
    glAttachShader(shader_program, shader_vert);

    int link_status;

    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &link_status);

    if (link_status != GL_TRUE)
    {
        char shader_message [512];
        glGetProgramInfoLog(shader_program, 512, NULL, shader_message);
        std::cout << "Failed to link shader program: ";
        std::cout << shader_message << std::endl;

        glDeleteProgram(shader_program);

        return 0;
    }


    glDeleteShader(shader_vert);
    glDeleteShader(shader_frag);

    return shader_program;
}

GLuint sunsim::compile_shaders(std::string vert_filepath, std::string frag_filepath)
{
    using namespace std;

    string vert_str, frag_str;

    ifstream ifs_vert(vert_filepath);

    while (ifs_vert.good())
    {
        string line;
        getline(ifs_vert, line);
        vert_str += line + '\n';
    }

    ifstream ifs_frag(frag_filepath);

    while (ifs_frag.good())
    {
        string line;
        getline(ifs_frag, line);
        frag_str += line + '\n';
    }

    return compile_shaders_source(vert_str, frag_str);
}

void sunsim::check_sunmap_edges_zero()
{
    for (int y = 0; y < ter.get_height(); y++)
    {
        //assert(fabs(sunmap[y * ter.get_width() + ter.get_width() - 1]) <= 1e-5);
        assert(fabs(sunmap.get(ter.get_width() - 1, y)) <= 1e-5);
    }
    for (int x = 0; x < ter.get_width(); x++)
    {
        //assert(fabs(sunmap[(ter.get_height() - 1) * ter.get_width() + x]) <= 1e-5);
        assert(fabs(sunmap.get(x, ter.get_height() - 1)) <= 1e-5);
    }
}

void sunsim::fix_sunmap_edges()
{
    for (int y = 0; y < ter.get_height(); y++)
    {
        //sunmap[y * ter.get_width() + ter.get_width() - 1] = sunmap[y * ter.get_width() + ter.get_width() - 2];
        sunmap.set(ter.get_width() - 1, y, sunmap.get(ter.get_width() - 2, y));
    }
    for (int x = 0; x < ter.get_width(); x++)
    {
        //sunmap[(ter.get_height() - 1) * ter.get_width() + x] = sunmap[(ter.get_height() - 2) * ter.get_width() + x];
        sunmap.set(x, ter.get_height() - 1, sunmap.get(x, ter.get_height() - 2));
    }

}

void sunsim::get_zero_column()
{
    for (int x = 0; x < ter.get_width(); x++)
    {
        bool all_zero = true;
        for (int y = 0; y < ter.get_height(); y++)
        {
            //if (abs(sunmap[y * ter.get_width() + x]) > 1e-5)
            if (abs(sunmap.get(x, y)) > 1e-5)
            {
                all_zero = false;
                break;
            }
        }
        if (all_zero)
        {
            std::cout << "Column " << x << " is all zero" << std::endl;
        }
    }
}

void sunsim::print_progress(int &dots_printed)
{
    float frac = curr_time / 1440.0f;
    int frac_div = int(std::round(frac / 0.02f));
    if (frac_div > dots_printed)
    {
        while (dots_printed < frac_div)
        {
            if (frac_div % 10 == 0)
                std::cout << frac_div * 2;
            else
                std::cout << ".";
            dots_printed++;
            std::cout.flush();
        }
    }
}

bool sunsim::render_day(int month, int time_incr)
{
    SDL_Event event;
    sunmap.fill(0.0f);
    curr_month = month;
    curr_time = 0;
    float hourstep = time_incr / 60.0f;
    int dots_printed = 0;
    while (curr_time < 60 * 24)
    {
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
                case SDL_QUIT:
                    return false;
                    break;
            }
        }
        print_progress(dots_printed);
        std::fill(visitmap.begin(), visitmap.end(), false);
        if (render(0, 0))
        {
            if (render_to_framebuf)
            {
                //render_to_quad();
            }
            else
            {
                SDL_GL_SwapWindow(window);
            }
            add_sunlight(hourstep);
        }
        curr_time += time_incr;
    }
    curr_time = 0;
    check_sunmap_edges_zero();
    fix_sunmap_edges();
    return true;
}

void sunsim::renderblock(bool show_time)
{
    curr_month = 1;
    int time_incr = 30;
    sunmap.fill(0.0f);
    while (curr_month <= 12)
    {
        std::cout << std::endl;
        std::cout << "month " << curr_month << ": 0";
        std::cout.flush();

        auto begin_time = std::chrono::steady_clock::now().time_since_epoch();
        bool ok = render_day(curr_month, time_incr);
        auto end_time = std::chrono::steady_clock::now().time_since_epoch();
        auto total_time = end_time - begin_time;
        if (show_time)
            std::cout << "Rendering and computation of day took " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() / 1000.0f << " seconds" << std::endl;

        sunmap_monthly[curr_month - 1] = sunmap;

        if (!ok)
        {
            break;
        }

        curr_month += 1;

    }
    std::cout << std::endl;
    //get_zero_column();
    //create_monthly_sunmap_hack();
    //write_png("/home/konrad/sunsim.png", sunmap, ter.get_width(), ter.get_height());
}

void sunsim::create_monthly_sunmap_hack()
{
    sunmap_monthly = std::vector< ValueMap< float > >(12, ValueMap<float>());
    std::for_each(sunmap_monthly.begin(), sunmap_monthly.end(),
                  [this](ValueMap<float> &vmap)
    {
        vmap = this->sunmap;
    });
}

/*
void sunsim::write_shaded_png(std::string filename)
{
	std::cout << "writing sunmap to file " << filename << "..." << std::endl;
	if (write_png(filename, sunmap, ter.get_width(), ter.get_height()))
	{
		std::cout << "Error: could not write to png file" << std::endl;
	}
}
*/

void sunsim::write_shaded_png_8bit(std::string filename)
{
	float maxval = *std::max_element(sunmap.begin(), sunmap.end());
	float minval = *std::min_element(sunmap.begin(), sunmap.end());
    std::vector<unsigned char> picvals(ter.get_width() * ter.get_height() * 4);

    for (int y = 0; y < ter.get_height(); y++)
    {
        for (int x = 0; x < ter.get_width(); x++)
        {
            uint32_t r = (sunmap.get(x, y) - minval) / (maxval - minval) * 255;
            if (r > 255) r = 255;
            if (r < 0) r = 0;
            int picidx = (y * ter.get_width() + x) * 4;		// image is always in row-major order
            for (int ch = 0; ch < 3; ch++)
                picvals[picidx + ch] = (uint8_t)r;
            picvals[picidx + 3] = 0xFF;
        }
    }

	std::vector<unsigned char> pngdata;

	unsigned err = lodepng::encode(pngdata, picvals, ter.get_width(), ter.get_height());
	if (!err) lodepng::save_file(pngdata, filename.c_str());
	else std::cout << "Error writing " << filename << ": " << lodepng_error_text(err) << std::endl;

	/*
	std::cout << "writing sunmap to file " << filename << "..." << std::endl;
	if (write_png(filename, sunmap, ter.get_width(), ter.get_height()))
	{
		std::cout << "Error: could not write to png file" << std::endl;
	}
	*/
}

/*
void sunsim::write_shaded_monthly_txt(std::string filename)
{
    std::ofstream ofs(filename);

    if (ofs.is_open() && ofs.good())
    {
        ofs << ter.get_width() << " " << ter.get_height() << "\n";
        for (int x = 0; x < ter.get_width(); x++)
        {
            for (int y = 0; y < ter.get_height(); y++)
            {
                for (int m = 0; m < 12; m++)
                {
                    //ofs << sunmap_monthly[m][y * ter.get_width() + x] << " ";
                    ofs << sunmap_monthly[m].get(x, y) << " ";
                }
            }
        }
    }
}
*/

void sunsim::write_shaded_monthly_txt(std::string filename, float step)
{
    std::ofstream ofs(filename);

    if (ofs.is_open() && ofs.good())
    {
        ofs << ter.get_width() << " " << ter.get_height() << " " << step << "\n";
        for (int y = 0; y < ter.get_width(); y++)
        {
            for (int x = 0; x < ter.get_height(); x++)
            {
                for (int m = 0; m < 12; m++)
                {
                    //ofs << sunmap_monthly[m][y * ter.get_width() + x] << " ";
                    ofs << sunmap_monthly[m].get(x, y) << " ";
                }
            }
        }
    }
}
