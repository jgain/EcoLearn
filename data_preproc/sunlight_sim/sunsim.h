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


#ifndef SUNSIM_H
#define SUNSIM_H

#include "terrain.h"

#include "count_pixels.h"

#include <SDL2/SDL.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <string>
#include <vector>
#include <array>

enum channel_depth
{
	CHDEPTH_8,
	CHDEPTH_16
};

struct triangle_square
{
    glm::vec3 botleft;;
    glm::vec3 topright;
    glm::vec3 topleft;
    glm::vec3 botleft2;
    glm::vec3 botright2;
    glm::vec3 topright2;
};

struct normal_square
{
    glm::vec3 normal1 [3];
    glm::vec3 normal2 [3];
};

struct square_color
{
    glm::vec3 color [6];
};

class sunsim
{
public:
    sunsim(terrain &ter, bool render_to_framebuf, int render_width, int render_height);
    ~sunsim();

    GLuint compile_shaders(std::string vert_filepath, std::string frag_filepath);
    GLint check_compile_status(GLuint shader);
    void compile_shaders();
    void generate_gl_objects();
    void allocate_gl_buffers();
    void send_gl_bufferdata();
    void init_gl_rendering_objects();
    void setup_vao();
    bool render(int curr_ortho_x, int curr_ortho_y);
    void renderblock(bool show_time);
    void create_landscape_triangles();
    void create_landscape_normals();
    void generate_rendering_data();
    void create_landscape_colors();
    int get_index_from_color(std::array<short, 3> color);
    std::array<short, 3> get_color_from_index(int idx);
    void setup_framebuffer();
    void setup_test_window();
    void add_sunlight(uint32_t *colors);
    void add_sunlight(float hourstep);
    void setup_quadrenderer();
    void render_to_quad();
    void check_sunmap_edges_zero();
    void fix_sunmap_edges();
    void get_zero_column();
    void write_shaded_png(std::string filename);
    void write_shaded_png_8bit(std::string filename);
    void create_monthly_sunmap_hack();
    void write_shaded_monthly_txt(std::string filename, float step);
    GLuint compile_shaders_source(std::string vert_src, std::string frag_src);

    static const float _half_day_in_minutes;
    static const float _quarter_day_in_minutes;
    static const float _axis_tilt;
    static const float _month_axis_tilt;
    float get_axis_tilt_angle(int month);
    float minutes_to_angle(float minutes);
    bool render_day(int month, int time_incr);
private:
    void init_sdl();
    void init_gl();
    void init();
    void close_sdl();

    void assign_sunlight_shader_source();
    void assign_quad_shader_source();
    void assign_shader_sources();

    SDL_Window *window;
    terrain ter;

    GLuint shader_program;
    GLuint terrain_vbo, terrain_normals_vbo, terrain_colors_vbo;
    GLuint terrain_vao;
    GLuint framebuf;
    GLuint targettex;
    GLuint framebuf_depthbuf;

    std::vector<triangle_square> landscape_triangles;
    std::vector<normal_square> landscape_normals;
    std::vector<square_color> landscape_colors;

    GLfloat rotate_angle = 0.0f;

    short base_color [3];

    int window_width, window_height;

    std::vector<bool> visitmap;
    ValueMap<float> sunmap;
    std::vector< ValueMap<float> > sunmap_monthly;
    float curr_time;	// time in minutes
    int curr_month;		// current month, from 1 to 12


    SDL_Window *test_window;
    SDL_Renderer *test_renderer;
    SDL_Surface *test_surface;

    GLuint quadrender_vbo;
    GLuint quadrender_vao;
    GLuint quadrender_program;

    bool render_to_framebuf;

    std::string sunlight_vert_shader_source, sunlight_frag_shader_source;
    std::string quad_vert_shader_source, quad_frag_shader_source;
    void print_progress(int &dots_printed);
    void calculate_sunpos(glm::vec3 &sunpos, glm::vec3 &up, glm::vec3 &center, float sun_radius);

    int ortho_width, ortho_height;

    gpumem cudamem;
};

#endif // SUNSIM_H
