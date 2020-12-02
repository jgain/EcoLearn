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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/



#ifndef GL_WRAPPER_H
#define GL_WRAPPER_H

#define GLM_ENABLE_EXPERIMENTAL

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>
#include <algorithm>

#include "data_importer/extract_png.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_texture_types.h>

static std::string get_errstring(GLuint errcode)
{
    switch (errcode)
    {
        case (GL_NO_ERROR):
            return "no error";
            break;
        case (GL_INVALID_ENUM):
            return "INVALID_ENUM";
            break;
        case (GL_INVALID_VALUE):
            return "INVALID_VALUE";
            break;
        case (GL_INVALID_OPERATION):
            return "INVALID_OPERATION";
            break;
        case (GL_INVALID_FRAMEBUFFER_OPERATION):
            return "GL_INVALID_FRAMEBUFFER_OPERATION";
            break;
        case (GL_OUT_OF_MEMORY):
            return "GL_OUT_OF_MEMORY";
            break;
        case (GL_STACK_OVERFLOW):
            return "GL_STACK_OVERFLOW";
            break;
        case (GL_STACK_UNDERFLOW):
            return "GL_STACK_UNDERFLOW";
            break;
        default:
            return "Unknown error";
            break;
    }
}

#define GL_ERRCHECK(show_noerr) \
    { \
        GLenum errcode = glGetError();	\
        if (errcode != GL_NO_ERROR) \
        { \
            std::cout << "GL error in file " << __FILE__ << ", line " << __LINE__ << ": " << get_errstring(errcode) << std::endl; \
        } \
        else if (show_noerr) \
        { \
            std::cout << "No GL errors on record in file " << __FILE__ << ", line " << __LINE__ << std::endl; \
        } \
    }


struct float_get
{};

struct uint32_get
{};


/*
 * gl_wrapper class encapsulates most of the lower-level OpenGL calls for managing and rendering to textures for the
 * canopy placement algorithm.
 */

class gl_wrapper
{
public:

    gl_wrapper(int width, int height, const std::vector<float> &example_chm_data);
    gl_wrapper(std::string object_filename, int width, int height, float *chm_data);
    gl_wrapper(std::string object_filepath, std::string example_filepath);
    gl_wrapper(float *chm, int width, int height);




public:

    /*
     * Render the current trees stored in this object as a top-down orthographic rendering
     */
    void render_via_gpu(int ntrees);
private:

    /*
     * Draw a texture with id texture_id to the SDL window held by this object.
     * The textype argument must be GL_INT for a normal colour texture, and GL_FLOAT
     * for a texture that indicates some form of data (will be greyscale)
     */
    void draw_texture_gl(GLuint texture_id, unsigned textype);

    /*
     * Draws depthbuffer for top-down rendering of tree spheres
     */
    void draw_chm_depthbuffer();

    /*
     * Draws colour texture for top-down rendering of tree spheres
     */
    void draw_chm_texture();

    /*
     * Initialize the OpenGL context, window, buffers, etc.
     */
    void init();

    void setup_gl_textures();

    /*
     * Vertex array object setup functions
     */
    void setup_chm_copy_vao();
    void setup_example_vao();
    void setup_main_vao();

    /*
     * Calls the above three vao setup functions
     */
    void setup_vertex_array_objects();

    /*
     * Framebuffer setup functions
     */
    void setup_chm_copy_framebuffer();
    void setup_chm_placement_framebuffer();
    void setup_diff_framebuffer();
    void setup_summation_framebuffer(GLuint tex_w, GLuint tex_h);

    void create_example_chm_texture();

    /*
     * Functions used in the import and compile of GLSL shaders
     */
    GLuint compile_shaders(std::string vert_filepath, std::string frag_filepath);
    GLint check_compile_status(GLuint shader);

    /*
     * Used in the initialization of the OpenGL context
     */
    void init_gl();
    void set_gl_attribs();
    void allocate_gl_buffers();

    /*
     * These functions are used to process the data imported from the .obj file
     */
    void process_all_nodes(const aiScene *scene);
    void process_all_meshes();
    void process_node(aiNode *node, const aiScene *scene);
    void process_mesh(aiMesh *mesh, const aiScene *scene);


    std::vector<glm::mat4> translate_matrices;
    std::vector<glm::mat4> scale_matrices;
    std::vector<glm::vec4> color_vecs;
    void set_sphere_model_from_string(std::string model_string);
public:
    SDL_Window * window;
    SDL_GLContext gl_context;
    GLuint * vbos;
    GLuint * vaos;
    GLuint * ebos;
    GLuint shader_frag, shader_vert;
    GLuint translate_matrix_vbo;
    GLuint scale_matrix_vbo;
    GLuint color_vec_vbo;

    GLuint batch_render_shader_program;

    GLuint shader_program,
        chm_example_shader_program,
        summation_program,
        sum_general_program,
        chm_copy_program;
    GLuint chm_placement_texture,
        diff_texture,
        diff_colors_texture,
        tree_placement_texture,
        summation_texture,
        diff_rem_texture,
        chm_gl_texture,
        chm_gl_texture_target;
    GLuint example_texture;
    GLuint chm_placement_depthbuffer;
    GLuint chm_placement_framebuffer,
        diff_framebuffer,
        summation_framebuffer,
        chm_copy_framebuffer;
    int width, height;
    cudaGraphicsResource *cudares;
    cudaTextureObject_t texture_obj_treecols;

    glm::vec3 verts_avg;
    glm::vec3 verts_max_dist;  // maximum distance of vertex from center of sphere for every dimension

    std::vector<aiMesh *> meshes;
    std::vector<GLfloat> vertices;
    std::vector<GLuint> faces_idxes;
    std::vector<float> example_chm_data;
    std::vector<uint32_t> example_chm_data_uint;

    float chm_max_val;

    Assimp::Importer assimp_import;
    const aiScene * scene;

    int num_meshes = 0;
    int num_vertices = 0;

    /*
     * Set a .obj model for the sphere that each tree uses for the top-down
     * orthographic rendering
     */
    void set_sphere_model(std::string model_filepath);

    /*
     * Setup the OpenGL buffers used for rendering
     */
    void setup_gl_objects_buffers();

    /*
     * Import and compile the OpenGL (GLSL) shaders
     */
    void init_gl_shaders();


    // Useful debugging functions
    // ------------------------------------

    /*
     * Get top-down orthographic rendering where each tree is represented by
     * a sphere with its own unique colour. Here each pixel is represented by a
     * uint32 integer
     */
    std::vector<uint32_t> get_color_id_data();

    /*
     * Get depthbuffer for the top-down orthographic rendering
     */
    std::vector<float> get_chm_placement_depthbuffer();

    /*
     * Get zvalues for top-down rendering
     */
    std::vector<float> get_chm_placement_zvals();


    /*
     * Get texture data as array of uint32 integers
     */
    uint32_t *get_pixel_data_chm();


    /*
     * Computes a squared difference texture for each point on the landscape
     */
    void compute_diff_texture();


    /*
     * Get the depthbuffer for the top-down rendering of canopy trees as an SDL surface
     */
    SDL_Surface *get_depth_texture();

    /*
     * This function needs some attention. It supposedly gets the colour texture, but the GL function calls
     * imply that the depth buffer is being retrieved
     */
    SDL_Surface *get_color_texture();


    /*
     * Get the original CHM as loaded on the GPU as an SDL surface object, stored on the CPU
     */
    SDL_Surface *get_example_surface();


    /*
     * Copy the CHM to the chm_copy_framebuffer fb object
     */
    void copy_gl_chm();


    /*
     * Get difference between ground truth CHM and rebuilt CHM from placed canopy trees as
     * vector of unsigned 8 bit integers. This function is used in the get_diff_rem_data functions
     * below, where each have their own formats
     */
    std::vector<uint8_t> get_diff_rem_data();
    std::vector<float> get_diff_rem_data(float_get);
    std::vector<uint32_t> get_diff_rem_data(uint32_get);

    ~gl_wrapper();


    /*
     * Get an SDL surface, i.e. a representation of a texture on the CPU via the SDL framework,
     * from a texture on the GPU. Useful for quick rendering of a texture for debugging purposes
     * with SDL's rendering functions
     */
    static SDL_Surface *get_surface_from_opengl_texture(GLuint opengl_tex);
protected:
    /*
     * Setup memory for translate, scale, and colour matrices
     */
    void allocate_transform_buffer_memory();
};

#endif
