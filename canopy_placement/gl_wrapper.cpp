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


#include "gl_wrapper.h"
#include "common/misc.h"

#include "sphere_obj_string.h"

#include <stdexcept>

using namespace std;

inline uint8_t * get_texture_data(GLuint gl_texture, GLint *width, GLint *height)
{
    glBindTexture(GL_TEXTURE_2D, gl_texture);

    GLint tex_w, tex_h;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &tex_h);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tex_w);

    uint8_t *buffer = (uint8_t *)malloc(sizeof(uint32_t) * tex_w * tex_h);
    //glGetTextureImage(gl_texture, 0, GL_RGBA, GL_UNSIGNED_BYTE, sizeof(uint32_t) * tex_w * tex_h, buffer);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

    if (width)
        *width = tex_w;
    if (height)
        *height = tex_h;

    return buffer;
}

SDL_Surface *gl_wrapper::get_surface_from_opengl_texture(GLuint opengl_tex)
{

    int tex_width, tex_height;

    uint8_t *buffer = get_texture_data(opengl_tex, &tex_width, &tex_height);

    SDL_Surface *surface = SDL_CreateRGBSurface(0, tex_width, tex_height, 32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);

    Uint8 * base_pix = (Uint8 *)surface->pixels;

    for (int row = 0; row < tex_height; row++)
    {
        for (int col = 0; col < tex_width; col++)
        {
            Uint32 * dest_pixel_ptr = (Uint32 *)(base_pix + row * surface->pitch + col * surface->format->BytesPerPixel);
            Uint32 * src_pixel_ptr = (uint32_t *)(buffer + row * tex_width * sizeof(uint32_t) + col * sizeof(uint32_t));

            Uint8 color, dummy;
            SDL_GetRGBA(*src_pixel_ptr, surface->format, &color, &color, &color, &dummy);

            *dest_pixel_ptr = *src_pixel_ptr;

            //cout << "Color value: " << (Uint32)color << endl;
        }
    }

    free(buffer);

    return surface;
}

gl_wrapper::gl_wrapper(int width, int height, const std::vector<float> &example_chm_data)
    : width(width), height(height), example_chm_data(example_chm_data), verts_avg(0, 0, 0),
      example_chm_data_uint(width * height)
{
    if (width * height != example_chm_data.size())
    {
        throw std::invalid_argument("width * height must equal example_chm_data.size()");
    }
    init_gl();
}

gl_wrapper::gl_wrapper(std::string object_filename, int width, int height, float *chm_data)
    : width(width),
      height(height),
      example_chm_data(std::vector<float>(width * height)),
      verts_avg(0, 0, 0),
      example_chm_data_uint(width * height)
{
    memcpy(example_chm_data.data(), chm_data, sizeof(float) * width * height);
    set_sphere_model(object_filename);
    init_gl();


    /*
     * / this is probably not necessary, if we render directly from the top
    if (!row_major)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float &s1 = example_chm_data[y * width + x];
                float &s2 = example_chm_data[x * height + y];
                std::swap(s1, s2);
            }
        }
    }
    */
}

gl_wrapper::gl_wrapper(float *chm_data, int width, int height)
    : width(width),
      height(height),
      example_chm_data(std::vector<float>(width * height)),
      verts_avg(0, 0, 0),
      example_chm_data_uint(width * height)
{
    memcpy(example_chm_data.data(), chm_data, sizeof(float) * width * height);
    //set_sphere_model_from_string(sphere_obj_contents);
    std::string sphere_filename = std::string(SPHEREMODEL_BASEDIR) + "/sphere.obj";
    set_sphere_model(sphere_filename);
    init_gl();

}

gl_wrapper::gl_wrapper(string object_filepath, string example_filepath)
    :
    verts_avg(0, 0, 0), example_chm_data(get_image_data_48bit(example_filepath, width, height)[0]),
    example_chm_data_uint(width * height)
    // XXX: previously the scene variable was initialized in the init list. Put it back here if issues arise
{
    set_sphere_model(object_filepath);
    init_gl();
}

void gl_wrapper::set_sphere_model(std::string model_filepath)
{
    scene = assimp_import.ReadFile(model_filepath, aiProcess_Triangulate);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw runtime_error("Scene object imported from " + string(model_filepath) + " is invalid");
    }

    process_all_nodes(scene);
    process_all_meshes();
}

void gl_wrapper::set_sphere_model_from_string(std::string model_string)
{
    scene = assimp_import.ReadFileFromMemory(model_string.c_str(), model_string.size() * sizeof(char), aiProcess_Triangulate);
    if (!scene)
    {

    }
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw runtime_error("Scene object imported from string" + model_string + " is invalid");
    }

    process_all_nodes(scene);
    process_all_meshes();
}

void gl_wrapper::init_gl()
{
    init();

    allocate_gl_buffers();
    glViewport(0, 0, width, height);

    init_gl_shaders();
    std::cout << "Allocating buffer memory..." << std::endl;
    setup_gl_objects_buffers();	// XXX: this is really messy, I know: example_chm_data basically get reassigned to itself in this function. Will
                                                                // think later of a better way of organising things.
    std::cout << "Done allocating buffer memory" << std::endl;

    setup_vertex_array_objects();

    setup_gl_textures();
    chm_max_val = *std::max_element(example_chm_data.begin(), example_chm_data.end()) * 0.3048;

    std::cout << "creating uint example chm data..." << std::endl;
    for (int i = 0; i < width * height; i++)
    {
        example_chm_data_uint[i] = example_chm_data[i] > 0 ? 256 * 256 * 256 * 255 - 1 + 256 * 256 * 256 : 255;
    }
    std::cout << "done creating uint example chm data" << std::endl;
}

void gl_wrapper::setup_vertex_array_objects()
{
    setup_main_vao();
    setup_example_vao();
    setup_chm_copy_vao();
}

void gl_wrapper::setup_main_vao()
{

    cout << "Generating vertex array" << endl;
    GL_ERRCHECK(false);

    glBindVertexArray(vaos[0]);


    glBindBuffer(GL_ARRAY_BUFFER, vbos[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, translate_matrix_vbo);
    int nattribs = 1;
    for (int attrib_num = 0; attrib_num < 4; attrib_num++, nattribs++)
    {
        glEnableVertexAttribArray(nattribs);
        glVertexAttribPointer(nattribs, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid *) (attrib_num * sizeof(glm::vec4)));
        glVertexAttribDivisor(nattribs, 1);
    }
    assert(nattribs == 5);

    glBindBuffer(GL_ARRAY_BUFFER, scale_matrix_vbo);
    for (int attrib_num = 0; attrib_num < 4; attrib_num++, nattribs++)
    {
        glEnableVertexAttribArray(nattribs);
        glVertexAttribPointer(nattribs, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid *) (attrib_num * sizeof(glm::vec4)));
        glVertexAttribDivisor(nattribs, 1);
    }
    assert(nattribs == 9);

    glBindBuffer(GL_ARRAY_BUFFER, color_vec_vbo);
    for (int attrib_num = 0; attrib_num < 1; attrib_num++, nattribs++)
    {
        glEnableVertexAttribArray(nattribs);
        glVertexAttribPointer(nattribs, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (GLvoid *)0);
        glVertexAttribDivisor(nattribs, 1);
    }
    assert(nattribs == 10);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[0]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * faces_idxes.size(), faces_idxes.data(), GL_STATIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    GL_ERRCHECK(false);
}

void gl_wrapper::render_via_gpu(int ntrees)
{
    bool display_noerr = false;

    // bind framebuffer set up for CHM placement rendering
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    GL_ERRCHECK(display_noerr);

    // enable and set depth test to greater than
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    GL_ERRCHECK(display_noerr);

    // set colour and depth buffer clear values
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(0.0f);
    GL_ERRCHECK(display_noerr);

    // clear colour and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GL_ERRCHECK(display_noerr);

    glUseProgram(batch_render_shader_program);

    // generate orthographic projection and view matrices
    glm::mat4 ortho_mat = glm::ortho(0.0f, (GLfloat)width, 0.0f, (GLfloat)height, chm_max_val, 2 * chm_max_val);
    glm::mat4 view_mat = glm::lookAt(glm::vec3(0.0f, 0.0f, -chm_max_val), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));

    // get location of ortho mat variable in shader
    GLint ortho_mat_loc = glGetUniformLocation(batch_render_shader_program, "ortho_mat");
    GL_ERRCHECK(display_noerr);

    // set the orthographic projection matrix
    glUniformMatrix4fv(ortho_mat_loc, 1, GL_FALSE, glm::value_ptr(ortho_mat));
    GL_ERRCHECK(display_noerr);

    // get location of view_mat variable in shader
    GLint view_mat_loc = glGetUniformLocation(batch_render_shader_program, "view_mat");
    // set the view matrix
    glUniformMatrix4fv(view_mat_loc, 1, GL_FALSE, glm::value_ptr(view_mat));
    GL_ERRCHECK(display_noerr);

    // get location of chm_texture variable in shader
    GLint chm_texture_loc = glGetUniformLocation(batch_render_shader_program, "chm_texture");
    GL_ERRCHECK(display_noerr);

    // set the CHM texture
    glUniform1i(chm_texture_loc, chm_gl_texture);
    GL_ERRCHECK(display_noerr);

    // get location of chm_max_val variable in shader
    GLint chm_max_val_loc = glGetUniformLocation(batch_render_shader_program, "chm_max_val");
    GL_ERRCHECK(display_noerr);

    // set the CHM maximum value
    glUniform1f(chm_max_val_loc, chm_max_val);

    // bind vertex array object
    glBindVertexArray(vaos[0]);
    GL_ERRCHECK(display_noerr);

    // draw
    glDrawElementsInstanced(GL_TRIANGLES, faces_idxes.size(), GL_UNSIGNED_INT, 0, ntrees);

    // detach framebuffer and vertex array object
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(0);

    // The results of this rendering are used after this in CUDA computations on the GPU. This call ensures
    // that all rendering operations in this function are complete before exiting this function, therefore
    // rendering will be done here before starting the CUDA operations
    glFinish();
}


void gl_wrapper::setup_gl_objects_buffers()
{
    allocate_transform_buffer_memory();

    create_example_chm_texture();

    setup_chm_placement_framebuffer();
    setup_diff_framebuffer();
    setup_chm_copy_framebuffer();

}

void gl_wrapper::init_gl_shaders()
{
    std::string shader_basedir = SHADER_BASEDIR;	// convert raw C string to std::string for easier concatenation
    shader_basedir += "/";
    batch_render_shader_program = compile_shaders(shader_basedir + "sphere_fast.vert", shader_basedir + "sphere_fast.frag");
    if (!batch_render_shader_program)
    {
        throw runtime_error("CHM rendering shader program for instanced drawing could not compile. Aborting");
    }
    shader_program = compile_shaders(shader_basedir + "sphere.vert", shader_basedir + "sphere.frag");
    if (!shader_program)
    {
        throw runtime_error("CHM rendering shader program could not compile. Aborting");
    }

    chm_example_shader_program = compile_shaders(shader_basedir + "diff_texture.vert", shader_basedir + "diff_texture.frag");
    if (!chm_example_shader_program)
    {
        throw runtime_error("Texture differencing shader program could not compile. Aborting");
    }

    sum_general_program = compile_shaders(shader_basedir + "sum_general.vert", shader_basedir + "sum_general.frag");
    if (!sum_general_program)
    {
        throw runtime_error("General texture summation shader program could not compile. Aborting");
    }

    summation_program = compile_shaders(shader_basedir + "summation.vert", shader_basedir + "summation.frag");
    if (!summation_program)
    {
        throw runtime_error("Texture summation shader program could not compile. Aborting");
    }

    chm_copy_program = compile_shaders(shader_basedir + "copy_chm_texture.vert", shader_basedir + "copy_chm_texture.frag");
    if (!chm_copy_program)
    {
        throw runtime_error("CHM copy shader program could not compile. Aborting");
    }

}
\
vector<uint32_t> gl_wrapper::get_color_id_data()
{
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    vector<uint32_t> pix_data(width * height, 0);
    GLint prev_read_buffer;
    glGetIntegerv(GL_READ_BUFFER, &prev_read_buffer);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, pix_data.data());
    glReadBuffer(prev_read_buffer);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    return pix_data;
}

void gl_wrapper::allocate_transform_buffer_memory()
{
    glBindBuffer(GL_ARRAY_BUFFER, translate_matrix_vbo); GL_ERRCHECK(false);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * width * height, NULL, GL_DYNAMIC_DRAW);GL_ERRCHECK(false);
    glBindBuffer(GL_ARRAY_BUFFER, scale_matrix_vbo);GL_ERRCHECK(false);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * width * height, NULL, GL_DYNAMIC_DRAW);GL_ERRCHECK(false);
    glBindBuffer(GL_ARRAY_BUFFER, color_vec_vbo);GL_ERRCHECK(false);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * width * height, NULL, GL_DYNAMIC_DRAW);GL_ERRCHECK(false);

    GL_ERRCHECK(true);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

/*
 * draw the texture given by texture_id to the screen (not to any custom framebuffer)
 */
void gl_wrapper::draw_texture_gl(GLuint texture_id, unsigned textype)
{
    if (textype != GL_FLOAT && textype != GL_INT)
    {
        throw std::invalid_argument("For gl_wrapper::draw_texture_gl: textype must be either GL_FLOAT or GL_INT");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);

    glClear(GL_COLOR_BUFFER_BIT);

    GLfloat pos_verts [] = { -1.0f, 1.0f, 0.0f,    0.0f, 1.0f,
                 -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
                 1.0f, -1.0f, 0.0f,    1.0f, 0.0f,
                 1.0f, 1.0f, 0.0f,     1.0f, 1.0f
    };

    std::string vertfname, fragfname;
    if (textype == GL_FLOAT)
    {
        vertfname = "show_float_texture.vert";
        fragfname = "show_float_texture.frag";
    }
    else
    {
        vertfname = "show_texture.vert";
        fragfname = "show_texture.frag";
    }

    GLuint shader_program = compile_shaders(vertfname, fragfname);

    glUseProgram(shader_program);

    GLuint idxes [] = {0, 1, 2, 0, 3, 2};

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo, ebo;

    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5, (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5, (void*) (3 * sizeof(GLfloat)) );

    glBufferData(GL_ARRAY_BUFFER, sizeof(pos_verts), pos_verts, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxes), idxes, GL_STATIC_DRAW);

    glDrawElements(GL_TRIANGLES, sizeof(idxes) / sizeof(GLuint), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    SDL_GL_SwapWindow(window);
}

void gl_wrapper::draw_chm_depthbuffer()
{
    draw_texture_gl(chm_placement_depthbuffer, GL_FLOAT);
}

void gl_wrapper::draw_chm_texture()
{
    draw_texture_gl(chm_placement_texture, GL_INT);
}

std::vector<float> gl_wrapper::get_chm_placement_depthbuffer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    std::vector<float> depth_data(width * height);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_data.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return depth_data;
}

std::vector<float> gl_wrapper::get_chm_placement_zvals()
{
    uint32_t rshift = 24;

    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    std::vector<uint32_t> zdata_uint(width * height);
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, zdata_uint.data());
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    std::vector<float> zdata(width * height);

    for (int i = 0; i < zdata_uint.size(); i++)
    {
        zdata[i] = (float)(zdata_uint[i] >> rshift);
    }

    return zdata;
}

uint32_t *gl_wrapper::get_pixel_data_chm()
{
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);

    uint32_t *data = (uint32_t *)malloc(sizeof(uint32_t) * width * height);

    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, data);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return data;
}

void gl_wrapper::compute_diff_texture()
{
    GLfloat tex_coords [] = {-1.0f, 1.0f, 0.0f,    0.0f, 1.0f,
                 -1.0f, -1.0f, 0.0f,   0.0f, 0.0f,
                 1.0f, -1.0f, 0.0f,    1.0f, 0.0f,
                 1.0f, 1.0f, 0.0f,     1.0f, 1.0f
    };

    glBindFramebuffer(GL_FRAMEBUFFER, diff_framebuffer);

    glClear(GL_COLOR_BUFFER_BIT);

    GLuint tex_idxes [] = {0, 1, 2, 0, 3, 2};

    glUseProgram(chm_example_shader_program);

    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(tex_idxes), tex_idxes, GL_STATIC_DRAW);

    GLuint tex_loc = glGetUniformLocation(chm_example_shader_program, "tex_orig");
    glUniform1i(tex_loc, 0);
    GLuint tex2_loc = glGetUniformLocation(chm_example_shader_program, "tex_optim");
    glUniform1i(tex2_loc, 1);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, example_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, chm_placement_depthbuffer);

    glBindVertexArray(vaos[1]);
    glDrawElements(GL_TRIANGLES, sizeof(tex_idxes) / sizeof(GLuint), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}



SDL_Surface *gl_wrapper::get_depth_texture()
{

    GLfloat *buffer = (GLfloat *)malloc(width * height * sizeof(GLfloat));
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, buffer);

    GLenum err = glGetError();

    if (err)
    {
        cout << "Error with GetnTexImage: ";
        cout << glewGetErrorString(err) << endl;
    }

    cout << "First few bytes of buffer into which depth texture was copied: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << (uint32_t)(*(((uint8_t *)buffer) + i)) << endl;
    }

    SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32,
                                                0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);

    Uint8 * base_pix = (Uint8 *)surface->pixels;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            Uint32 * dest_pixel_ptr = (Uint32 *)(base_pix + row * surface->pitch + col * surface->format->BytesPerPixel);

            GLfloat * curr_pix = buffer + width * row + col;
            uint8_t color = (*curr_pix) * 255;
            *dest_pixel_ptr = SDL_MapRGBA(surface->format, color, color, color, 255);
        }
    }

    free(buffer);

    return surface;
}

SDL_Surface *gl_wrapper::get_color_texture()
{
    GLuint bufsize = sizeof(GLfloat) * width * height;
    GLfloat *buffer = (GLfloat *)malloc(bufsize);
    memset((void*)buffer, 128, bufsize);
    glBindTexture(GL_TEXTURE_2D, chm_placement_depthbuffer);

    // why GL_DEPTH_COMPONENT? We need the colour texture
    glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, (void*)buffer);

    GLenum err = glGetError();

    if (err)
    {
        cout << "Error with GetnTexImage: ";
        cout << glewGetErrorString(err) << endl;
    }

    cout << "First few bytes of buffer into which color texture was copied: " << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << (uint32_t)(*(((uint8_t *)buffer) + i)) << endl;
    }

    SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);

    Uint8 * base_pix = (Uint8 *)surface->pixels;

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            Uint32 * dest_pixel_ptr = (Uint32 *)(base_pix + row * surface->pitch + col * surface->format->BytesPerPixel);
            GLfloat src_pixel_val = *(buffer + row * width + col);
            Uint32 src_pixel_intval = src_pixel_val * 255;
            if (src_pixel_intval > 255)
                src_pixel_intval = 255;
            if (src_pixel_intval < 0)
                src_pixel_intval = 0;
            Uint8 bit8 = (Uint8)src_pixel_intval;

            *dest_pixel_ptr = SDL_MapRGBA(surface->format, bit8, bit8, bit8, 255);
        }
    }

    free(buffer);

    return surface;
}



SDL_Surface *gl_wrapper::get_example_surface()
{
    return get_surface_from_opengl_texture(example_texture);
}

void gl_wrapper::copy_gl_chm()
{
    glBindFramebuffer(GL_FRAMEBUFFER, chm_copy_framebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(chm_copy_program);

    GLuint chm_texture_loc = glGetUniformLocation( chm_copy_program, "chm_texture");
    glUniform1ui(chm_texture_loc, chm_gl_texture);

    glBindVertexArray(vaos[2]);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

std::vector<uint8_t> gl_wrapper::get_diff_rem_data()
{
    int fw = ((width - 1) / 4 + 1) * 4;

    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    vector<uint8_t> pix_data(fw * height);
    glReadBuffer(GL_COLOR_ATTACHMENT1);		// specify the source for the glReadPixels function below
    glReadPixels(0, 0, fw, height, GL_RED, GL_UNSIGNED_BYTE, pix_data.data());
    glReadBuffer(GL_COLOR_ATTACHMENT0);		// reset read buffer to default
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return pix_data;
}

std::vector<float> gl_wrapper::get_diff_rem_data(float_get)
{
    std::vector<uint8_t> pix_data = get_diff_rem_data();
    std::vector<float> fpix_data(pix_data.size());

    for (int i = 0; i < pix_data.size(); i++)
    {
        float val = ((uint32_t)pix_data[i]) / 255;
        fpix_data[i] = val;
    }

    return fpix_data;
}

std::vector<uint32_t> gl_wrapper::get_diff_rem_data(uint32_get)
{

    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);
    vector<uint32_t> pix_data(width * height);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, pix_data.data());
    glReadBuffer(GL_COLOR_ATTACHMENT0);		// is this the right default to restore?
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return pix_data;

}

gl_wrapper::~gl_wrapper()
{
    delete [] vaos;
    delete [] vbos;
    delete [] ebos;

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

// all these methods are private

void gl_wrapper::init()
{
    std::cout << "Initializing SDL and creating gl context..." << std::endl;

    SDL_Init(SDL_INIT_EVERYTHING);

    set_gl_attribs();
    window = SDL_CreateWindow("Sphere", 100, 100, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
    gl_context = SDL_GL_CreateContext(window);
    //SDL_HideWindow(window);

    if (!gl_context)
    {
        throw std::runtime_error("Failed to create SDL GL context");
    }
    else
    {
        std::cout << "SDL GL context created successfully" << std::endl;
    }

    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLuint glew_return = glewInit();
    if (glew_return)
    {

        cout << "could not initialize glew: " << glewGetErrorString(glew_return) << endl;
    }


}

void gl_wrapper::setup_gl_textures()
{
    float min_val, max_val;
    min_val = *std::min_element(example_chm_data.begin(), example_chm_data.end());
    max_val = *std::max_element(example_chm_data.begin(), example_chm_data.end());
    float range = max_val - min_val;
    vector<float> datacopy = example_chm_data;
    std::for_each(datacopy.begin(), datacopy.end(), [this, range, min_val](float &el){
        el = (el - min_val) / range;
    });

    int skip = 1;
    int fw = ((width - 1) / 4 + 1) * 4;
    vector<uint32_t> texdata(width * height);
    for (int i = 0; i < width * height; i += skip)
    {
        //cout << "Datacopy[i] = " << datacopy[i] << endl;
        uint32_t val = datacopy[i] * 255;
        trim(val, (uint32_t)0, (uint32_t)255);
        uint32_t bitval = (uint32_t)255 + val * 256 + val * 256 * 256 + val * 256 * 256 * 256;
        texdata.at(i) = bitval;
        int x = i % width;
        int y = i / width;
    }

    glGenTextures(1, &chm_gl_texture);
    glBindTexture(GL_TEXTURE_2D, chm_gl_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, fw, height, 0, GL_RED, GL_UNSIGNED_BYTE, (uint8_t*)texdata.data());
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, texdata.data());

    int w, h;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
    cout << "w, h of texture: " << w << ", " << h << endl;
    glBindTexture(GL_TEXTURE_2D, 0);


    glGenTextures(1, &chm_gl_texture_target);
    glBindTexture(GL_TEXTURE_2D, chm_gl_texture_target);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void gl_wrapper::setup_chm_copy_vao()
{
    GLfloat coords [4 * 3] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f
    };

    GLfloat tex_coords [4 * 2] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f
    };

    GLuint idxes [6] = {
        0, 2, 3,
        0, 1, 2
    };

    GLuint vbo_coord;
    GLuint vbo_tex_coord;

    glGenBuffers(1, &vbo_coord);
    glGenBuffers(1, &vbo_tex_coord);

    //glGenVertexArrays(1, vaos + 2);
    //glGenBuffers(1, vbos + 2);
    //glGenBuffers(1, ebos + 2);

    glBindVertexArray(vaos[2]);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_coord);
    glBufferData(GL_ARRAY_BUFFER, sizeof(coords), coords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_tex_coord);
    glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idxes), idxes, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


void gl_wrapper::setup_example_vao()
{
    //glGenVertexArrays(1, vaos + 1);
    //glGenBuffers(1, vbos + 1);
    //glGenBuffers(1, ebos + 1);

    glBindVertexArray(vaos[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebos[1]);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5, (void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 5, (void*) (3 * sizeof(GLfloat)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(0);
}

void gl_wrapper::setup_chm_copy_framebuffer()
{
    glGenFramebuffers(1, &chm_copy_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, chm_copy_framebuffer);

    glBindTexture(GL_TEXTURE_2D, diff_rem_texture);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D,
                           diff_rem_texture,
                           0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    GLenum status = 0;

    if ((status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "CHM placement framebuffer construction failed: ";
        switch (status)
        {
            case (GL_FRAMEBUFFER_UNDEFINED):
                std::cout << "framebuffer undefined" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT):
                std::cout << "framebuffer contains incomplete attachment" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER):
                std::cout << "framebuffer contains an incomplete draw buffer" << std::endl;
                break;
            case (GL_FRAMEBUFFER_UNSUPPORTED):
                std::cout << "invalid combination of internal formats of framebuffer images" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT):
                std::cout << "framebuffer misses an attachment" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER):
                std::cout << "framebuffer contains incomplete read buffer" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE):
            case (GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS):
                std::cout << "Either incomplete multisample or incomplete layer targets " << std::endl;
                break;
            case (GL_INVALID_ENUM):
            case (GL_INVALID_OPERATION):
                std::cout << "invalid framebuffer argument" << std::endl;
                break;
            default:
                cout << "unspecified error, code " << status << std::endl;
                break;
        }
        return;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void gl_wrapper::setup_chm_placement_framebuffer()
{
    glGenFramebuffers(1, &chm_placement_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, chm_placement_framebuffer);


    /*
        generate texture for depthbuffer
    */
    glGenTextures(1, &chm_placement_depthbuffer);
    glBindTexture(GL_TEXTURE_2D, chm_placement_depthbuffer);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    /*
        bind generated depthbuffer texture to the framebuffer
    */
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, chm_placement_depthbuffer, 0);

    glGenTextures(1, &chm_placement_texture);
    glBindTexture(GL_TEXTURE_2D, chm_placement_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, chm_placement_texture, 0);

    // XXX: TODO: These color attachments below were more for debugging and some metrics (such as difference between
    // rebuilt CHM and original, ground truth CHM. Remove these for performance

    glGenTextures(1, &diff_rem_texture);
    glBindTexture(GL_TEXTURE_2D, diff_rem_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, diff_rem_texture, 0);

    glGenTextures(1, &diff_colors_texture);
    glBindTexture(GL_TEXTURE_2D, diff_colors_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, diff_colors_texture, 0);

    glGenTextures(1, &tree_placement_texture);
    glBindTexture(GL_TEXTURE_2D, tree_placement_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, 0);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, tree_placement_texture, 0);


    GLenum drawbuff [4] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3};
    glDrawBuffers(4, drawbuff);


    GLenum status = 0;

    if ((status = glCheckFramebufferStatus(GL_FRAMEBUFFER)) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cout << "CHM placement framebuffer construction failed: ";
        switch (status)
        {
            case (GL_FRAMEBUFFER_UNDEFINED):
                std::cout << "framebuffer undefined" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT):
                std::cout << "framebuffer contains incomplete attachment" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER):
                std::cout << "framebuffer contains an incomplete draw buffer" << std::endl;
                break;
            case (GL_FRAMEBUFFER_UNSUPPORTED):
                std::cout << "invalid combination of internal formats of framebuffer images" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT):
                std::cout << "framebuffer misses an attachment" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER):
                std::cout << "framebuffer contains incomplete read buffer" << std::endl;
                break;
            case (GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE):
            case (GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS):
                std::cout << "Either incomplete multisample or incomplete layer targets " << std::endl;
                break;
            case (GL_INVALID_ENUM):
            case (GL_INVALID_OPERATION):
                std::cout << "invalid framebuffer argument" << std::endl;
                break;
            default:
                cout << "unspecified error, code " << status << std::endl;
                break;
        }
        return;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void gl_wrapper::setup_diff_framebuffer()
{
    glGenFramebuffers(1, &diff_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, diff_framebuffer);

    glGenTextures(1, &diff_texture);

    glBindTexture(GL_TEXTURE_2D, diff_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, diff_texture, 0);

    GLenum drawbuffs [1] = {GL_COLOR_ATTACHMENT0};

    glDrawBuffers(1, drawbuffs);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        cout << "Framebuffer construction failed" << endl;
        return;
    }
}

void gl_wrapper::setup_summation_framebuffer(GLuint tex_w, GLuint tex_h)
{
    glGenFramebuffers(1, &summation_framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, summation_framebuffer);

    glGenTextures(1, &summation_texture);

    glBindTexture(GL_TEXTURE_2D, summation_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, tex_w, tex_h, 0, GL_RED, GL_FLOAT, 0);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, summation_texture, 0);

    GLenum drawbuffs [1] = {GL_COLOR_ATTACHMENT0};

    glDrawBuffers(1, drawbuffs);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        cout << "Summation framebuffer construction failed" << endl;
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void gl_wrapper::create_example_chm_texture()
{
    GLuint bufsize = width * height * sizeof(GLfloat);
    GLfloat *buffer = (GLfloat *)malloc(bufsize);

    memcpy(buffer, example_chm_data.data(), bufsize);

    glGenTextures(1, &example_texture);

    glBindTexture(GL_TEXTURE_2D, example_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, buffer);
    glBindTexture(GL_TEXTURE_2D, 0);

    free(buffer);
}

GLuint gl_wrapper::compile_shaders(string vert_filepath, string frag_filepath)
{
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

    shader_vert = glCreateShader(GL_VERTEX_SHADER);
    shader_frag = glCreateShader(GL_FRAGMENT_SHADER);

    const char * vert_c_str = vert_str.c_str();
    const char * frag_c_str = frag_str.c_str();

    glShaderSource(shader_vert, 1, &vert_c_str, NULL);
    glShaderSource(shader_frag, 1, &frag_c_str, NULL);

    glCompileShader(shader_vert);
    glCompileShader(shader_frag);

    if (check_compile_status(shader_vert) != GL_TRUE)
    {
        cout << "Vertex shader from file " << vert_filepath << " failed to compile" << endl;
        return 0;
    }
    if (check_compile_status(shader_frag) != GL_TRUE)
    {
        cout << "Fragment shader from file " << frag_filepath << " failed to compile" << endl;
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
        cout << "Failed to link shader program: ";
        cout << shader_message << endl;

        glDeleteProgram(shader_program);

        return 0;
    }


    glDeleteShader(shader_vert);
    glDeleteShader(shader_frag);

    return shader_program;
}

GLint gl_wrapper::check_compile_status(GLuint shader)
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
            cout << "compiler log: " << compiler_log << endl;
            free(compiler_log);
        }
    }
    return compiled;
}

void gl_wrapper::set_gl_attribs()
{
    if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4))
    {
        cout << "Could not set GL major version" << endl;
    }
    if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4))
    {
        cout << "Could not set GL minor version" << endl;
    }
    // XXX: Do some reading on this. It may noticeably affect the Z buffering
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE))
    {
        cout << "Could not set core profile" << endl;
    }


    /*
    cout << "Shading language version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
    cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
    cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << endl;
    */
}

void gl_wrapper::allocate_gl_buffers()
{
    cout << "generating buffers" << endl;
    //vaos = new GLuint [5];
    //vbos = new GLuint [5];
    vaos = (GLuint *)malloc(sizeof(GLuint) * 5);
    vbos = (GLuint *)malloc(sizeof(GLuint) * 5);
    ebos = (GLuint *)malloc(sizeof(GLuint) * 5);

    GLuint a;

    // move this to its own function later

    cout << "Generating vertex buffer" << endl;

    glGenVertexArrays(5, vaos);
    GL_ERRCHECK(false);

    glGenBuffers(5, vbos);
    GL_ERRCHECK(false);
    glGenBuffers(5, ebos);
    GL_ERRCHECK(false);

    glGenBuffers(1, &translate_matrix_vbo);
    GL_ERRCHECK(false);
    glGenBuffers(1, &scale_matrix_vbo);
    GL_ERRCHECK(false);
    glGenBuffers(1, &color_vec_vbo);
    GL_ERRCHECK(false);

    cout << "generated buffers" << endl;
}

void gl_wrapper::process_all_nodes(const aiScene *scene)
{
    process_node(scene->mRootNode, scene);
}

void gl_wrapper::process_all_meshes()
{
    for (auto &mesh : meshes)
    {
        process_mesh(mesh, scene);
    }
    verts_avg = glm::vec3(0.0, 0.0, 0.0);
    verts_max_dist = glm::vec3(0.0, 0.0, 0.0);
    if (vertices.size() >= 3)
        verts_avg /= (vertices.size() / 3);
    for (int i = 0; i < vertices.size(); i++)
    {
        int dim = i % 3;
        float dist = abs(vertices[i] - verts_avg[dim]);
        if (dist > verts_max_dist[dim])
        {
            verts_max_dist[dim] = dist;
        }
    }
    std::cout << "Finished processing all meshes" << std::endl;
}

void gl_wrapper::process_node(aiNode *node, const aiScene *scene)
{
    for (int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(mesh);
    }

    for (int i = 0; i < node->mNumChildren; i++)
    {
        process_node(node->mChildren[i], scene);
    }
}

void gl_wrapper::process_mesh(aiMesh *mesh, const aiScene *scene)
{
    num_meshes++;

    num_vertices = vertices.size() / 3;

    for (int i = 0; i < mesh->mNumVertices; i++)
    {
        vertices.push_back(mesh->mVertices[i].x);
        vertices.push_back(mesh->mVertices[i].y);
        vertices.push_back(mesh->mVertices[i].z);
        verts_avg += glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
    }

    for (int i = 0; i < mesh->mNumFaces; i++)
    {
        for (int j = 0; j < mesh->mFaces[i].mNumIndices; j++)
        {
            faces_idxes.push_back(num_vertices + mesh->mFaces[i].mIndices[j]);
        }
    }
}
