#ifndef MODEL_IMPORTER_H
#define MODEL_IMPORTER_H

#define GL_IMPORTED

#ifndef GL_IMPORTED
#define GL_IMPORTED
#include <GL/glew.h>
#include <GL/gl.h>
#endif

//#include <GL/glew.h>

#define cimg_display 0
#include "CImg.h"

#include <assimp/mesh.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <glm/glm.hpp>

#include <string>
#include <vector>
#include <map>

struct teximage
{
    unsigned char * data;
    int width, height;
    int channels;
};

class model_importer
{
public:
    model_importer();
    model_importer(std::string model_filename);
    //~model_importer();
    void process_all_nodes(const aiScene *scene);
    void process_all_meshes(const aiScene *scene);
    void process_node(aiNode *node, const aiScene *scene);
    void process_mesh(aiMesh *mesh, const aiScene *scene);

    std::vector<float> &get_vertices();
    std::vector<int> &get_idxes();
    std::vector<int> &get_tex_idxes();
    std::vector<cimg_library::CImg<unsigned char> > &get_teximages();
    std::vector<float> &get_texcoords();
    void normalize_vertices_height();
    static uint8_t *get_img_gldata(cimg_library::CImg<unsigned char> &img);
    void import(std::string model_filename, std::string directory);
private:
    std::vector<float> vertices;
    std::vector<int> faces_idxes;
    std::vector<aiMesh *> meshes;
    std::vector<float> texcoords;
    std::vector<int> sampler_idxes;
    std::vector<int> textures;
    std::vector<cimg_library::CImg<unsigned char> > teximages;
    //Assimp::Importer assimp_importer;

    int num_meshes;
    int num_vertices;
    int num_triangles = 0;
    int num_polygons = 0;

    glm::vec3 verts_max_dist;
    glm::vec3 verts_avg;
    glm::vec3 verts_max_dims;
    glm::vec3 verts_min_dims;

    std::string directory;
};

#endif
