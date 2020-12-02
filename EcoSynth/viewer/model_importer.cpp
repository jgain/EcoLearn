#include "model_importer.h"

#include "CImg.h"

#include <assimp/types.h>
#include <assimp/scene.h>
#include <assimp/mesh.h>
#include <assimp/postprocess.h>

//#include <SOIL.h>

#include <glm/glm.hpp>

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

#include <iostream>

using namespace cimg_library;

model_importer::model_importer()
{}

model_importer::model_importer(std::string model_filename)
{
    size_t leaf_idx = model_filename.find_last_of("/\\");
    std::string directory = model_filename.substr(0, leaf_idx) + "/";
    import(model_filename, directory);
}

void model_importer::import(std::string model_filename, std::string directory)
{
    Assimp::Importer assimp_importer;
    this->directory = directory;
    const aiScene *scene = assimp_importer.ReadFile(model_filename, aiProcess_Triangulate | aiProcess_GenUVCoords | aiProcess_TransformUVCoords | aiProcess_FlipUVs);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error("Could not import assimp model at " + model_filename);
    }
    process_all_nodes(scene);
    process_all_meshes(scene);

    if (scene->HasTextures())
    {
        std::cerr << "Assimp model already has textures" << std::endl;
    }
    else
    {
        /*
        aiString str;
        scene->mMaterials[0]->GetTexture(aiTextureType_DIFFUSE, 0, &str);
        std::cerr << "Texture path: " << str.C_Str() << std::endl;
        */
    }

    std::cerr << "Number of materials: " << scene->mNumMaterials << std::endl;

    using namespace cimg_library;

    for (int i = 0; i < scene->mNumMaterials; i++)
    {
        aiColor3D color;
        aiString name;
        aiString texpath;
        aiTextureMapping texmap;
        std::cerr << "Number of diffuse, specular, ambient, emissive, bump: ";
        std::cerr << scene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE) << ", ";
        std::cerr << scene->mMaterials[i]->GetTextureCount(aiTextureType_SPECULAR) << ", ";
        std::cerr << scene->mMaterials[i]->GetTextureCount(aiTextureType_EMISSIVE) << ", ";
        std::cerr << scene->mMaterials[i]->GetTextureCount(aiTextureType_HEIGHT) << std::endl;
        //scene->mMaterials[i]->Get(AI_MATKEY_TEXTURE(0, 0), texpath);
        //std::cerr << "texture path: " << texpath.C_Str() << std::endl;
        scene->mMaterials[i]->Get(AI_MATKEY_NAME, name);
        std::cerr << "Material name: " << name.C_Str() << std::endl;
        scene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, 0, &texpath, &texmap);
        if (texpath.length > 0)
        {
            std::string texture_path = directory + texpath.C_Str();
            std::cerr << "texture path: " << texture_path << std::endl;
            teximage teximg;
            //teximg.data = SOIL_load_image(texture_path.c_str(), &teximg.width, &teximg.height, &teximg.channels, SOIL_LOAD_AUTO);
            CImg<unsigned char> img(texture_path.c_str());
            assert(img.data());
            //img.display();
            /*
            teximg.data = img.data();
            teximg.width = img.width();
            teximg.height = img.height();
            teximg.channels = img.spectrum();
            */

            teximages.push_back(img);
            std::cerr << "Texture width, height, channels: " << img.width() << ", " << img.height() << ", " << img.spectrum() << std::endl;
            //textures.push_back(SOIL_load_OGL_texture(texture_path.c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, 0));
            //std::cerr << "loaded texture id: " << textures.back() << std::endl;
        }
        else
        {
            std::cerr << "Material has no texture path" << std::endl;
            teximages.push_back(CImg<unsigned char> ());
        }
        std::cerr << "Texture mapping: " << texmap << std::endl;
        scene->mMaterials[i]->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        std::cerr << "Material diffuse color: " << color.r << ", " << color.g << ", " << color.b << ", " << std::endl;
        scene->mMaterials[i]->Get(AI_MATKEY_COLOR_SPECULAR, color);
        std::cerr << "Material specular color: " << color.r << ", " << color.g << ", " << color.b << ", " << std::endl;
        scene->mMaterials[i]->Get(AI_MATKEY_COLOR_REFLECTIVE, color);
        std::cerr << "Material reflective color: " << color.r << ", " << color.g << ", " << color.b << ", " << std::endl;
        std::cerr << "-------------------------------------------------------------------------" << std::endl;
    }

    std::cerr << "vertices average position: " << verts_avg.x << ", " << verts_avg.y << ", " << verts_avg.z << std::endl;
    std::cerr << "vertices maximum distance from origin: " << verts_max_dist.x << ", " << verts_max_dist.y << ", " << verts_max_dist.z << std::endl;
    std::cerr << "Number of vertices: " << vertices.size() << std::endl;
    std::cerr << "Number of indicies: " << faces_idxes.size() << std::endl;

}

/*
model_importer::~model_importer()
{
    for (auto &teximg : teximages)
    {
        free(teximg.data);
    }
}
*/

void model_importer::normalize_vertices_height()
{
    float dist = verts_max_dims.y - verts_min_dims.y;
    float scale = 1 / dist;

    for (int i = 0; i < vertices.size(); i++)
    {
        int dim = i % 3;
        vertices[i] = (vertices[i] - verts_avg[dim]) * scale + verts_avg[dim];
    }
}

uint8_t *model_importer::get_img_gldata(CImg<unsigned char> &img)
{
    uint8_t *pixels = (uint8_t *)malloc(sizeof(uint8_t) * img.width() * img.height() * img.spectrum());
    int i = 0;
    for (int channel = 0; channel < img.spectrum(); channel++)
    {
        for (int j = channel; i < img.width() * img.height() * (channel + 1); i++, j += img.spectrum())
        {
            pixels[j] = *(img.data() + i);
        }
    }
    return pixels;
}

void model_importer::process_all_nodes(const aiScene *scene)
{
    process_node(scene->mRootNode, scene);
}

void model_importer::process_all_meshes(const aiScene *scene)
{
    for (auto &mesh : meshes)
    {
        process_mesh(mesh, scene);
    }
    verts_avg = glm::vec3(0.0, 0.0, 0.0);
    verts_max_dist = glm::vec3(0.0, 0.0, 0.0);
    verts_max_dims = glm::vec3(-std::numeric_limits<float>::max());
    verts_min_dims = glm::vec3(std::numeric_limits<float>::max());
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
        float &v = vertices[i];
        if (v < verts_min_dims[dim])
            verts_min_dims[dim] = v;
        if (v > verts_max_dims[dim])
            verts_max_dims[dim] = v;
    }
    std::cout << "Finished processing all meshes" << std::endl;
    std::cout << "Number of triangles, polygons: " << num_triangles << ", " << num_polygons << std::endl;
}

void model_importer::process_node(aiNode *node, const aiScene *scene)
{
    std::cerr << "Processing node: " << node->mName.C_Str() << std::endl;

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

void model_importer::process_mesh(aiMesh *mesh, const aiScene *scene)
{
    num_meshes++;

    num_vertices = vertices.size() / 3;

    for (int i = 0; i < mesh->mNumVertices; i++)
    {
        vertices.push_back(mesh->mVertices[i].x);
        vertices.push_back(mesh->mVertices[i].y);
        vertices.push_back(mesh->mVertices[i].z);
        verts_avg += glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);

        //std::cerr << "Number of UV channels: " << mesh->GetNumUVChannels() << std::endl;
        for (int j = 0; j < mesh->GetNumUVChannels(); j++)
            if (mesh->HasTextureCoords(j))
            {
                //std::cerr << "mesh has texture coordinate set" << std::endl;
                aiVector3D texcoord = mesh->mTextureCoords[j][i];
                texcoords.push_back(texcoord.x);
                texcoords.push_back(texcoord.y);
                //std::cerr << "texcoord: " << texcoord.x << ", " << texcoord.y << ", " << texcoord.z << std::endl;
                assert(texcoord.z == 0);	// ensure that texture coordinates have only two value. Otherwise we not working with texture coordinates...
                sampler_idxes.push_back(mesh->mMaterialIndex);
                //std::cerr << "sampler index: " << sampler_idxes.back() << std::endl;
                assert(sampler_idxes.back() != 0);	// ensure that sampler indexes start at 1, not 0
            }
    }

    for (int i = 0; i < mesh->mNumFaces; i++)
    {
        if (mesh->mFaces[i].mNumIndices == 3)
        {
            num_triangles++;
        }
        else if (mesh->mFaces[i].mNumIndices > 3)
        {
            num_polygons++;
        }
        for (int j = 0; j < mesh->mFaces[i].mNumIndices; j++)
        {
            faces_idxes.push_back(num_vertices + mesh->mFaces[i].mIndices[j]);
        }
    }
}

std::vector<float> &model_importer::get_vertices()
{
    return vertices;
}

std::vector<int> &model_importer::get_idxes()
{
    return faces_idxes;
}

std::vector<int> &model_importer::get_tex_idxes()
{
    return sampler_idxes;
}

std::vector<CImg<unsigned char> > &model_importer::get_teximages()
{
    return teximages;
}

std::vector<float> &model_importer::get_texcoords()
{
    return texcoords;
}
