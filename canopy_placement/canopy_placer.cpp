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




#include "canopy_placer.h"
#include "gl_wrapper.h"
#include "gpu_procs.h"
#include "common/basic_types.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "data_importer/data_importer.h"

#include <chrono>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

int canopy_placer::memorydiv = 8;

#define MAX_SPECIES 64

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s (code: %d) %s %d\n", cudaGetErrorString(code), code, file, line);
      if (abort) exit(code);
   }
}

#define kernelCheck() gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

using namespace basic_types;

canopy_placer::canopy_placer(basic_types::MapFloat *chm, ValueMap<int> *species, const std::map<int, data_importer::species> &params_species, const data_importer::common_data &cdata)
    : data_w(chm->width()),
      data_h(chm->height()),
      niters(0),
      gl_renderer( new gl_wrapper(chm->data(), chm->width(), chm->height())),
      params_species(params_species),
      duplmap(chm->width(), chm->height()),
      el_alloc(data_w * data_h / memorydiv),
      orig_specmap(species),
      cdata(cdata)
{
    init_gpu();
    std::cout << "Initializing maps in canopy_placer..." << std::endl;
    init_maps(chm, species, params_species);
    std::cout << "canopy_placer construction done" << std::endl;
}

canopy_placer::~canopy_placer()
{
    free_cuda_resources();
    unregister_gl_resources();
    delete gl_renderer;
    // call unregister_gl_resources here?
}

void canopy_placer::free_cuda_resources()
{
    gpuErrchk(cudaFree(d_trees));
    gpuErrchk(cudaFree(d_new_trees));
    gpuErrchk(cudaFree(compaction_temp));
    gpuErrchk(cudaFree(d_ntrees));
    gpuErrchk(cudaFree(d_nlocs));
    gpuErrchk(cudaFree(d_nsampled));
    gpuErrchk(cudaFree(d_indicator_memspace));
    gpuErrchk(cudaFree(d_centers));
    gpuErrchk(cudaFree(d_translate_matrices));
    gpuErrchk(cudaFree(d_scale_matrices));
    gpuErrchk(cudaFree(d_color_vecs));
    gpuErrchk(cudaFree(d_chm));
    gpuErrchk(cudaFree(d_rstate_arr));
    gpuErrchk(cudaFree(d_species_map));
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));

    gpuErrchk(cudaFree(mem.whratios));

    std::vector<specmodels> models(nspecmodels);
    gpuErrchk(cudaMemcpy(models.data(), mem.models, sizeof(specmodels) * nspecmodels, cudaMemcpyDeviceToHost));
    for (auto &m : models)
    {
        m.free_memory();
    }
}

void canopy_placer::init_gpu()
{
    std::cout << "Registering GPU resources..." << std::endl;
    register_gl_resources();

    std::cout << "Allocating cuda buffers..." << std::endl;
    allocate_cuda_buffers();

    std::cout << "Initializing cuda curand..." << std::endl;
    ::init_curand_gpu(mem.d_rstate_arr, std::chrono::steady_clock::now().time_since_epoch().count(), el_alloc);
}


void canopy_placer::save_to_file(std::string filepath)
{
    gpuErrchk(cudaMemcpy(&ntrees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<mosaic_tree> trees(ntrees);
    cudaMemcpy(trees.data(), d_trees, sizeof(mosaic_tree) * ntrees, cudaMemcpyDeviceToHost);

    data_importer::write_pdb(filepath, trees.data(), trees.data() + trees.size());
}

std::map<int, int> canopy_placer::create_idx_to_species()
{
    std::map<int, int> idx_to_species;

    int count = 0;
    for (auto &pspec : params_species)
    {
        idx_to_species[count] = pspec.first;
        assert(pspec.first == pspec.second.idx);
        count++;
    }
    return idx_to_species;
}

std::map<int, int> canopy_placer::create_species_to_idx()
{
    std::map<int, int> species_to_idx;

    int count = 0;
    for (auto &pspec : params_species)
    {
        species_to_idx[pspec.first] = count;
        assert(pspec.first == pspec.second.idx);
        count++;
    }
    return species_to_idx;
}


bool canopy_placer::convert_trees_species(spec_convert convert)
{
    get_trees();		// ensure that treesholder has trees in
    if (convert == spec_convert::TO_ID)
    {
        auto idx_to_species = create_idx_to_species();
        for (auto &tree : treesholder)
        {
            assert(tree.species >= 0);
            tree.species = idx_to_species.at(tree.species);
        }
    }
    else
    {
        auto species_to_idx = create_species_to_idx();
        for (auto &tree : treesholder)
        {
            assert(tree.species >= 0);
            tree.species = species_to_idx.at(tree.species);
        }
    }
    if (treesholder.size() > 0)
        return true;
    else
        return false;
}

void canopy_placer::init_maps(basic_types::MapFloat *chm, ValueMap<int> *species, std::map<int, data_importer::species> params_species)
{
    data_w = chm->width();
    data_h = chm->height();
    cudaMemcpy(d_chm, chm->data(), sizeof(float) * data_w * data_h, cudaMemcpyHostToDevice);

    if (!species)
    {
        // if a species map is not given, we create a species map with only one species
        species = new ValueMap<int>();
        species->setDim(*chm);
        for (int y = 0; y < data_h; y++)
        {
            for (int x = 0; x < data_w; x++)
            {
                if (chm->get(x, y) > 1e-5)
                    species->set(x, y, 0);
                else
                    species->set(x, y, -1);
            }
        }
        data_importer::species dummy_sp = params_species.at(7);
        params_species.clear();
        //params_species.emplace_back("Species1", -2.0f, 1.0f);
        params_species.insert({dummy_sp.idx, dummy_sp});
    }
    else
    {
        // ensure that the species map is consistent with the CHM, and that it contains valid values
        /*
        for (int y = 0; y < data_h; y++)
        {
            for (int x = 0; x < data_w; x++)
            {
                if (chm->get(x, y) < 1e-5)
                {
                    species->set(x, y, -1);
                }
                else
                {
                    //if (species->get(x, y) >= params_species.size())
                    int spec_idx = species->get(x, y);
                    if (!params_species.count(spec_idx))
                    {
                        std::string errstr = "Species map given to canopy placer contains species indices that are not in existence: index: " + std::to_string(spec_idx);
                        errstr += ", x, y: " + std::to_string(x) + ", " + std::to_string(y);
                        throw std::invalid_argument(errstr.c_str());
                    }
                }
            }
        }
        */
    }

    this->params_species = params_species;

    std::vector<float> as, bs;

    all_species.clear();
    std::map<int, int> tempmap;
    int count = 0;
    for (auto &p : params_species)
    {
        data_importer::species &sp = p.second;
        all_species.push_back(sp);
        as.push_back(sp.a);
        bs.push_back(sp.b);
        tempmap.insert({sp.idx, count});
        count++;
    }

    // since the canopy placement algorithm only works with contiguous indices from 0 to nspecies - 1, we create a map that contains
    // only these contiguous indices. After the canopy placement is done, we can map from these contiguous indices to actual indices again
    // TODO: change the algorithm so that we can work with actual indices, not contiguous ones. (Could do this by sending an array to the GPU,
    // that maps from real to contiguous indices. The 'fake' contiguous indices can then be normally used to access the a and b parameters)
    ValueMap<int> temp_intmap;
    temp_intmap.setDim(*species);
    int w, h;
    temp_intmap.getDim(w, h);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int val = species->get(x, y);
            if (val > -1 && chm->get(x, y) > 1e-5)
                temp_intmap.set(x, y, tempmap.at(val));
            else
                temp_intmap.set(x, y, -1);
        }
    }

    assert(w == data_w && h == data_h);

    gpuErrchk(cudaMemcpy(d_species_map, temp_intmap.data(), sizeof(int) * data_w * data_h, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy(d_species_params, params_species.data(), sizeof(species_params) * params_species.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_a, as.data(), sizeof(float) * as.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, bs.data(), sizeof(float) * bs.size(), cudaMemcpyHostToDevice));

    duplmap.fill((unsigned char)0);
}

/*
 * REQUIRED:
 * - nothing (class only needs to be initialized)
 *
 * YIELDS:
 * - d_trees
 * - d_ntrees
 * - d_rendered_chm_texture
 */
void canopy_placer::init_optim()
{
    // TODO: create data_struct variable for chm
    std::cout << "entering find_local_maxima_trees..." << std::endl;
    find_local_maxima_trees();
    std::cout << "entering do_rendering..." << std::endl;
    do_rendering();
    std::cout << "done with do_rendering and init_optim" << std::endl;
}

void canopy_placer::allocate_cuda_buffers()
{
    gpuErrchk(cudaMalloc(&d_trees, sizeof(mosaic_tree) * el_alloc));
    gpuErrchk(cudaMalloc(&d_new_trees, sizeof(mosaic_tree) * el_alloc));
    gpuErrchk(cudaMalloc(&compaction_temp, sizeof(mosaic_tree) * el_alloc));
    gpuErrchk(cudaMalloc(&d_ntrees, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_nlocs, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_nsampled, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_indicator_memspace, sizeof(int) * el_alloc));
    gpuErrchk(cudaMalloc(&d_centers, sizeof(xy_avg) * el_alloc));
    gpuErrchk(cudaMalloc(&d_translate_matrices, sizeof(glm::mat4) * el_alloc));
    gpuErrchk(cudaMalloc(&d_scale_matrices, sizeof(glm::mat4) * el_alloc));
    gpuErrchk(cudaMalloc(&d_color_vecs, sizeof(glm::vec4) * el_alloc));
    gpuErrchk(cudaMalloc(&d_chm, sizeof(float) * data_w * data_h));
    std::cout << "Allocating " << el_alloc << " elements for curandstate array" << std::endl;
    gpuErrchk(cudaMalloc(&d_rstate_arr, sizeof(curandState) * el_alloc));
    gpuErrchk(cudaMalloc(&d_species_map, sizeof(int) * data_w * data_h));
    //gpuErrchk(cudaMalloc(&d_species_params, sizeof(species_params) * MAX_SPECIES));
    gpuErrchk(cudaMalloc(&d_a, sizeof(float) * MAX_SPECIES));
    gpuErrchk(cudaMalloc(&d_b, sizeof(float) * MAX_SPECIES));

    mem.d_trees = d_trees;
    mem.d_new_trees = d_new_trees;
    mem.compaction_temp = compaction_temp;
    mem.d_ntrees = d_ntrees;
    mem.d_nlocs = d_nlocs;
    mem.d_nsampled = d_nsampled;
    mem.d_indicator_memspace = d_indicator_memspace;
    mem.d_centers = d_centers;
    mem.d_translate_matrices = d_translate_matrices;
    mem.d_scale_matrices = d_scale_matrices;
    mem.d_color_vecs = d_color_vecs;
    mem.d_chm = d_chm;
    mem.d_rstate_arr = d_rstate_arr;
    mem.d_species_map = d_species_map;
    mem.d_a = d_a;
    mem.d_b = d_b;

    init_specmodels(cdata, create_species_to_idx(), &mem.whratios, &mem.models, nwhratios, nspecmodels);
}

void canopy_placer::register_gl_resources()
{
    glFlush();

    GL_ERRCHECK(true);

    SDL_GLContext ctx = SDL_GL_GetCurrentContext();
    if (!ctx)
    {
        std::runtime_error("No GL context is active");
    }
    else
    {
        std::cout << "GL context is active and valid" << std::endl;
    }

    gpuErrchk(cudaGetLastError());
    //while (cudaGetLastError() != cudaSuccess) {} // flush error flags before we start


    std::cout << "registering translate_matrix_vbo....." << std::endl;
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&bufres_translation, gl_renderer->translate_matrix_vbo, cudaGraphicsRegisterFlagsNone));
    std::cout << "registering scale_matrix_vbo..." << std::endl;
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&bufres_scale, gl_renderer->scale_matrix_vbo, cudaGraphicsRegisterFlagsNone));
    std::cout << "registering color_vec_vbo..." << std::endl;
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&bufres_color, gl_renderer->color_vec_vbo, cudaGraphicsRegisterFlagsNone));

    std::cout << "registering chm_placement_texture..." << std::endl;
    gpuErrchk(cudaGraphicsGLRegisterImage(&texres_chm_rendered, gl_renderer->chm_placement_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
    std::cout << "done with register_gl_resources" << std::endl;
}

void canopy_placer::unregister_gl_resources()
{
    cudaGraphicsUnregisterResource(bufres_translation);
    cudaGraphicsUnregisterResource(bufres_scale);
    cudaGraphicsUnregisterResource(bufres_color);
    cudaGraphicsUnregisterResource(texres_chm_rendered);
}

void canopy_placer::ready_cuda_texture_access()
{
    cudaArray_t inarr;
    cudaGraphicsMapResources(1, &texres_chm_rendered, 0);	// unmap this resource before rendering again???
    cudaGraphicsSubResourceGetMappedArray(&inarr, texres_chm_rendered, 0, 0);

    cudaResourceDesc descript;
    memset(&descript, 0, sizeof(cudaResourceDesc));
    descript.resType = cudaResourceTypeArray;
    descript.res.array.array = inarr;

    cudaTextureDesc texdesc;
    memset(&texdesc, 0, sizeof(cudaTextureDesc));
    texdesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&d_rendered_chm_texture, &descript, &texdesc, NULL);

    mem.d_rendered_chm_texture = d_rendered_chm_texture;

}

void canopy_placer::finish_cuda_texture_access()
{
    gpuErrchk(cudaDestroyTextureObject(d_rendered_chm_texture));
    gpuErrchk(cudaGraphicsUnmapResources(1, &texres_chm_rendered, 0));

}

/*
 * REQUIRED
 * - d_centers
 * - d_trees
 * - d_ntrees
 */
void canopy_placer::find_centers()
{
    //::find_centers_gpu(d_centers, d_trees, d_ntrees);		// TODO: define dcenters and dtrees device pointers
    ::find_centers_gpu(mem);
}

/*
 * REQUIRED:
 * - d_trees
 * - d_centers
 * - d_nlocs
 *
 * NOTE:
 * - also resets d_centers
 */
void canopy_placer::move_trees()
{
    //::move_trees_gpu(d_trees, d_centers, d_chm, data_w, d_species_map, d_a, d_b, d_nlocs);
    ::move_trees_gpu(mem, data_w);
}

/*
 * REQUIRED:
 * - d_trees
 * - d_centers
 * - d_ntrees
 *
 * d_nlocs will be assigned into and will contain a copy of the old number of locations
 */
void canopy_placer::remove_dominated_trees()
{
    //::rm_dominated_trees_gpu(d_trees, d_centers, d_ntrees, d_nlocs);
    ::rm_dominated_trees_gpu(mem);
}

void canopy_placer::get_chm_rendered_texture(std::vector<uint32_t> &result)
{
    result.clear();
    result.resize(data_w * data_h);
    std::fill(result.begin(), result.end(), 32);

    uint32_t *d_result;
    gpuErrchk(cudaMalloc(&d_result, sizeof(uint32_t) * data_w * data_h));

    get_cuda_texture_object_data(d_rendered_chm_texture, data_w, data_h, d_result);

    cudaMemcpy(result.data(), d_result, data_w * data_h * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaFree(d_result));
}


void canopy_placer::erase_duplicates_fast(std::vector<basic_tree> &trees, float rwidth, float rheight)
{
    ValueGridMap<bool> occmap;
    occmap.setDim(data_w, data_h);
    occmap.setDimReal(rwidth, rheight);

    std::vector<int> rmidxes;
    for (int i = trees.size() - 1; i >= 0; i--)
    {
        if (occmap.get_fromreal(trees.at(i).x, trees.at(i).y))
            rmidxes.push_back(i);
    }

    for (auto &idx : rmidxes)
        trees.erase(std::next(trees.begin(), idx));
}

void canopy_placer::erase_duplicates(std::vector<basic_tree> &trees, float rwidth, float rheight)
{
    float cellsize = 10.0f * 2;		// cellsize is equal to x2 maximum trunk radius
    int nccols = std::ceil(rwidth / cellsize) + 1e-5f;
    int ncrows = std::ceil(rheight / cellsize) + 1e-5f;
    float trunk_ratio = 0.2f;		// trunk radius to canopy radius ratio

    std::vector< std::vector< std::vector<int> > > tree_idxes(ncrows, std::vector< std::vector<int> >(nccols));

    //auto trees = get_trees_basic_rw_coords();

    for (int i = 0; i < trees.size(); i++)
    {
        auto &t = trees[i];
        int cx = t.x / cellsize;
        int cy = t.y / cellsize;
        tree_idxes.at(cy).at(cx).push_back(i);
    }

    std::vector<int> to_remove(trees.size(), 0);

    std::vector<int> rm_idxes;

    int q = 0;
    int longtree_rems = 0;
    int close_rems = 0;

    for (int cy = 0; cy < ncrows; cy++)
    {
        for (int cx = 0; cx < nccols; cx++)
        {
            auto &cellvec = tree_idxes.at(cy).at(cx);
            for (auto &tidx : cellvec)
            {
                if (to_remove.at(tidx))
                    continue;
                float tx = trees.at(tidx).x;
                float ty = trees.at(tidx).y;
                float tr = trees.at(tidx).radius * trunk_ratio;
                float rad = trees.at(tidx).radius;
                for (int y = cy - 1; y <= cy + 1; y++)
                {
                    for (int x = cx - 1; x <= cx + 1; x++)
                    {
                        if (x < 0 || x >= nccols || y < 0 || y >= ncrows)
                        {
                            continue;
                        }

                        auto &ocellvec = tree_idxes.at(y).at(x);
                        for (auto &otidx : ocellvec)
                        {
                            if (otidx == tidx)
                            {
                                if (x != cx || y != cy)
                                {
                                    throw std::runtime_error("Error: same tree index found in different cells, in canopy_placer::check_duplicates");
                                }
                                continue;
                            }
                            else if (to_remove.at(otidx))
                            {
                                continue;
                            }
                            float otx = trees.at(otidx).x;
                            float oty = trees.at(otidx).y;
                            float otr = trees.at(otidx).radius * trunk_ratio;
                            float orad = trees.at(otidx).radius;
                            //float mindistsq = (otr + tr) * (otr + tr);
                            float mindist = 1.0f;
                            float distsq = (tx - otx) * (tx - otx) + (ty - oty) * (ty - oty);
                            float dist = sqrt(distsq);
                            float largerad = rad >= orad ? rad : orad;
                            float smallrad = rad >= orad ? orad : rad;
                            int smallradidx = rad >= orad ? otidx : tidx;
                            int bigradidx = smallradidx == tidx ? otidx : tidx;
                            if (dist < mindist)		// trees are within the absolute minimum allowable distance of each other. Remove one with smallest radius
                            {
                                bool remove = true;
                                if (largerad + smallrad < mindist * 1.5f && dist > mindist * 0.5f)		// if both trees' radiuses are small, then we relax the minimum distance by a half of itself
                                    remove = false;
                                if (remove)
                                {
                                    to_remove.at(smallradidx) = 1;
                                    close_rems++;
                                }
                            }
                            else if (dist < largerad - smallrad)		// tree with smaller radius is completely inside larger radius tree, if viewed from above. Remove smaller radius tree
                            {
                                to_remove.at(smallradidx) = 1;
                                if (trees.at(smallradidx).height < trees.at(bigradidx).height * 0.85f)
                                {
                                    q++;
                                }
                                else
                                {
                                    longtree_rems++;
                                }
                                /*
                                if (trees.at(smallradidx).height < trees.at(bigradidx).height)
                                {
                                    throw std::runtime_error("Smaller tree completely under larger tree in canopy_placer::check_duplicates. Canopy placement algorithm should have picked this up");
                                }
                                */
                            }
                        }
                    }
                }
            }
        }
    }

    int nrem = 0;
    for (int i = to_remove.size() - 1; i >= 0; i--)
    {
        if (to_remove.at(i))
        {
            trees.erase(std::next(trees.begin(), i));
            nrem++;
        }
    }

    std::cout << nrem << " trees removed in canopy_placer::erase_duplicates" << std::endl;
    std::cout << "Questionable removals (should have been detected by canopy placement): " << q << std::endl;
    std::cout << "Removals due to long, thin trees: " << longtree_rems << std::endl;
    std::cout << "Removals due to minimum distance violations: " << close_rems << std::endl;

}


/*
 * REQUIRED:
 * - d_chm						(type: float *)
 * - d_chm_rendered_texture		(type: cudaTextureObject_t)
 * - data_w						(type: int)
 * - data_h						(type: int)
 * - d_rstate_arr				(type: curandState *)
 */
void canopy_placer::sample_new_trees(bool radially)
{
    ready_cuda_texture_access();

    int nsampled = el_alloc;
    if (!radially)
    {
        ::sample_new_trees_gpu(d_chm, d_rendered_chm_texture, data_w, data_h, d_species_map, d_a, d_b, d_new_trees, d_rstate_arr);
    }
    else
    {
        std::cout << "Doing radial sample..." << std::endl;
        int tempsmult = sample_mult;
        if (ntrees * sample_mult >= el_alloc)
        {
            tempsmult = el_alloc / ntrees;
        }
        ::sample_radial_trees_gpu(mem, data_w, data_h, ntrees, tempsmult);
        nsampled = ntrees * tempsmult;
    }

    std::cout << "Doing memset for " << nsampled << " mosaic trees, " << el_alloc << " mosaic_trees allocated..." << std::endl;
    nsampled = compact_and_assign(d_trees + ntrees, d_new_trees, nsampled);

    std::cout << "Number of trees sampled: " << nsampled << std::endl;

    ntrees += nsampled;
    gpuErrchk(cudaMemcpy(d_ntrees, &ntrees, sizeof(int), cudaMemcpyHostToDevice));

    finish_cuda_texture_access();
}




/*
 * REQUIRED:
 * - d_chm
 * - data_w, data_h, a, b
 * - d_species_map
 * - d_species_params
 *
 * writes into:
 * - d_trees
 * - d_ntrees
 * ntrees;
 */
void canopy_placer::find_local_maxima_trees()
{
    mosaic_tree *d_temp_trees;
    gpuErrchk(cudaMalloc(&d_temp_trees, sizeof(mosaic_tree) * data_w * data_h));
    gpuErrchk(cudaMemset(d_temp_trees, 0, sizeof(mosaic_tree) * data_w * data_h));
    int nlocal_maxima = ::find_local_maxima_trees_gpu(mem, d_temp_trees, data_w, data_h, 10.0f);
    gpuErrchk(cudaMemcpy(&ntrees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost));
    ntrees = compact_and_assign(mem.d_trees, d_temp_trees, nlocal_maxima);
    std::cout << "Number of trees found initially: " << ntrees << std::endl;
    gpuErrchk(cudaFree(d_temp_trees));

    int test_ntrees;
    cudaMemcpy(&test_ntrees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost);
    assert(test_ntrees == ntrees);

}

/*
 * REQUIRED:
 * - d_rendered_chm_texture		(type: cudaTextureObject_t)
 * - d_chm						(type: float *)
 * - data_w						(type: int)
 * - data_h						(type: int)
 *
 * Requires cuda texture from gl interop to already be rendered with the current tree locations
 */
void canopy_placer::populate_centers()
{
    ready_cuda_texture_access();

    //::populate_centers_gpu(d_rendered_chm_texture, d_chm, d_centers, data_w, data_h);
    ::populate_centers_gpu(mem, data_w, data_h);

    finish_cuda_texture_access();
}

ValueMap<int> canopy_placer::get_species_texture_raw()
{
    ready_cuda_texture_access();

    auto species_texture = ::apply_species_to_rendered_texture_gpu(d_rendered_chm_texture, d_trees, data_w, data_h);

    finish_cuda_texture_access();

    return species_texture;
}

ValueMap<int> canopy_placer::get_species_texture()
{
    auto texdata = get_species_texture_raw();
    int w, h;
    texdata.getDim(w, h);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int idx = texdata.get(x, y);
            int species_idx;
            if (idx > -1)
            {
                species_idx = all_species[idx].idx;
            }
            else
            {
                species_idx = -1;
            }
            texdata.set(x, y, species_idx);
        }
    }
    return texdata;
}

void canopy_placer::save_species_texture(std::string out_filename)
{
    auto texdata = get_species_texture();
    data_importer::write_txt(out_filename, &texdata);
}

void canopy_placer::save_rendered_texture(std::string out_filename)
{
    std::vector<uint32_t> result;
    get_chm_rendered_texture(result);

    ValueMap<uint32_t> vmap_result(data_w, data_h);

    assert(data_w * data_h == result.size());

    memcpy(vmap_result.data(), result.data(), sizeof(uint32_t) * data_w * data_h);

    data_importer::write_txt(out_filename, &vmap_result);
}

int canopy_placer::get_ntrees()
{
    return ntrees;
}

std::vector<mosaic_tree> canopy_placer::get_trees()
{
    if (treesholder.size() == 0)
        treesholder = get_trees_from_gpu();
    return treesholder;
}

std::vector<basic_tree> canopy_placer::get_trees_basic_rw_coords()
{
    get_trees();
    std::vector<basic_tree> btrees;
    for (auto &mtree : treesholder)
    {
        float modx = fmod(mtree.x, 1.0f);
        float modx2 = fmod(mtree.x, 0.5f);
        btrees.push_back(basic_tree(mtree.x * 0.9144f, mtree.y * 0.9144f, mtree.radius * 0.9144f, mtree.height));
        btrees.back().species = mtree.species;
    }
    return btrees;
}


std::vector<basic_tree> canopy_placer::get_trees_basic()
{
    get_trees();
    std::vector<basic_tree> btrees;
    for (auto &mtree : treesholder)
    {
        btrees.push_back(basic_tree(mtree.x, mtree.y, mtree.radius, mtree.height));
        btrees.back().species = mtree.species;
    }
    return btrees;
}

void canopy_placer::update_treesholder()
{
    treesholder = get_trees_from_gpu();
}

/*
 * REQUIRED:
 * - d_centers
 *
 * dummy function - centers are reset in move_trees
 */
void canopy_placer::reset_centers()
{
}

/*
 * REQUIRED:
 * - d_trees		(type: mosaic_tree *, array)
 * - ntrees			(type: int)
 * - gl_renderer	(type: std::unique_ptr<gl_wrapper>)
 *
 * cuda gl interop resources must also be initialized
 */
void canopy_placer::do_rendering()
{
    glFlush();
    glFinish();

    std::vector<float> arr(100, 15);
    GL_ERRCHECK(false);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL_ERRCHECK(false);

    std::vector<glm::mat4> cpumatrices(el_alloc, glm::mat4(0.0f));
    std::vector<glm::vec4> cpuvecs(el_alloc, glm::vec4(0.0f));
    gpuErrchk(cudaMemcpy(mem.d_translate_matrices, cpumatrices.data(), el_alloc * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mem.d_scale_matrices, cpumatrices.data(), el_alloc * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mem.d_color_vecs, cpuvecs.data(), el_alloc * sizeof(glm::vec4), cudaMemcpyHostToDevice));

    ::create_world_matrices_and_colvecs_gpu(mem, ntrees);
    GL_ERRCHECK(false);

    ::send_gl_buffer_data_gpu(&bufres_translation, mem.d_translate_matrices, ntrees * sizeof(glm::mat4));
    ::send_gl_buffer_data_gpu(&bufres_scale, mem.d_scale_matrices, ntrees * sizeof(glm::mat4));
    ::send_gl_buffer_data_gpu(&bufres_color, mem.d_color_vecs, ntrees * sizeof(glm::vec4));

    int printnum = 100;

    GL_ERRCHECK(false);
    glBindBuffer(GL_ARRAY_BUFFER, gl_renderer->translate_matrix_vbo);
    float *dataptr = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, gl_renderer->scale_matrix_vbo);
    dataptr = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, gl_renderer->color_vec_vbo);
    dataptr = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    glUnmapBuffer(GL_ARRAY_BUFFER);

    GL_ERRCHECK(false);

    cudaDeviceSynchronize();
    gl_renderer->render_via_gpu(ntrees);
}

/*
 * REQUIRED
 * - d_trees
 * - d_ntrees
 *
 * updates: d_ntrees, d_trees
 */
void canopy_placer::compact_trees()
{
    std::cerr << "Running compact_and_assign in compact_trees to copy " << ntrees << "..." << std::endl;
    ntrees = compact_and_assign(mem.compaction_temp, mem.d_trees, ntrees);
    std::cerr << "Done running compact_and_assign in compact_trees..." << std::endl;
    cudaMemcpy(mem.d_trees, mem.compaction_temp, sizeof(mosaic_tree) * ntrees, cudaMemcpyDeviceToDevice);
    cudaMemcpy(mem.d_ntrees, &ntrees, sizeof(int), cudaMemcpyHostToDevice);
}

void canopy_placer::optimise(int max_iters)
{
    init_optim();	// this initializes d_trees, d_ntrees, and d_rendered_chm_texture
    for (int i = 0; i < max_iters; i++)
    {
        iteration();
    }
    final_adjustments_gpu();

    treesholder = get_trees_from_gpu();
}


void canopy_placer::final_adjustments_gpu()
{
}

void canopy_placer::write_chm_data_to_file(std::string out_file)
{
    std::vector<uint32_t> values;
    get_chm_rendered_texture(values);
    ValueMap<uint32_t> outmap;
    outmap.setDim(data_w, data_h);
    memcpy(outmap.data(), values.data(), sizeof(uint32_t) * data_w * data_h);

    data_importer::write_txt(out_file, &outmap);
}

std::vector<mosaic_tree> canopy_placer::get_trees_from_gpu()
{
    int num_trees;
    cudaMemcpy(&num_trees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<mosaic_tree> retval(num_trees);
    cudaMemcpy(retval.data(), d_trees, sizeof(mosaic_tree) * (num_trees), cudaMemcpyDeviceToHost);
    return retval;
}

void canopy_placer::check_species_outofbounds(std::string locdesc)
{
    std::string msg = "Species index out of bounds ";
    msg += locdesc;

    int maxspec = orig_specmap->calcmax();
    int minspec = orig_specmap->calcmin();

    auto trees = get_trees_from_gpu();

    for (auto &t : trees)
    {
        if (t.species > maxspec || t.species < minspec)
        {
            std::cout << "maxspec, minspec: " << maxspec << ", " << minspec << std::endl;
            std::cout << "t.species: " << t.species << std::endl;
            throw std::runtime_error(msg.c_str());
        }
    }

}

void canopy_placer::iteration()
{
    GL_ERRCHECK(false);

    niters++;

    gpuErrchk(cudaGetLastError());
    reset_centers_gpu(d_centers, el_alloc);

    gpuErrchk(cudaGetLastError());
    // requires d_rendered_chm_texture and d_chm, both initialized by init_optim (and the former is updated by do_rendering)
    // updates: d_centers
    populate_centers();


    gpuErrchk(cudaGetLastError());
    // requires: d_centers and d_trees
    // updates: d_centers
    find_centers();

    std::cout << "trees left before removing dominated trees: " << ntrees << std::endl;
    gpuErrchk(cudaGetLastError());
    //check_species_outofbounds(" before remove_dominated_trees in canopy_placer::iteration");
    // requires: d_trees, d_centers
    // updates: d_ntrees and d_trees	(note, this does NOT compact the tree array: it simply updates the state of each tree in d_trees)
    remove_dominated_trees();	// after this, *d_nlocs == *d_ntrees


    gpuErrchk(cudaGetLastError());
    //check_species_outofbounds(" before move_trees in canopy_placer::iteration (after remove_dominated_trees)");
    // requires: d_trees, d_centers
    // updates: d_trees
    move_trees();
    //reset_centers_gpu(d_centers, ntrees);
    gpuErrchk(cudaMemcpy(&ntrees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost));
    //std::cout << "ntrees before removing dominated trees: " << ntrees << std::endl;

    //check_species_outofbounds(" after move_trees in canopy_placer::iteration");

    gpuErrchk(cudaGetLastError());
    // requires: d_trees, d_ntrees
    // updates: d_trees, d_ntrees
    compact_trees();	// compacting trees erases the correlation between color and location
                        // once we compact, we cannot locate a tree in the array anymore based on its color.
                        // We therefore compact only when operations on existing trees with the rendered texture is done.

    gpuErrchk(cudaMemcpy(&ntrees, d_ntrees, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "trees left after removing dominated trees and compacting: " << ntrees << std::endl;
                        // after this, *d_nlocs != *d_ntrees (d_ntrees refers to the actual number of trees now)
    //std::cout << "ntrees after removing dominated trees: " << ntrees << std::endl;

    gpuErrchk(cudaGetLastError());
    do_rendering();
    // requires: d_rendered_chm_texture, d_chm	(note we can use the d_rendered_chm_texture here, despite color indices not correlating with
    // 												the right trees anymore, because we only need to check if a tree canopy exists in a given location
    //												- it does not matter which tree it is)
    //check_species_outofbounds(" before sample_new_trees in canopy_placer::iteration");
    gpuErrchk(cudaGetLastError());
    std::cout << "number of trees before sampling new trees: " << ntrees << std::endl;
    if (niters == 1)
        sample_new_trees(true);	// d_ntrees refers to old ntrees + new ntrees
    else
        sample_new_trees(true);
    std::cout << "number of trees after sampling new trees: " << ntrees << std::endl;

    gpuErrchk(cudaGetLastError());
    //check_species_outofbounds(" after sample_new_trees in canopy_placer::iteration");
    // requires: d_trees, d_ntrees
    do_rendering();

    gpuErrchk(cudaGetLastError());
    populate_centers();
    gpuErrchk(cudaGetLastError());
    remove_dominated_trees();
    gpuErrchk(cudaGetLastError());
    compact_trees();

    std::cout << "Number of trees after culling newly sampled trees: " << ntrees << std::endl;
    //eliminate_proxims();

    gpuErrchk(cudaGetLastError());

    do_rendering();
}
