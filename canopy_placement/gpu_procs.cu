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


//#include "mosaic_spacing.h"
#include "common/basic_types.h"
#include <iostream>

#include "gpu_procs.h"

#include "canopy_placement/canopy_placer.h"
#include "data_importer/data_importer.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include <vector_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include <thrust/transform_scan.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

#include <glm/glm.hpp>


#include <vector>

#define CUDA_ERRCHECK(errcode) \
    if (cudaSuccess != errcode) { \
        std::cout << cudaGetErrorString(errcode) << " in file " << __FILE__ << ", line " << __LINE__ << std::endl; \
    }
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define kernelCheck() gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());


modelselection::modelselection(const std::vector<int> &select)
{
    if (select.size() > 0)
    {
        nmodels = select.size();
        gpuErrchk(cudaMalloc(&vueids, sizeof(int) * nmodels));
        gpuErrchk(cudaMemcpy(vueids, select.data(), sizeof(int) * select.size(), cudaMemcpyHostToDevice));
    }
}

__device__
int modelselection::sample(curandState *rstate)
{
    float unif = curand_uniform(rstate);
    int idx = nmodels * unif;
    return vueids[idx];
}

__host__
void modelselection::free_memory()
{
    gpuErrchk(cudaFree(vueids));
}

specmodels::specmodels(const data_importer::modelset &mset, float *model_whratios)
{
    binsize = mset.binsize;
    nbins = mset.ranges.size() - 1;
    this->model_whratios = model_whratios;

    if (nbins < 0)
    {
        ranges = nullptr;
        selects = nullptr;
        samplemap = nullptr;
    }
    else
    {
        gpuErrchk(cudaMalloc(&ranges, mset.ranges.size() * sizeof(float)));
        gpuErrchk(cudaMemcpy(ranges, mset.ranges.data(), sizeof(float) * mset.ranges.size(), cudaMemcpyHostToDevice));
        std::cout << "Ranges: ";
        for (auto &r : mset.ranges)
        {
            std::cout << r << " ";
        }
        std::cout << "      (nbins = " << nbins << ")" << std::endl;

        std::vector<modelselection> mselects;
        for (auto &sel : mset.selections)
        {
            mselects.push_back(modelselection(sel));
        }
        gpuErrchk(cudaMalloc(&selects, mset.selections.size() * sizeof(modelselection)));
        gpuErrchk(cudaMemcpy(selects, mselects.data(), sizeof(modelselection) * mselects.size(), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMalloc(&samplemap, mset.samplemap.size() * sizeof(int)));
        gpuErrchk(cudaMemcpy(samplemap, mset.samplemap.data(), sizeof(int) * mset.samplemap.size(), cudaMemcpyHostToDevice));
        samplemap_size = mset.samplemap.size();
    }
}

specmodels::specmodels()
{
    binsize = 0.0f;
    nbins = -1;
    samplemap_size = -1;

    ranges = nullptr;
    selects = nullptr;
    samplemap = nullptr;
    model_whratios = nullptr;

}

__device__
float specmodels::sample(curandState *rstate, float height)
{
    if (nbins < 0 || samplemap_size < 0)
    {
        printf("Error in specmodels::sample device function: attempting to sample from an empty specmodel instance\n");
        asm("trap;");
    }

    if (height > ranges[nbins])
    {
        printf("Error in specmodels::sample device function: plant height out of bounds - plant height: %f, upper range height: %f\n", height, ranges[nbins]);
        asm("trap;");
    }

    int idx = height / binsize;
    if (idx >= samplemap_size)
    {
        printf("Error in specmodels::sample device function: sample index out of bounds - plant height: %f, binsize: %f, samplemap size: %d\n", height, binsize, samplemap_size);
        asm("trap;");
    }

    int selidx = samplemap[idx];
    if (idx < nbins - 1 && samplemap[idx + 1] != samplemap[idx])
    {
        if (height >= ranges[samplemap[idx + 1]])
        {
            selidx = samplemap[idx + 1];
        }
    }
    if (selidx >= nbins)
    {
        printf("Error in specmodels::sample device function: selection index out of bounds\n");
        asm("trap;");
    }

    int vidx = selects[selidx].sample(rstate);
    if (vidx > 50 || vidx < 0)
    {
        printf("vue model index out of bounds: %d\n", vidx);
        asm("trap;");
    }
    //return 0.2f;
    return model_whratios[vidx] / 2.0f;
}

void specmodels::free_memory()
{
    std::vector<modelselection> cpuselects(nbins, modelselection({}));
    gpuErrchk(cudaMemcpy(cpuselects.data(), selects, sizeof(modelselection) * nbins, cudaMemcpyDeviceToHost));
    for (auto &sel : cpuselects)
    {
        sel.free_memory();
    }

    gpuErrchk(cudaFree(ranges));
    gpuErrchk(cudaFree(samplemap));
    //gpuErrchk(cudaFree(model_whratios));
}

void init_specmodels(const data_importer::common_data &cdata, const std::map<int, int> &species_to_idx, float **ratios, specmodels **models, int &nratios, int &nspecmodels)
{
    //specmodels *models;

    const std::map<int, data_importer::modelset> &msamplers = cdata.modelsamplers;
    std::map<int, float> ratiosmap = data_importer::get_whratios(msamplers);
    std::vector<float> ratvec;
    float lowest_rat = std::numeric_limits<float>::max();
    for (auto &p : ratiosmap)
    {
        if (p.first + 1 > ratvec.size())
        {
            ratvec.resize(p.first + 1, -1.0f);
            ratvec[p.first] = p.second;
            if (ratvec[p.first] < lowest_rat)
                lowest_rat = ratvec[p.first];
        }
    }

    for (auto &v : ratvec)
        if (v < lowest_rat) lowest_rat = v;

    std::cout << "lowest w/h ratio: " << lowest_rat << std::endl;


    int maxidx = -1;
    for (const auto &p : msamplers)
    {
        int idx;
        if (species_to_idx.count(p.first))
        {
            idx = species_to_idx.at(p.first);
            if (idx > maxidx)
                maxidx = idx;
        }
    }
    if (maxidx == -1)
    {
        printf("Error in init_specmodels: either species indices incorrect or modelsamplers map is empty\n");
        *models = nullptr;
        return;
    }
    if (maxidx > 200)
    {
        printf("Error in init_specmodels: maximum species index higher than 200!");
        *models = nullptr;
        return;
    }

    gpuErrchk(cudaMalloc(ratios, sizeof(float) * ratvec.size()));
    gpuErrchk(cudaMemcpy(*ratios, ratvec.data(), sizeof(float) * ratvec.size(), cudaMemcpyHostToDevice));
    nratios = ratvec.size();

    std::vector<specmodels> cpumodels(maxidx + 1);

    for (const auto &p : msamplers)
    {
        if (species_to_idx.count(p.first))
            cpumodels.at(species_to_idx.at(p.first)) = specmodels(p.second, *ratios);
    }

    gpuErrchk(cudaMalloc(models, sizeof(specmodels) * cpumodels.size()));
    gpuErrchk(cudaMemcpy(*models, cpumodels.data(), sizeof(specmodels) * cpumodels.size(), cudaMemcpyHostToDevice));
    nspecmodels = cpumodels.size();

    //return models;
}



__global__
void find_local_maxima_kernel(float *d_data, int width, int height, xy<int> *d_result, float minval);

__device__
void remove_tree(mosaic_tree *tree);


struct xy_equals_minus_one
{
    __device__ __host__
    bool operator ()(const xy<int> &xy)
    {
        return xy.x == -1 && xy.y == -1;
    }
};

struct xy_equals_zero
{
    __device__ __host__
    bool operator ()(const xy<int> &xy)
    {
        return xy.x == 0 && xy.y == 0;
    }
};

__global__
void get_opengl_texture_data(cudaTextureObject_t tex_in, uint32_t *texture_data, int width, int height)
{
    int y = blockIdx.x;
    int x = threadIdx.x;
    int idx = y * width + x;

    if (y >= 0 && y < height && x >= 0 && x < width)
    {
        texture_data[idx] = (tex2D<uint4>(tex_in, x, y).x << 24) | (tex2D<uint4>(tex_in, x, y).y << 16) | (tex2D<uint4>(tex_in, x, y).z << 8) | (tex2D<uint4>(tex_in, x, y).w << 0);
    }
}

cudaTextureObject_t makeCudaTextureObject(cudaGraphicsResource *cudares, cudaResourceDesc &rdesc, cudaTextureDesc &tdesc)
{
    cudaArray *cudaarr;

    gpuErrchk(cudaGraphicsMapResources(1, &cudares, 0));
    gpuErrchk(cudaGraphicsSubResourceGetMappedArray(&cudaarr, cudares, 0, 0));

    memset(&rdesc, 0, sizeof(cudaResourceDesc));
    rdesc.resType = cudaResourceTypeArray;
    rdesc.res.array.array = cudaarr;

    memset(&tdesc, 0, sizeof(cudaTextureDesc));
    tdesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texobj = 0;
    gpuErrchk(cudaCreateTextureObject(&texobj, &rdesc, &tdesc, NULL));

    return texobj;

}

void test_opengl_texture(cudaTextureObject_t texobj, int width, int height, uint32_t *texture_data)
{
    uint32_t *d_texture_data;
    gpuErrchk(cudaMalloc(&d_texture_data, sizeof(uint32_t) * width * height));

    get_opengl_texture_data<<<height, width>>>(texobj, d_texture_data, width, height);

    gpuErrchk(cudaMemcpy(texture_data, d_texture_data, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_texture_data));
}

int get_nblocks(int nthreads, int data_size)
{
    return (data_size - 1) / nthreads + 1;
}

__global__
void add(const float *a, const float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        result[idx] = a[idx] + b[idx];
    }
}

__device__
int gpu_rgba_to_idx(int r, int g, int b, int a)
{
    int idx = b * 256 * 256 + g * 256 + r;
    idx = idx - 1;

    return idx;
}

__device__
int gpu_color_int_to_idx(uint32_t colval)
{
    uint32_t r = colval >> 24;
    uint32_t g = (colval << 8) >> 24;
    uint32_t b = (colval << 16) >> 24;

    return gpu_rgba_to_idx(r, g, b, 255);

}


// we do the reverse of what we do in the gpu_color_int_to_idx function, above
// for the alpha channel, we just assign 255 (0xFF).
__device__
void gpu_idx_to_color_int(int idx, uint32_t *colval)
{
    idx = idx + 1;
    uint32_t r = idx % 256;
    idx -= r;
    uint32_t g = idx % (256 * 256);
    idx -= g;
    uint32_t b = idx % (256 * 256 * 256);
    *colval = (r << 24) | (g << 16) | (b << 8) | 0xFF;
}

__device__
void gpu_color_int_to_rgba(uint32_t colval, uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *a)
{
    uint32_t rtemp = colval >> 24;
    uint32_t gtemp = (colval << 8) >> 24;
    uint32_t btemp = (colval << 16) >> 24;

    *r = rtemp;
    *g = gtemp;
    *b = btemp;
    *a = 0xFF;
}

__device__
void gpu_idx_to_rgba(int idx, uint32_t *r, uint32_t *g, uint32_t *b, uint32_t *a)
{
    *r = 0;
    *g = 0;
    *b = 0;
    *a = 0;

    idx = idx + 1;
    *r = idx % 256;
    idx -= *r;
    *g = (idx % (256 * 256)) / 256;
    idx -= *g * 256;
    *b = (idx % (256 * 256 * 256)) / (256 * 256);
    *a = 255;
}

// XXX: assumes that height is in units of 3 feet, instead of meters
__device__
float d_get_radius_from_height(float height, float a, float b)
{
    float m_per_foot = 0.3048;
    float radius = exp(a + b * log(height));
    return radius / (m_per_foot * 3);
}

__device__
long get_idx()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
void d_get_indicator_array(mosaic_tree *trees, int *indicators, int nlocs)
{
    int idx = get_idx();

    if (idx < nlocs)
    {
        indicators[idx] = trees[idx].valid ? 1 : 0;
    }
}

__global__
void apply_species_to_rendered_texture(cudaTextureObject_t color_grid, int *species_texture, mosaic_tree * trees, int width, int height)
{
    int maxidx = width * height;
    int idx = get_idx();
    if (idx < maxidx)
    {
        int x = idx % width;
        int y = idx / width;
        uint32_t r = tex2D<uint4>(color_grid, x, y).x;
        uint32_t g = tex2D<uint4>(color_grid, x, y).y;
        uint32_t b = tex2D<uint4>(color_grid, x, y).z;
        int treeidx = gpu_rgba_to_idx(r, g, b, 255);
        if (treeidx >= 0)
        {
            mosaic_tree *tptr = trees + treeidx;
            species_texture[idx] = tptr->species;
        }
        else
        {
            species_texture[idx] = -1;
        }
    }
}

ValueMap<int> apply_species_to_rendered_texture_gpu(cudaTextureObject_t color_grid, mosaic_tree * trees, int width, int height)
{
    ValueMap<int> species_texture;
    species_texture.setDim(width, height);

    int *d_species_texture;
    gpuErrchk(cudaMalloc(&d_species_texture, sizeof(uint32_t) * width * height));

    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;
    cudaGetLastError();
    apply_species_to_rendered_texture<<<nblocks, nthreads>>>(color_grid, d_species_texture, trees, width, height);
    kernelCheck();

    gpuErrchk(cudaMemcpy(species_texture.data(), d_species_texture, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_species_texture));

    return species_texture;
}

__global__
void get_indicator_array(mosaic_tree *trees, int *indicators, int nlocs)
{
    d_get_indicator_array(trees, indicators, nlocs);
}

__global__
void assign_to_proper_idxes(mosaic_tree *trees, mosaic_tree *new_trees, int *indicators, int *nlocs, int old_nlocs)
{
    int idx = get_idx();
    if (idx < old_nlocs)
    {
        if (trees[idx].valid)
        {
            int newidx = indicators[idx];
            new_trees[newidx] = trees[idx];
        }
    }
    __syncthreads();
    if (idx == 0)
    {
        if (trees[old_nlocs - 1].valid)
        {
            *nlocs = indicators[old_nlocs - 1] + 1;
        }
        else
        {
            *nlocs = indicators[old_nlocs - 1];
        }
    }
}

// assume that nlocs is an even number
__device__
void prefix_sum(int *indicator_array, int *result, int nlocs)
{
    int idx = get_idx();

    if (idx < nlocs && idx > 0)
        result[idx] = indicator_array[idx - 1];
    else if (idx < nlocs)
        result[idx] = 0;

    __syncthreads();

    int offset = 1;

    int iters = __float2int_ru(log2f(nlocs));

    for (int i = 0; i < iters; i++)
    {
        int add = 0;
        if (idx < nlocs && offset <= idx)
        {
            add = result[idx - offset];
            offset *= 2;
        }

        __syncthreads();

        if (idx < nlocs)
        {
            result[idx] += add;
        }

        __syncthreads();
    }
}


// nlocs I would have to determine in some way...perhaps from the previous iteration?
// We could assign the return value from this function (number of trees) to the nlocs variable passed to this function
// this would only work for a single block, though
// this function gets called after add_trees and rm_dominated_trees
__device__
int stream_compact_trees(mosaic_tree *trees, mosaic_tree *new_trees, int *indicators, int *prefix_result, int nlocs)
{
    d_get_indicator_array(trees, indicators, nlocs);
    prefix_sum(indicators, prefix_result, nlocs);

    int idx = get_idx();

    if (idx < nlocs)
        new_trees[idx].valid = false;

    if (idx < nlocs && trees[idx].valid)
    {
        int new_idx = prefix_result[idx];
        new_trees[new_idx] = trees[idx];
    }

    int ntrees = indicators[nlocs - 1] + prefix_result[nlocs - 1];

    return ntrees;
}

__global__
void prefix_sum_kernel(int *indicator_array, int *result, int nlocs)
{
    prefix_sum(indicator_array, result, nlocs);
}

std::vector<int> prefix_sum_testfunc(std::vector<int> indicator)
{
    int *d_indicator_array, *d_result;

    std::vector<int> result(indicator.size());

    gpuErrchk(cudaMalloc(&d_indicator_array, sizeof(int) * indicator.size()));
    gpuErrchk(cudaMalloc(&d_result, sizeof(int) * indicator.size()));

    cudaMemcpy(d_indicator_array, indicator.data(), sizeof(int) * indicator.size(), cudaMemcpyHostToDevice);

    prefix_sum_kernel<<<1, 1024>>>(d_indicator_array, d_result, indicator.size());

    cudaMemcpy(result.data(), d_result, sizeof(int) * result.size(), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaFree(d_indicator_array));
    gpuErrchk(cudaFree(d_result));

    return result;
}

void test_prefix_sum()
{
    std::vector<int> indicator = {1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1};

    auto result = prefix_sum_testfunc(indicator);

    for (auto &v : result)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}


__global__
void populate_centers(canopy_placer::gpumem mem, int width, int height)
{
    int idx = get_idx();
    int x = idx % width;
    int y = idx / width;

    cudaTextureObject_t color_grid = mem.d_rendered_chm_texture;
    float *chm_grid = mem.d_chm;
    canopy_placer::xy_avg *centers = mem.d_centers;

    if (x < width && x >= 0 && y < height && y >= 0)
    {
        int grid_idx = y * width + x;

        int r, g, b;

        r = tex2D<uint4>(color_grid, x, y).x;
        g = tex2D<uint4>(color_grid, x, y).y;
        b = tex2D<uint4>(color_grid, x, y).z;


        int center_idx = gpu_rgba_to_idx(r, g, b, 255);
        if (center_idx >= 0)
        {
            if (center_idx >= width * height)
            {
                printf("Out of range index found. Index: %d, r, g, b: %d, %d, %d\n", center_idx, r, g, b);
            }
            atomicAdd(&centers[center_idx].visible_count, 1);
            if (chm_grid[grid_idx] > 1e-5)
            {
                atomicAdd(&centers[center_idx].count, 1);
                atomicAdd(&centers[center_idx].x, x);
                atomicAdd(&centers[center_idx].y, y);
            }
        }
    }
}

/*
 * REQUIRED:
 * - color_grid
 * - chm_grid
 * - width
 * - height
 *
 * RETURNS (INTO):
 * centers
 */
void populate_centers_gpu(canopy_placer::gpumem mem, int width, int height)
{
    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;
    cudaGetLastError();
    populate_centers<<<nblocks, nthreads>>>(mem, width, height);

    kernelCheck();

}

/*
 *
 * height is in units of meters, but radius must be returned in units of 3 feet, since the canopy height model's
 * dimensions are in units of three feet
 */
__device__
float get_radius_from_height(float height, float a, float b)
{
    float m_per_foot = 0.3048;
    float radius = exp(a + b * log(height));
    return radius / (m_per_foot * 3);
}

__global__
void create_trees_from_posses(canopy_placer::gpumem mem,
                              mosaic_tree *trees_begin,
                              xy<int> *pos_begin,
                              int nposses,
                              int width,
                              int height)
{
    int idx = get_idx();		// each idx corresponds to a tree, not a pixel on the map

    float *chm = mem.d_chm;
    int *species_map = mem.d_species_map;
    float *a = mem.d_a;
    float *b = mem.d_b;
    int *ntrees = mem.d_ntrees;
    curandState *rstate_arr = mem.d_rstate_arr;

    if (idx < nposses)
    {
        int chm_idx = pos_begin[idx].y * width + pos_begin[idx].x;
        int species_idx = species_map[chm_idx];
        if (species_idx < 0)
        {
            atomicAdd(ntrees, -1);
            return;
        }
        trees_begin[idx].height = chm[chm_idx] * 0.3048;
        trees_begin[idx].x = pos_begin[idx].x;
        trees_begin[idx].y = pos_begin[idx].y;
        trees_begin[idx].x += curand_uniform(rstate_arr + idx) * 0.9144f / 3.0f;
        trees_begin[idx].y += curand_uniform(rstate_arr + idx) * 0.9144f / 3.0f;
        //trees_begin[idx].r = trees_begin[idx].radius = get_radius_from_height(trees_begin[idx].height, a[species_idx], b[species_idx]);
        trees_begin[idx].r = trees_begin[idx].radius = mem.models[species_idx].sample(mem.d_rstate_arr + idx, trees_begin[idx].height) * trees_begin[idx].height;

        // some sanity checks - remove for better performance
        if (trees_begin[idx].radius / trees_begin[idx].height < 0.05f)
        {
            printf("Very low ratio detected: %f\n",trees_begin[idx].radius / trees_begin[idx].height );
            asm("trap;");
        }
        if (trees_begin[idx].radius / trees_begin[idx].height > 5.0f)
        {
            printf("Very high ratio detected: %f\n",trees_begin[idx].radius / trees_begin[idx].height );
            asm("trap;");
        }

        trees_begin[idx].species = species_idx;
        if (trees_begin[idx].radius < 1)
        {
            trees_begin[idx].radius = 1;
        }
        trees_begin[idx].valid = true;
        trees_begin[idx].local_max = true;

    }
}

xy<int> *find_local_maxima_gpu(float *d_data, int width, int height, xy<int> *d_result_begin, float minval)
{
    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;

    find_local_maxima_kernel<<<nblocks, nthreads>>>(d_data, width, height, d_result_begin, minval);
    kernelCheck();

    thrust::device_ptr<xy<int> > result_d_ptr = thrust::device_pointer_cast(d_result_begin);
    thrust::device_ptr<xy<int> > last_pos = thrust::remove_if(result_d_ptr, result_d_ptr + width * height,
                      xy_equals_zero());
    return thrust::raw_pointer_cast(last_pos);
}

int find_local_maxima_trees_gpu(canopy_placer::gpumem mem, mosaic_tree *d_temp_trees, int width, int height, float minval)
{
    xy<int> *maxima_xy_begin;
    gpuErrchk(cudaMalloc(&maxima_xy_begin, sizeof(xy<int>) * width * height));
    gpuErrchk(cudaMemset(maxima_xy_begin, 0, sizeof(xy<int>) * width * height));
    xy<int> *last_maxima = find_local_maxima_gpu(mem.d_chm, width, height, maxima_xy_begin, minval);
    int nposses = last_maxima - maxima_xy_begin;
    gpuErrchk(cudaMemcpy(mem.d_ntrees, &nposses, sizeof(int), cudaMemcpyHostToDevice));

    int nthreads = 1024;
    int nblocks = get_nblocks(nthreads, nposses);

    std::cout << "nposses: " << nposses << std::endl;

    create_trees_from_posses<<<nblocks, nthreads>>>(mem, d_temp_trees, maxima_xy_begin, nposses, width, height);
    kernelCheck();

    gpuErrchk(cudaFree(maxima_xy_begin));

    return nposses;
}

__global__
void find_local_maxima_kernel(float *d_data, int width, int height, xy<int> *d_result, float minval)
{
    int idx = get_idx();
    int maxidx = width * height;
    int x = idx % width;
    int y = idx / width;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        bool is_locmax = true;
        if (d_data[idx] < minval)
        {
            is_locmax = false;
            return;
        }
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int didx = (y + dy) * width + (x + dx);
                if (didx != idx && d_data[didx] >= d_data[idx])
                {
                    is_locmax = false;
                    break;
                }
            }
            if (!is_locmax) break;
        }
        if (is_locmax)
        {
            d_result[idx].x = x;
            d_result[idx].y = y;
        }
    }
}

__global__
void compute_centers_kernel(canopy_placer::gpumem mem)
{
    int idx = get_idx();

    canopy_placer::xy_avg *centers = mem.d_centers;
    mosaic_tree *trees = mem.d_trees;
    int *ntrees = mem.d_ntrees;

    if (idx < *ntrees)
    {
        if (trees[idx].valid)
        {

            if (centers[idx].count == 0)
            {
                remove_tree(&trees[idx]);
                return;
            }

            float sumx, sumy;
            sumx = centers[idx].x;
            sumy = centers[idx].y;
            float xcenter, ycenter;
            xcenter = centers[idx].x / centers[idx].count;
            ycenter = centers[idx].y / centers[idx].count;
            centers[idx].x = xcenter;
            centers[idx].y = ycenter;

            bool invalid = !(fabs(xcenter - trees[idx].x) <= trees[idx].radius
                   && abs(ycenter - trees[idx].y) <= trees[idx].radius);

        }
    }
}

void find_centers_gpu(canopy_placer::gpumem mem)
{
    int ntrees_host;
    cudaMemcpy(&ntrees_host, mem.d_ntrees, sizeof(int), cudaMemcpyDeviceToHost);

    int nthreads = 1024;
    int nblocks = (ntrees_host - 1) / nthreads + 1;

    cudaGetLastError();
    compute_centers_kernel<<<nblocks, nthreads>>>(mem);

    kernelCheck();
}

__global__
void move_trees(canopy_placer::gpumem mem, int width)
{
    mosaic_tree *trees = mem.d_trees;
    canopy_placer::xy_avg *centers = mem.d_centers;
    float *chm = mem.d_chm;
    int *species_map = mem.d_species_map;
    float *a = mem.d_a;
    float *b = mem.d_b;
    int *nlocs = mem.d_nlocs;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < *nlocs && trees[idx].valid)
    {
        if (!trees[idx].local_max && centers[idx].count > 0)	// only move the tree if it is not a local maximum,
                                                                // and if at least one pixel is on nonzero section of CHM
        {
            float ydiff = centers[idx].y - trees[idx].y;
            float xdiff = centers[idx].x - trees[idx].x;
            float ydelta = ydiff * 0.25f;
            float xdelta = xdiff * 0.25f;
            float prevx, prevy;
            prevx = trees[idx].x;
            prevy = trees[idx].y;
            trees[idx].x += xdelta;
            trees[idx].y += ydelta;


            int mapidx = ((int)trees[idx].y) * width + (int)trees[idx].x;
            int species_idx = species_map[mapidx];
            if (species_idx < 0)
            {
                remove_tree(&trees[idx]);
            }
            else
            {
                trees[idx].height = chm[mapidx] * 0.3048;
                trees[idx].radius = mem.models[species_idx].sample(mem.d_rstate_arr + idx, trees[idx].height) * trees[idx].height;

                // some sanity checks - remove for better performance
                if (trees[idx].radius / trees[idx].height < 0.05f)
                {
                    printf("Very low ratio detected: %f\n",trees[idx].radius / trees[idx].height );
                    asm("trap;");
                }
                if (trees[idx].radius / trees[idx].height > 5.0f)
                {
                    printf("Very high ratio detected: %f\n",trees[idx].radius / trees[idx].height );
                    asm("trap;");
                }
                if (trees[idx].radius < 1)
                    trees[idx].radius = 1;
                trees[idx].species = species_idx;
            }
        }
    }
}

__global__
void reset_centers(canopy_placer::xy_avg *centers, int nlocs)
{
    int idx = get_idx();

    if (idx < nlocs)
    {
        centers[idx].reset();
    }
}

void reset_centers_gpu(canopy_placer::xy_avg *centers, int nlocs)
{
    int nthreads = 1024;
    int nblocks = (nlocs - 1) / nthreads + 1;

    reset_centers<<<nblocks, nthreads>>>(centers, nlocs);
}

void move_trees_gpu(canopy_placer::gpumem mem, int width)
{
    int nlocs_host;
    cudaMemcpy(&nlocs_host, mem.d_nlocs, sizeof(int), cudaMemcpyDeviceToHost);
    int nthreads = 1024;
    int nblocks = (nlocs_host - 1) / nthreads + 1;
    move_trees<<<nblocks, nthreads>>>(mem, width);
    kernelCheck();
}

__device__
void remove_tree(mosaic_tree *tree)
{
    tree->valid = false;
}

/*
 * REQUIRED:
 * - trees
 * - centers
 * - d_ntrees
 *
 * d_nlocs will containa copy of the old number of locations
 */
__global__
void rm_dominated_trees(canopy_placer::gpumem mem)
{
    mosaic_tree *trees = mem.d_trees;
    canopy_placer::xy_avg *centers = mem.d_centers;
    int *d_ntrees = mem.d_ntrees;
    int *d_nlocs = mem.d_nlocs;

    int idx = get_idx();
    float min_canopy_exposed = 0.5f;

    if (idx == 0)
        *d_nlocs = *d_ntrees;

    if (idx < *d_ntrees && trees[idx].valid)
    {
        mosaic_tree &tree = trees[idx];
        float area = tree.radius * tree.radius * M_PI;
        float prop = centers[idx].visible_count / area;
        if (prop < 0.3f)
        {
            remove_tree(&tree);
        }
    }
}


/*
 * REQUIRED:
 * - trees
 * - centers
 * - d_ntrees
 *
 * d_nlocs will contain a copy of the old number of locations
 */
void rm_dominated_trees_gpu(canopy_placer::gpumem mem)
{
    int nlocs_host;
    cudaMemcpy(&nlocs_host, mem.d_ntrees, sizeof(int), cudaMemcpyDeviceToHost);
    int nthreads = 1024;
    int nblocks = (nlocs_host - 1) / nthreads + 1;
    rm_dominated_trees<<<nblocks, nthreads>>>(mem);
    kernelCheck();
}

__device__
void d_get_xy(int idx, int width, int *x, int *y)
{
    *x = idx % width;
    *y = idx / width;
}

__device__
int get_surr_idx_at_iter(int iter, int x, int y, int width, int height)
{
    int ymod = iter / 3 - 1;
    int xmod = iter % 3 - 1;
    int cy = y + ymod;
    int cx = x + xmod;
    if (cx >= 0 && cx < width && cy >= 0 && cy < height)
        return cy * width + cx;
    else
        return -1;
}

__device__
float get_height(int idx, float *data)
{
    return data[idx] * 0.3048;
}


// this samples new trees into the new_trees array. Basically, a thread is run for each pixel on the CHM, and if the pixel is uncovered and the CHM > 0, then
// with a certain probability, a new tree is put into the new_trees array at the index corresponding to the thread.
// Afterwards, we can compact this new trees array, with the result of the compaction being inserted at the end of the compacted array of existing trees
__global__
void sample_new_trees(float *chm, cudaTextureObject_t color_grid, int width, int height, int *species_map, float *a, float *b, mosaic_tree *new_trees, curandState *rstate_arr)
{
    int max_idx = width * height;

    int idx = get_idx();

    if (idx < max_idx)
    {
        bool nonzero_render;
        int x, y;
        d_get_xy(idx, width, &x, &y);
        if ( tex2D<uint4>(color_grid, x, y).x ||
        tex2D<uint4>(color_grid, x, y).y ||
        tex2D<uint4>(color_grid, x, y).z)
        {
            nonzero_render = true;
            // could zero out all pixels here, instead of clearing the color buffer bit via opengl
            //tex2D<uint4>(color_grid, x, y).x = 0;
            //tex2D<uint4>(color_grid, x, y).y = 0;
            //tex2D<uint4>(color_grid, x, y).z = 0;
        }
        else
        {
            nonzero_render = false;
        }

        if (chm[idx] >= 10.0 && species_map[idx] >= 0 && !nonzero_render)
        {
            float randnum = curand_uniform(rstate_arr + idx);
            if (randnum < 1.0f / chm[idx])
            {
                int x, y;
                d_get_xy(idx, width, &x, &y);
                float tree_height = chm[idx] * 0.3048;
                int species_idx = species_map[idx];
                float radius = get_radius_from_height(tree_height, a[species_idx], b[species_idx]);
                radius = radius >= 1 ? radius : 1;	// make sure radius is at least one
                new_trees[idx] = mosaic_tree(x + curand_uniform(rstate_arr + idx) * 0.9144f / 3.0f, y + curand_uniform(rstate_arr + idx) * 0.9144f / 3.0f, radius, tree_height, false);
                new_trees[idx].species = species_idx;
            }
        }
    }
}

__global__
void sample_radial_trees(canopy_placer::gpumem mem, int width, int height, int ncur_trees, int sample_mult)
{
    int max_idx = ncur_trees;
    float *chm = mem.d_chm;
    cudaTextureObject_t color_grid = mem.d_rendered_chm_texture;
    int *species_map = mem.d_species_map;
    float *a = mem.d_a;
    float *b = mem.d_b;
    mosaic_tree *current_trees = mem.d_trees;
    mosaic_tree *new_trees = mem.d_new_trees;
    curandState *rstate_arr = mem.d_rstate_arr;

    int idx = get_idx();

    if (idx < max_idx)
    {
        for (int mult_iter = 0; mult_iter < sample_mult; mult_iter++)
        {
            mosaic_tree *thistree = &current_trees[idx];
            float x, y;
            x = thistree->x, y = thistree->y;
            float newx = -1, newy = -1;
            int attempts = 0;
            while ((newx < 0 || newx >= width || newy < 0 || newy >= height))
            {
                float sample_range = thistree->radius;
                if (sample_range < 3)
                    sample_range = 3;
                float randdir = curand_uniform(rstate_arr + idx) * 2 * M_PI;
                float randdist = curand_uniform(rstate_arr + idx) * sample_range + thistree->radius;
                float dx = cos(randdir) * randdist;
                float dy = sin(randdir) * randdist;

                newx = x + dx;
                newy = y + dy;
                attempts++;
            }

            int mapidx = ((int)newy) * width + (int)newx;

            bool nonzero_render;
            if ( tex2D<uint4>(color_grid, newx, newy).x ||
            tex2D<uint4>(color_grid, newx, newy).y ||
            tex2D<uint4>(color_grid, newx, newy).z)
            {
                nonzero_render = true;
                // could zero out all pixels here, instead of clearing the color buffer bit via opengl
                //tex2D<uint4>(color_grid, x, y).x = 0;
                //tex2D<uint4>(color_grid, x, y).y = 0;
                //tex2D<uint4>(color_grid, x, y).z = 0;
            }
            else
            {
                nonzero_render = false;
            }

            if (chm[mapidx] >= 5.0 && species_map[mapidx] >= 0 && !nonzero_render)
            {
                float tree_height = chm[mapidx] * 0.3048;
                int species_idx = species_map[mapidx];
                //float radius = height_to_width[species_idx] * tree_height;
                float radius = mem.models[species_idx].sample(mem.d_rstate_arr + idx * sample_mult + mult_iter, tree_height) * tree_height;
                if (radius / tree_height < 0.15f)
                {
                    printf("Very low ratio detected: %f\n", radius / tree_height );
                    asm("trap;");
                }
                if (radius / height > 5.0f)
                {
                    printf("Very high ratio detected: %f\n",radius / height );
                    asm("trap;");
                }
                //float radius = get_radius_from_height(tree_height, a[species_idx], b[species_idx]);
                new_trees[idx * sample_mult + mult_iter] = mosaic_tree(newx, newy, radius, tree_height, false);
                new_trees[idx * sample_mult + mult_iter].species = species_idx;
                if (radius < 1.0f) radius = 1.0f;
                new_trees[idx * sample_mult + mult_iter].radius = radius;
                if (species_idx > 15)
                {
                    printf("SPECIES IDX ABOVE 15 IN SAMPLE_RADIAL_TREES\n");
                }
            }
            // else: the new_trees array will be filled with zeros, so if a new_tree's valid variable = false, we know a tree was not sampled
        }
    }
}

__global__
void init_curand(curandState *states, long base_seed, long nthreads_total)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nthreads_total)
    {
        curand_init(base_seed + idx, 0, 0, states + idx);
    }
}

void sample_radial_trees_gpu(canopy_placer::gpumem mem, int width, int height, int ncur_trees, int sample_mult)
{
    int nthreads = 1024;
    int nblocks = (ncur_trees - 1) / nthreads + 1;

    gpuErrchk(cudaMemset(mem.d_new_trees, 0, sizeof(mosaic_tree) * ncur_trees * sample_mult));

    sample_radial_trees<<<nblocks, nthreads>>>(mem, width, height, ncur_trees, sample_mult);
    kernelCheck();
}

void sample_new_trees_gpu(float *chm, cudaTextureObject_t color_grid, int width, int height, int *species_map, float *a, float *b, mosaic_tree *new_trees, curandState *rstate_arr)
{
    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;

    cudaMemset(new_trees, 0, sizeof(mosaic_tree) * width * height);

    sample_new_trees<<<nblocks, nthreads>>>(chm, color_grid, width, height, species_map, a, b, new_trees, rstate_arr);
    kernelCheck();
}

__global__
void get_texture_object_data(cudaTextureObject_t color_grid, int width, int height, uint32_t *result)
{
    int max_idx = width * height;

    int idx = get_idx();

    if (idx < max_idx)
    {
        int x = idx % width;
        int y = idx / width;
        uint32_t r = tex2D<uint4>(color_grid, x, y).x;
        uint32_t g = tex2D<uint4>(color_grid, x, y).y;
        uint32_t b = tex2D<uint4>(color_grid, x, y).z;
        uint32_t a = tex2D<uint4>(color_grid, x, y).w;
        uint32_t colint = r | (g << 8) | (b << 16) | (a << 24);
        result[idx] = colint;
    }
}

void get_cuda_texture_object_data(cudaTextureObject_t color_grid, int width, int height, uint32_t *result)
{
    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;

    get_texture_object_data<<<nblocks, nthreads>>>(color_grid, width, height, result);
}

void init_curand_gpu(curandState *states, long base_seed, long nthreads_total)
{
    int nthreads = 1024;
    int nblocks = (nthreads_total - 1) / nthreads + 1;
    init_curand<<<nblocks, nthreads>>>(states, base_seed, nthreads_total);
    kernelCheck();
    //int idx = blockDim.x * blockIdx.x + threadIdx.x;
}


void compact_tree_array(mosaic_tree *trees, int ntrees)
{
    const int nblocks = (ntrees - 1) / 1024 + 1;

    int *d_indicators, *d_new_ntrees;
    int new_ntrees;
    mosaic_tree *d_new_trees, *d_trees;
    std::vector<int> indicators(ntrees);
    for (int i = 0; i < ntrees; i++)
    {
        indicators[i] = trees[i].valid ? 1 : 0;
    }

    gpuErrchk(cudaMalloc(&d_indicators, sizeof(int) * ntrees));
    gpuErrchk(cudaMalloc(&d_new_trees, sizeof(mosaic_tree) * ntrees));
    gpuErrchk(cudaMalloc(&d_new_ntrees, sizeof(int)));
    gpuErrchk(cudaMalloc(&d_trees, sizeof(mosaic_tree) * ntrees));

    gpuErrchk(cudaMemcpy(d_indicators, indicators.data(), sizeof(int) * ntrees, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_trees, trees, sizeof(mosaic_tree) * ntrees, cudaMemcpyHostToDevice));

    thrust::device_ptr<int> dptr = thrust::device_pointer_cast(d_indicators);
    thrust::device_ptr<mosaic_tree> trees_dptr = thrust::device_pointer_cast(d_trees);
    thrust::transform_exclusive_scan(trees_dptr, trees_dptr + ntrees, dptr, getvalid(), 0, thrust::plus<int>());

    assign_to_proper_idxes<<<nblocks, 1024>>>(d_trees, d_new_trees, d_indicators, d_new_ntrees, ntrees);
    kernelCheck();

    gpuErrchk(cudaMemcpy(&new_ntrees, d_new_ntrees, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(trees, d_new_trees, sizeof(mosaic_tree) * new_ntrees, cudaMemcpyDeviceToHost));
    memset(trees + new_ntrees, 0, sizeof(mosaic_tree) * (ntrees - new_ntrees));		// zero out the remainder of the array, to be
                                                                                    // sure that each 'valid' variable of remaining trees are false
    gpuErrchk(cudaMemcpy(indicators.data(), d_indicators, sizeof(int) * ntrees, cudaMemcpyDeviceToHost));

    std::cout << "Indicators: ";
    for (int i = 0; i < ntrees; i++)
    {
        std::cout << indicators[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Number of actual trees: " << new_ntrees << std::endl;

    gpuErrchk(cudaFree(d_indicators));
    gpuErrchk(cudaFree(d_new_trees));
    gpuErrchk(cudaFree(d_new_ntrees));
    gpuErrchk(cudaFree(d_trees));
}


// this function is not really necessary - thrust::transform_exclusive_scan takes care of this in any case
__global__
void create_indicator_array(mosaic_tree *trees, int *indicators, int nlocs)
{
    int idx = get_idx();
    if (idx < nlocs)
    {
        if (trees[idx].valid)
        {
            indicators[idx] = 1;
        }
        else
        {
            indicators[idx] = 0;
        }
    }
}

void scandebug(mosaic_tree *trees, int *indicators, int *nlocs)
{
    int nlocs_host;
    gpuErrchk(cudaMemcpy(&nlocs_host, nlocs, sizeof(int), cudaMemcpyDeviceToHost));
    thrust::device_ptr<int> indic_ptr = thrust::device_pointer_cast(indicators);
    thrust::device_ptr<mosaic_tree> trees_ptr = thrust::device_pointer_cast(trees);
    thrust::transform_exclusive_scan(trees_ptr, trees_ptr + nlocs_host, indic_ptr, getvalid(), 0, thrust::plus<int>());
}

int compact_and_assign(mosaic_tree *target, mosaic_tree *source, int nsource)
{
    thrust::device_ptr<mosaic_tree> source_ptr = thrust::device_pointer_cast(source);
    thrust::device_ptr<mosaic_tree> target_endptr = thrust::device_pointer_cast(target);
    thrust::device_ptr<mosaic_tree> outptr = thrust::copy_if(source_ptr, source_ptr + nsource, target_endptr, getvalid());
    int ncopied = (int)(outptr - target_endptr);

    gpuErrchk(cudaMemset(source, 0, sizeof(mosaic_tree) * nsource));
    return ncopied;
}

__global__
void create_world_matrices_and_colvecs(canopy_placer::gpumem mem, int ntrees)
{
    mosaic_tree *d_trees = mem.d_trees;
    glm::mat4 *d_translate_matrices = mem.d_translate_matrices;
    glm::mat4 *d_scale_matrices = mem.d_scale_matrices;
    glm::vec4 *colvecs = mem.d_color_vecs;

    float range_scale = 1.0f;
    float orig_radius = 0.5f;
    float scale_base = 1 / orig_radius;
    float height_scale = scale_base / range_scale;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < ntrees)
    {
        float add_height = d_trees[idx].height / range_scale - height_scale * d_trees[idx].radius * 0.5f;
        glm::mat4 *tm = d_translate_matrices;
        glm::mat4 *sm = d_scale_matrices;
        tm[idx][0][0] = 1.0f;
        tm[idx][1][1] = 1.0f;
        tm[idx][2][2] = 1.0f;
        tm[idx][3][3] = 1.0f;
        tm[idx][3][0] = -d_trees[idx].x;
        tm[idx][3][1] = d_trees[idx].y;
        tm[idx][3][2] = add_height;

        sm[idx][0][0] = d_trees[idx].radius * scale_base;
        sm[idx][1][1] = d_trees[idx].radius * scale_base;
        sm[idx][2][2] = d_trees[idx].radius * height_scale;
        sm[idx][3][3] = 1.0f;

        for (int i = 0; i < 3; i++)
        {
            if (sm[idx][i][i] < 1.0f)
            {
                sm[idx][i][i] = 2.0f;
            }
        }

        uint32_t colval;
        uint32_t r, g, b, a;
        gpu_idx_to_rgba(idx, &r, &g, &b, &a);

        colvecs[idx][0] = (r + 0.1f) / 255.0f;
        colvecs[idx][1] = (g + 0.1f) / 255.0f;
        colvecs[idx][2] = (b + 0.1f) / 255.0f;
        colvecs[idx][3] = (a + 0.1f) / 255.0f;
    }
}

void create_world_matrices_and_colvecs_gpu(canopy_placer::gpumem mem, int ntrees)
{
    gpuErrchk(cudaGetLastError());		// check if any errors remain in the queue first

    int nthreads = 1024;
    int nblocks = (ntrees - 1) / nthreads + 1;
    create_world_matrices_and_colvecs<<<nblocks, nthreads>>>(mem, ntrees);
    kernelCheck();
}

__global__
void dummy_kernel(float *data)
{
    int idx = get_idx();
    data[idx] = 3215;
}

// NOTE: buffers needs to be registered at some point, preferably with other opengl code
void send_gl_buffer_data_gpu(cudaGraphicsResource **bufres, void *data, int data_nbytes)
{
    glFinish();

    std::vector<float> testbuf(data_nbytes / sizeof(float));

    float *dptr;
    cudaGraphicsMapResources(1, bufres, 0);
    size_t nbytes;
    cudaError_t errcode = cudaGraphicsResourceGetMappedPointer((void **)&dptr, &nbytes, *bufres);
    gpuErrchk(errcode);

    gpuErrchk(cudaGetLastError());

    assert(nbytes >= data_nbytes);

    cudaMemcpy(dptr, data, data_nbytes, cudaMemcpyDeviceToDevice);


    gpuErrchk(cudaGraphicsUnmapResources(1, bufres, 0));
    gpuErrchk(cudaGetLastError());
}

void test_compact_tree_array()
{
    int ntrees = 20;
    const int nthreads_per_block = 1024;
    const int nblocks = (ntrees - 1) / nthreads_per_block + 1;
    std::vector<mosaic_tree> trees(ntrees);
    for (int i = 0; i < ntrees; i++)
    {
        trees[i].valid = false;
    }
    trees[5].valid = true;
    trees[8].valid = true;
    trees[19].valid = true;

    for (int i = 0; i < ntrees; i++)
    {
        std::cout << "Valid: " << trees[i].valid << std::endl;
    }

    std::cout << "RUNNING CUDA FUNCTION" << std::endl << "----------------------------" << std::endl;
    compact_tree_array(trees.data(), ntrees);
    std::cout << "DONE RUNNING CUDA FUNCTION" << std::endl << "----------------------------" << std::endl;

    for (int i = 0; i < ntrees; i++)
    {
        std::cout << "Valid: " << trees[i].valid << std::endl;
    }
}

void run_gpu_tests()
{
    test_compact_tree_array();
}

/*
 * Do bilinear interpolation, assuming srcdata array is column-major
 */
__device__
float interpolate_colmajor(float *srcdata, int upwidth, int upheight, int factor)
{
    //printf("running interpolate colmajor\n");
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int x = idx / upheight;
    int y = idx % upheight;
    int srcx = x / factor;
    int srcy = y / factor;
    int srcwidth = upwidth / factor;
    int srcheight = upheight / factor;
    int xd = x % factor;
    int yd = y % factor;
    float xratio = xd / (float)factor;
    float yratio = yd / (float)factor;

    float lefttop = srcdata[srcx * srcheight + srcy];
    float righttop;
    float leftbot;
    float rightbot;
    bool on_right_edge = false;
    bool on_bot_edge = false;
    if (srcx == srcwidth - 1)
    {
        on_right_edge = true;
        righttop = lefttop;
    }
    else
    {
        righttop = srcdata[(srcx + 1) * srcheight + srcy];
    }

    if (srcy == srcheight - 1)
    {
        on_bot_edge = true;
        leftbot = lefttop;
    }
    else
    {
        leftbot = srcdata[srcx * srcheight + srcy + 1];
    }

    if (on_right_edge)
    {
        rightbot = leftbot;
    }
    else if (on_bot_edge)
    {
        rightbot = righttop;
    }
    else
    {
        rightbot = srcdata[(srcx + 1) * srcheight + srcy + 1];
    }

    float leftinterp = (1 - yratio) * lefttop + yratio * leftbot;
    float rightinterp = (1 - yratio) * righttop + yratio * rightbot;

    //if (idx < 100000 && rightinterp > 0)
    //    printf("leftinterp, rightinterp, xratio, yratio: %f, %f, %f, %f\n", leftinterp, rightinterp, xratio, yratio);
    return (1 - xratio) * leftinterp + xratio * rightinterp;

}

/*
 * Bilinear upsample of srcdata array assuming column major ordering of data. Upsample by factor of 'factor'
 */
__global__
void bilinear_upsample_colmajor(float *srcdata, float *destdata, int srcw, int srch, int factor)
{
    int upwidth = srcw * factor;
    int upheight = srch * factor;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < upwidth * upheight)
        destdata[idx] = interpolate_colmajor(srcdata, upwidth, upheight, factor);
}

void bilinear_upsample_colmajor_gpu(float *srcdata, float *destdata, int srcw, int srch, int factor)
{
    int nthreads = 1024;
    int nblocks = (srcw * srch * factor * factor - 1) / nthreads + 1;

    printf("nblocks, nthreads: %d, %d\n", nblocks, nthreads);

    bilinear_upsample_colmajor<<<nblocks, nthreads>>>(srcdata, destdata, srcw, srch, factor);

    gpuErrchk(cudaGetLastError());
}

void bilinear_upsample_colmajor_allocate_gpu(float *srcdata, float *destdata, int srcw, int srch, int factor)
{
    float *d_srcdata, *d_destdata;

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMalloc(&d_destdata, sizeof(float) * srcw * srch * factor * factor));
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMalloc(&d_srcdata, sizeof(float) * srcw * srch));
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaMemcpy(d_srcdata, srcdata, sizeof(float) * srcw * srch, cudaMemcpyHostToDevice));
    gpuErrchk(cudaGetLastError());

    bilinear_upsample_colmajor_gpu(d_srcdata, d_destdata, srcw, srch, factor);

    std::cout << "Copying width, height, factor: " << srcw << ", " << srch << ", " << factor << std::endl;
    gpuErrchk(cudaMemcpy(destdata, d_destdata, sizeof(float) * srcw * srch * factor * factor, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_destdata));
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_srcdata));
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaDeviceSynchronize());
}

__global__
void smooth_uniform_radial_kernel(int radius, float *data, float *result, int width, int height)
{
    int idx = get_idx();
    if (idx < width * height)
    {
        float maxdist = radius * radius;
        int x = idx % width;
        int y = idx / width;
        int sx = x - radius >= 0 ? x - radius : 0;
        int ex = x + radius < width ? x + radius : width - 1;
        int sy = y - radius >= 0 ? y - radius : 0;
        int ey = y + radius < height ? y + radius : height - 1;
        float sum = 0.0f;
        int count = 0;
        for (int cy = sy; cy <= ey; cy++)
        {
            for (int cx = sx; cx <= ex; cx++)
            {
                float dy = cy - y;
                float dx = cx - x;
                if (dy * dy + dx * dx <= maxdist)
                {
                    int cidx = cy * width + cx;
                    sum += data[cidx];
                    count++;
                }
            }
        }
        sum /= count;
        result[idx] = sum;
    }
}

void smooth_uniform_radial(int radius, float *data, float *result, int width, int height)
{
    float *d_data, *d_result;
    cudaMalloc(&d_data, sizeof(float) * width * height);
    cudaMalloc(&d_result, sizeof(float) * width * height);
    cudaMemcpy(d_data, data, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    int nthreads = 1024;
    int nblocks = (width * height - 1) / nthreads + 1;
    smooth_uniform_radial_kernel<<<nblocks, nthreads>>>(radius, d_data, d_result, width, height);

    cudaMemcpy(result, d_result, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_data);
}

void bilinear_upsample_colmajor_test(float *srcdata, float *destdata, int srcw, int srch, int factor)
{
    float *d_srcdata, *d_destdata;
    cudaMalloc(&d_destdata, sizeof(float) * srcw * srch * factor * factor);
    cudaMalloc(&d_srcdata, sizeof(float) * srcw * srch);
    cudaMemset(d_destdata, -1, sizeof(float) * srcw * srch * factor * factor);
    cudaMemcpy(d_srcdata, srcdata, sizeof(float) * srcw * srch, cudaMemcpyHostToDevice);
    bilinear_upsample_colmajor_gpu(d_srcdata, d_destdata, srcw, srch, factor);
    cudaMemcpy(destdata, d_destdata, sizeof(float) * srcw * srch * factor * factor, cudaMemcpyDeviceToHost);
    cudaFree(d_destdata);
    cudaFree(d_srcdata);
}


bool test_find_local_maxima_gpu(verbosity v)
{
    std::cerr << "in test_find_local_maxima_gpu" << std::endl;
    int width = 7;
    int height = 7;
    std::vector<float> data(width * height);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            if (x % 2 == 0)
                data[idx] = 2;
            else
                data[idx] = 1;
        }
    data[1 * width + 1] = 2;

    std::vector<bool> checkarr(width * height, false);
    for (int i = 0; i < width * height; i++)
    {
        int y = i / width;
        int x = i % width;

        if (y > 0 && y < height - 1 && x > 0 && x < width - 1
                && data[i] == 2)
        {
            checkarr[i] = true;
        }
    }

    float *d_data;
    xy<int> *d_result;
    std::vector<xy<int> > result;

    std::cerr << "Running cudaMallocs..." << std::endl;
    gpuErrchk(cudaMalloc(&d_data, sizeof(float) * width * height));
    gpuErrchk(cudaMalloc(&d_result, sizeof(xy<int>) * width * height));

    std::cerr << "Running cudaMemcpy..." << std::endl;
    gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(float) * width * height, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_result, 0, sizeof(xy<int>) * width * height));

    std::cerr << "Running kernel..." << std::endl;
    xy<int> *last_pos = find_local_maxima_gpu(d_data, width, height, d_result, 0.0f);

    int nelements = last_pos - d_result;

    result.resize(nelements);

    gpuErrchk(cudaMemcpy(result.data(), d_result, sizeof(xy<int>) * nelements, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_result));

    //result.pop_back();

    if ((int)v > 1)
    {
        std::cout << "test data: " << std::endl;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                std::cout << data[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    if ((int)v > 0)
    {
        std::cout << "local maxima: " << std::endl;
        for (auto &r : result)
        {
            std::cout << r.x << ", " << r.y << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    for (int i = 0; i < nelements; i++)
    {
        int idx = result[i].y * width + result[i].x;
        checkarr[idx] = !checkarr[idx];
        if (checkarr[idx])
        {
            return false;
        }
    }
    for (int i = 0; i < width * height; i++)
    {
        if (checkarr[i])
            return false;
    }
    return true;
}
