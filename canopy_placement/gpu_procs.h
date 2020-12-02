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


#ifndef GPU_PROCS_H
#define GPU_PROCS_H

#include "common/basic_types.h"

#include <vector>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>

#include "canopy_placement/canopy_placer.h"
#include "data_importer/data_importer.h"

/*
 * Defines what a valid tree is for the stream compaction function compact_and_assign
 */
struct getvalid
{
    typedef mosaic_tree argument_type;
    typedef int result_type;

    __device__ __host__
    int operator () (const mosaic_tree &tree) const {return tree.valid ? 1 : 0; }
};

/*
 * Represents a selection choice between a set of different tree models. For example, for a given height of 30
 * meters, two models might be valid. This struct selects randomly between such two models
 */
struct modelselection
{
    modelselection(const std::vector<int> &select);

    __device__
    int sample(curandState *rstate);

    __host__
    void free_memory();

    int nmodels;
    int *vueids;
};

/*
 * Device-side struct that samples a radius for a given height, for a given tree model
 */
struct specmodels
{
    specmodels(const data_importer::modelset &mset, float *model_whratios);

    specmodels();

    __device__
    float sample(curandState *rstate, float height);

    __host__
    void free_memory();

    int nbins;
    float binsize;
    float *ranges;
    modelselection *selects;
    int samplemap_size;
    int *samplemap;
    float *model_whratios;
};

/*
 * Initialize 3d model attributes for species on the GPU, as well as functionality to sample attributes when necessary.
 * These are mainly the radius/height ratios, which are important for the canopy placement algorithm
 */
void init_specmodels(const data_importer::common_data &cdata, const std::map<int, int> &species_to_idx, float **ratios, specmodels **models, int &nratios, int &nspecmodels);
//void find_centers_gpu(mosaic_spacing::xy_avg *centers, mosaic_tree *trees, int *d_ntrees);

/*
 * Compute centers of each tree (center of mass) by dividing sum of x and y pixels by the number found.
 * Note the actual summation of x, y pixels is not done by this function,
 * but by the populate_centers_gpu function
 */
void find_centers_gpu(canopy_placer::gpumem mem);

/*
 * Sums visible x, y locations for each tree's sphere, and keeps track of number of visible pixels, in preparation for
 * computing the center of mass for each tree
 */
void populate_centers_gpu(canopy_placer::gpumem mem, int width, int height);

/*
 * Move each tree towards center of mass of its visible portion
 */
void move_trees_gpu(canopy_placer::gpumem mem, int width);

/*
 * Remove trees that are dominated, i.e., where not enough of the tree is visible
 */
void rm_dominated_trees_gpu(canopy_placer::gpumem mem);

/*
 * Sample trees based on a probability for each cell. Not used currently in canopy placement
 */
void sample_new_trees_gpu(float *chm, cudaTextureObject_t color_grid, int width, int height,
                          int *species_map, float *a, float *b, mosaic_tree *new_trees, curandState *rstate_arr);

/*
 * create translation, scale matrices and colour vectors for each tree
 */
void create_world_matrices_and_colvecs_gpu(canopy_placer::gpumem mem, int ntrees);

/*
 * Send translation and colour matrix data to GPU (using CUDA ops), for use by the OpenGL pipeline
 */
void send_gl_buffer_data_gpu(cudaGraphicsResource **bufres, void *data, int data_nbytes);

/*
 * Compact trees in array 'source', into array 'target'. Compact here means that there are no trees that are invalid
 * between valid trees. In other words, valid trees make up the first part of the target array, invalid trees the rest.
 * 'valid' and 'invalid' trees are defined by the struct (functor) getvalid
 */
int compact_and_assign(mosaic_tree *target, mosaic_tree *source, int nsource);

/*
 * Initialize random number generator array 'states' on GPU
 */
void init_curand_gpu(curandState *states, long base_seed, long nthreads);

/*
 * Find local maxima in floating point array 'd_data', with specified width and height.
 * Local maxima are stored as x, y coordinates in device array 'd_result'
 */
xy<int> *find_local_maxima_gpu(float *d_data, int width, int height, xy<int> *d_result, float minval);
//void find_local_maxima_trees_gpu(float *d_data, int width, int height, int *species_map, float *a, float *b, mosaic_tree *d_result_begin, int *ntrees);

/*
 * In CHM array contained in 'mem' struct, find local maxima locations and assign trees to those locations. Store them in
 * 'd_temp_trees'
 */
int find_local_maxima_trees_gpu(canopy_placer::gpumem mem, mosaic_tree *d_temp_trees, int width, int height, float minval);

/*
 * Get data contained in color_grid, and store as unsigned 32 bit integers in 'result' array
 */
void get_cuda_texture_object_data(cudaTextureObject_t color_grid, int width, int height, uint32_t *result);

/*
 * Reset computed canopy tree centers, so that we can restart a computation on them in the next iteration
 */
void reset_centers_gpu(canopy_placer::xy_avg *centers, int nlocs);

/*
 * Smooth 'data' array with radial, uniform kernel
 */
void smooth_uniform_radial(int radius, float *data, float *result, int width, int height);

/*
 * Sample new trees in radial pattern around current trees, stored in the 'mem' structure.
 * Used directly in canopy placement algorithm in 'canopy_placer' class
 */
void sample_radial_trees_gpu(canopy_placer::gpumem mem, int width, int height, int ncur_trees, int sample_mult);

void scandebug(mosaic_tree *trees, int *indicators, int *nlocs);

/*
 * Bilinear upsample of srcdata array assuming column major ordering of data. Upsample by factor of 'factor'.
 * This is done on the GPU
 */
void bilinear_upsample_colmajor_gpu(float *srcdata, float *destdata, int srcw, int srch, int factor);
void bilinear_upsample_colmajor_test(float *srcdata, float *destdata, int srcw, int srch, int factor);

/*
 * Allocate data on GPU in preperation for upsampling on GPU via bilinear_upsample_colmajor_gpu
 */
void bilinear_upsample_colmajor_allocate_gpu(float *srcdata, float *destdata, int srcw, int srch, int factor);

/*
 * Convert colours in rendered texture, to integers indicating to what species each tree belongs
 * (Pretty much only for debuggin purposes, not really useful in the pipeline itself)
 */
ValueMap<int> apply_species_to_rendered_texture_gpu(cudaTextureObject_t color_grid, mosaic_tree * trees, int width, int height);


enum verbosity
{
    NO,
    SOLUTION,
    ALL
};

bool test_find_local_maxima_gpu(verbosity v);
#endif	 // GPU_PROCS_H
