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


#ifndef GPU_EVAL_H
#define GPU_EVAL_H

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <common/basic_types.h>

#define AVAR 0.2
#define DCONST 0.01
#define POWCONST 4.5
#define ADAPT_CUTOFF 0.01f		// This used to be 0.166, in accordance to how James translates the adaptability function. Might have to change it back at some point...

struct interm_memory
{
    float *maps;
    float *species_adapts;
    float *species_minvals;
    int *species_winners;
    float *species_percentages;
    float *winner_values;
    int *prev_winners;
    int *real_idxes;
    int *specdraw_map;
    float *specadv_smoothmap;
    curandState *random_states;
    float **spec_multmaps;
    float *spec_maxheights;
    float *chm_map;

    int map_width;
    int map_height;
    int map_size;
    int nmaps;
    int nspecies;

    __device__
    int get_species_minvals_size()
    {
        return get_map_size() * nspecies;
    }

    __device__
    int get_species_adapts_size()
    {
        return get_map_size() * nspecies * nmaps;
    }

    __device__
    int get_map_size()
    {
        return map_size;
    }
};

/*
 * Assign species, then compute and return a percentage representation for each species over the area defined by the 'mem' struct.
 * Note that this function also assigns the species to the interm_memory object 'mem' (species_winners member), so this function can also be
 * used for simple species assignment, not necessarily optimisation of a brushstroke/landscape
 */
std::vector<float> evaluate_gpu(std::vector<float> &species_locs, std::vector<float> &species_scales, int *nzero_winners_arg, int *minvalue_counts_arg, interm_memory mem, std::vector<float> &mult_maps, int *species_winners = nullptr);

/*
 * Initialize CUDA memory for a created interm_memory object, and return the object from this function
 */
interm_memory create_cuda_memory(float ** maps, float *chmmap, int msize, int nmaps, int nspecies, int *nonzero_idxes, int map_width, int map_height, float *spec_maxheights);

/*
 * Free CUDA memory from the interm_memory object 'gpu_mem'
 */
void free_cuda_memory(interm_memory gpu_mem);

/*
 * Assign minimum values for each species to 'result'. Could be useful for some debugging of intermediate values
 */
void get_species_minvals(std::map<int, ValueMap<float> > &result, std::vector<int> spec_idx_map, interm_memory gpu_mem, int width, int height);

#endif
