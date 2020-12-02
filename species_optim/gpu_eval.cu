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


#include "gpu_eval.h"

#include <vector>
#include <cassert>
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>

#include "assert.h"

#include "common/basic_types.h"


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

#define THREADS_PER_BLOCK 1024

__device__
int get_thread_idx()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__
float gauss_kernel(float x, float d, float loc)
{
    return (1 + AVAR) * expf(log(DCONST) / powf(d, POWCONST) * powf(fabs(x - loc), POWCONST)) - AVAR;
}

/*
 * Note that maps array contains each map's value in succession. I.e., it is ordered:
 * m11 m21 m31 .. mn1 m12 m22 m32 .. mn2 ...  where mnh is the hth value of the nth map
 */
__global__
void transform_adapt_eval_each(interm_memory mem, float *locs, float *scales)
{
    int mapsarr_size = mem.nmaps * mem.map_size;

    int thread_idx = get_thread_idx();

    if (thread_idx < mem.map_size)
    {
        float *maps = mem.maps;
        int nmaps = mem.nmaps;
        int nspecies = mem.nspecies;
        int index_on_map = mem.real_idxes[thread_idx];
        float spec_adv = -10000;		// XXX: remove the smoothmap stuff later
        float adv_quant = 0;
        if (mem.specdraw_map && mem.specadv_smoothmap)
        {
            spec_adv = mem.specdraw_map[index_on_map];
            adv_quant = mem.specadv_smoothmap[index_on_map];
        }
        for (int i = 0; i < nspecies; i++)
        {
            for (int map_idx = 0; map_idx < nmaps; map_idx++)
            {
                float scale = scales[map_idx * nspecies + i];
                float loc = locs[map_idx * nspecies + i];
                float adv;
                if (i == spec_adv)
                    adv = adv_quant;
                else
                    adv = 0.0f;
                float adapt_val = gauss_kernel(maps[thread_idx * nmaps + map_idx], scale, loc) + adv;
                if (adapt_val < ADAPT_CUTOFF)
                    adapt_val = 0.0f;
                mem.species_adapts[thread_idx * nmaps * nspecies + i * nmaps + map_idx] = adapt_val;
            }
        }
    }
}

// in the species_adapts array, map index changes the fastest, then species index, then map value (xy location)
__global__
void create_species_min_map(interm_memory mem, int *minvalue_counts, float *multmaps)
{
    int thread_idx = get_thread_idx();
    int result_map_size = mem.get_species_minvals_size();

    int nthreads = blockDim.x * gridDim.x;

    int niters = (result_map_size - 1) / nthreads + 1;

    for (int i = 0; i < niters; i++)
    {
        int idx = thread_idx + i * nthreads;
        if (idx < result_map_size)
        {
            int idx_onmap = idx / mem.nspecies;
            int spec_idx = idx % mem.nspecies;
            float minval = 1e20;
            int minidx = -1;
            // for each map, we check to see if the species' adaptation value is less than the current minimum
            for (int mapidx = 0; mapidx < mem.nmaps; mapidx++)
            {
                if (mem.species_adapts[idx * mem.nmaps + mapidx] < minval)
                {
                    minval = mem.species_adapts[idx * mem.nmaps + mapidx];
                    minidx = mapidx;
                }
            }
            float mult = multmaps[idx];
            float cheight = mem.chm_map[idx_onmap] * 0.3048f;
            // if we want to test for this (mem.chm_map not nullptr) and the chm value is higher than maximum species value, disregard this species
            if (mem.chm_map && cheight > mem.spec_maxheights[spec_idx])
            {
                minval = 0.0f;
            }
            mem.species_minvals[idx] = minval * mult;
            atomicAdd(minvalue_counts + spec_idx * mem.nmaps + minidx, 1);
        }
    }
}

void get_species_minvals(std::map<int, ValueMap<float> > &result, std::vector<int> spec_idx_map, interm_memory gpu_mem, int width, int height)
{
    std::vector<int> nonzero_idxes(gpu_mem.map_size);
    std::vector<float> result_arr(gpu_mem.map_size * gpu_mem.nspecies);
    result.clear();
    for (auto &idx : spec_idx_map)
    {
        result[idx].setDim(width, height);
        result[idx].fill(-2.0f);
    }
    gpuErrchk(cudaMemcpy(nonzero_idxes.data(), gpu_mem.real_idxes, sizeof(int) * gpu_mem.map_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(result_arr.data(), gpu_mem.species_minvals, sizeof(float) * gpu_mem.map_size * gpu_mem.nspecies, cudaMemcpyDeviceToHost));
    for (int i = 0; i < gpu_mem.map_size * gpu_mem.nspecies; i++)
    {
        int x, y;
        int idx = i / gpu_mem.nspecies;
        int real_idx = nonzero_idxes[idx];
        int fake_specidx = i % gpu_mem.nspecies;
        int specidx = spec_idx_map[fake_specidx];
        result.at(specidx).idx_to_xy(real_idx, x, y);
        result.at(specidx).set(x, y, result_arr[i]);
    }
}

__global__
void create_species_winners(interm_memory mem, int *nzero_winners)
{
    int thread_idx = get_thread_idx();
    int result_map_size = mem.get_map_size();

    if (thread_idx < result_map_size)
    {
        float maxval = 0.0f;
        int max_idx = -1;
        for (int i = 0; i < mem.nspecies; i++)
        {
            float val;
            if ((val = mem.species_minvals[thread_idx * mem.nspecies + i]) > maxval)
            {
                maxval = val;
                max_idx = i;
            }
        }
        if (max_idx == -1)
        {
            atomicAdd(nzero_winners, 1);
            mem.species_winners[thread_idx] = -1;
        }
        else
        {
            mem.species_winners[thread_idx] = max_idx;
        }
    }
}

__global__
void calc_species_winners_percentage(interm_memory mem)
{
    int thread_idx = get_thread_idx();
    int maxthreads = blockDim.x;

    int niters = (mem.get_map_size() - 1) / maxthreads + 1;

    for (int i = 0; i < niters; i++)
    {
        int idx = thread_idx + maxthreads * i;
        if (idx < mem.get_map_size())
        {
            int win_idx = mem.species_winners[idx];
            if (win_idx > -1)
                atomicAdd(mem.species_percentages + win_idx, 1);
            if (win_idx >= mem.nspecies)
            {
                assert(false);
            }
        }
    }
}

/*
 *
 * species_locs and species_scales are arranged as follows:
 * m1sp1, m1sp2, m1sp3, ..., m2sp1, m2sp2, m2sp3, ...		(i.e., species changes the quickest)
 * maps are interlaced, i.e., they are arranged as
 * m1(x=0, y=0), m2(x=0, y=0), ..., m1(x=1, y=0), m2(x=1, y=0), ...		(i.e. maps change the fastest, then coordinates)
 */

interm_memory create_cuda_memory(float ** maps, float *chmmap, int msize, int nmaps, int nspecies, int *nonzero_idxes, int map_width, int map_height, float *spec_maxheights)
{
    interm_memory gpu_mem;
    memset(&gpu_mem, 0, sizeof(gpu_mem));
    gpu_mem.map_height = map_height;
    gpu_mem.map_width = map_width;
    gpu_mem.map_size = msize;
    gpu_mem.nmaps = nmaps;
    gpu_mem.nspecies = nspecies;

    std::vector<float> temp_maps(nmaps * msize);
    for (int i = 0; i < msize; i++)
    {
        for (int mapi = 0; mapi < nmaps; mapi++)
        {
            temp_maps[i * nmaps + mapi] = maps[mapi][i];
        }
    }

    std::cout << "Nspecies when creating cuda memory: " << nspecies << std::endl;
    gpuErrchk(cudaMalloc(&gpu_mem.maps, sizeof(float) * temp_maps.size()));
    gpuErrchk(cudaMalloc(&gpu_mem.species_adapts, sizeof(float) * nmaps * nspecies * msize));
    gpuErrchk(cudaMalloc(&gpu_mem.species_minvals, sizeof(float) * nspecies * msize));
    gpuErrchk(cudaMalloc(&gpu_mem.species_winners, sizeof(int) * msize));
    gpuErrchk(cudaMalloc(&gpu_mem.species_percentages, sizeof(float) * nspecies));
    gpuErrchk(cudaMalloc(&gpu_mem.real_idxes, sizeof(int) * msize));
    gpuErrchk(cudaMalloc(&gpu_mem.spec_maxheights, sizeof(float) * nspecies));
    gpuErrchk(cudaMalloc(&gpu_mem.chm_map, sizeof(float) * msize));
    if (map_width > 0 && map_height > 0)
    {
        gpuErrchk(cudaMalloc(&gpu_mem.specdraw_map, sizeof(int) * map_width * map_height));
        gpuErrchk(cudaMalloc(&gpu_mem.specadv_smoothmap, sizeof(float) * map_width * map_height));
    }

    gpuErrchk(cudaMemcpy(gpu_mem.maps, temp_maps.data(), sizeof(float) * temp_maps.size(), cudaMemcpyHostToDevice));

    gpu_mem.nmaps = nmaps;
    gpu_mem.nspecies = nspecies;
    gpu_mem.map_size = msize;
    gpu_mem.map_width = map_width;
    gpu_mem.map_height = map_height;
    gpuErrchk(cudaMemcpy(gpu_mem.real_idxes, nonzero_idxes, sizeof(int) * msize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(gpu_mem.spec_maxheights, spec_maxheights, sizeof(float) * nspecies, cudaMemcpyHostToDevice));
    if (chmmap)		// we are already running short on gpu memory - if memory is an issue, we disable this experimental feature
        gpuErrchk(cudaMemcpy(gpu_mem.chm_map, chmmap, sizeof(float) * msize, cudaMemcpyHostToDevice))
    else
        gpu_mem.chm_map = nullptr;

    return gpu_mem;
}

void free_cuda_memory(interm_memory gpu_mem)
{
    gpuErrchk(cudaFree(gpu_mem.maps));
    gpuErrchk(cudaFree(gpu_mem.species_adapts));
    gpuErrchk(cudaFree(gpu_mem.species_minvals));
    gpuErrchk(cudaFree(gpu_mem.species_winners));
    gpuErrchk(cudaFree(gpu_mem.species_percentages));
    gpuErrchk(cudaFree(gpu_mem.real_idxes));
    gpuErrchk(cudaFree(gpu_mem.specdraw_map));
    gpuErrchk(cudaFree(gpu_mem.specadv_smoothmap));
    gpuErrchk(cudaFree(gpu_mem.spec_maxheights));
    gpuErrchk(cudaFree(gpu_mem.chm_map));
}

std::vector<float> evaluate_gpu(std::vector<float> &species_locs,
                                std::vector<float> &species_scales,
                                int *nzero_winners_arg,		// can be null
                                int *minvalue_counts_arg,		// can be null
                                interm_memory mem,
                                std::vector<float> &mult_maps,
                                int *species_winners	// can be null
                                )
{
    int nmaps = mem.nmaps;
    int nspecies = mem.nspecies;

    assert(species_locs.size() == nmaps * nspecies);
    assert(species_scales.size() == nmaps * nspecies);

    float *mult_maps_gpu;

    assert(mult_maps.size() == mem.map_size * mem.nspecies);

    gpuErrchk(cudaMalloc(&mult_maps_gpu, sizeof(float) * mem.map_size * mem.nspecies));
    gpuErrchk(cudaMemcpy(mult_maps_gpu, mult_maps.data(), sizeof(float) * mem.map_size * mem.nspecies, cudaMemcpyHostToDevice));

    std::cout << "nspecies: " << nspecies << std::endl;
    for (int i = 0; i <= nspecies; i++)
    {
        std::cout << "Setting memory for i = " << i << std::endl;
        gpuErrchk(cudaMemset(mem.species_percentages, 0, sizeof(float) * i));
    }

    float *species_locs_gpu;
    float *species_scales_gpu;
    gpuErrchk(cudaMalloc(&species_locs_gpu, sizeof(float) * nspecies * nmaps));
    gpuErrchk(cudaMalloc(&species_scales_gpu, sizeof(float) * nspecies * nmaps));

    gpuErrchk(cudaMemcpy(species_locs_gpu, species_locs.data(), sizeof(float) * nspecies * nmaps, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(species_scales_gpu, species_scales.data(), sizeof(float) * nspecies * nmaps, cudaMemcpyHostToDevice));

    int zero = 0;
    int *nzero_winners;
    gpuErrchk(cudaMalloc(&nzero_winners, sizeof(int)));
    gpuErrchk(cudaMemcpy(nzero_winners, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int *minvalue_counts;
    gpuErrchk(cudaMalloc(&minvalue_counts, sizeof(int) * nspecies * nmaps));
    gpuErrchk(cudaMemset(minvalue_counts, 0, sizeof(int) * nspecies * nmaps));

    int nblocks = mem.map_size / 1024 + (((mem.nmaps * mem.map_size) % 1024) ? 1 : 0);
    int nthreads = 1024;

    printf("nblocks, nthreads: %d, %d\n", nblocks, nthreads);
    printf("nmaps, map_size: %d, %d\n", mem.nmaps, mem.map_size);

    transform_adapt_eval_each<<<nblocks, nthreads>>>(mem, species_locs_gpu, species_scales_gpu);
    kernelCheck();
    nblocks = 1;
    nthreads = 1024;
    create_species_min_map<<<nblocks, nthreads>>>(mem, minvalue_counts, mult_maps_gpu);
    kernelCheck();
    nblocks = (mem.map_size - 1) / 1024 + 1;
    nthreads = 1024;
    create_species_winners<<<nblocks, nthreads>>>(mem, nzero_winners);
    kernelCheck();
    calc_species_winners_percentage<<<1, 1024>>>(mem);
    kernelCheck();

    std::vector<float> species_perc(nspecies);

    gpuErrchk(cudaMemcpy(species_perc.data(), mem.species_percentages, sizeof(float) * nspecies, cudaMemcpyDeviceToHost));

    if (species_winners)
    {
        gpuErrchk(cudaMemcpy(species_winners, mem.species_winners, sizeof(float) * mem.map_size, cudaMemcpyDeviceToHost));
    }

    if (nzero_winners_arg)
        gpuErrchk(cudaMemcpy(nzero_winners_arg, nzero_winners, sizeof(int), cudaMemcpyDeviceToHost));
    if (minvalue_counts_arg)
        gpuErrchk(cudaMemcpy(minvalue_counts_arg, minvalue_counts, sizeof(int) * nspecies * nmaps, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(species_locs_gpu));
    gpuErrchk(cudaFree(species_scales_gpu));

    gpuErrchk(cudaFree(nzero_winners));
    gpuErrchk(cudaFree(minvalue_counts));
    gpuErrchk(cudaFree(mult_maps_gpu));

    return species_perc;
}
