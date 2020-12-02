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


#ifndef KMEANS_ASSIGN_H
#define KMEANS_ASSIGN_H

#include <data_importer/AbioticMapper.h>
#include <cluster_distribs/src/ClusterAssign.h>
#include <array>
#include "common/constants.h"

#include "cudatypes.h"




__device__
void assign_dists(AbioticMapsGpu amaps, KmeansData kdata, int *idxarr, float *distarr, int row, int pitch)
{
    int clidx = threadIdx.x;
    if (clidx < kdata.nmeans)
    {
        int mapidx = blockIdx.x + row * pitch;
        float fvals[4];

        idxarr[clidx] = clidx;
        fvals[0] = (amaps.wet.data[mapidx] - kdata.fmins[0]) / (kdata.fmaxs[0] - kdata.fmins[0]);
        fvals[1] = (amaps.sun.data[mapidx] - kdata.fmins[1]) / (kdata.fmaxs[1] - kdata.fmins[1]);
        fvals[2] = (amaps.slope.data[mapidx] - kdata.fmins[2]) / (kdata.fmaxs[2] - kdata.fmins[2]);
        fvals[3] = (amaps.temp.data[mapidx] - kdata.fmins[3]) / (kdata.fmaxs[3] - kdata.fmins[3]);

        float *f = kdata.get_cluster_ptr(clidx);
        float dist = 0.0f;
        for (int i = 0; i < kdata.nfeatures; i++)
        {
            float v = fvals[i] - f[i];
            dist += v * v;
        }
        //printf("Assigning distance %f to distarr[%d]\n", dist, clidx);
        distarr[clidx] = dist;
    }
}

__global__
void assign_dists_row(AbioticMapsGpu amaps, KmeansData kdata, int *row_idxarr, float *row_distarr, int row, int pitch)
{
    int colidx = blockIdx.x;
    if (colidx < pitch)
        assign_dists(amaps, kdata, row_idxarr + colidx * kdata.nmeans, row_distarr + colidx * kdata.nmeans, row, pitch);
}

__global__
void find_minima_idxes(float *alldata, int *allidxes, int *minidxes, int set_size, int nsets)
{
    int currsize = set_size;
    float *data = alldata + set_size * blockIdx.x;
    int *idxes = allidxes + set_size * blockIdx.x;
    int idx = threadIdx.x;
    if (idx < set_size / 2)
    {
        while (currsize > 1)
        {
            if (currsize % 2 == 1)
            {
                if (data[currsize - 1] < data[currsize - 2])
                {
                    data[currsize - 2] = data[currsize - 1];
                    idxes[currsize - 2] = idxes[currsize - 1];
                }
                currsize--;
            }
            else
            {
                if (data[currsize / 2 + idx] < data[idx])
                {
                    data[idx] = data[currsize / 2 + idx];
                    idxes[idx] = idxes[currsize / 2 + idx];
                }
                currsize /= 2;
            }
            __syncthreads();
        }
        if (idx == 0)
        {
            //printf("Assigning idx %d to minidxes[%d]\n", idxes[0], blockIdx.x);
            minidxes[blockIdx.x] = idxes[0];
        }
    }
}

__global__
void create_sbmap_gpu(basic_tree *trees, int ntrees, ValueGridMapGPU<unsigned int> sbmap, int *canopyspec_to_sb, float sample_mult)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < ntrees)
    {
        basic_tree curr = trees[idx];
        xy<int> gridxy = sbmap.togrid_safe(curr.x, curr.y);
        xy<int> start = sbmap.togrid_safe(curr.x - curr.radius * sample_mult, curr.y - curr.radius * sample_mult);
        xy<int> end = sbmap.togrid_safe(curr.x + curr.radius * sample_mult, curr.y + curr.radius * sample_mult);
        int maxdistsq = sbmap.togrid_safe(curr.radius * sample_mult, 0.0f).x;
        maxdistsq *= maxdistsq;

        //printf("Tree at idx %d: start xy: %d, %d. End xy: %d, %d. Tree xy: %f, %f. \n", idx, start.x, start.y, end.x, end.y, curr.x, curr.y);

        for (int y = start.y; y <= end.y; y++)
        {
            for (int x = start.x; x <= end.x; x++)
            {
                int distsq = (y - gridxy.y) * (y - gridxy.y) + (x - gridxy.x) * (x - gridxy.x);
                if (distsq < maxdistsq)
                {
                    int sbcode = canopyspec_to_sb[curr.species];
                    unsigned int prevval = sbmap.get(x, y);
                    unsigned int orval = 1;
                    orval <<= sbcode;
                    unsigned int newval = prevval | orval;
                    sbmap.set(x, y, newval);
                }
            }
        }
    }
}

__global__
void create_duplmap_gpu(basic_tree *trees, int ntrees, ValueGridMapGPU<unsigned char> duplmap, float sample_mult)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < ntrees)
    {
        basic_tree curr = trees[idx];
        xy<int> gridxy = duplmap.togrid_safe(curr.x, curr.y);
        xy<int> start = duplmap.togrid_safe(curr.x - curr.radius * sample_mult, curr.y - curr.radius * sample_mult);
        xy<int> end = duplmap.togrid_safe(curr.x + curr.radius * sample_mult, curr.y + curr.radius * sample_mult);
        int maxdistsq = duplmap.togrid_safe(0.75f, 0.0f).x;
        maxdistsq *= maxdistsq;

        //printf("Tree at idx %d: start xy: %d, %d. End xy: %d, %d. Tree xy: %f, %f. \n", idx, start.x, start.y, end.x, end.y, curr.x, curr.y);

        for (int y = start.y; y <= end.y; y++)
        {
            for (int x = start.x; x <= end.x; x++)
            {
                int distsq = (y - gridxy.y) * (y - gridxy.y) + (x - gridxy.x) * (x - gridxy.x);
                if (distsq < maxdistsq)
                {
                    duplmap.set(x, y, 1);
                }
            }
        }
    }
}

ValueGridMapGPU<unsigned char> create_duplmap(basic_tree *d_trees, int ntrees, int gw, int gh, float rw, float rh)
{
    float sample_mult = common_constants::undersim_sample_mult;

    ValueGridMapGPU<unsigned char> duplmap;
    duplmap.set_dims(gw, gh, rw, rh);
    duplmap.allocate();

    int nthreads = 1024;
    int nblocks = (gw * gh - 1) / nthreads + 1;
    create_duplmap_gpu<<<nblocks, nthreads>>>(d_trees, ntrees, duplmap, sample_mult);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return duplmap;
}

__global__
void create_clustermap_gpu(ValueGridMapGPU<unsigned int> sbmap, ValueGridMapGPU<int> clidxes, int nmeans)
{
    int maxidx = sbmap.gw * sbmap.gh;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < maxidx)
    {
        if (sbmap.data[idx] > 0)
            clidxes.data[idx] = clidxes.data[idx] + (sbmap.data[idx] - 1) * nmeans;
        else
            clidxes.data[idx] = -1;
    }
}

__global__
void compute_region_target_counts(ValueGridMapGPU<int> regionmap, float *densities, float *targetcounts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < regionmap.gw * regionmap.gh)
    {
        int clusteridx = regionmap.data[idx];
        if (clusteridx >= 0)
        {
            float d = densities[clusteridx];
            if (d >= 0)
            {
                atomicAdd(targetcounts + clusteridx, d);
            }
            else
            {
                atomicExch(targetcounts + clusteridx, -1.0f);
                //targetcounts[clusteridx] = -1.0f;
            }
        }
    }
}


__global__
void cast_region_target_counts(float *fcounts, int *icounts, int nclusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nclusters)
    {
        icounts[idx] = int(fcounts[idx]);
    }
}

ValueGridMapGPU<int> assign_clusteridxes(const abiotic_maps_package &amaps, const ClusterAssign &classign, const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata)
{
    AbioticMapsGpu amaps_gpu(amaps);
    KmeansData kmeans_gpu(classign);

    basic_tree *d_trees;
    int ntrees = canopytrees.size();
    gpuErrchk(cudaMalloc(&d_trees, sizeof(basic_tree) * ntrees));
    gpuErrchk(cudaMemcpy(d_trees, canopytrees.data(), sizeof(basic_tree) * ntrees, cudaMemcpyHostToDevice));

    std::vector<int> canopyspec_to_subbiome;
    for (auto &ctosub : cdata.canopyspec_to_subbiome)
    {
        if (ctosub.first >= canopyspec_to_subbiome.size())
        {
            canopyspec_to_subbiome.resize(ctosub.first + 1);
        }
        canopyspec_to_subbiome.at(ctosub.first) = ctosub.second;
    }
    int *d_canopyspec_to_subbiome;
    gpuErrchk(cudaMalloc(&d_canopyspec_to_subbiome, sizeof(int) * canopyspec_to_subbiome.size()));
    gpuErrchk(cudaMemcpy(d_canopyspec_to_subbiome, canopyspec_to_subbiome.data(), sizeof(int) * canopyspec_to_subbiome.size(), cudaMemcpyHostToDevice));

    ValueGridMapGPU<unsigned int> sbmap;
    sbmap.set_dims(amaps.gw, amaps.gh, amaps.rw, amaps.rh);
    gpuErrchk(cudaMalloc(&sbmap.data, sizeof(unsigned int) * amaps.gw * amaps.gh));
    gpuErrchk(cudaMemset(sbmap.data, 0, sizeof(unsigned int) * amaps.gw * amaps.gh));

    int nthreads = 1024;
    int nblocks = (ntrees - 1) / nthreads + 1;
    create_sbmap_gpu<<<nblocks, nthreads>>>(d_trees, ntrees, sbmap, d_canopyspec_to_subbiome, common_constants::undersim_sample_mult);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ValueGridMap<unsigned int> sbmap_cpu;
    sbmap_cpu.setDim(amaps.wet);
    sbmap_cpu.setDimReal(amaps.wet);
    gpuErrchk(cudaMemcpy(sbmap_cpu.data(), sbmap.data, sizeof(unsigned int) * amaps.gw * amaps.gh, cudaMemcpyDeviceToHost));
    data_importer::write_txt<ValueGridMap<unsigned int> >("/home/konrad/sbmap_gpu.txt", &sbmap_cpu);

    int *row_idxarr;
    float *row_distarr;

    gpuErrchk(cudaMalloc(&row_idxarr, sizeof(int) * amaps.gw * classign.get_nmeans()));
    gpuErrchk(cudaMalloc(&row_distarr, sizeof(float) * amaps.gw * classign.get_nmeans()));

    ValueGridMapGPU<int> clustermap_gpu;
    clustermap_gpu.set_dims(amaps.gw, amaps.gh, amaps.rw, amaps.rh);
    clustermap_gpu.allocate();

    int nrows = amaps.gh;
    int pitch = amaps.gw;

    for (int row = 0; row < nrows; row++)
    {
        assign_dists_row<<<pitch, 1024>>>(amaps_gpu, kmeans_gpu, row_idxarr, row_distarr, row, pitch);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        find_minima_idxes<<<pitch, 512>>>(row_distarr, row_idxarr, clustermap_gpu.data + row * pitch, classign.get_nmeans(), pitch);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

    nthreads = 1024;
    nblocks = (amaps.gw * amaps.gh - 1) / nthreads + 1;
    create_clustermap_gpu<<<nblocks, nthreads>>>(sbmap, clustermap_gpu, classign.get_nmeans());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(row_idxarr));
    gpuErrchk(cudaFree(row_distarr));
    amaps_gpu.free_data();
    kmeans_gpu.free_data();


    gpuErrchk(cudaFree(d_trees));
    gpuErrchk(cudaFree(d_canopyspec_to_subbiome));
    gpuErrchk(cudaFree(sbmap.data));

    return clustermap_gpu;
}

#endif // KMEANS_ASSIGN_H
