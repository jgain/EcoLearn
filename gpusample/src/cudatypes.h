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


#ifndef CUDATYPES_H
#define CUDATYPES_H

#include <cuda_runtime.h>
#include <cuda.h>

#include "common/basic_types.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void fill(float value, float *data, int gw, int gh)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < gw * gh)
        data[tidx] = value;
}

template<typename T>
struct ValueGridMapGPU
{
    float rtog;
    float gtor;
    int gw, gh;
    float rw, rh;
    T *data = nullptr;

    ValueGridMapGPU(const ValueGridMap<T> &map)
    {
        map.getDim(gw, gh);
        map.getDimReal(rw, rh);
        rtog = gw / rw;
        gtor = rw / gw;

        gpuErrchk(cudaMalloc(&data, sizeof(float) * gw * gh));
        gpuErrchk(cudaMemcpy(data, map.data(), sizeof(float) * gw * gh, cudaMemcpyHostToDevice));
    }

    ValueGridMapGPU()
    {}

    __host__
    void set_dims(int gw, int gh, float rw, float rh)
    {
        this->gw = gw, this->gh = gh, this->rw = rw, this->rh = rh;
        rtog = gw / rw;
        gtor = rw / gw;
    }

    __host__
    void allocate()
    {
        if (data)
            free_data();
        else
            gpuErrchk(cudaMalloc(&data, sizeof(T) * gw * gh));
    }

    __host__
    void free_data()
    {
        gpuErrchk(cudaFree(data));
        data = nullptr;
    }

    __device__
    xy<int> togrid_safe(float x, float y)
    {
        int gx = x * rtog;
        int gy = y * rtog;
        if (gx >= gw) gx = gw - 1;
        if (gx < 0) gx = 0;
        if (gy >= gh) gy = gh - 1;
        if (gy < 0) gy = 0;

        return xy<int>(gx, gy);
    }

    __device__
    xy<float> toreal_safe(int x, int y)
    {
        float rx = x * gtor;
        float ry = y * gtor;
        if (rx >= rw) rx = rw - 1;
        if (rx < 0) rx = 0;
        if (ry >= rh) ry = rh - 1;
        if (ry < 0) ry = 0;

        return xy<float>(rx, ry);
    }

    __device__
    void set(int x, int y, T value)
    {
        int idx = y * gw + x;
        data[idx] = value;
    }

    __device__
    T get(int x, int y)
    {
        int idx = y * gw + x;
        return data[idx];
    }

    void fill_hostcall(float value)
    {
        int nthreads = 1024;
        int nblocks = (gw * gh - 1) / nthreads + 1;
        fill<<<nblocks, nthreads>>>(value, data, gw, gh);
    }

    ValueGridMap<T> toValueGridMap()
    {
        ValueGridMap<T> other;
        other.setDim(gw, gh);
        other.setDimReal(rw, rh);
        gpuErrchk(cudaMemcpy(other.data(), data, sizeof(T) * gw * gh, cudaMemcpyDeviceToHost));
        return other;
    }
};

struct KmeansData
{
        float *means;
        int nmeans;
        int nfeatures;		// this should be 4: wet, sun, slope and temp
        float *fmins, *fmaxs;	// minima and maxima for each feature, respectively

        KmeansData(const ClusterAssign &classign)
        {
            std::vector<std::array<float, 4> > means_obj = classign.get_means();
            nmeans = means_obj.size();
            nfeatures = 4;
            std::array<std::pair<float, float>, 4> minmax_ranges = classign.get_minmax_ranges();

            std::vector<float> means_cpu(nmeans * nfeatures, 0.0f);
            std::vector<float> fmins_cpu(4, std::numeric_limits<float>::max());
            std::vector<float> fmaxs_cpu(4, -std::numeric_limits<float>::max());

            for (int i = 0; i < nmeans; i++)
            {
                for (int j = 0; j < nfeatures; j++)
                {
                    means_cpu.at(i * nfeatures + j) = means_obj.at(i).at(j);
                }
            }
            for (int i = 0; i < nfeatures; i++)
            {
                fmins_cpu.at(i) = minmax_ranges.at(i).first;
                fmaxs_cpu.at(i) = minmax_ranges.at(i).second;
            }

            gpuErrchk(cudaMalloc(&means, sizeof(float) * nmeans * nfeatures));
            gpuErrchk(cudaMalloc(&fmins, sizeof(float) * nfeatures));
            gpuErrchk(cudaMalloc(&fmaxs, sizeof(float) * nfeatures));

            gpuErrchk(cudaMemcpy(means, means_cpu.data(), sizeof(float) * nmeans * nfeatures, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(fmins, fmins_cpu.data(), sizeof(float) * nfeatures, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(fmaxs, fmaxs_cpu.data(), sizeof(float) * nfeatures, cudaMemcpyHostToDevice));
        }

        __device__
        float *get_cluster_ptr(int clidx)
        {
            return means + nfeatures * clidx;
        }

        void free_data()
        {
            gpuErrchk(cudaFree(means));
            gpuErrchk(cudaFree(fmins));
            gpuErrchk(cudaFree(fmaxs));
        }
};

struct AbioticMapsGpu
{
        AbioticMapsGpu(const abiotic_maps_package &amaps)
            : wet(amaps.wet), sun(amaps.sun), slope(amaps.slope), temp(amaps.temp)
        {
        }

        void free_data()
        {
            gpuErrchk(cudaFree(wet.data));
            gpuErrchk(cudaFree(sun.data));
            gpuErrchk(cudaFree(slope.data));
            gpuErrchk(cudaFree(temp.data));
        }

        ValueGridMapGPU<float> wet, sun, slope, temp;
};

#endif     // CUDATYPES_H
