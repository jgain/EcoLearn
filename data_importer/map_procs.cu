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


#include "data_importer/map_procs.h"
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
void clear_sumdata(float *data, int w, int h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int maxidx = w * h;

	if (idx < maxidx)
	{
		data[idx] = 0.0f;
	}
}

__global__
void sum_month_data(float *data, int w, int h, float *sumdata)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int maxidx = w * h * 12;

	if (idx < maxidx)
	{
		int sumidx = idx % (w * h);
		float val = data[idx];
		atomicAdd(sumdata + sumidx, val);
	}
}

__global__
void average_month_data(float *data, int w, int h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int maxidx = w * h;

	if (idx < maxidx)
	{
		data[idx] /= 12.0f;
	}
}

ValueGridMap<float> average_monthly_data_hostcall(const std::vector<ValueGridMap<float> > &data)
{
	int w, h;
	float rw, rh;
	for (auto &dm : data)
	{
		dm.getDim(w, h);
		dm.getDimReal(rw, rh);
	}

	ValueGridMap<float> ret;
	ret.setDim(w, h);
	ret.setDimReal(rw, rh);

	float *d_alldata;
	gpuErrchk(cudaMalloc(&d_alldata, sizeof(float) * w * h * 12));
	for (int i = 0; i < 12; i++)
	{
		gpuErrchk(cudaMemcpy(d_alldata + w * h * i, data.at(i).data(), sizeof(float) * w * h, cudaMemcpyHostToDevice));
	}

	float *d_sumdata;
	gpuErrchk(cudaMalloc(&d_sumdata, sizeof(float) * w * h));

	int nthreads = 1024;
	int nblocks = (w * h - 1) / nthreads + 1;
	clear_sumdata<<<nblocks, nthreads>>>(d_sumdata, w, h);

	nblocks = (w * h * 12 - 1) / nthreads + 1;
	sum_month_data<<<nblocks, nthreads>>>(d_alldata, w, h, d_sumdata);

	nblocks = (w * h - 1) / nthreads + 1;
    average_month_data<<<nblocks, nthreads>>>(d_sumdata, w, h);

	gpuErrchk(cudaMemcpy(ret.data(), d_sumdata, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

	return ret;
}

ValueGridMap<float> average_monthly_data_hostcall(const std::vector<float> &data, int w, int h, float rw, float rh)
{
	ValueGridMap<float> ret;
	ret.setDim(w, h);
	ret.setDimReal(rw, rh);

	float *d_alldata;
	gpuErrchk(cudaMalloc(&d_alldata, sizeof(float) * w * h * 12));
	gpuErrchk(cudaMemcpy(d_alldata, data.data(), sizeof(float) * w * h * 12, cudaMemcpyHostToDevice));

	float *d_sumdata;
	gpuErrchk(cudaMalloc(&d_sumdata, sizeof(float) * w * h));

	int nthreads = 1024;
	int nblocks = (w * h - 1) / nthreads + 1;
	clear_sumdata<<<nblocks, nthreads>>>(d_sumdata, w, h);

	nblocks = (w * h * 12 - 1) / nthreads + 1;
	sum_month_data<<<nblocks, nthreads>>>(d_alldata, w, h, d_sumdata);

	nblocks = (w * h - 1) / nthreads + 1;
    average_month_data<<<nblocks, nthreads>>>(d_sumdata, w, h);

	gpuErrchk(cudaMemcpy(ret.data(), d_sumdata, sizeof(float) * w * h, cudaMemcpyDeviceToHost));

	return ret;
}
