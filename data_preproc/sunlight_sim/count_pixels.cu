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
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cstdio>
#include <assert.h>

#include "count_pixels.h"

#define blocksize 10

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define kernelCheck() gpuErrchk(cudaPeekAtLastError()); gpuErrchk(cudaDeviceSynchronize());

void printDevProp(cudaDeviceProp devProp)
{
    fprintf(stderr, "%s\n", devProp.name);
    fprintf(stderr, "Major revision number:         %d\n", devProp.major);
    fprintf(stderr, "Minor revision number:         %d\n", devProp.minor);
    fprintf(stderr, "Total global memory:           %zu", devProp.totalGlobalMem);
    fprintf(stderr, " bytes\n");
    fprintf(stderr, "Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    fprintf(stderr, "Total amount of shared memory per block: %zu\n", devProp.sharedMemPerBlock);
    fprintf(stderr, "Total registers per block:     %d\n", devProp.regsPerBlock);
    fprintf(stderr, "Warp size:                     %d\n", devProp.warpSize);
    fprintf(stderr, "Maximum memory pitch:          %zu\n", devProp.memPitch);
    fprintf(stderr, "Total amount of constant memory:         %zu\n", devProp.totalConstMem);
    return;
}

__device__
int get_map_idx_from_color(short *color, short *base_color)
{
    color[0] -= base_color[0];
    if (color[0] < 0)
    {
        color[0] += 256;
        color[1]--;
    }
    color[1] -= base_color[1];
    if (color[1] < 0)
    {
        color[1] += 256;
        color[2]--;
    }
    color[2] -= base_color[2];

    return color[0] + color[1] * 256 + color[2] * 256 * 256;
}
	
__global__
void count_pixels(float incr, int pass, gpumem mem, short basecol_r, short basecol_g, short basecol_b)
{
    uint32_t *pixels = mem.pixels;
    float *sums = mem.sums;
    int *visited = mem.visited;
    int width = mem.tex_width;
    int height = mem.tex_height;
    int mapw = mem.map_width;
    int maph = mem.map_height;
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int ncols = (width - 1) / (blocksize * 2) + 1;
    int nrows = (height - 1) / blocksize + 1;
    int y = tidx / ncols;
    int x = tidx % ncols;
    int ncells = ncols * nrows;

    y *= blocksize;
	x *= blocksize * 2;
    if ((pass % 2) == (y % 2))
	{
        x += blocksize;
	}

	short color[3];
	short base_color[3];
	base_color[0] = basecol_r;
	base_color[1] = basecol_g;
    base_color[2] = basecol_b;

    if (tidx < ncells)
    {
        for (int cy = y; cy < y + blocksize; cy++)
        {
            for (int cx = x; cx < x + blocksize; cx++)
            {
                int idx = cy * width + cx;
                if (idx < width * height)
                {
                    uint32_t pixval = pixels[idx];
                    short r = pixval >> 24;
                    short g = (pixval << 8) >> 24;
                    short b = (pixval << 16) >> 24;
                    color[0] = r;
                    color[1] = g;
                    color[2] = b;
                    int mapidx = get_map_idx_from_color(color, base_color);
                    if (mapidx < 0)
                        continue;
                    if (mapidx >= mapw * maph)
                        printf("mapidx: %d. rgb: %d, %d, %d. idx: %d\n", mapidx, (int)r, (int)g, (int)b, idx);
                    assert(mapidx < mapw * maph);
                    if (!atomicCAS(visited + mapidx, 0, 1))
                    {
                        sums[mapidx] += incr;
                    }
                }
            }
        }
    }
}

void count_pixels_gpu(const std::vector<uint32_t> &pixels, float *sums, float incr, short *base_color, gpumem mem)
{
    /*
    cudaSetDevice(0);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printDevProp(props);
    */

    /*
	uint32_t *dpixels;
    float *dsums;
    int *dvisited;
    */

    kernelCheck();

    /*
    gpuErrchk(cudaMalloc(&dsums, sizeof(float) * mapw * maph));
    gpuErrchk(cudaMalloc(&dvisited, sizeof(int) * mapw * maph));
    gpuErrchk(cudaMalloc(&dpixels, sizeof(uint32_t) * width * height));
    */

    int width = mem.tex_width;
    int height = mem.tex_height;
    int mapw = mem.map_width;
    int maph = mem.map_height;

    gpuErrchk(cudaMemcpy(mem.pixels, pixels.data(), sizeof(uint32_t) * width * height, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(mem.sums, sums, sizeof(float) * mapw * maph, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(mem.visited, 0, sizeof(int) * mapw * maph));

    int n_tot_threads = (width * height) / (blocksize * 2);

    const int threads_per_cudablock = 1024;

    int nblocks = (n_tot_threads - 1) / threads_per_cudablock + 1;

    int pass = 0;
    count_pixels<<<nblocks, threads_per_cudablock>>>(incr, pass, mem, base_color[0], base_color[1], base_color[2]); kernelCheck();
    gpuErrchk(cudaDeviceSynchronize());
    pass = 1;
    count_pixels<<<nblocks, threads_per_cudablock>>>(incr, pass, mem, base_color[0], base_color[1], base_color[2]); kernelCheck();

    gpuErrchk(cudaMemcpy(sums, mem.sums, sizeof(float) * mapw * maph, cudaMemcpyDeviceToHost));

    /*
    gpuErrchk(cudaFree(dpixels));
    gpuErrchk(cudaFree(dsums));
    gpuErrchk(cudaFree(dvisited));
    */
}

gpumem alloc_gpu_memory(int tex_width, int tex_height, int map_width, int map_height)
{
    gpumem mem;
    gpuErrchk(cudaMalloc(&mem.pixels, sizeof(uint32_t) * tex_width * tex_height));
    gpuErrchk(cudaMalloc(&mem.sums, sizeof(float) * map_width * map_height));
    gpuErrchk(cudaMalloc(&mem.visited, sizeof(int) * map_width * map_height));
    mem.tex_height = tex_height;
    mem.tex_width = tex_width;
    mem.map_width = map_width;
    mem.map_height = map_height;

    return mem;
}

void free_gpu_memory(gpumem mem)
{
    gpuErrchk(cudaFree(mem.pixels));
    gpuErrchk(cudaFree(mem.sums));
    gpuErrchk(cudaFree(mem.visited));
}
