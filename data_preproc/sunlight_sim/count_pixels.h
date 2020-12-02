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


#ifndef COUNT_PIXELS_H
#define COUNT_PIXELS_H

#include <vector>
#include <cinttypes>

struct gpumem
{
    float *sums;
    uint32_t *pixels;
    int *visited;
    int tex_width;
    int tex_height;
    int map_width;
    int map_height;
};

void count_pixels_gpu(const std::vector<uint32_t> &pixels, float *sums, float incr, short *base_color, gpumem mem);
gpumem alloc_gpu_memory(int tex_width, int tex_height, int map_width, int map_height);
void free_gpu_memory(gpumem mem);

#endif
