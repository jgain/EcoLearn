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


#pragma once
#ifndef MISC_H
#define MISC_H

#include "common/basic_types.h"
//#include "grass.h"

//#include <SDL2/SDL.h>
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <boost/filesystem.hpp>


struct rgba
{
    int r, g, b, a;
};

template<typename T>
void trim(T &target, T min, T max)
{
    if (target < min)
        target = min;
    else if (target > max)
        target = max;
}

template<typename T>
inline T sq_distance(T x0, T x1, T y0, T y1)
{
    T xdiff = x1 - x0;
    T ydiff = y1 - y0;
    xdiff *= xdiff;
    ydiff *= ydiff;
    return xdiff + ydiff;
}

template<typename T>
inline T distance(T x0, T x1, T y0, T y1)
{
    T xdiff = x1 - x0;
    T ydiff = y1 - y0;
    xdiff *= xdiff;
    ydiff *= ydiff;
    return sqrt(sq_distance(x0, x1, y0, y1));
}

inline rgba idx_to_rgba(int idx)
{
    idx++;
    int r = idx % 256;
    idx -= r;
    int g = (idx % (256 * 256)) / 256;
    idx -= g * 256;
    int b = (idx % (256 * 256 * 256)) / (256 * 256);
    int a = 255;
    rgba color = {r: r, g: g, b: b, a: a};
    return color;
}

inline int rgba_to_idx(rgba color)
{

    int idx = color.b * 256 * 256 + color.g * 256 + color.r - 1;

    return idx;
}

inline int color_int_to_idx(uint32_t colval)
{
        uint32_t r = colval >> 24;
        uint32_t g = (colval << 8) >> 24;
        uint32_t b = (colval << 16) >> 24;
        uint32_t a = (colval << 24) >> 24;
        return rgba_to_idx({r, g, b, a});
}

inline float stdnorm_cdf(float x)
{
    return 0.5 + 0.5 * erf(x * sqrt(0.5));
}

template<typename T>
inline float is_between(T val, T a, T b)
{
    if (a > b)
    {
        std::swap(a, b);
    }
    return val > a && val < b;
}

#endif
