/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
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


#include "outimage.h"

#include <iostream>
#include <time.h>
#ifdef SLOWBUILD

// using namespace cimg_library;
#define cimg_use_magick

#endif
using namespace std;

bool OutImage::write(std::string filename, int width, int height, std::vector<float> flimg)
{
#ifdef SLOWBUILD

    int i = 0;
    CImg<unsigned char> img(width, height,1,1);  // Define a width * height single channel color image with 8 bits per color component

    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            img(x,y) = (int) (flimg[i] * 255.0f);
            i++;
        }
    img.save(filename.c_str());

#endif
    return true;
}
