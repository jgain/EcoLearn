
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


// outimage.h: simple wrapper for cimg functionality
// author: James Gain
// date: 17 August 2016

#include <string>
#include <vector>

//#define SLOWBUILD

class OutImage
{
public:

    /**
     * @brief write Write to a grascale PNG image
     * @param width     image width
     * @param height    image height
     * @param flimg     flattened image of [0-1] range grayscale pixels
     * @return      true if the write succeeded, false otherwise
     */
    bool write(std::string filename, int width, int height, std::vector<float> flimg);
};
