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

// descriptor.h: provides a map encapsulating all the terrain conditions
// author: James Gain
// date: 20 September 2016

#ifndef _descriptor_h
#define _descriptor_h

#include <vector>
#include <string>

struct SampleDescriptor // Information about the generating parameters for a sample
{
    float slope;
    int moisture[2];
    int sunlight[2];
    int temperature[2];
    int age;
};

class MapDescriptor
{
private:
    int gx, gy;                                //< grid dimensions
    std::vector<SampleDescriptor> dmap;        //< grid of descriptor values

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

public:

    MapDescriptor(){ gx = 0; gy = 0; initMap(); }

    ~MapDescriptor(){ delMap(); }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy){ dx = gx; dy = gy; }

    /// setter for grid dimensions
    void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }

    /// clear the contents of the grid to empty
    void initMap(){ dmap.clear(); dmap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ dmap.clear(); }

    /// getter and setter for map elements
    SampleDescriptor & get(int x, int y){ return dmap[flatten(x, y)]; }
    void set(int x, int y, SampleDescriptor sd){ dmap[flatten(x, y)] = sd; }
    // TO DO TEST SET

    /**
     * @brief read  read a descriptor data grid from file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool read(std::string filename);

    /**
     * @brief read  write a descriptor grid to file
     * @param filename  name of file to write
     * @return true if the file is successfully written, false otherwise
     */
    bool write(std::string filename);
};

#endif
