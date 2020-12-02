/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
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


// grass.h: seperate simulator for grass
// author: James Gain
// date: 17 August 2016

#ifndef _grass_h
#define _grass_h

#include "eco.h"

#define MAXGRASSHGHT 60.0f    // maximum grass height in cm

class MapFloat
{
private:
    int gx, gy;                     //< grid dimensions
    std::vector<float> fmap;        //< grid of floating point values

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy) const { return dx * gy + dy; }

public:

    MapFloat(){ gx = 0; gy = 0; initMap(); }

    ~MapFloat(){ delMap(); }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy) const { dx = gx; dy = gy; }

    /// setter for grid dimensions
    void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }

    /// clear the contents of the grid to empty
    void initMap(){ fmap.clear(); fmap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ fmap.clear(); }

    /// set grass heights to a uniform value
    void fill(float h){ fmap.clear(); fmap.resize(gx*gy, h); }

    /// getter and setter for map elements
    float get(int x, int y) const { return fmap[flatten(x, y)]; }
    void set(int x, int y, float val){ fmap[flatten(x, y)] = val; }

    /**
     * @brief read  read a floating point data grid from file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool read(std::string filename);

    float *data() { return fmap.data(); }
};

class EcoSystem;

#endif
