
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


// grass.h: seperate simulator for grass
// author: James Gain
// date: 17 August 2016

#ifndef _grass_h
#define _grass_h

//#include "eco.h"
#include "MapFloat.h"
#include "data_importer/data_importer.h"

#define MAXGRASSHGHT 60.0f    // maximum grass height in cm

class Terrain;
class TypeMap;
class EcoSystem;

/*
class MapFloat
{
private:
    int gx, gy;                     //< grid dimensions
    std::vector<float> fmap;        //< grid of floating point values


public:

    MapFloat(){ gx = 0; gy = 0; initMap(); }

    ~MapFloat(){ delMap(); }

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy){ dx = gx; dy = gy; }

    int height(){ return gy; }
    int width(){ return gx; }

    /// setter for grid dimensions
    void setDim(int dx, int dy){ gx = dx; gy = dy; initMap(); }

    /// clear the contents of the grid to empty
    void initMap(){ fmap.clear(); fmap.resize(gx*gy); }

    /// completely delete map
    void delMap(){ fmap.clear(); }

    /// set grass heights to a uniform value
    void fill(float h){ fmap.clear(); fmap.resize(gx*gy, h); }

    /// getter and setter for map elements
    float get(int x, int y){ return fmap[flatten(x, y)]; }
    float get(int idx){ return fmap[idx]; }
    void set(int x, int y, float val){ fmap[flatten(x, y)] = val; }

    // * @brief read  read a floating point data grid from file
    // * @param filename  name of file to be read
    // * @return true if the file is found and has the correct format, false otherwise
    bool read(std::string filename);

    float *data() { return fmap.data(); }

};
*/


class GrassSim
{
private:
    MapFloat grasshght;   //< linearized map of grass heights
    MapFloat backup_grass;
    MapFloat * moisture, * illumination, * temperature, * landsun; //< condition maps
    MapFloat litterfall_density;
    float scx, scy;       //< size of an individual grass cell in meters
    float hscx, hscy;     //< half a grass cell in meters
    bool has_backup = false;
    std::map<std::string, data_importer::grass_viability> viability;

    /// map terrain condition grid coordinates to grass coordinates
    void convertCoord(Terrain * ter, int x, int y, int & sx, int & sy, int &ex, int &ey);

    /// convert from real terrain coordinates (in metres) to a grass grid cell index
    void toGrassGrid(float x, float y, int &i, int &j);

    /// convert from a grass grid cell index to real terrain coordinates
    void toTerrain(int i, int j, float &x, float &y);

    /**
     * @brief suitability   Calculate a suitability score based on a terrain condition range and condition value
     * @param inval         terrain condition value
     * @param absmin        lowest allowable survival value
     * @param innermin      start of ideal range
     * @param innermax      end of ideal range
     * @param absmax        highest allowable survival value
     * @return              suitability score in [0,1]
     */
    float suitability(float inval, float absmin, float innermin, float innermax, float absmax);

    /**
     * @brief smooth    Apply an averaging smoothing kernel to the grass heights
     * @param filterwidth   extent of the kernel in a single direction. Total extent is filterwidth*2+1
     * @param passes        number of smoothing iterations
     * @param noblank       if true, avoid smoothing into areas that are empty of grass
     */
    void smooth(int filterwidth, int passes, bool noblank);

    void smooth_general(int filterwidth, int passes, bool noblank, MapFloat &srcdest);
public:

    GrassSim(){ grasshght.initMap(); }

    GrassSim(Terrain *ter);

    ~GrassSim(){ grasshght.delMap(); }

    /**
     * @brief matchDim  Set all maps to match the underlying terrain
     * @param ter   Underlying terrain
     * @param scale size of terrain in meters
     * @param mult  factor by which to multiply the underlying terrain in grid size
     */
    void matchDim(Terrain *ter, float scale, int mult);

    /**
     * @brief burnInPlant Reduce grass height in the shade of a plant
     * @param x terrain x-coordinate of plant
     * @param y terrain y-coordinate of plant
     * @param r radius of plant in terrain coordinates
     */
    void burnInPlant(float x, float y, float r, float alpha);

    /**
     * @brief grow Simulate grass growth using information from the supplied ecosystem and terrain conditions
     * @param ter   Underlying terrain
     * @param ecs   Ecosystem with influencing plants
     * @param painted   Type map with user painted regions, such as rock, sand and water
     */
    void grow(Terrain * ter, std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float scale);

    /**
     * @brief setConditions    set all terrain conditions
     * @param wetfile          map containing moisture values
     * @param sunfile          map containing illumination values
     * @param tempfile         map containing temperature values
     */
    void setConditions(MapFloat * wetfile, MapFloat * sunfile, MapFloat *landsun_file, MapFloat * tempfile);

    /**
     * @brief write Write to a grascale PNG image
     * @param filename  name of the file to write to
     * @return      true if the write succeeded, false otherwise
     */
    bool write(std::string filename);

    MapFloat * get_data()
    {
        return &grasshght;
    }
    MapFloat * get_litterfall_data()
    {
        return &litterfall_density;
    }
    void set_viability_params(const std::map<std::string, data_importer::grass_viability> &viability_params);
    bool write_litterfall(std::string filename);
    void burnGrass(Terrain *ter, const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float scale);
};

#endif
