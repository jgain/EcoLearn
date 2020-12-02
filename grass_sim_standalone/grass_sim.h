
#ifndef GRASS_SIM_GRASS_H
#define GRASS_SIM_GRASS_H

// grass.h: seperate simulator for grass
// author: James Gain
// date: 17 August 2016

// adapted by K. Kapp July 2019

//#include "eco.h"
#include "MapFloat.h"
//#include "../../viewer/canopy_placement/basic_types.h"
#include "basic_types.h"
#include "data_importer.h"

#include <memory>

#define MAXGRASSHGHT 60.0f    // maximum grass height in cm

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

class EcoSystem;

class GrassSim
{
private:
    MapFloat grasshght;   //< linearized map of grass heights
    MapFloat * moisture, * illumination, * temperature; //< condition maps
    MapFloat litterfall_density;
    float scx, scy;       //< size of an individual grass cell in meters
    float hscx, hscy;     //< half a grass cell in meters
    int cellmult;
    bool conditions_set = false;
    bool params_set = false;
    std::unique_ptr<data_importer::common_data> cdata_ptr;

    /// map terrain condition grid coordinates to grass coordinates
    void convertCoord(int x, int y, int &sx, int &sy, int &ex, int &ey);

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

    std::map<std::string, data_importer::grass_viability> viability_params;

public:

    GrassSim(){ grasshght.initMap(); litterfall_density.initMap(); }

    GrassSim(ValueGridMap<float> &ter, int cellmult);

    ~GrassSim(){ grasshght.delMap(); litterfall_density.delMap(); }

    /**
     * @brief matchDim  Set all maps to match the underlying terrain
     * @param ter   Underlying terrain
     * @param scale size of terrain in meters
     * @param mult  factor by which to multiply the underlying terrain in grid size
     */
    void matchDim(ValueGridMap<float> &ter, int cellmult);

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
    void grow(ValueGridMap<float> &ter, const std::vector<basic_tree *> &plnt_pointers);

    /**
     * @brief setConditions    set all terrain conditions
     * @param wetfile          map containing moisture values
     * @param sunfile          map containing illumination values
     * @param tempfile         map containing temperature values
     */
    void setConditions(MapFloat * wetfile, MapFloat * sunfile, MapFloat * tempfile);

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

    void burnGrass(const std::vector<basic_tree *> &plnt_pointers);

    void set_viability_params(const std::map<std::string, data_importer::grass_viability> &viability_params);

    void set_commondata(std::string cdata_pathname);
    bool write_litterfall(std::string filename);
    void smooth_general(int filterwidth, int passes, bool noblank, MapFloat &srcdest);
};



#endif //GRASS_SIM_GRASS_H
