/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp (konrad.p.kapp@gmail.com)
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

// TypeMap class for passing around terrian type information

#ifndef _TYPEMAP
#define _TYPEMAP

#include "glheaders.h"
#include <vector>
#include <memory>
#include <common/map.h>
#include <common/debug_string.h>
#include <common/region.h>

#include <iostream>

#include "grass.h"
#include "vecpnt.h"

namespace data_importer
{
    struct common_data;
}

enum class TypeMapType
{
    EMPTY,          //< default colour for terrain
    PAINT,          //< to display brush painting
    CATEGORY,       //< to show terrain attribute categories
    SLOPE,          //< to capture slope information
    WATER,          //< to show water areas
    SUNLIGHT,       //< to show illumination
    TEMPERATURE,    //< to show heat levels
    CHM,            //< canopy height model for tree heights
    CDM,            //< canopy density model for ground visibility
    SUITABILITY,    //< shows discrepancy between terrain and user painted distributions
    GRASS,
    ROCKS,
    PRETTY_PAINTED,
    PRETTY,
    SPECIES,
    CLUSTER,
    CLUSTERDENSITY,
    TMTEND
};
const std::array<TypeMapType, 17> all_typemaps = {TypeMapType::EMPTY, TypeMapType::PAINT, TypeMapType::CATEGORY, TypeMapType::SLOPE, TypeMapType::WATER, TypeMapType::SUNLIGHT, TypeMapType::TEMPERATURE, TypeMapType::CHM, TypeMapType::CDM, TypeMapType::SUITABILITY, TypeMapType::GRASS, TypeMapType::ROCKS, TypeMapType::PRETTY_PAINTED, TypeMapType::PRETTY, TypeMapType::SPECIES, TypeMapType::CLUSTER, TypeMapType::CLUSTERDENSITY}; // to allow iteration over the typemaps

class MapFloat;

class TypeMap
{
private:
    MemMap<int> * tmap;             ///< a map corresponding to the terrain storing integer types
    std::vector<GLfloat *> colmap;  ///< a 32-element lookup table for converting type indices to colours
    Region dirtyreg;                ///< bounding box in terrain grid integer coordinates (e.g, x=[0-width), y=[0-hieght))
    TypeMapType usage;              ///< indicates map purpose
    int numSamples;                 ///< number of active entries in lookup table

    /// Set up the colour table with colours appropriate to categories
    void initCategoryColTable();

    /// Set up the colour table with natural USGS inspired map colours
    void initNaturalColTable();

    /// Set up the colour table with colours appropriate to the initial ecosystem pallete of operations
    void initPaletteColTable();

    // Set up colour table for species painting
    void initSpeciesColTable(std::string dbname);

    /**
     * @brief initPerceptualColTable Set up a colour table sampled from a perceptually uniform colour map stored in a CSV file
     * @param colmapfile        file on disk containing the colour map
     * @param samples           number of samples taken from the colour map
     * @param truncend          proportion of the colourmap to select, truncating from the upper end
     */
    void initPerceptualColTable(std::string colmapfile, int samples, float truncend = 1.0f);

    /// clip a region to the bounds of the map
    void clipRegion(Region &reg);

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & tmap;
        ar & colmap;
        ar & dirtyreg;
        ar & usage;
    }

public:

    TypeMap(){ usage = TypeMapType::EMPTY; }

    TypeMap(TypeMapType purpose);

    /**
     * Create type map that matches the terrain dimensions
     * @param w         map width
     * @param h         map height
     * @param purpose   map purpose, to represent different kinds of layers
     */
    TypeMap(int w, int h, TypeMapType purpose);

    virtual ~TypeMap();

    template <typename T>
    int convert(T * map, TypeMapType purpose, float range)
    {
        int species_count = 0;
        int tp, maxtp = 0;
        int width, height;
        float val, maxval = 0.0f;

        map->getDim(width, height);
        matchDim(width, height);
        // convert to internal type map format
        int mincm, maxcm;
        mincm = 100; maxcm = -1;

        for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            tp = 0;
            switch(purpose)
            {
            case TypeMapType::EMPTY: // do nothing
                break;
            case TypeMapType::PAINT: // do nothing
                break;
            case TypeMapType::CATEGORY: // do nothing, since categories are integers not floats
                break;
            case TypeMapType::SLOPE:
                val = map->get(x, y);
                if(val > maxval)
                maxval = val;

                // discretise into ranges of illumination values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range) val = range;
                tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                break;
            case TypeMapType::WATER:
                val = map->get(x, y);
                if(val > maxval)
                maxval = val;

                // discretise into ranges of water values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range) val = range;
                tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                break;
            case TypeMapType::SUNLIGHT:
                 val = map->get(x, y);
                if(val > maxval)
                maxval = val;

                // discretise into ranges of illumination values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range) val = range;
                tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                break;
            case TypeMapType::TEMPERATURE:
                val = map->get(x, y);
                if(val > maxval)
                maxval = val;

                // discretise into ranges of temperature values
                // clamp values to range, temperature is bidrectional
                if(val < -range) val = -range;
                if(val > range) val = range;
                tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-2)) + 1;

                break;
            case TypeMapType::CHM:
                 val = map->get(y, x);
                 if(val > maxval)
                maxval = val;

                // discretise into ranges of tree height values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range)
                {
                val = range;
                //std::cout << "clamping value to range (upper): " << range << std::endl;
                }
                //tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                tp = (int) (val / (range+pluszero) * (400)) + 1;		// I am assuming we are not categorising this? Multiplying by 400 here will bring the heights back to their proper values in feet
                if(tp < mincm)
                mincm = tp;
                if(tp > maxcm)
                maxcm = tp;
                break;
            case TypeMapType::CDM:
                val = map->get(y, x);
                if(val > maxval)
                maxval = val;

                // discretise into ranges of tree density values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range) val = range;
                tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                if(tp < mincm)
                mincm = tp;
                if(tp > maxcm)
                maxcm = tp;
                break;
            case TypeMapType::SUITABILITY:
                val = map->get(x, y);
                if(val > maxval)
                maxval = val;
                // discretise into ranges of illumination values
                // clamp values to range
                if(val < 0.0f) val = 0.0f;
                if(val > range) val = range;
                tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                break;

            case TypeMapType::GRASS:
                tp = (int)(map->get(x, y) / MAXGRASSHGHT * 255.0f);
                break;
            case TypeMapType::PRETTY_PAINTED:
                tp = (int)(map->get(x, y));
                break;
            case TypeMapType::PRETTY:
                tp = (int)(map->get(x, y));
                break;
            case TypeMapType::SPECIES:
                tp = (int)(map->get(y, x)) + 1;		// XXX: why do I need to swap x, y here...?
                if (tp >= 0)
                    species_count++;
                break;
            default:
                break;
            }
            (* tmap)[y][x] = tp;

            if(tp > maxtp)
                maxtp = tp;
        }
        if(purpose == TypeMapType::CDM)
        {
            std::cerr << "Minimum colour value = " << mincm << endl;
            std::cerr << "Maxiumum colour value = " << maxcm << endl;
        }
        if (purpose == TypeMapType::SPECIES)
        {
            std::cerr << "Proportion taken by species assignment: " << species_count / ((float)width * height) << std::endl;
        }
        return maxtp;
    }

    /// getters for width and height
    int width(){ return tmap->width(); }
    int height(){ return tmap->height(); }

    /// fill map with a certain colour
    void fill(int val){ tmap->fill(val); }

    /// Match type map dimensions to @a w (width) and @a h (height)
    void matchDim(int w, int h);
    
    /// clear typemap to unconstrained
    void clear();

    /// getter for underlying map
    MemMap<int> * getMap(void){ return tmap; }
    
    /// getter for individual value
    int get(int x, int y){ return (* tmap)[y][x]; }
    void set(int x, int y, int val){ (* tmap)[y][x] = val; }

    void replace_value(int orig, int newval);
    
    /// replace underlying map
    void replaceMap(MemMap<int> * newmap);

    /// load from file, return number of clusters
    int load(const uts::string &filename, TypeMapType purpose);
    int load(const QImage &img, TypeMapType purpose);

    /// load from category data from PNG file, return number of clusters
    bool loadCategoryImage(const uts::string &filename);

    /// convert a floating point map into a discretized type map
    int convert(MapFloat * map, TypeMapType purpose, float range);

    /// save a mask file version of the type map
    void save(const uts::string &filename);

    /**
     * @brief saveToPainrImage   Save paint map out as a greyscale image
     * @param filename      Name of file to save to
     */
    void saveToPaintImage(const uts::string &filename);

    /**
     * @brief saveToImage   Save map out as a binary mask
     * @param filename      Name of file to save to
     * @param maskval       Map value to select as white of mask
     */
    void saveToBinaryImage(const uts::string &filename, int maskval);

    /**
     * @brief saveToImage   Save map out as a greyscale image
     * @param filename      Name of file to save to
     * @param maxrange      upper limit of values (mapped to 1.0 in image)
     */
    void saveToGreyscaleImage(const uts::string &filename, float maxrange, bool row_major = false);

    /// getter for colour table
    std::vector<GLfloat *> * getColourTable(void) { return &colmap; }

    /// getter for sampling range
    int getTopSample(){ return numSamples-1; }

    /// getter for update region
    Region getRegion(void) { return dirtyreg; }

    TypeMapType getPurpose() { return usage; }

    /// setter for update region
    void setRegion(const Region& toupdate)
    {
        dirtyreg = toupdate;
        clipRegion(dirtyreg);
    }

    /// return region that covers the entire type map
    Region coverRegion()
    {
        return Region(0,0,width(),height());
    }

    /// setter for purpose
    void setPurpose(TypeMapType purpose);

    /// Reset the indicated type to zero everywhere it appears in the map
    void resetType(int ind);

    /**
     * Index to colour translation. First element is the erase colour
     * @param ind   index for a type (potentially) stored in the type map
     * @retval      4-element RGBA colour associated with @a ind
     */
    GLfloat * getColour(int ind)
    {
        if(ind >= 0 && ind < 32) // limit on number of available colours
            return colmap[ind];
        else
            return NULL;
    }

    /**
     * Set the colour associated with a particular index
     * @param ind   index for the type that is to be given a new colour association
     * @param col   4-element RBGA colour to associate with @a ind
     */
    void setColour(int ind, GLfloat * col);

    /**
     * @brief bandCHMMap    adjust the texture display of the canopy height model to a band of tree heights
     * @param chm   Canopy Height Model map
     * @param mint  minimum tree height (below this black or transparent)
     * @param maxt  maximum tree height (above this red)
     */
    void bandCHMMap(MapFloat * chm, float mint, float maxt);

    /**
     * @brief bandCHMMapEric    adjust the texture display of the canopy height model to a binary display 
     * @param chm   Canopy Height Model map
     * @param mint  minimum tree height (below this black or transparent)
     * @param maxt  maximum tree height (above this red)
     */
    void bandCHMMapEric(MapFloat * chm, float mint, float maxt);

    /**
     * @brief setWater  Set areas of the map to a special water cluster based on moisture conditions
     * @param wet       Map of moisture conditions
     * @param wetthresh Moisture value above which pixel is considered as open water
     */
    void setWater(MapFloat * wet, float wetthresh);

private:

    data_importer::common_data *cdata_ptr = nullptr;
};

#endif
