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


#ifndef HISTOGRAMDISTRIB_H
#define HISTOGRAMDISTRIB_H

#include <vector>
#include "dice.h"
#include "common.h"

class QPainter;

enum normalize_method
{
    NONE,
    BYREF,
    COMPLETE
};

/*
 * Class representing a single distribution for either distances between plants/canopy trees, or plant sizes
 */

class HistogramDistrib
{
public:
    // ideally, we should have separate HistogramMatrix class definitions for canopy-undergrowth and undergrowth-undergrowth mtxes,
    // because they have different metadata, such as maximum distances and as a result different bin widths.
    // But at this stage we will just distinguish between these two distributions' metadata via this metadata struct
    // (see canopy_metadata and under_metadata members)
    struct Metadata
    {
        float maxdist;
        int nreserved_bins;
        int nreal_bins;
        int ntotal_bins;		// not really necessary, but can be used as a sanity check (nbins + nreserved_bins = nbins_total) and to reduce bugs
        float binwidth;			// ditto

        bool operator == (const Metadata &other)
        {
            return fabs(maxdist - other.maxdist) < 1e-5f
                    && nreserved_bins == other.nreserved_bins
                    && nreal_bins == other.nreal_bins
                    && ntotal_bins == other.ntotal_bins
                    && fabs(binwidth - other.binwidth) < 1e-5f;
        }

        bool operator != (const Metadata &other)
        {
            return !(*this == other);
        }
    };
private:
    common_types::decimal min;                    //< first bin value
    common_types::decimal max;                    //< last bin value
    int numbins;                //< number of non-reserved bins
    int reserved;               //< number of bins reserved for special distributions properties
    int numrnd;                 //< number of elements in distribution acceleration array
    std::vector<common_types::decimal> bins;    //< normalized histogram for distribution
    std::vector<float> rnddistrib; //< for accelerated generation of a distribution that matches the histogram
    Dice roller;                //< uniform random number generator

        common_types::decimal normal_factor;			//< normalization factor >
    int refcount;
    normalize_method nmeth = normalize_method::NONE;

    common_types::decimal binwidth;

    static std::vector<float> sinlookup;


public:

    typedef float value_type;

    HistogramDistrib(){ min = 0.0f; max = 0.0f; numbins = 0; reserved = 0; numrnd = 0; bins.clear(); normal_factor = 1.0f; refcount = 0;}
    HistogramDistrib(int hnumbins, int reservedbins, common_types::decimal min, common_types::decimal max, const std::vector<common_types::decimal>  &bin_data);
    HistogramDistrib(int hnumbins, int nreserved_bins, common_types::decimal min, common_types::decimal max);

    ~HistogramDistrib()
    {
        bins.clear();
        rnddistrib.clear();
    }

    void draw(QPainter *paint, int sx, int sy, int ex, int ey, int r, int g, int b, int a) const;

    const std::vector<common_types::decimal> &getbins() const { return bins; }

    /// getter for total number of bins
    int getnumtotbins() const { return numbins+reserved; }

    Metadata get_metadata();

    /// getter for minimum value
    common_types::decimal getmin() const { return min; }

    /// getter for maximum value
    common_types::decimal getmax() const { return max; }

    /// getter for number of reserved bins
    int getnumreserved() const { return reserved; }
    int getnumbins() const { return numbins; }

    /// getter for value in particular histogram bin, no range checking
    common_types::decimal getbin(int b) const { return bins[b]; }

    common_types::decimal getnormalfactor() const { return normal_factor; }

    common_types::decimal calc_binsum() const;

    common_types::decimal get_binwidth() const;

    const std::vector<float> &get_rnddistrib() const { return rnddistrib; }

    /**
     * Initialise the histogram parameters
     * @param minb  starting bin value (inclusive)
     * @param maxb  ending bin value (inclusive)
     * @param numb  number of histogram bins
     * @param skip  number of special prefix bins
     */
    void init(common_types::decimal minb, common_types::decimal maxb, int numb, int skip);

    /**
     * Map a data value to its corresponding histogram bin
     * @param val data value
     * @return correponsding histogram bin index
     */
    int maptobin(common_types::decimal val) const;
    int maptobin(int val) const;

    /**
     * Map a bin index to the corresponding min data value
     * @param histogram bin index
     * @return correponsding data value
     */
    common_types::decimal maptoval(int bin) const;

    /**
     * Map a bin index to the corresponding max data value
     * @param histogram bin index
     * @return correponsding data value
     */
    common_types::decimal maptomaxval(int bin) const;

    /**
     * Get the probability associated with a particular histogram value
     * @param val   histogram data value
     * @return      associated probability
     */
    //common_types::decimal probability(common_types::decimal val);

    /**
     * Create a histogram given an input distribution with integer values
     * @param data  vector of distribution data that maps to bins in the histogram
     */
    void buildbins(std::vector<int> & data);

    /**
     * Create a histogram given an input distribution with integer values but compensate for annular area
     * @param data  vector of distribution data that maps to bins in the histogram
     * @param areas  vector of annular disk areas for normalization
     */
    void buildannularbins(std::vector<common_types::decimal> & data, std::vector<common_types::decimal> & areas);

    /**
     * Merge a raw histogram with matching parameters into a single normalized histogram
     * @param h  source raw histogram
     */
    void accumulatebins(HistogramDistrib & h);

    /**
     * Create an acellerated lookup structure for mapping random integer values to bins
     * @param rndmax    upper limit on random numbers, lower limit is assumed to be zero
     */
    void buildrnddistrib(int rndmax, int seed = 0);

        /**
         * Set the seed for generating from the histogram distribution
         * @param seed		seed value
         */
        void setRndDistribSeed(int seed);

    /**
     * Create a probability density by normalizing the bin values
     */
    void normalize();

    /**
     * Randomly generate an element according to the histogram distribution
     * Reserved elements have negative values, others are in the range [min, max]
     * @return Random element that obeys the histogram probability distribution
     */
    common_types::decimal rndgen();

    /**
     * @brief diff Calculate the RMS difference between two normalized histograms
     * @param cmp   Histogram to compare against
     * @return      RMS difference
     */
    common_types::decimal diff(const HistogramDistrib &cmp) const;

    /**
     * Draw a histogram as part of a histogram matrix onto an existing canvas
     * @param paint    Qt paint controller
     * @param cx    corner x-offset
     * @param cy    corner y-offset
     * @param dx    half dimension of the canvas in x
     * @param dy    dimension of the canvas in y
     * @param offx  index of histogram in x
     * @param offy  index of histogram in y
     * @param numh  number of rows in histogram matrix
     */
    //void draw(QPainter * paint, int cx, int cy, int dx, int dy, int offx, int offy, int numh);

    /**
     * Print histogram on standard output
     */
    void print();

    /**
     * @brief reset
     * make all bins zero, however, metadata (such as min, max, etc) are untouched.
     */
    void reset();

    /**
     * @brief validitytest
     * @retval true if the histogram distribution is fundamentally sound
     * @retval false otherwise
     */
    bool validitytest();

    /**
     * Run unit tests on this class
     * @retval true if the unit test succeeds,
     * @retval false otherwise
     */
    bool unittest();

    /**
     * function for modification of histogram based on a single plant movement
     * @param prev_ds		distances based on previous position of moved plant
     * @param new_ds		distances based on new position of moved plant
     * @param old_areas		old annular areas for each bin
     * @param new_areas		new annular areas for each bin
     */
    void modify(const std::vector<common_types::decimal> &prev_dists, const std::vector<common_types::decimal> &new_dists, const std::vector<common_types::decimal> &old_areas, const std::vector<common_types::decimal> &new_areas, int repeat = 1);
    void modify(const std::vector<int> &prev_bins, const std::vector<int> &new_bins, const std::vector<common_types::decimal> &prev_areas, const std::vector<common_types::decimal> &new_areas);
    void clean_bins();
    void unnormalize();
    common_types::decimal rndgen_binmax();

    /*
     * Calculate area of a disk, centered at location x, y with radius 'r'. 'width' and 'height' are necessary so that
     * intersections with the edge of the landscape can be accounted for
     */
    common_types::decimal calcDiskArea(float x, float y, common_types::decimal r, float width, float height) const;

    /*
     * Calculate the area of a ring, where the outer radius is 'r2' and the inner radius 'r1'
     */
    common_types::decimal calcRingArea(float x, float y, common_types::decimal r1, common_types::decimal r2, float width, float height) const;

    /*
     * Calculate the area of a ring, where the outer radius is the upper distance of bin 'binnum' and the inner radius
     * the lower distance of 'binnum'
     */
    common_types::decimal calcRingArea(float x, float y, int binnum, float width, float height) const;

    /*
     * Calculate area covered by a plnt with radius 'plnt_radius'
     */
    common_types::decimal calcReservedArea(float x, float y, float plnt_radius, float width, float height) const;


    /*
     * Add the plant distance between 'refplnt' and 'oplnt' to the current histogram
     */
    bool add_plantdist(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, normalize_method nmeth);

    /*
     * Remove the plant distance between 'refplnt' and 'oplnt' to the current histogram
     */
    bool remove_plantdist(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, normalize_method nmeth);

    /*
     * Add the plant distance 'dist' to the current histogram
     */
    bool add_dist(float dist, normalize_method nmeth);

    /*
     * Remove the plant distance 'dist' from the current histogram
     */
    void remove_dist(float dist, normalize_method nmeth);

    /*
     * Normalize the histogram with normalize method 'nmeth'
     */
    void normalize(normalize_method nmeth);

    /*
     * Normalize by dividing by number of plants added to this distribution
     */
    void normalize_by_ref();

    /*
     * Unnormalize by multiplying by number of plants added to this distribution
     */
    void unnormalize_by_ref();

    /*
     * Unnormalize the histogram, assuming histogram was normalized with normalize method nmeth
     */
    void unnormalize(normalize_method nmeth);

    /*
     * Same as maptobin, except this takes the separation distance as an argument, directly
     */
    int maptobin_raw(float refrad, float otherrad, float sep, float diff);

    /*
     * Compute a distance code, which will be the separation distance if greater than zero.
     * If less than zero (indicating overlap), we have different codes for partial overlap, full overlap, etc.
     */
    common_types::decimal distCodeNew(float refx, float refy, float refrad, float otherx, float othery, float otherrad) const;
    common_types::decimal distCodeNew(float refrad, float otherrad, float sep, float diff) const;

    /*
     * Check if bins equal 1 when summed, return true if so, false otherwise
     */
    bool bins_eqone() const;

    /*
     * Check if bins equal 0 when summed, return true if so, false otherwise
     */
    bool bins_eqzero() const;

    /*
     * Multiplier for converting from radians to degrees
     */
    static constexpr float convmult = 360.0f / (2 * M_PI);

    /*
     * Get the number of plants added to this distribution
     */
    int get_refcount() const;

    /*
     * Obtain a summation of all bins in this distribution
     */
    float sumbins() const;

    /*
     * Calculate area by which distance between refplnt and oplnt will be normalized when added to distribution
     */
    void calc_div_area(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, int &binnum, common_types::decimal &divarea);

    /*
     * Check if this distribution is equal to distribution 'other'
     */
    bool is_equal(const HistogramDistrib &other) const;

    /*
     * Set bins of this distribution equal to 'bins'
     */
    void setbins(const std::vector<common_types::decimal> &bins);
protected:
    common_types::decimal calcQuadrantArea(common_types::decimal dv, common_types::decimal dh, common_types::decimal r) const;
    common_types::decimal distCode(float refx, float refy, float refrad, float otherx, float othery, float otherrad) const;
    common_types::decimal distCode(float refrad, float otherrad, float sep, float diff) const;

};

#endif
