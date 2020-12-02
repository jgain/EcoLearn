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


#ifndef HISTOGRAM_MATRIX_H
#define HISTOGRAM_MATRIX_H

#include "HistogramDistrib.h"
#include <fstream>
#include <unordered_map>
#include <list>
#include <set>

class HistogramMatrix
{
public:

    static HistogramDistrib::Metadata global_canopy_metadata;
    static HistogramDistrib::Metadata global_under_metadata;

public:
    HistogramMatrix(const std::vector<int> &plant_ids, const std::vector<int> &canopy_ids,
                    HistogramDistrib::Metadata under_metadata, HistogramDistrib::Metadata canopy_metadata);

    HistogramMatrix(const std::vector<int> &plant_ids,
                    const std::vector<int> &canopy_ids,
                    const std::vector<std::vector<bool> > &active_distribs,
                    const std::vector<std::vector<HistogramDistrib> > &matrix);

    /*
     * Add distances between plant 'ref' and the plants contained in 'others', to the relevant
     * distributions (histograms).
     */
    void addPlantsUndergrowth(const basic_tree &ref, const std::vector<basic_tree> &others,
                              float width, float height,
                              const HistogramMatrix *benchmtx);

    /*
     * Add distances between plant 'ref' and the canopy trees contained in 'others',
     * to the relevant distributions (histograms).
     */
    void addPlantsCanopy(const basic_tree &ref, const std::vector<basic_tree> &others,
                         float width, float height,
                         const HistogramMatrix *benchmatrix);

    /*
     * normalize all distributions in matrix based on normalize method 'nmeth'
     */
    void normalizeAll(normalize_method nmeth);

    /*
     * Unnormalize all distributions in matrix. 'nmeth' parameter has no use actually, except if
     * normalize_method::NONE is passed, in which case nothing happens.
     * TODO: remove this parameter and arguments for all instances of this function
     */
    void unnormalizeAll(normalize_method nmeth);

    /*
     * write this matrix to the file opened by ofstream 'out'.
     */
    void write_matrix(std::ofstream &out) const;

    /*
     * write this matrix to the filename 'filename'
     */
    void write_matrix(std::string filename) const;

    /*
     * Get the distribution for interactions between undergrowth species spec1 and spec2
     */
    HistogramDistrib &get_distrib_undergrowth(int spec1, int spec2);

    /*
     * Get the distribution for interactions between canopy species 'canopyspec' and undergrowth species 'underspec'
     */
    HistogramDistrib &get_distrib_canopy(int canopyspec, int underspec);

    /*
     * Get the distribution for interactions between species at row 'row' and column 'col' in this matrix
     */
    HistogramDistrib &get_distrib_rowcol(int row, int col);

    /*
     * Flattened version of function above
     */
    const HistogramDistrib &get_distrib(int idx) const;

    /*
     * Get ids of all undergrowth plants
     */
    const std::vector<int> &get_plant_ids();

    /*
     * Get ids of all canopy trees
     */
    const std::vector<int> &get_canopy_ids();

    /*
     * Get the total number of bins for all histograms in this matrix
     */
    int get_ntotal_bins() const;

    /*
     * Get the number of real bins (i.e. bins that don't encode overlap between plants/trees) for
     * all histograms in this matrix
     */
    int get_nreal_bins() const;

    /*
     * Get the number of reserved bins (i.e. bins that encode overlap between plants/trees) for
     * all histograms in this matrix
     */
    int get_nreserved_bins() const;

    /*
     * Get bin value for a distance sep and raw_distance between undergrowth species spec1 and spec2
     */
    float get_bin_value_undergrowth(float refrad, float orad, int spec1, int spec2, float sep, float raw_distance);

    /*
     * Get bin value for a distance sep and raw_distance between undergrowth species refspec and
     * canopy species canopyspec
     */
    float get_bin_value_canopy(float refrad, float canopyrad, int refspec, int canopyspec, float sep, float raw_distance);

    /*
     * Check if distribution between undergrowth species spec1 and spec2 is active
     */
    bool is_active_undergrowth(int spec1, int spec2) const;

    /*
     * Check if distribution between canopy species canopyspec and undergrowth species underspec is active
     */
    bool is_active_canopy(int canopyspec, int underspec) const;

    /*
     * Convert from canopy species id, to the index in the matrix
     */
    int get_canopy_idx(int id);

    /*
     * Convert from undergrowth species id, to the index in the matrix
     */
    int get_undergrowth_idx(int id);

    /*
     * Get a vector that provides a mapping from undergrowth id to its index in the matrix
     */
    std::vector<int> get_underid_to_idx() const;

    /*
     * Get a vector that provides a mapping from canopy id to its index in the matrix
     */
    std::vector<int> get_canopyid_to_idx() const;

    /*
     * Check if this matrix has at least one active distribution
     */
    bool has_active_distribs() const;

    /*
     * Get distribution matrix
     */
    const std::vector< std::vector< HistogramDistrib > > &get_distribs();

    /*
     * Read and return a distribution matrix from std::ifstream ifs
     */
    static HistogramMatrix read_matrix(std::ifstream &ifs);

    /*
     * Get flat index of distribution for canopy species canopyspec and undergrowth species underspec
     */
    int get_distrib_canopy_flatidx(int canopyspec, int underspec) const;

    /*
     * Get flat index of distribution for undergrowth species spec1 and spec2
     */
    int get_distrib_under_flatidx(int spec1, int spec2);

    /*
     * Get bin width for all distributions
     * TODO: have a member variable for this, instead of getting it from the first distribution
     */
    int get_binwidth() const;


    /*
     * Remove the effect of a single distance between refplnt and oplnt from the relevant distribution
     * Optionally, benchmtx is a pointer to the benchmark matrix for this distribution (in the same cluster).
     * Also optional is the benefit pointer, which points to memory that stores the benefit of removing this distance
     * from the histogram, with the purpose of making this distribution more similar to the benchmark matrix.
     */
    bool remove_from_bin_undergrowth(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx, float *benefit);

    /*
     * Add the effect of a single distance between refplnt and oplnt from the relevant distribution
     * Same as above for 'benchmtx' and 'benefit' pointers
     */
    bool add_to_bin_undergrowth(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx, float *benefit, std::vector<common_types::decimal> *bins);

    /*
     * Same as HistogramMatrix::remove_from_bin_undergrowth, except distance between undergrowth plant 'uplant' and canopy tree 'cplant' is used
     */
    bool remove_from_bin_canopy(const basic_tree &uplant, const basic_tree &cplant, float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx = nullptr, float *benefit = nullptr);

    /*
     * Same as HistogramMatrix::add_to_bin_undergrowth, except distance between undergrowth plant 'uplant' and canopy tree 'cplant' is used
     */
    bool add_to_bin_canopy(const basic_tree &uplant, const basic_tree &cplant, float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx = nullptr, float *benefit = nullptr, std::vector<common_types::decimal> *bins = nullptr);

    /*
     * Compute L1 difference between all distributions' histograms in this matrix and 'other' matrix
     */
    float diff(const HistogramMatrix &other, float showdiff) const;

    /*
     * Compute L1 difference between distributions' histograms in rowcol_check only
     */
    float diff(const HistogramMatrix &other, const std::set<std::pair<int, int> > &rowcol_check) const;

    /*
     * Reset all histograms (assign zero to all bins)
     */
    void zero_out_histograms();

    /*
     * get all the row, col indices of distributions that are empty (all bins equal to zero), but active
     */
    std::vector<std::pair<int, int> > get_zero_active() const;

    /*
     * get all the row, col indices of distributions where sum of bins do not equal one
     * FIXME: need to exclude zero bins
     */
    std::vector<std::pair<int, int> > get_noteqone_active() const;

    /*
     * Set the distribution between undergrowth plants species spec1 and spec2 as active
     */
    void set_active_undergrowth(int spec1, int spec2);

    /*
     * Set the distribution between canopy tree species canopyspec and undergrowth species underspec as active
     */
    void set_active_canopy(int canopyspec, int underspec);

    /*
     * Assign the lower priority species between spec1 and spec2 to 'low', and higher priority to 'high'
     */
    void set_low_high(int spec1, int spec2, int &low, int &high) const;

    /*
     * Get difference between distributions between canopy and undergrowth only
     */
    float canopymtx_diff(const HistogramMatrix &other, int &nactive) const;

    /*
     * Check if undergrowth plant id 'left' is higher or equal priority to undergrowth plant id 'right'
     * If a plant is lower priority, then we calculate distances from it to the higher or equal priority plant
     */
    bool isHigherOrEqualPriority(int left, int right) const;

    /*
     * Sum all histogram bins of the entire matrix
     */
    float sum() const;

    /*
     * Run some consistency checks on the object. If an inconsistency is found an 'invalid_argument' exception is thrown
     */
    void validate_distribs();

    /*
     * Get maximum distance for which we analyse undergrowth-undergrowth distances
     */
    float get_under_maxdist();

    /*
     * Get maximum distance for which we analyse undergrowth-canopy distances
     */
    float get_canopy_maxdist();

    /*
     * Check if this object is equal to another.
     * Currently it just checks if all distributions are the same in the two matrices.
     */
    bool is_equal(const HistogramMatrix &other) const;

    /*
     * Get the actual row, column value in the matrix for relationship between canopy species canopyspec and
     * undergrowth species underspec
     */
    std::pair<int, int> get_cspecies_rowcol(int canopyspec, int underspec) const;

    /*
     * Get the actual row, column value in the matrix for relationship between undergrowth species spec1 and spec2
     */
    std::pair<int, int> get_uspecies_rowcol(int spec1, int spec2) const;
private:

    std::vector< std::vector< HistogramDistrib > > distribs;
    std::vector< std::vector< bool > > active_distribs;
    std::vector<int> plant_ids;
    std::unordered_map<int, int> plnt_id_to_idx;
    std::vector<int> canopy_ids;
    std::map<int, int> canopy_id_to_idx;
    int nreserved_bins;
    int nreal_bins;
    float max_dist;

    HistogramDistrib::Metadata canopy_metadata;		// metadata for canopy-undergrowth distributions
    HistogramDistrib::Metadata under_metadata;		// metadata for undergrowth-undergrowth distributions

    std::map<int, int> species_counts_under;


};

#endif
