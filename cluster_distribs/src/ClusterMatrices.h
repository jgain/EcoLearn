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


#ifndef CLUSTERMATRICES_H
#define CLUSTERMATRICES_H

#include <common/basic_types.h>
#include <data_importer/data_importer.h>
#include "HistogramMatrix.h"
#include "ClusterMaps.h"
#include "ClusterData.h"
#include "data_importer/AbioticMapper.h"
#include "EcoSynth/kmeans/src/kmeans.h"
#include <list>
#include <memory>
#include <unordered_map>

#define RW_SELECTION_DEBUG

class QLabel;

enum abiotic_factor
{
    MOISTURE,
    SUN,
    TEMP,
    SLOPE
};

struct abiotic_ranges
{
    std::vector<float> moisture;
    std::vector<float> sun;
    std::vector<float> slope;
    std::vector<float> temp;

    bool operator == (const abiotic_ranges &other)
    {
        bool sizes_equal = other.moisture.size() == moisture.size() &&
                other.sun.size() == sun.size() &&
                other.slope.size() == slope.size() &&
                other.temp.size() == temp.size();

        if (sizes_equal)
        {
            for (int i = 0; i < moisture.size(); i++)
                if (fabs(moisture.at(i) - other.moisture.at(i)) > 1e-4) return false;
            for (int i = 0; i < sun.size(); i++)
                if (fabs(sun.at(i) - other.sun.at(i)) > 1e-4) return false;
            for (int i = 0; i < temp.size(); i++)
                if (fabs(temp.at(i) - other.temp.at(i)) > 1e-4) return false;
            for (int i = 0; i < slope.size(); i++)
                if (fabs(slope.at(i) - other.slope.at(i)) > 1e-4) return false;
            return true;
        }
        else
            return false;
    }

    bool operator != (const abiotic_ranges &other)
    {
        return !(*this == other);
    }
};

enum class validcheck
{
    UNDERGROWTH,
    CANOPY,
    BOTH,
    NONE
};



/*
 * This class represents a model of how a simulation distributed plants across a landscape.
 * It is based on segmenting the landscape based on abiotic conditions and subbiomes (a measure
 * of nearby canopy trees' species), and recording undergrowth plant statistics for each segment,
 * such as number of plants per square meter/yard, percentage of overall plants that each species etc.
 * comprises, etc.
 */

class ClusterMatrices
{
public:
    ClusterMatrices(AllClusterInfo &clusterinfo, const data_importer::common_data &cdata);
    ClusterMatrices(AllClusterInfo &&clusterinfo, const data_importer::common_data &cdata);
    ClusterMatrices(std::vector<std::string> clusterfilenames, const data_importer::common_data &cdata);
    ClusterMatrices(const data_importer::common_data &cdata);		// creates an empty ClusterMatrices class, except for cdata member

private:

    enum class HistAction
    {
        ADD,
        REMOVE
    };

public:
    enum class layerspec
    {
        UNDERGROWTH,
        CANOPY,
        ALL
    };

public:

    /*
     * Write the clusters and their data contained in this model to the file at 'out_filename'
     */
    void write_clusters(std::string out_filename);

    /*
     * Write the distributions for undergrowth plant sizes 'distribs' to open filestream 'ofs'.
     * Used in the 'write_clusters' function
     */
    static void write_sizedistribs(std::ofstream &ofs, const std::map<int, HistogramDistrib> &distribs);

    /*
     * If more than one clusterfile has been imported (i.e. a set of related clusterfiles), then for each cluster, sample a random one from
     * each of these files
     */
    static void select_random_clusters(const AllClusterInfo &info, std::map<int, HistogramMatrix> &distribs, std::map<int, std::map<int, HistogramDistrib> > &sizemtxes, std::unordered_map<int, std::unordered_map<int, float> > &specratios);

    /*
     * Write species proportions for cluster with index 'clusterid' to open filestream 'ofs'
     * Used in 'write_clusters' function.
     */
    void write_species_props(ofstream &ofs, int clusterid);

    /*
     * Set common data on species, such as responses to abiotic conditions, rainfall, etc, imported from database
     */
    void set_cdata(const data_importer::common_data &cdata_arg);

    /*
     * Set kmeans cluster parameters, i.e. cluster means and minmax ranges for scaling abiotic conditions appropriately
     */
    void set_cluster_params(const std::vector<std::array<float, kMeans::ndim> > &kclusters, const std::array<std::pair<float, float>, kMeans::ndim> &minmax_pairs);
    void set_cluster_params(const ClusterAssign &classign);

    /*
     * Fill empty clusters/regions with closest matching one based on hamming distance of bitstring
     * derived from which subbiomes are present at each point
     */
    void fill_empty(int nmeans);

    /*
     * Normalize all distributions with normalize method 'nmeth'
     */
    void normalizeAll(normalize_method nmeth);

    /*
     * Get undergrowth plant density (per square yard) for region with index 'idx'
     */
    float
    get_region_density(int idx);
    float
    get_region_density(int idx) const;

    /*
     * Get size distributions for region index 'idx'
     */
    std::map<int, HistogramDistrib> &
    get_sizematrix(int idx);

    /*
     * Get spatial distributions for region index 'idx'
     */
    HistogramMatrix &
    get_locmatrix(int idx);

    /*
     * Get total number of regions
     */
    int
    get_nclusters() const;

    /*
     * get structures that map from undergrowth and canopy id to their indices
     */
    std::vector<int> get_under_to_idx();
    std::vector<int> get_canopy_to_idx();

    /*
     * get all canopy and undergrowth plant ids
     */
    const std::vector<int> &get_canopy_ids() const;
    const std::vector<int> &get_plant_ids() const;

    /*
     * Check if this model's spatial distributions are equal to that of model 'other'
     */
    bool is_equal(const ClusterMatrices &other) const;

    /*
     * Difference between this model and the other one for only undergrowth-canopy spatial distributions
     */
    float canopymtx_diff(const ClusterMatrices *other) const;

    void show_region_densities();

    /*
     * Update spatial distribution matrix based on reftree's distance from 'othertrees' ('reftree' can only
     * be an undergrowth plant, while othertrees could contain undergrowth and canopy trees).
     */
    void updateHistogramLocMatrix(const basic_tree &reftree, const std::vector<basic_tree> &othertrees, const ClusterMaps &clmaps, ClusterMatrices *benchmark);

    /*
     * Check matrices and print to console possible issues, such as distributions that do not sum to 1 or active
     * distributions that do not have any values for its bins
     */
    void check_matrices();

    /*
     * Difference spatial distributions of this model with those of 'other' model. Useful for
     * checking if synthesis is actually working
     * Second function, containing parameter 'rowcol_check', does the differencing for distributions specified only in
     * 'rowcol_check'. The format is std::unordered_map< region idx, std::set< std::pair< species id1, species id2> > >
     */
    float diff_other(ClusterMatrices *other);
    float diff_other(ClusterMatrices *other, const std::unordered_map<int, std::set<std::pair<int, int> > > &rowcol_check);

    /*
     * Update the size matrix based on the size of plant 'reftree'
     */
    void updateHistogramSizeMatrix(const basic_tree &reftree, const ClusterMaps &clmaps);

    /*
     * Get metadata for undergrowth and canopy trees
     */
    HistogramDistrib::Metadata get_undergrowth_meta() const;
    HistogramDistrib::Metadata get_canopy_meta() const;

    /*
     * Get data on region with index 'idx'
     */
    ClusterData &get_cluster(int idx);

    /*
     * Get data on region with index 'idx', without checking first if it exists. (const also)
     */
    const ClusterData &get_cluster_nocheck(int idx) const;

    /*
     * Check if region with index 'idx' exists in this model
     */
    bool clusterdata_exists(int idx) const;

    /*
     * Get spatial distribution matrix for region with index 'idx', without checking (const)
     */
    const HistogramMatrix &get_locmatrix_nocheck(int idx) const;

    /*
     * Set plant and canopy ids
     */
    void set_plant_ids(std::set<int> plant_ids);
    void set_canopy_ids(std::set<int> canopy_ids);

    /*
     * Get common data for species assigned to this object
     */
    const data_importer::common_data &get_cdata() const;

    /*
     * for each species size distribution, finds the highest nonzero bin, and sets a maximum height from this
     */
    void set_allspecies_maxheights();


    void check_maxdists();
    void unnormalizeAll(normalize_method nmeth);
private:
private:
    // kept in ClusterMatrices class
    std::map<int, ClusterData> clusters_data;
    data_importer::common_data cdata;
    ClusterAssign classign;
    std::map<int, float> allspecies_maxheights;		// used for sampling a height if a distribution for a height doesnt exist (move to sampler class? Yes, has nothing to do with distribs)
    std::vector<int> plant_ids;
    std::vector<int> canopy_ids;
    float maxdist = 5.0f;
    std::shared_ptr<QLabel> histcomp;		// perhaps refactor this to the interface?

};

#endif // CLUSTERMATRICES_H
