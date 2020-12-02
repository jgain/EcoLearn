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


#ifndef CLUSTERDISTRIBDERIVATOR_H
#define CLUSTERDISTRIBDERIVATOR_H

#include "ClusterMatrices.h"
#include "ClusterMaps.h"
#include "PlantSpatialHashmap.h"

/*
 * This class does the derivation of distributions, as well as kmeans clustering (via delegation)
 * to return a ClusterMatrices object, which contains the model for the undergrowth data.
 * It can derive models from multiple input datasets, although it will do kmeans clustering over all
 * landscapes in one go, so that the cluster means and minmax ranges will be consistent over all
 * resulting clusterfiles that are written to disk.
 */

// XXX TODO: still need to do kmeans clustering, or account for it in some way for this class
class ClusterDistribDerivator
{
private:
    struct Dataset
    {
        ClusterMaps *clmaps;		// XXX: use shared_ptr for these?
        PlantSpatialHashmap plantmap;
    };
public:
    ClusterDistribDerivator(std::vector<std::string> datasets_paths, const data_importer::common_data &cdata);
    ClusterDistribDerivator(ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth);
    ClusterDistribDerivator();

    /*
     * Add a new dataset, given a landscape with a clustermeans model and undergrowth plants
     */
    void add_dataset(ClusterMaps &clmaps, const PlantSpatialHashmap &plantmap);
    void add_dataset(ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth);

    /*
     * Remove datasets, by index (first) or by giving source data for the cluster (second, overload)
     */
    bool remove_dataset(int idx);
    bool remove_dataset(const ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth);

    /*
     * If there are cluster-subbiome combinations that are empty, without any data (as there probably will be),
     * then copy another distribution from the closest matching cluster-subbiome combination
     */
    void fill_empty(ClusterMatrices &clm);

    /*
     * Set benchmark model. This is useful if distributions that are inactive in the model, will be ignored in the derivation,
     * as is desired with synthesis models
     */
    void set_benchmark(ClusterMatrices *benchmark);

    /*
     * Remove benchmark model, and return it
     */
    ClusterMatrices *remove_benchmark();

    /*
     * Derive undergrowth models for all datasets, and return them as separate models
     */
    std::list<ClusterMatrices> deriveHistogramMatricesSeparate(const data_importer::common_data &cdata);

    /*
     * Derive an undergrowth model for a single dataset, and return the model
     */
    ClusterMatrices deriveHistogramMatrices(const Dataset &ds, const data_importer::common_data &cdata, ClusterMatrices *benchmark);
    ClusterMatrices deriveHistogramMatrices(ClusterMaps &clmaps,
                                            std::vector<basic_tree> undergrowth,
                                            std::vector<basic_tree> canopy,
                                            data_importer::common_data &cdata,
                                            ClusterMatrices *benchmark);

    /*
     * ensures that all datasets have the same kmeans model
     * Note that this function does kmeans over all datasets' abiotic maps, then assigns this model to each dataset.
     * Due to the kmeans clustering, it may be an expensive function
     */
    void do_kmeans(int nmeans, int niters, const data_importer::common_data &cdata);

    /*
     * Compute density for each region (number of undergrowth plants / region size) then return in a std::map structure
     * that maps from region id to density
     */
    std::map<int, float> compute_region_densities(const Dataset &ds, const std::vector<basic_tree> &undergrowth, bool allow_zero_region);

    std::list<ClusterMaps> allocd_maps;		// if we allocate our own clustermaps, we store them here. Used only for initialization and storage, in that case.
                                            // ClusterMap objects will still be accessed via the Dataset objects' clmaps pointers


    std::vector<Dataset> datasets;
};

#endif   // CLUSTERDISTRIBDERIVATOR_H
