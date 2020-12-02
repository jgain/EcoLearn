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


#ifndef UNDERGROWTH_SAMPLER_H
#define UNDERGROWTH_SAMPLER_H

#include "ClusterMaps.h"
#include "ClusterMatrices.h"

/*
 * UndergrowthSampler
 * Class that handles undergrowth plant sampling. This randomly samples plants from each canopy
 * tree until each cluster has its required count satisfied.
 *
 * Most class methods here deal with the CPU sampling, as well as setup of necessary resources
 * such as maps, counters, etc., some which also get used for the GPU accelerated version
 * (GPU accelerated sampling is called via UndergrowthSampler::sample_undergrowth_gpu
 */

class UndergrowthSampler
{
public:
    UndergrowthSampler(std::vector<std::string> cluster_filenames,
                       abiotic_maps_package amaps,
                       std::vector<basic_tree> canopytrees,
                       data_importer::common_data cdata);

    /*
     * Sample undergrowth plants based on computed cluster maps, with a required
     * density for each cluster as well as specified proportions of each species.
     * This version uses the CPU only
     */
    std::vector<basic_tree> sample_undergrowth();

    /*
     * Sample undergrowth plants using the GPU also
     */
    std::vector<basic_tree>
    sample_undergrowth_gpu(const std::vector<basic_tree> &canopytrees);

    /*
     * Sample one undergrowth plant from a single tree
     */
    bool sample_from_tree(const basic_tree *ptr,
                          basic_tree &new_underplant,
                          ValueGridMap<bool> *occgrid);

    /*
     * Get the number of plants overall to be sampled for this terrain
     */
    int get_overall_target_count();

    /*
     * Get computed clustermaps from this object
     */
    const ClusterMaps &get_clustermaps() const;

    // XXX: perhaps it will be a better idea to write functions in this class
    //	    that can modify its members? Because this returns non-const member
    // 	    reference of this class

    /*
     * Non-const version of getter function above
     */
    ClusterMaps &get_clustermaps();

    /*
     * Get the model for undergrowth plants, such as the densities and distributions
     * for each cluster.
     */
    const ClusterMatrices &get_model() const;

    /*
     * Update canopy trees' impact on cluster maps, such as shading and required total counts
     * for undergrowth plants
     */
    void
    update_canopytrees(const std::vector<basic_tree> &canopytrees,
                            const ValueGridMap<float> &sunmap);

    /*
     * Update the sunmap used for assigning clusters
     */
    void
    update_sunmap(const ValueGridMap<float> &sunmap);

    /*
     * Set callback function that reports progress for sampling of undergrowth plants,
     * as well as refinement
     */
    void set_progress_callback(std::function< void (int) > callback);
protected:
    ClusterMaps clmaps;
    ClusterMatrices model;

    std::unordered_map<int, int> region_target_counts;
    std::unordered_map<int, int> region_synthcount;
    ValueGridMap<bool> occgrid;
    ValueGridMap<float> sunbackup;

    std::uniform_real_distribution<float> unif;
    std::default_random_engine gen;

    void init_occgrid();
    void init_synthcount();

    std::function< void (int) > progress_callback;
private:
};

#endif
