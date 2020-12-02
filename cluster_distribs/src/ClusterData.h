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


#ifndef CLUSTERDATA_H
#define CLUSTERDATA_H

#include "HistogramMatrix.h"
#include "HistogramDistrib.h"
#include "data_importer/data_importer.h"
//#include "AllClusterInfo.h"

/*
 * This class holds the relevant data for a single cluster, such as the distribution matrix for spatial relationships
 * between species, a size distribution for each species, the cluster density, and so on.
 */

class ClusterData
{
public:

    ClusterData(std::set<int> plant_ids, std::set<int> canopy_ids, HistogramDistrib::Metadata undergrowth_meta, HistogramDistrib::Metadata canopy_meta, const data_importer::common_data &cdata);
    ClusterData(const HistogramMatrix &spatial_data, const std::map<int, HistogramDistrib> &size_data, float density, std::unordered_map<int, float> species_ratios,
                const data_importer::common_data &cdata);

    // getter functions
    float
    get_density() const;

    /*
     * Get the structure that maps from plant id to the species' size distribution
     */
    std::map<int, HistogramDistrib> &
    get_sizematrix();

    /*
     * Same as above, but const
     */
    const std::map<int, HistogramDistrib> &
    get_sizematrix() const;

    /*
     * Get size distribution of undergrowth species 'specid'
     */
    HistogramDistrib &
    get_sizedistrib(int specid);

    /*
     * Same as above, but non-const
     */
    const HistogramDistrib &
    get_sizedistrib(int specid) const;

    /*
     * Get matrix of spatial relationships between plants
     */
    HistogramMatrix &
    get_locmatrix();

    /*
     * Same as above, but non-const
     */
    const HistogramMatrix &
    get_locmatrix() const;

    /*
     * Get distribution of spatial relationship between undergrowth species 'specid1' and 'specid2'
     */
    HistogramDistrib &
    get_locdistrib(int specid1, int specid2);

    /*
     * Get the structure that maps from species id to its ratio (proportion of plant in this cluster belonging
     * to this species)
     */
    const std::unordered_map<int, float> &
    get_species_ratios();

    /*
     * Get species 'specid' ratio
     */
    float
    get_species_ratio(int specid);

    float get_maxdist() const;
    int get_ntotbins() const;
    int get_nreal_bins() const;
    int get_nreserved_bins() const;
    float get_binwidth() const;
    // global constants for all histogram matrices?


    /*
     * Get vectors that map from undergrowth and canopy id to indicies
     */
    std::vector<int> get_underid_to_idx() const;
    std::vector<int> get_canopyid_to_idx() const;

    /*
     * Get all plant and canopy ids
     */
    std::vector<int> get_canopy_ids() const;
    std::vector<int> get_plant_ids() const;

    // setter functions
    void set_species_ratios(const std::unordered_map<int, float> &ratios);
    void set_species_ratio(int specid, float ratio);
    void set_density(float density);

    // XXX: maybe we don't need cdata for this class?
    void set_cdata(const data_importer::common_data &cdata_arg);

    //void updateHistogramSizeMatrix(const basic_tree &reftree, const ClusterMaps &clmaps);
    //void updateHistogramLocMatrix(const basic_tree &reftree, const std::vector<basic_tree> &othertrees, const ClusterMaps &clmaps, ClusterMatrices *benchmark);

public:
    struct subbiome_clusters_type
    {
        subbiome_clusters_type(const data_importer::common_data &cdata, int canopyspecies, const std::unordered_map<int, float> &species_props);
        std::map<int, float> species_ratios;
    };

    void fix_species_ratios();
    void check_plant_and_canopy_ids() const;
    void check_plantids() const;
    void check_canopyids() const;
    float get_subspecies_ratio(int canopyspec_id, int underspec_id) const;
    const std::map<int, float> &get_subspecies_ratios(int canopyspec_id) const;
private:
    data_importer::common_data cdata;
    HistogramMatrix spatial_data;
    std::map<int, HistogramDistrib> size_data;
    float density;
    std::unordered_map<int, float> species_ratios;
    std::unordered_map<int, subbiome_clusters_type> subbiome_species_ratios;
    std::set<int> plant_ids, canopy_ids;

};

#endif 		//CLUSTERDATA_H
