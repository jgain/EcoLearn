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


#ifndef CLUSTERMAPS_H
#define CLUSTERMAPS_H

//#include "PlantSpatialHashmap.h"
//#include "kmeans.h"
#include <common/basic_types.h>
#include <data_importer/data_importer.h>
#include "data_importer/AbioticMapper.h"
#include "ClusterAssign.h"
#include <list>
#include <memory>
#include <unordered_map>


#define RW_SELECTION_DEBUG

class QLabel;
class PlantSpatialHashmap;
class kMeans;



class ClusterMatrices;

/*
 * This class holds a specific 'implementation' of a model (as defined in ClusterMatrices) applied to a landscape.
 * For example, in this class, we have exact required counts for undergrowth plants, and for each species, and
 * clusters (regions) are assign to each point of the landscape
 */

class ClusterMaps
{
public:

    //ClusterMaps(const ClusterMatrices &model, const abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees);
    ClusterMaps(const abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata);
    ClusterMaps(const ClusterAssign &classign, const abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata);

public:

    using treeitype = std::list<basic_tree>::iterator;

    static void do_kmeans_separate(std::vector< std::string > targetdirs, int nmeans, int niters, std::vector< std::array<float, kMeans::ndim> > &clusters, std::array<std::pair<float, float>, kMeans::ndim> &minmax_ranges, float undersim_sample_mult, const data_importer::common_data &cdata);

    static void compute_sampleprob_map(ValueGridMap<float> &map, const std::vector<basic_tree> &trees, float sample_mult, float seedprob);

    // Setter functions - ClusterMaps
    // ------------------
    void set_seedprob_map(const ValueGridMap<float> &probmap);
    void set_clustermap(const ValueGridMap<int> &clmap);
    void set_plant_ids();		// set from undergrowth plants
    void set_canopy_ids();		// set from canopy trees

    // getter functions - ClusterMaps
    // ----------------------------------

    /*
     * Get cluster index at world location on landscape at x, y
     */
    int
    get_cluster_idx(float x, float y) const;

    /*
     * Get size of the landscape in real terms (as opposed to grid/cell terms)
     */
    void
    get_real_size(float &width, float &height) const;

    /*
     * Get width of landscape in real terms
     */
    float
    get_width() const;

    /*
     * Get height of landscape in real terms
     */
    float
    get_height() const;


    /*
     * Get common data assigned to this object
     */
    const data_importer::common_data &
    get_cdata() const;


    /*
     * Get 2D map that indicates for each cell on landscape which cluster it belongs to
     */
    const ValueGridMap<int> &
    get_clustermap() const;

    /*
     * Get 2D map of seeding probabilities, corresponding to simulation model
     * We need the seeding probability to compute a 'size' for each cluster  (?)
     */
    const ValueGridMap<float> &
    get_seedprob_map() const;

    /*
     * Get subbiome hash via grid coordinates
     */
    int
    get_subbiome_hash_fromgrid(int gx, int gy);

    /*
     * Get subbiome hash via real coordinates
     */
    int
    get_subbiome_hash(float rx, float ry);

    /*
     * Get size for each region (which is a combination of cluster means and subbiome hash)
     */
    std::map<int, float>
    get_region_sizes();

    /*
     * Get total number of clusters (number of cluster means times number of subbiome combinations)
     */
    int
    get_nclusters();

    /*
     * Get number of cluster means
     */
    int get_nmeans();


    // Refactor to ClusterMaps class
    // -------------------------

    void fill_seedprob_map(float value);

    /*
     * Compute size for each region
     */
    void compute_region_sizes();

    /*
     * Initialize the clustermap.
     * Abiotic maps and ClusterAssign object (classign) need to be initialized
     */
    void create_clustermap();

    /*
     * Compute and return mapping from region index, to its target count of undergrowth plants
     */
    std::unordered_map<int, int> compute_region_target_counts(const ClusterMatrices &model);

    /*
     * Compute map of sample probabilities, based on sample probabilities in simulation model
     */
    void compute_sampleprob_map();

    /*
     * Compute and return target count for all undergrowth plants over whole landscape
     */
    int compute_overall_target_count(const ClusterMatrices &model);

    /*
     * Compute and return subbiome map (no side-effects for object)
     */
    static std::map<int, ValueGridMap<unsigned char> > create_subbiome_map(const std::vector<basic_tree> &canopytrees, int gw, int gh, float rw, float rh, float undersim_sample_mult, const data_importer::common_data &cdata);

    /*
     * Initialize subbiome map for this object
     */
    void create_subbiome_map();

    /*
     * Compute, based on argument, what the plant count for each region is
     */
    std::map<int, int> compute_region_plantcounts(const std::vector<basic_tree> &undergrowth);

    /*
     * Erase plants that are out of bounds in landscape
     */
    void erase_outofbounds_plants(std::list<basic_tree> &plnts);


    // (abiotic map operations)

    /*
     * Setter for each abiotic map
     */
    void set_moisturemap(const ValueGridMap<float> &moisture);
    void set_sunmap(const ValueGridMap<float> &sun);
    void set_slopemap(const ValueGridMap<float> &slope);
    void set_tempmap(const ValueGridMap<float> &temp);

    /*
     * Getter from real location for each abiotic map
     */
    float get_moisture(float x, float y) const;
    float get_sun(float x, float y) const;
    float get_temp(float x, float y) const;
    float get_slope(float x, float y) const;

    /*
     * Getter for each abiotic map
     */
    const ValueGridMap<float> &get_tempmap() const;
    const ValueGridMap<float> &get_sunmap() const;
    const ValueGridMap<float> &get_slopemap() const;
    const ValueGridMap<float> &get_moisturemap() const;

    /*
     * Set all abiotic maps based on directory name where they reside, and database filename
     * for common data
     */
    void set_maps(std::string target_dirname, std::string db_filename);

    /*
     * Set all abiotic maps based on abiotic_maps_package object, which contains
     * all imported abiotic maps
     */
    void set_maps(const abiotic_maps_package &amaps);

    /*
     * Getter for abiotic_maps_package object, amaps (const, then non-const)
     */
    const abiotic_maps_package &get_maps() const;
    abiotic_maps_package &get_maps();

    /*
     * Calculate viability for undergrowth species 'specid' at location x, y
     */
    float calcviability(float x, float y, int specid);

    /*
     * More general viability function, based on parameters of function
     */
    float viability(float val, float c, float r);

    /*
     * Set ClusterAssign object for this object
     */
    void set_classign(const ClusterAssign &classign);




    // Move undergrowth and canopy plants to another, separate class?


    // Move sampling and synthesis to a separate class. This class must have a ClusterMatrices member, since it needs data already computed/imported by the ClusterMatrices class
    // ----------------------------------------------
    // sampling helper functions
    std::unordered_map<int, std::unordered_map<int, int> > calc_species_counts(const PlantSpatialHashmap &undergrowth);


    // New feature - adding/removing canopy tree
    // make new class for this?
    // -------------------------------------
    bool remove_canopytree(const basic_tree &tree);
    void add_canopytree(const basic_tree &tree);


    // Discard  (and make appropriate changes)
    // -----------------
    const std::vector<basic_tree> &get_canopytrees() const;

    /*
     * Compute sample probability map, subbiome map, clustermap and region sizes
     */
    void compute_maps();

    /*
     * Set common data on species and subbiomes, imported from a SQL database, for this object
     */
    void set_cdata(const data_importer::common_data &cdata_arg);

    /*
     * Get object that assigns clusters (contains the trained kmeans model)
     */
    ClusterAssign get_classign();

    /*
     * Update the canopy trees for this object. Also updates computed clustermaps
     */
    void update_canopytrees(const std::vector<basic_tree> canopytrees, const ValueGridMap<float> &canopysun);
    void update_canopytrees(const std::vector<basic_tree> canopytrees);
private:
    struct subbiome_clusters_type
    {
        subbiome_clusters_type(const data_importer::common_data &cdata, int canopyspecies, const std::unordered_map<int, std::unordered_map<int, float> > &species_props)
        {
            int subbiome_id = cdata.canopyspec_to_subbiome.at(canopyspecies);
            data_importer::sub_biome sb = cdata.subbiomes_all_species.at(subbiome_id);
            std::vector<int> subspecies;
#ifndef RW_SELECTION_DEBUG
            for (auto &spenc : sb.species)
            {
                int specid = spenc.id;
                subspecies.push_back(specid);
            }
#else
             // to debug the roulette wheel selection of species in sample_from_tree
            for (auto &specpair : cdata.canopy_and_under_species)
            {
                subspecies.push_back(specpair.first);
            }
#endif 	// RW_SELECTION_DEBUG
            for (const auto &regionpair : species_props)
            {
                int region_idx = regionpair.first;
                std::map<int, float> &species_perc = clusters_species[region_idx];
                float sum = 0.0f;

                // for each species in this subbiome, we get the proportion of the cluster it makes up, then add that proportion to the probability map for this cluster (species_perc)
                for (auto &specid : subspecies)
                {
                    float prop;
                    if (regionpair.second.count(specid))
                        prop = regionpair.second.at(specid);
                    else
                        prop = 0.0f;
                    species_perc.insert({specid, prop});
                    sum += prop;
                }
                if (sum > 1e-5f)
                {
                    // if at least one species from the subbiome occurs in this cluster, we proceed to normalize the percentages
                    for (auto &proppair : species_perc)
                    {
                        proppair.second /= sum;
                    }
                }
                else
                {
                    species_perc.clear();		// this cluster contains no species from this subbiome, so we clear the map
                }
            }
        }

        std::unordered_map<int, std::map<int, float> > clusters_species;
    };
private:

    float cellsize_canopy = 20.0f;

    // to be refactored to ClusterMaps class
    // --------------------------------------
    data_importer::common_data cdata;
    ClusterAssign classign;
    //PlantSpatialHashmap canopyhashmap;
    std::vector<basic_tree> canopytrees;
    // maps structures
    float width, height;
    std::map<std::string, bool> maps_set;
    ValueGridMap<float> dem;
    abiotic_maps_package amaps;
    ValueGridMap<int> cluster_assigns;
    ValueGridMap<float> sampleprob_map;
    ValueGridMap<int> sbvals;
    // summary structures
    std::map<int, float> region_size;										// involved in derivation and sampling/synthesis

    bool maps_computed = false;


};

#endif // CLUSTERMAPS_H
