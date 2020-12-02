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


#include "UndergrowthSampler.h"

#include <functional>

/*
 * Class that refines initially sampled locations, performed by base class, based
 * on pairwise correlation functions calculated on benchmark simulated data
 */

class UndergrowthRefiner : public UndergrowthSampler
{
public:
    UndergrowthRefiner(std::vector<std::string> 	cluster_filenames,
                       abiotic_maps_package 		amaps,
                       std::vector<basic_tree> 		canopytrees,
                       data_importer::common_data 	cdata);

    /*
     * sets the undergrowth plants on which refinement will be performed
       this is used if the undergrowth plants already exist elsewhere, and must not be sampled still
     */
    void set_undergrowth(const std::vector<basic_tree> &undergrowth);

    /*
     * does an initial sampling of undergrowth plants, then stores them in this object
     */
    void sample_init();

    /* do the refinement of undergrowth plants
     */
    void refine();

    /*
     * test the encode-decode function for storing affected distributions in UndergrowthRefiner::refine
     */
    static void test_encode_decode();

    /*
     * Restore affected distributions in refine function, due to the changes made being temporary changes made by
     * candidate plants only
     */
    void restore_distribs();

    /*
     * Retrieve stored undergrowth plants
     */
    std::vector<basic_tree> get_undergrowth();
protected:

    /*
     * Do a complete derivation for pairwise correlation functions, based on current undergrowth plants.
     * Used before refinement starts for an initial set of synthesized distributions to work from
     */
    void derive_complete();

    /*
     * Add plant effects to histograms based on adding plnt to landscape. This version registers the distribution
     * if no perturbs for the current plant has affect this distribution yet, so that we may restore the original
     * distribution after discarding this candidate plant
     */
    void add_plant_effects_for_restore(const basic_tree &		plnt,
                                       const basic_tree &		ignore,
                                       PlantSpatialHashmap &	underghash,
                                       PlantSpatialHashmap &	canopyhash,
                                       float *					rembenefit);

    /*
     * version of plant effect addition that does not update the tally of which distributions need to be restored.
     * See comments of add_plant_effects_for_restore above for comments on relevant sections
     */
    void add_plant_effects(const basic_tree &		plnt,
                           const basic_tree &		ignore,
                           PlantSpatialHashmap &	underghash,
                           PlantSpatialHashmap &	canopyhash,
                           float *					rembenefit);

    void remove_plant_effects(const basic_tree &	plnt,
                              const basic_tree &	ignore,
                              PlantSpatialHashmap &	underghash,
                              PlantSpatialHashmap &	canopyhash,
                              float *				rembenefit);

    /*
     * computes and encoded integer representing the cluster, row and column (in the distribution matrix)
     * that a distribution belongs to
     */
    static unsigned encode_distrib(unsigned clidx, unsigned row, unsigned col);

    /*
     * decodes the integer computed by the function above
     */
    static void decode_distrib(unsigned code, int &clidx, int &row, int &col);

    std::unique_ptr<ClusterMatrices> derivmodel;
    std::vector<basic_tree> undergrowth;
    std::set<HistogramDistrib *> normdistribs;

    std::unordered_map<unsigned, std::vector<common_types::decimal> > backup_distribs;

};
