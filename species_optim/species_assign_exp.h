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


#ifndef SPECIES_ASSIGN_EXP
#define SPECIES_ASSIGN_EXP

#include "data_importer/data_importer.h"
#include "common/basic_types.h"
#include "species_optim/gpu_eval.h"
#include <vector>
#include <map>
#include <functional>

enum paint_type
{
    STRENGTH,
    PERC
};

class suit_func
{
public:
    suit_func(float loc, float d);
    suit_func(float loc, float d, float mult);

    void set_d(float d);
    void set_loc(float loc);
    void set_values(float loc, float d);

    float get_d() const;
    float get_loc() const;

    float operator () (float x) const;
private:
    float d;
    float loc;
    float mult;
};

// class for a single species' adaptations
class species
{
public:
    species(data_importer::species specimport, float maxheight);
    species(const std::vector< suit_func > &adaptations, float maxheight, int seed);
    float operator () (const std::vector<float> &values);
    float get_loc_for_map(int idx) const;
    float get_scale_for_map(int idx) const;
private:
    std::vector<suit_func> adaptations;
    float maxheight;
};

namespace data_importer
{
    class species;
}

class species_assign
{
public:
    species_assign(const ValueMap<float> &chm, const std::vector<ValueMap<float> > &abiotics, const std::vector<species> &species_vec, const std::vector<float> &max_heights);
    species_assign(const ValueMap<float> &chm, const std::vector<ValueMap<float> > &abiotics, const std::map<int, data_importer::species> &specmap_raw);
    ~species_assign();

    /*
     * Assign species at location x, y based on abiotic maps and species viability functions held by this object
     */
    void assign_to(int x, int y);

    // exactly the same as assign_gpu - check comments for assign_gpu
    void assign();

    /*
     * Get species assigned to location x, y
     */
    int get(int x, int y);

    void add_drawn_circle(int x, int y, int radius, float mult, int specie);

    /*
     * Runs species assignment (only) on GPU, then sets each element in the assigned array to the assigned species - creates a map of species
     */
    void assign_gpu();

    /*
     * Create maps corresponding to nonzero CHM cells. This saves some memory, as normally not the entire landscape
     * is filled with nonzero CHM values
     */
    void create_nonzero();

    /*
     * Get the multiplier for species index idx at location x, y
     */
    float get_mult_at(int spec_idx, int x, int y);

    /*
     * Clear brushstroke data, so that we can start anew for the next optimisation to be done
     */
    void clear_brushstroke_data();

    /*
     * Optimise current brushstroke area according to the required percentage 'req_perc' for species index 'spec_idx'
     */
    void optimise_brushstroke(int spec_idx, float req_perc, std::string outfile = "");

    /*
     * This is the inner function that gets called by 'optimise_brushstroke'
     * Optimise over species percentage for species spec_idx, so that it satisfies percentage 'req_perc', given abiotic conditions 'abiotics',
     * corresponding to vector 'nonzero_idxes', indicating indices into real map.
     */
    void optimise(int spec_idx, float req_perc, std::vector<std::vector<float> > &abiotics, std::vector<int> &nonzero_idxes, float max_mult, std::vector<int> all_indices, ValueMap<float> smoothmap);

    /*
     * Get 2D map containing assigned species
     */
    const ValueMap<int> &get_assigned();

    /*
     * set the canopy height model for this object, and initialize the nonzero map by calling 'create_nonzero'
     */
    void set_chm(const ValueMap<float> &chm);

    /*
     * Get and set maps that contain the drawings for species optimisation, as well as multipliers for each such species
     */
    void get_mult_maps(std::map<int, ValueMap<float> > &drawmap, std::map<int, ValueMap<bool> > &draw_indicator);
    void set_mult_maps(const std::map<int, ValueMap<float> > &drawmap, const std::map<int, ValueMap<bool> > &draw_indicator);

    /*
     * Write drawing for species optimisation for specidx out to filename 'outfile'
     */
    void write_species_drawing(int specidx, std::string outfile);

    /*
     * Get grid width and height for the landscape we are applying species optimisation to
     */
    void get_dims(int &w, int &h) const;

    /*
     * Set the optional callback function, generally used for keeping track of progress of species optimisatoin
     */
    void set_progress_func(std::function<void(int)> progress_func);
private:
    std::vector<float> max_heights;
    int nspecies;
    std::map<int, ValueMap<float> > raw_adapt_vals;
    ValueMap<int> assigned;
    ValueMap<float> chm;
    std::map<int, ValueMap<float> > noise;
    std::map<int, ValueMap<float> > backup_noise;
    int last_seed = 0;
    std::map<int, ValueMap<bool> > spec_seeding;
    std::map<int, ValueMap<float> > seeding;
    std::map<int, ValueMap<float> > drawing;
    std::map<int, ValueMap<bool> > drawing_indicator;
    ValueMap<bool> bs_indicator;	// shows where we already added values for current brushstroke. Gets nulled out each time a new brush stroke is begun
    std::vector<std::vector<float> > bs_abiotics;	// abiotics for brush stroke area
    std::vector<int> bs_nonzero_indices;	// nonzero idxes for brush stroke area
    std::vector<int> bs_all_indices;
    std::vector<float> bs_multmaps;
    float bs_max_mult = 0.0f;
    int bs_npixels = 0;
    int bs_nonzero_npixels = 0;
    std::vector<float> chmsorted;
    std::vector<int> sorted_idxes;
    std::vector<ValueMap<float> > abiotics;
    std::vector< std::vector<float> > nonzero_abiotics;
    std::vector<int> nonzero_idxes;
    std::vector<float> nonzero_chm;
    std::vector<species> species_vec;
    std::vector<float> species_locs;
    std::vector<float> species_scales;

    std::vector<float> species_percs;

    std::function<void(int)> progress_func;

    interm_memory gpu_mem;
    int width, height;
    bool gpu_initialized = false;
    bool first_eval = true;
};


#endif //SPECIES_ASSIGN_EXP
