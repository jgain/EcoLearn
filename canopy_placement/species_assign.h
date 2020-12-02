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


#include "basic_types.h"

#include <functional>
#include <vector>

using species_adapt = std::vector<std::function<float(float)> >;
using pt_species_adaptlist = std::vector<float>;	// list of minimum adaptation values for each species for a given point (the minimum being over all adaptation maps,
                                                        // since the minimum adaptation value is the final adaptation value for a species)

class species_assign
{
public:
    species_assign(std::vector<basic_tree *> &tree_ptrs, int nspecies, int width, int height, const std::vector<float *> &adapt_maps_data, const std::vector<bool> &adapt_maps_ownership, std::vector<bool> row_major, const std::vector<species_adapt> &map_adapts_per_species);

    void assign_species();		// pass params here which will determine whether we are assigning unspecified or specified

    int get_nspecies();

protected:
    void generate_pt_randoms();		// exactly like the function of the same name in the Python version, except it does not return random maps (will also change the name later to something more intuitive)
    void generate_pt_likelihoods();	// will not do anything to the point species likelihoods atm
    void assign_species_unspecified();	// assigns species based purely on random maps and conditions. No specification on required percentages of each species

    void create_adapt_maps_datastructs(const std::vector<float *> &adapt_maps_data, const std::vector<bool> &adapt_maps_ownership_arg);

    void assign_unassigned();
private:
    int width;
    int height;
    int nspecies;
    int ntrees;
    int nmaps;
    std::vector<bool> row_major;
    std::vector<data_struct> adapt_maps;
    std::vector<data_struct> random_maps;
    std::vector<basic_tree *> tree_ptrs;
    std::vector<pt_species_adaptlist> pts_species_adapts;
    std::vector<int> unassigned_idxes;
    std::vector<species_adapt> map_adapts_per_species;	// vector containing, for each species, a vector of functions corresponding to each adaptation map


};
