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


#include "species_assign.h"

#include <algorithm>
#include <iostream>


// will probably not need the nspecies variable -  can determine number of species from the map_adapts_per_species vector
species_assign::species_assign(std::vector<basic_tree *> &tree_ptrs,
                               int nspecies,
                               int width,
                               int height,
                               const std::vector<float *> &adapt_maps_data,
                               const std::vector<bool> &adapt_maps_ownership,
                               std::vector<bool> row_major,
                               const std::vector<species_adapt> &map_adapts_per_species)
    : width(width),
      height(height),
      nspecies(map_adapts_per_species.size()),
      ntrees(tree_ptrs.size()),
      nmaps(adapt_maps_data.size()),
      row_major(row_major),
      tree_ptrs(tree_ptrs),
      map_adapts_per_species(map_adapts_per_species)

{
    create_adapt_maps_datastructs(adapt_maps_data, adapt_maps_ownership);
}

void species_assign::assign_species()
{
    generate_pt_randoms();
    assign_species_unspecified();
}

int species_assign::get_nspecies()
{
    return nspecies;
}

// XXX: still have to decide whether I want to multiply with random maps before taking minimum condition value, or after. If before, can do all minimum/maximum value taking in one function

void species_assign::generate_pt_randoms()
{
    std::vector<float> mapvals(nmaps);
    pts_species_adapts = std::vector<pt_species_adaptlist>(ntrees, pt_species_adaptlist(nspecies));
    for (int pt_idx = 0; pt_idx < ntrees; pt_idx++)
    {
        for (int spec_idx = 0; spec_idx < nspecies; spec_idx++)
        {
            //std::fill(mapvals.begin(), mapvals.end(), 0);
            for (int midx = 0; midx < nmaps; midx++)
            {
                int x, y;
                x = tree_ptrs[pt_idx]->x, y = tree_ptrs[pt_idx]->y;
                float mapval = adapt_maps[midx](x, y);
                mapvals[midx] = map_adapts_per_species[spec_idx][midx](mapval);
            }
            float minval = *std::min_element(mapvals.begin(), mapvals.end());
            pts_species_adapts[pt_idx][spec_idx] = minval;
        }
    }
}

void species_assign::assign_species_unspecified()
{
    for (int i = 0; i < tree_ptrs.size(); i++)
    {
        auto max_el_iter = std::max_element(pts_species_adapts[i].begin(), pts_species_adapts[i].end());
        int spec_idx = max_el_iter - pts_species_adapts[i].begin();
        if (*max_el_iter == 0)
        {
            //std::cout << "Index unassigned" << std::endl;
            unassigned_idxes.push_back(i);
            continue;
        }
        else
        {
            tree_ptrs[i]->species = (all_species)spec_idx;		// TODO: find a way to ensure that number of specified species is more than species_idx
        }
    }
}

void species_assign::assign_unassigned()
{

}

void species_assign::create_adapt_maps_datastructs(const std::vector<float *> &adapt_maps_data, const std::vector<bool> &adapt_maps_ownership_arg)
{
    auto adapt_maps_ownership = adapt_maps_ownership_arg;
    if (adapt_maps_data.size() == 0)
        throw std::runtime_error("Ownership for at least one adaptation map must be supplied (in species_assign::create_adapt_maps_datastructs)");
    if (row_major.size() == 0)
        throw std::runtime_error("Row or column major specification must be supplied for at leat one adaptation map");

    while (adapt_maps_ownership.size() < adapt_maps_data.size())
    {
        std::cerr << "Warning: ownership for only some of the adaptation maps are supplied. Will assume all maps have the same ownership as the last indicated map" << std::endl;
        adapt_maps_ownership.push_back(adapt_maps_ownership.back());
    }
    while (row_major.size() < adapt_maps_data.size())
    {
        std::cerr << "Warning: ownership for only some of the adaptation maps are supplied. Will assume all maps have the same ownership as the last indicated map" << std::endl;
        row_major.push_back(row_major.back());
    }

    for (int i = 0; i < adapt_maps_data.size(); i++)
    {
        adapt_maps.emplace_back(adapt_maps_data[i], width, height, adapt_maps_ownership[i]);
    }

}

void species_assign::generate_pt_likelihoods()
{
    // empty for now
}
