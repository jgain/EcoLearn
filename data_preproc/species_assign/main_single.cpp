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
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include "species_assign_exp.h"
#include "data_importer.h"
#include "common/basic_types.h"

#include <iostream>
#include <stdexcept>
#include <map>
#include <cassert>
#include <chrono>
#include <random>

std::set<data_importer::species_encoded> get_species_perc(std::map<int, data_importer::sub_biome> biomes)
{
    using namespace data_importer;
    std::set<species_encoded> species;
    for (auto &map_el : biomes)
    {
        sub_biome &bm = map_el.second;
        for (auto &sp : bm.species)
        {
            species_encoded newsp = sp;
            newsp.percentage *= bm.percentage;		// we weigh each species' percentage with its subbiome's percentage
            auto result_pair = species.insert(newsp);
            if (!result_pair.second)	// the current species has already been found in a different subbiome. Add to the percentage instead
            {
                //result_pair.first->percentage += newsp.percentage;	// normally we would do it like this, but since the elements of a set are immutable, we
                                                                        // have to calculate the new percentage, remove the old element, and insert the element
                                                                        // with the right percentage and the same key
                newsp.percentage += result_pair.first->percentage;
                assert(newsp.id == result_pair.first->id);
                species.erase(*result_pair.first);
                species.insert(newsp);
            }
        }
    }

    float sum = 0.0f;
    for (auto &el : species)
    {
        sum += el.percentage;
        assert(el.percentage >= 0.0f && el.percentage <= 1.0f);
    }
    assert(sum >= 1 - 1e-5 && sum <= 1 + 1e-5);

    return species;
}

void validate_sim_details(std::map<int, data_importer::sub_biome> biomes)
{
    for (auto &bpair : biomes)
    {
        assert(bpair.second.percentage >= 0.0f && bpair.second.percentage <= 1.0f);
    }
}

int main(int argc, char * argv [])
{
    using namespace basic_types;

    if (argc != 3)
    {
        std::cout << "usage: species_optim <directory> <database filename>" << std::endl;
        return 1;
    }

    std::string data_directory = argv[1];
    std::string database_filename = argv[2];
    data_importer::common_data simdata(database_filename);
    data_importer::data_dir ddir(data_directory);

    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.first << ", " << sp_pair.second.idx << ", " << sp_pair.second.a << ", " << sp_pair.second.b << std::endl;
    }

    int width, height;
    auto month_moisture = data_importer::read_monthly_map< ValueMap<float> >(ddir.wet_fname);
    ValueMap<float> moisturemap = data_importer::average_mmap< ValueMap<float>, ValueMap<float> >(month_moisture);
    ValueMap<float> slopemap = data_importer::load_txt<ValueMap<float> >(ddir.slope_fname);
    //std::vector<MapFloat> month_sun = data_importer::read_monthly_map(sun_filename, width, height);
    auto month_sun = data_importer::read_monthly_map< ValueMap<float> >(ddir.sun_fname);
    ValueMap<float> sunmap = data_importer::average_mmap< ValueMap<float>, ValueMap<float> >(month_sun);
    auto month_temp = data_importer::read_monthly_map< ValueMap<float> > (ddir.temp_fname);
    ValueMap<float> tempmap = data_importer::average_mmap<ValueMap<float>, ValueMap<float> >(month_temp);
    tempmap.getDim(width, height);

    int chmw, chmh;
    //MapFloat chmdata = data_importer::load_txt(ddir.chm_fname, chmw, chmh);
    ValueMap<float> chmdata = data_importer::load_txt<ValueMap<float> >(ddir.chm_fname);
    std::vector< ValueMap<float> > abiotic_maps = {tempmap, slopemap, moisturemap, sunmap};

    std::cout << "Number of species: " << simdata.all_species.size() << std::endl;
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.second.sun.c << ", " << sp_pair.second.sun.r << std::endl;
        std::cout << sp_pair.second.temp.c << ", " << sp_pair.second.temp.r << std::endl;
    }

    std::set<data_importer::species_encoded> species_to_sim;

    for (auto &subb_pair : simdata.subbiomes)
    {
        data_importer::sub_biome &subb = subb_pair.second;
        for (const auto &sp : subb.species)
        {
            species_to_sim.insert(sp);
        }
        //species_to_sim.insert(subb.species.begin(), subb.species.end());
    }
    std::vector<species> species_vec;
    std::vector<int> species_indices;
    std::vector<float> species_perc;
    std::vector<float> spec_maxheights;
    for (auto &sp_tosim : species_to_sim)
    {
        species_perc.push_back(sp_tosim.percentage);
        species_indices.push_back(sp_tosim.id);
        data_importer::species sp = simdata.all_species[sp_tosim.id];

        spec_maxheights.push_back(simdata.canopy_and_under_species[sp_tosim.id].maxhght);

        //species_vec.emplace_back(adapt_params, 0);
        species_vec.emplace_back(sp, spec_maxheights.back());
    }

    std::string species_fname = ddir.species_fnames[0];
    std::cout << "Creating species optim..." << std::endl;
    //auto optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, false);
    //auto optim_ptr = species_set::create_evaluator(&chmdata, maps_ptrs, species_vec, false, spec_maxheights);
    auto optimobj = species_assign(chmdata, abiotic_maps, species_vec, spec_maxheights);
    std::cout << "Starting species optim..." << std::endl;
    optimobj.assign();
    auto specmap = optimobj.get_assigned();
    int gw, gh;
    specmap.getDim(gw, gh);
    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            int val = specmap.get(x, y);
            if (val >= 0)
                specmap.set(x, y, species_indices.at(val));
        }
    }
    std::cout << "writing species map..." << std::endl;
    data_importer::write_txt<ValueMap<int> >(species_fname, &specmap);
    //optim_ptr->evaluate_gpu();
    std::cout << "Done" << std::endl;

    return 0;
}
