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


#include "species_optim.h"
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

    /*
    std::string dir(argv[1]);
    while (dir[dir.size() - 1] == '/')
    {
        dir.pop_back();
    }
    size_t fwsl_pos = dir.find_last_of('/');
    std::string dataset_name = dir.substr(fwsl_pos + 1);

    std::string chm_filename = dir + "/" + dataset_name + ".chm";
    std::string dem_filename = dir + "/" + dataset_name + ".elv";
    std::string species_filename = dir + "/" + dataset_name + "_species_map.txt";
    std::string species_params_filename = dir + "/" + dataset_name + "_species_params.txt";

    std::string wet_filename = dir + "/" + dataset_name + "_wet.txt";
    std::string slope_filename = dir + "/" + dataset_name + "_slope.txt";
    std::string sun_filename = dir + "/" + dataset_name + "_sun.txt";
    std::string temp_filename = dir + "/" + dataset_name + "_temp.txt";
    */

    std::default_random_engine generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> unif;

    std::string data_directory = argv[1];
    std::string database_filename = argv[2];
    data_importer::common_data simdata(database_filename);
    data_importer::data_dir ddir(data_directory, simdata);

    std::vector< std::map<int, data_importer::sub_biome > > req_sims = ddir.required_simulations;

    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.first << ", " << sp_pair.second.idx << ", " << sp_pair.second.a << ", " << sp_pair.second.b << std::endl;
    }

    int width, height;
    auto month_moisture = data_importer::read_monthly_map< ValueMap<float> >(ddir.wet_fname);
    MapFloat moisturemap = data_importer::average_mmap< MapFloat, ValueMap<float> >(month_moisture);
    MapFloat slopemap = data_importer::load_txt<MapFloat>(ddir.slope_fname);
    //std::vector<MapFloat> month_sun = data_importer::read_monthly_map(sun_filename, width, height);
    auto month_sun = data_importer::read_monthly_map< ValueMap<float> >(ddir.sun_fname);
    auto sunmap = data_importer::average_mmap< MapFloat, ValueMap<float> >(month_sun);
    auto month_temp = data_importer::read_monthly_map< ValueMap<float> > (ddir.temp_fname);
    MapFloat tempmap = data_importer::average_mmap<MapFloat, ValueMap<float> >(month_temp);
    tempmap.getDim(width, height);

    int chmw, chmh;
    MapFloat chmdata = data_importer::load_txt(ddir.chm_fname, chmw, chmh);
    std::vector< MapFloat > abiotic_maps = {moisturemap, slopemap, sunmap, tempmap};
    std::vector<MapFloat *> maps_ptrs;
    for (auto &amap : abiotic_maps)
    {
        maps_ptrs.push_back(&amap);
    }

    std::cout << "Number of species: " << simdata.all_species.size() << std::endl;
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.second.sun.c << ", " << sp_pair.second.sun.r << std::endl;
        std::cout << sp_pair.second.temp.c << ", " << sp_pair.second.temp.r << std::endl;
    }

    std::vector<species> species_vec;
    std::vector<int> species_indices;
    std::vector<abiotic_adapt_params> adapt_params;
    std::vector<float> maxheights;

    for (std::pair<const int, data_importer::species> &sp_pair : simdata.all_species)
    {
        data_importer::species &sp = sp_pair.second;
        abiotic_adapt_params simp;
        float adapt_mod = 1.0f;
        simp.distance = sp.wet.r * adapt_mod;
        simp.center = sp.wet.c;
        //species_sim_params.push_back(simp);
        adapt_params.push_back(simp);
        simp.distance = sp.slope.r * adapt_mod;
        simp.center = sp.slope.c;
        //species_sim_params.push_back(simp);
        adapt_params.push_back(simp);
        simp.distance = sp.sun.r * adapt_mod;
        simp.center = sp.sun.c;
        //species_sim_params.push_back(simp);
        adapt_params.push_back(simp);
        simp.distance = sp.temp.r * adapt_mod;
        simp.center = sp.temp.c;
        //species_sim_params.push_back(simp);
        adapt_params.push_back(simp);

        species_vec.emplace_back(adapt_params, 0);
        species_indices.push_back(sp.idx);

        maxheights.push_back(sp_pair.second.maxhght);
    }

    std::cout << "Creating species optim..." << std::endl;
    //auto optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, false);
    auto optim_ptr = species_set::create_evaluator(&chmdata, maps_ptrs, species_vec, false, maxheights);
    std::cout << "Starting species optim..." << std::endl;
    //auto simparams = species_sim_params;
    float best_eval = 0.0f;
    optim_ptr->evaluate_gpu();
    std::cout << "writing species map..." << std::endl;
    optim_ptr->write_species_map(ddir.species_fnames.at(0), &chmdata, species_indices);
    std::cout << "Done" << std::endl;

    return 0;
}

    /*
    for (int i = 3; i < argc; i++)
    {
        MapFloat amap;
        int width, height;
        try
        {
            amap = data_importer::load_txt(argv[i], width, height);
        }
        catch (std::runtime_error &err)
        {
            auto mmap = data_importer::read_monthly_map(argv[i], width, height);
            amap = data_importer::average_mmap(mmap);
        }
        if (width != chmw || height != chmh)
        {
            throw std::invalid_argument(std::string("width of abiotic map at ") + argv[i] + " does not have the same width or height as the CHM");
        }
        abiotic_maps.push_back(amap);
    }
    */
