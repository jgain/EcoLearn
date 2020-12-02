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
#include "basic_types.h"

#include <iostream>
#include <stdexcept>
#include <map>
#include <cassert>
#include <chrono>
#include <random>

#include <SDL2/SDL.h>

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

template<typename T>
void trim(T &val, T min, T max)
{
    if (val < min) val = min;
    if (val > max) val = max;
}

template<typename T>
void mask_zero_map(T &map, int size)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> unif_real;

    int w, h;
    map.getDim(w, h);

    int pad = size;

    int x = unif_real(generator) * (w - 2 * size) + size;
    int y = unif_real(generator) * (h - 2 * size) + size;

    int sx, ex, sy, ey;
    sx = x - size;
    ex = x + size;
    sy = y - size;
    ey = y + size;

    trim(sx, 0, w - 1);
    trim(ex, 0, w - 1);
    trim(sy, 0, h - 1);
    trim(ey, 0, h - 1);

    for (int cx = sx; cx < ex; cx++)
    {
        for (int cy = sy; cy < ey; cy++)
        {
            map.set(cx, cy, 0);
        }
    }
}

template<typename T>
void mask_nonzero_map(T &map, int size)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> unif_real;

    int w, h;
    map.getDim(w, h);

    int pad = size;

    int x = unif_real(generator) * (w - 2 * size) + size;
    int y = unif_real(generator) * (h - 2 * size) + size;

    int sx, ex, sy, ey;
    sx = x - size;
    ex = x + size;
    sy = y - size;
    ey = y + size;

    trim(sx, 0, w - 1);
    trim(ex, 0, w - 1);
    trim(sy, 0, h - 1);
    trim(ey, 0, h - 1);

    for (int cy = 0; cy < h; cy++)
        for (int cx = 0; cx < w; cx++)
        {
            if (cx > sx && cx < ex && cy > sy && cy < ey)
            {
                continue;
            }
            else
            {
                map.set(cx, cy, 0);
            }
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
    //data_importer::common_data simdata(database_filename);
    int nsims = 3;
    data_importer::data_dir ddir(data_directory, 3);

    std::vector< std::map<int, data_importer::sub_biome > > req_sims = ddir.required_simulations;

    /*
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.first << ", " << sp_pair.second.idx << ", " << sp_pair.second.a << ", " << sp_pair.second.b << std::endl;
    }
    */

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
    //mask_nonzero_map(chmdata, 100);
    std::vector< MapFloat > abiotic_maps = {moisturemap, slopemap, sunmap, tempmap};
    std::vector<MapFloat *> maps_ptrs;
    for (auto &amap : abiotic_maps)
    {
        maps_ptrs.push_back(&amap);
    }

    /*
    std::cout << "Number of species: " << simdata.all_species.size() << std::endl;
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.second.sun.c << ", " << sp_pair.second.sun.r << std::endl;
        std::cout << sp_pair.second.temp.c << ", " << sp_pair.second.temp.r << std::endl;
    }
    */

    std::vector<species> solutions;
    std::vector<float> species_perc;
    std::vector<float> maxheights(3, 100.0f);
    species_perc.push_back(0.4);
    species_perc.push_back(0.4);
    species_perc.push_back(0.2);

    const int ntexs = 20;

    std::string base_fname = "/home/konrad/PhDStuff/data/species_optim";

    /*
    for (int i = 0; i < ntexs; i++)
    {
        std::unique_ptr<species_set> optim_ptr;
        if (i == 0)
        {
            optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, true);
        }
        else
        {
            optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, solutions, true);
        }
        optim_ptr->optimise(1000);
        solutions = optim_ptr->get_species_vec();
        species_perc[0] -= 0.075;
        species_perc[1] += 0.075;

        auto species_perc = optim_ptr->get_curr_species_perc();
        for (auto &f : species_perc)
        {
            std::cout << f << " ";
        }
        std::cout << std::endl;

        std::string fname = base_fname + std::to_string(i) + ".txt";
        optim_ptr->write_species_map(fname, &chmdata, {0, 1, 2});
        specmaps.push_back(optim_ptr->create_species_map(&chmdata, {0, 1, 2}));
    }
    */

    std::vector<float> evals;
    std::vector<ValueMap<int> > specmaps;

    species_set::test_smoothness_smooth(&chmdata, maps_ptrs, ntexs, evals, specmaps, maxheights);


    SDL_Window *window = SDL_CreateWindow("Species", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Surface *sfc;
    sfc = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA8888);
    std::vector<uint32_t> colors = {0x000000FF, 0xFF0000FF, 0x00FF00FF, 0x0000FFFF};

    auto modify_surface = [sfc, colors, width, height](const ValueMap<int> &vals) {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = sfc->pitch * y + x * sizeof(uint8_t) * 4;
                uint32_t *curr_pos = (uint32_t *)(((uint8_t *)sfc->pixels) + idx);
                *curr_pos = colors[vals.get(x, y) + 1];
            }
        }
    };

    auto render_surface = [sfc, renderer]() {
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, sfc);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, tex, NULL, NULL);

        SDL_RenderPresent(renderer);
        SDL_DestroyTexture(tex);
    };

    auto modify_and_render_surface = [&modify_surface, &render_surface] (const ValueMap<int> &vals) {
        modify_surface(vals);
        render_surface();
    };


    int curr_idx = 0;
    modify_and_render_surface(specmaps[curr_idx]);


    SDL_Event event;
    bool quit = false;
    while (!quit)
    {
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_QUIT)
                quit = true;
            if (event.type == SDL_KEYDOWN)
            {
                switch (event.key.keysym.sym)
                {
                    case SDLK_RIGHT:
                        curr_idx++;
                        if (curr_idx >= ntexs)
                            curr_idx--;
                        modify_and_render_surface(specmaps[curr_idx]);
                        std::cout << "Roughness: " << evals[curr_idx] << std::endl;
                        break;
                    case SDLK_LEFT:
                        curr_idx--;
                        if (curr_idx < 0)
                            curr_idx++;
                        modify_and_render_surface(specmaps[curr_idx]);
                        break;
                    default:
                        break;
                }
            }
        }
    }

    SDL_Quit();



    /*

    //for (auto &biome_fname : ddir.biome_fnames)
    //for (auto &sim_details : req_sims)
    //for (int sim_idx = 0; sim_idx < 1; sim_idx++)
    //for (int sim_idx = 0; sim_idx < req_sims.size(); sim_idx++)
    for (auto &subb_pair : simdata.subbiomes)
    {
        int subb_idx = subb_pair.first;
        data_importer::sub_biome &subb = subb_pair.second;


        //auto &sim_details = req_sims[sim_idx];
        //validate_sim_details(sim_details);

        //std::set<data_importer::species_encoded> species_to_sim = get_species_perc(sim_details);
        std::set<data_importer::species_encoded> species_to_sim = subb.species;
        std::vector<float> species_perc;
        std::vector<int> species_indices;
        std::vector<abiotic_adapt_params> species_sim_params;
        std::vector<species> species_vec;
        for (auto &sp_tosim : species_to_sim)
        {
            std::vector<abiotic_adapt_params> adapt_params;

            species_perc.push_back(sp_tosim.percentage);
            species_indices.push_back(sp_tosim.idx);
            abiotic_adapt_params simp;
            data_importer::species sp = simdata.all_species[sp_tosim.idx];
            float adapt_mod = 0.5f;
            simp.distance = sp.wet.r * adapt_mod;
            simp.center = sp.wet.c;
            species_sim_params.push_back(simp);
            adapt_params.push_back(simp);
            simp.distance = sp.slope.r * adapt_mod;
            simp.center = sp.slope.c;
            species_sim_params.push_back(simp);
            adapt_params.push_back(simp);
            simp.distance = sp.sun.r * adapt_mod;
            simp.center = sp.sun.c;
            species_sim_params.push_back(simp);
            adapt_params.push_back(simp);
            simp.distance = sp.temp.r * adapt_mod;
            simp.center = sp.temp.c;
            species_sim_params.push_back(simp);
            adapt_params.push_back(simp);

            species_vec.emplace_back(adapt_params, 0);
        }
        //std::string species_fname = ddir.get_species_filename(biome_fname);
        //std::string species_fname = ddir.species_fnames[sim_idx];
        std::string species_fname = ddir.generate_species_fname(subb_idx);


        //std::vector<species_params> all_params = data_importer::read_species_params(biome_fname);

        std::vector<abiotic_adapt_params> params;


        std::cout << "Creating species optim..." << std::endl;
        auto optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, false);
        //auto optim_ptr = species_set::create_evaluator(&chmdata, maps_ptrs, species_vec);
        std::cout << "Starting species optim..." << std::endl;
        auto simparams = species_sim_params;
        float best_eval = 0.0f;
        //optim_ptr->set_params(species_sim_params);
        optim_ptr->evaluate_gpu();
        std::cout << "writing species map..." << std::endl;
        optim_ptr->write_species_map(species_fname, &chmdata, species_indices);
        std::cout << "Done" << std::endl;

    }
    */

    return 0;
}
