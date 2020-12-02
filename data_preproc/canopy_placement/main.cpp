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


#include "gpu_procs.h"
#include "data_importer.h"
#include <string>
#include "common/basic_types.h"

void test_local_maxima_find()
{
    std::cerr << "test_find_local_maxima_gpu: " << std::endl;
    if (test_find_local_maxima_gpu(verbosity::ALL))
    {
        std::cerr << "PASS" << std::endl;
    }
    else
    {
        std::cerr << "FAIL" << std::endl;
    }
}

int main(int argc, char * argv [])
{
    using namespace basic_types;

    if (argc != 3)
    {
        std::cout << "Usage: canopy_placement <dataset directory> <database filename>" << std::endl;
        return 1;
    }

    int nsims = 1;
    data_importer::common_data simdata(argv[2]);
    data_importer::data_dir ddir(argv[1], nsims);
    //int nsims = ddir.required_simulations.size();
    std::map<int, data_importer::species> all_species = simdata.all_species;
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
    */

    int width, height;
    MapFloat chm = data_importer::load_txt< MapFloat >(ddir.chm_fname);
    ValueGridMap<float> dem = data_importer::load_elv< ValueGridMap<float> >(ddir.dem_fname);
    //for (auto &biome_fname : ddir.biome_fnames)
    for (int i = 0; i < nsims; i++)
    {
        /*
        std::string species_fname = ddir.get_species_filename(biome_fname);
        std::string canopy_fname = ddir.get_canopy_filename(biome_fname);
        // make sure that the filenames are also in the ddir vector containing valid species and canopy filenames, just for internal consistency
        bool in_vector = std::any_of(ddir.species_fnames.begin(), ddir.species_fnames.end(), [&species_fname] (std::string &fname){
            return fname == species_fname;
        });
        assert(in_vector);
        in_vector 	   = std::any_of(ddir.canopy_fnames.begin(), ddir.canopy_fnames.end(), [&canopy_fname] (std::string &fname){
            return fname == canopy_fname;
        });
        assert(in_vector);
        */

        //int idx = i;
        int idx = 0;

        std::string canopy_fname = ddir.canopy_fnames.at(idx);
        std::string species_fname = ddir.species_fnames.at(idx);
        std::string canopy_texture_fname = ddir.canopy_texture_fnames.at(idx);
        std::string rendertexture_fname = ddir.rendertexture_fnames.at(idx);

        ValueMap<int> species = data_importer::load_txt< ValueMap<int> >(species_fname);
        //std::vector<species_params> all_params = data_importer::read_species_params(biome_fname);
        canopy_placer placer(&chm, &species, all_species, simdata);

        placer.optimise(50);

        placer.save_to_file(canopy_fname);
        placer.save_species_texture(canopy_texture_fname);
        placer.save_rendered_texture(rendertexture_fname);

    }

    cudaDeviceReset();

    return 0;
}

