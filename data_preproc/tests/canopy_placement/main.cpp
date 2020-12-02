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
#include "basic_types.h"

int main(int argc, char * argv [])
{
    using namespace basic_types;

    if (argc != 3)
    {
        std::cout << "Usage: canopy_placement <dataset directory> <database filename>" << std::endl;
        return 1;
    }

    int nsims = 4;
    data_importer::common_data simdata(argv[2]);
    data_importer::data_dir ddir(argv[1], nsims);
    std::map<int, data_importer::species> all_species = simdata.all_species;

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

        std::string canopy_fname = ddir.canopy_fnames[i];
        std::string species_fname = ddir.species_fnames[i];

        ValueMap<int> species = data_importer::load_txt< ValueMap<int> >(species_fname);
        //std::vector<species_params> all_params = data_importer::read_species_params(biome_fname);
        canopy_placer placer(&chm, &species, all_species, simdata);
        placer.optimise(20);

        //std::vector<uint32_t> rendered_texture;
        //placer.get_chm_rendered_texture(rendered_texture);
        ValueMap<int> tex = placer.get_species_texture_raw();
        /*
        tex.setDim(chm);
        int nelements;
        int texw, texh;
        tex.getDim(texw, texh);
        nelements = texw * texh;
        memcpy(tex.data(), rendered_texture.data(), sizeof(uint32_t) * nelements);
        */

        std::string out_filename = "/home/konrad/PhDStuff/data/test_canopy_placement";
        std::string tex_out_filename = "/home/konrad/PhDStuff/data/test_canopy_placement_texture";
        out_filename += std::to_string(i) + ".pdb";
        tex_out_filename += std::to_string(i) + ".txt";

        placer.save_to_file(out_filename, dem.data());

        data_importer::write_txt(tex_out_filename, &tex);
    }

    return 0;
}

