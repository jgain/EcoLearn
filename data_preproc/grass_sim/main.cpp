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


#include "grass_sim.h"
#include "data_importer.h"

#include <iostream>
#include <map>

int main(int argc, char * argv []) {
    bool use_treeshade = false;
    if (argc < 2 || argc > 3)
    {
        std::cout << "Usage: grass_sim <data directory> [--use_treeshade]" << std::endl;
        return 1;
    }
    if (argc == 3)
    {
        if (strcmp(argv[2], "--use_treeshade"))
        {
            std::cout << "Usage: grass_sim <data directory> [--use_treeshade]" << std::endl;
            return 1;
        }
        use_treeshade = true;
    }
    data_importer::data_dir ddir(argv[1], 1);
    std::string shade_filename;
    if (use_treeshade)
        shade_filename = ddir.sun_tree_fname;
    else
        shade_filename = ddir.sun_fname;
    ValueGridMap<float> ter = data_importer::load_elv<ValueGridMap<float> >(ddir.dem_fname);
    std::map<int, std::vector<MinimalPlant> > minplants;
    data_importer::read_pdb(ddir.canopy_fnames[0], minplants);
    std::vector<basic_tree> trees = data_importer::minimal_to_basic(minplants);
    std::vector<basic_tree *> treeptrs;
    for (auto &tr : trees)
        treeptrs.push_back(&tr);

    std::vector<MapFloat> moisture, sunlight, temperature;
    moisture = data_importer::read_monthly_map<MapFloat>(ddir.wet_fname);
    sunlight = data_importer::read_monthly_map<MapFloat>(shade_filename);
    temperature = data_importer::read_monthly_map<MapFloat>(ddir.temp_fname);

    auto viability_params = data_importer::read_grass_viability(ddir.grass_params_fname);

    std::cout << "Constructing grass sim object..." << std::endl;
    GrassSim gsim(ter, 2);
    gsim.setConditions(&moisture[5], &sunlight[5], &temperature[5]);
    gsim.set_viability_params(viability_params);
    std::cout << "Running grass simulation..." << std::endl;
    gsim.grow(ter, treeptrs);
    std::cout << "All done" << std::endl;

    gsim.write(ddir.grass_fname);

    //std::cout << "Hello, World!" << std::endl;
    return 0;
}
