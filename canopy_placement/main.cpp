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


#include "data_importer.h"
//#include "mosaic_spacing.h"
#include "canopy_placer.h"
#include "basic_types.h"

int main(int argc, char * argv [])
{
    if (argc < 5)
    {
        std::cout << "usage: ./main <chm filename> <dem filename> <species map filename> <output pdb filename>" << std::endl;
        return 1;
    }

    int width, height;
    //MapFloat chm_data = data_importer::load_txt("/home/konrad/EcoLearn/Data/Sonoma_Sim_Input/test/test.chm", width, height);
    MapFloat chm_data = data_importer::load_txt(argv[1], width, height);
    MapFloat dem_data = data_importer::load_elv(argv[2], width, height);
    ValueMap<int> species_map = data_importer::load_txt< ValueMap<int> >(argv[3]);

    std::vector<species_params> params;
    params.push_back(species_params(-2.0f, 1.0f));
    params.push_back(species_params(-1.9f, 1.05f));
    params.push_back(species_params(-1.9f, 1.00f));
    params.push_back(species_params(-1.9f, 1.00f));

    int w, h;
    chm_data.getDim(w, h);

    std::cout << "width, height: " << w << ", " << h << std::endl;

    //canopy_placer placer(&chm_data, 50);
    canopy_placer placer(&chm_data, &species_map, params);
    placer.optimise(50);
    placer.save_to_file(argv[4], dem_data.data());

    return 0;
}
