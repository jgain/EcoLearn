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


#include "terrain.h"
#include "sunsim.h"
#include "extract_png.h"
#include "basic_types.h"
#include "data_importer.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char * argv [])
{
    //int w = 500, h = 500;
    //std::vector<float> data(w * h, 0.0f);

    if (argc != 3)
    {
        std::cout << "usage: sunlight_sim <dem filename> <db filename>" << std::endl;
    }

    std::string dem_filename = argv[1];
    std::string db_filename = argv[2];

    data_importer::common_data biomedata(db_filename);

    float latitude = biomedata.latitude;
    //float latitude = 0.0f;
    std::cout << "Latitude: " << latitude << std::endl;

    int w, h;
    //std::vector<float> data = get_image_data_48bit(argv[1], w, h)[0];
    //std::vector<float> data = load_elv(argv[1], w, h);
    auto data = data_importer::load_elv< ValueGridMap<float> >(dem_filename);
    float minval = *std::min_element(data.begin(), data.end());
    std::for_each(data.begin(), data.end(), [&minval](float &val) { val -= minval; /*val *= 0.3048;*/ });
    std::cout << "done importing data" << std::endl;
    glm::vec3 north(0, 0, -1);

    //data.getDim(w, h);

    //terrain ter(data.data(), w, h, glm::vec3(0, 0, -1));
    terrain ter(data, north, latitude);

    auto begin_time = std::chrono::steady_clock::now().time_since_epoch();
    sunsim sim(ter, true, 6000, 6000);
    sim.renderblock(true);
    auto end_time = std::chrono::steady_clock::now().time_since_epoch();
    auto total_time = end_time - begin_time;
    std::cout << "Sunlight simulation took " << std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count() / 1000.0f << " seconds" << std::endl;

    //sim.write_shaded_png_8bit("/home/konrad/shading_test.png");
    //sim.write_shaded_monthly_txt("/home/konrad/monthly_sunmap_test.txt");
    sim.write_shaded_monthly_txt("/home/konrad/PhDStuff/data/test_sunsim_out.txt", 0.9144f);

    return 0;
}

