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
#include "common/basic_types.h"
#include "data_importer.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

/*
std::vector<float> load_elv(std::string filename, int &width, int &height)
{
    using namespace std;

    float step, lat;
    int dx, dy;

    float val;
    ifstream infile;

    std::vector<float> retval;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> dx >> dy;
        infile >> step;
        infile >> lat;
        width = dx;
        height = dy;
        retval.resize(width * height);
        //init(dx, dy, (float) dx * step, (float) dy * step);
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                infile >> val;
                retval[y * width + x] = val * 0.3048f;
                //grid[x][y] = val * 0.3048f; // convert from feet to metres
            }
        }
        infile.close();
    }
    else
    {
        throw runtime_error("Error Terrain::loadElv:unable to open file " + filename);
    }
    return retval;
}
*/

int main(int argc, char * argv [])
{
    //int w = 500, h = 500;
    //std::vector<float> data(w * h, 0.0f);
	
    if (argc != 3)
	{
        std::cout << "usage: sunlight_sim <data directory> <db filename>" << std::endl;
    }

    data_importer::data_dir ddir(argv[1]);
    data_importer::common_data biomedata(argv[2]);

    float latitude = biomedata.latitude;
    std::cout << "Latitude: " << latitude << std::endl;
	
    int w, h;
    //std::vector<float> data = get_image_data_48bit(argv[1], w, h)[0];
    //std::vector<float> data = load_elv(argv[1], w, h);
    auto data = data_importer::load_elv< ValueGridMap<float> >(ddir.dem_fname);
    float minval = *std::min_element(data.begin(), data.end());
    std::for_each(data.begin(), data.end(), [&minval](float &val) { val -= minval; /*val *= 0.3048;*/ });
    std::cout << "done importing data" << std::endl;
    glm::vec3 north(0, 0, -1);

    //data.getDim(w, h);

    //terrain ter(data.data(), w, h, glm::vec3(0, 0, -1));
    terrain ter(data, north, latitude);

    sunsim sim(ter, true, 6000, 6000);

    sim.renderblock(false);

    //sim.write_shaded_png_8bit("/home/konrad/shading_test.png");
    //sim.write_shaded_monthly_txt("/home/konrad/monthly_sunmap_test.txt");
    sim.write_shaded_monthly_txt(ddir.sun_fname, 0.9144f);

	return 0;
}
