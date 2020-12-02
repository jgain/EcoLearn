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


#include "data_importer.h"
#include "common/basic_types.h"

#include <algorithm>
#include <cmath>

void compute_temp(const ValueGridMap<float> &input, std::array<float, 12> basetemps, float alttemp, std::vector<ValueGridMap<float> > &output)
{
    if (output.size() != 12)
    {
        output.resize(12);
    }

    std::for_each(output.begin(), output.end(), [&input](ValueGridMap<float> &outmap){
        outmap.setDim(input);
        outmap.setDimReal(input);
    });

    int width, height;
    input.getDim(width, height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int m = 0; m < 12; m++)
            {
                float temp = input.get(x, y) / 1000.0f * (-fabs(alttemp)) + basetemps[m];
                output[m].set(x, y, temp);
            }
        }
    }
}

int main(int argc, char * argv [])
{
    if (argc != 3)
	{
        //std::cout << "usage: temp_compute <DEM elv> <climate file> <lapse rate temp> <OUTPUT txt>" << std::endl;
        std::cout << "usage: temp_compute <data directory> <db filename>" << std::endl;
        return 1;
    }

    data_importer::common_data simdata(argv[2]);
    data_importer::data_dir ddir(argv[1]);

    ValueGridMap<float> dem = data_importer::load_elv< ValueGridMap<float> >(ddir.dem_fname);
    //std::vector<float> temps = data_importer::read_temperature(ddir.clim_fname);
    //float basetemp = std::stof(argv[2]);

    //float alttemp = std::stof(argv[2]);
    float alttemp = simdata.temp_lapse_rate;
    std::cout << "Temp lapse rate: " << alttemp << std::endl;
    std::vector<ValueGridMap<float> > tempmap;

    compute_temp(dem, simdata.temperature, alttemp, tempmap);

    //data_importer::write_txt(argv[4], &tempmap);
    data_importer::write_monthly_map(ddir.temp_fname, tempmap);

	return 0;
}
