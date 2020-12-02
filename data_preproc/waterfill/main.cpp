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




#include "waterfill.h"
#include "moisture.h"
#include "extract_png.h"
#include "data_importer.h"
#include "common/basic_types.h"
#include <png.h>
#include <fstream>

#include <iostream>

#include <stdexcept>


struct terrain_info
{
    terrain_info(std::vector<float> &terrain,
                 float step,
                 int gw,
                 int gh,
                 float lat)
        :
          terrain(terrain),
          step(step),
          gw(gw),
          gh(gh),
          lat(lat)
    {}

    terrain_info()
    {}

    std::vector<float> terrain;
    float step;
    int gw, gh;
    float lat;
};

struct mflow_info
{
    std::vector<float> mflow;
    int width, height;
};

std::vector<float> calc_average_vecs(const std::vector<std::vector<float> > &moisture)
{
    if (moisture.size() == 0)
        return {};
    std::vector<float> avgs(moisture[0].size());
    for (int i = 0; i < moisture[0].size(); i++)
    {
        float sum = 0.0f;
        for (int m = 0; m < moisture.size(); m++)
        {
            sum += moisture[m][i];
        }
        avgs[i] = sum / moisture.size();
    }
    return avgs;
}

std::vector<uint16_t> get_16bit_img_greyscale_data(std::vector<float> &data, int width, int height)
{
    std::vector<uint16_t> img_data(width * height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            uint32_t val = data[idx];
            if (val > std::numeric_limits<uint16_t>::max())
            {
                val = std::numeric_limits<uint16_t>::max();
            }
            img_data[idx] = val;
        }
    }
    return img_data;
}

bool write_to_greyscale_png(std::string filename, std::vector<float> &raw_data, int width, int height)
{
    auto write_png = [](std::string str, const std::vector<uint16_t> &data, int width, int height)
    {
        FILE *fp = fopen(str.c_str(), "wb");

        if (!fp)
        {
            std::cout << "Error: Could not open image at " << str << std::endl;
            return false;
        }

        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png) return false;

        png_infop info = png_create_info_struct(png);

        if (setjmp(png_jmpbuf(png))) return false;

        png_init_io(png, fp);

        uint8_t *row_ptrs[height];
        for (int row = 0; row < height; row++)
        {
            row_ptrs[row] = (unsigned char *)&data.data()[row * width];
        }

        png_set_IHDR(
                png,
                info,
                width, height,
                16,
                PNG_COLOR_TYPE_GRAY,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT
        );
        png_write_info(png, info);

        png_write_image(png, row_ptrs);
        png_write_end(png, NULL);

        fclose(fp);

        return true;
    };

    auto img_data = get_16bit_img_greyscale_data(raw_data, width, height);

    return write_png(filename, img_data, width, height);
}
/*
void read_params(std::string biome_filename, std::string climate_filename, sim_info &sinfo)
{
    int nb, nc;
    std::ifstream infile;
    float o1, o2, z1, z2, i1, i2;

    infile.open((char *) biome_filename.c_str(), std::ios_base::in);
    if(infile.is_open())
    {
        infile >> name;

        // plant functional types
        infile >> nb; // number of pft
        for(int t = 0; t < nb; t++)
        {
            //PFType pft;
            std::string pftcode;
            float junk;
            std::string shapestr;

            infile >> pftcode >> junk >> junk >> junk >> junk >> junk >> junk >> junk;
            infile >> shapestr;

            // viability response values
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            //pft.sun.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            //pft.wet.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            //pft.temp.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            //pft.slope.setValues(o1, o2, z1, z2, i1, i2);

            //infile >> pft.alpha;
            //infile >> pft.maxage;

            //< growth parameters
            infile >> pft.grow_months;
            infile >> pft.grow_m >> pft.grow_c1 >> pft.grow_c2;

            //< allometry parameters
            infile >> pft.alm_m >> pft.alm_c1;
            infile >> pft.alm_rootmult;
        }

        // category names
        infile >> nc; // number of categories
        for(int c = 0; c < nc; c++)
        {
            std::string str;
            infile >> str;
            //catTable.push_back(str);
        }

        // soil moisture parameters
        infile >> sinfo.slopethresh;
        infile >> sinfo.slopemax;
        infile >> sinfo.evap;
        infile >> sinfo.runofflim;
        infile >> sinfo.soilsat;
        infile >> sinfo.riverlevel;

        infile.close();
        return true;
    }
    else
    {
        cerr << "Error Biome::read: unable to open file" << filename << endl;
        return false;
    }

    infile.open((char *)climate_filename.c_str(), std::ios_base::in);
    if (infile.is_open())
    {
        std::string line;
        std::getline(infile, line);
        std::getline(infile, line);
        std::getline(infile, line);
        std::stringstream strs(line);

        std::string valstr;

        for (int i = 0; i < 12; i++)
        {
            std::getline(strs, valstr, ' ');
            sinfo.rainfall.push_back(std::strtof(valstr));
        }
    }

}

void get_elv_data(std::string elv_filename, terrain_info &terinfo)
{
    std::ifstream ifs(elv_filename);

    if (ifs.is_open())
    {
        ifs >> terinfo.gw >> terinfo.gh;
        ifs >> terinfo.step >> terinfo.lat;
        for (int x = 0; x < terinfo.gw; x++)
        {
            for (int y = 0; y < terinfo.gh; y++)
            {
                float val;
                infile >> val;
                terinfo.terrain[y * terinfo.gh + x] = val * 0.3048f; // convert from feet to metres
            }
        }
    }
}

mflow_info runsim(const std::vector<float> &img_data, int width, int height, int twidth, int theight, sim_info &sinfo)
{
    std::vector<std::vector<float> > result;
    MoistureSim msim(img_data, width, height, twidth, theight);
    msim.simSoilCycle(sinfo.rainfall, sinfo.slopethresh, sinfo.slopemax, sinfo.evap,
                      sinfo.runofflim, sinfo.soilsat, sinfo.riverlevel, result);
    mflow_info minfo;

    minfo.mflow = calc_average_vecs(result);
    minfo.width = width;
    minfo.height = height;
    return minfo;
}

mflow_info runsim(const std::vector<float> &img_data, int width, int height, int twidth, int theight,
                          std::string biome_filename, std::string climate_filename)
{
    sim_info sinfo;
    read_params(biome_filename, climate_filename, sinfo);
    return runsim(img_data, width, height, twidth, theight, sinfo);
}

mflow_info runsim(std::string elv_filename, std::string biome_filename, std::string climate_filename)
{
    terrain_info terinfo;
    get_elv_data(elv_filename, terinfo);
    return runsim(terinfo.terrain, terinfo.gw, terinfo.gh, terinfo.gw * terinfo.step, terinfo.gh * terinfo.step, biome_filename, climate_filename);
}

void write_mflow(std::string out_filename, mflow_info &mflow)
{
    std::ofstream ofs(out_filename);

    const int width = mflow.width;
    const int height = mflow.height;

    ifs << width << " " << height << "\n";
    for (int i = 0; i < width * height; i++)
    {
        ofs << mflow.mflow[i];
        if (i < width * height - 1)
            ofs << " ";
    }
    ofs.close();
}
*/

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

/*
bool write_to_monthly_txt(std::string filename, std::vector< std::vector<float > > &mmap, int width, int height)
{
    std::ofstream ofs(filename);

    if (ofs.is_open() && ofs.good())
    {
        ofs << width << " " << height << "\n";
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int m = 0; m < 12; m++)
                {
                    ofs << mmap[m][y * width + x] << " ";
                }
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}
*/

int main(int argc, char * argv [])
{
    if (argc != 3)
    {
        //std::cout << "Usage: ./main <input txt> <rainfall file (clim)> <soil condition file (biome)> <output txt>" << std::endl;
        std::cout << "Usage: waterfill <data directory> <db filename>" << std::endl;
        return 1;
    }

	float mod = 0.3048 * 3;

    int width, height;
    float rw, rh;
    //std::vector<float> img_data = get_image_data_48bit("/home/konrad/PhDStuff/data/heightmap_png_cuts/H2.png", width, height)[0];
    //std::vector<float> img_data = get_image_data_48bit(argv[1], width, height)[0];
    //std::vector<float> img_data = load_elv(argv[1], width, height);

    data_importer::data_dir ddir(argv[1]);

    auto img_data = data_importer::load_elv< ValueGridMap<float> >(ddir.dem_fname);
    img_data.getDim(width, height);
    img_data.getDimReal(rw, rh);
    std::cout << "image width, height: " << width << ", " << height << std::endl;
    std::cout << "Real width, height: " << rw << ", " << rh << std::endl;
    /*
    for (int  y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            std::cout << img_data[y * width + x] << " ";
        }
        std::cout << std::endl;
    }
    */
    std::cout << "width: " << width << std::endl;
    /*
    auto wf = WaterFill(img_data, width, height, width * mod, height * mod);
    wf.setAbsorbsion(5000.0f);
    wf.reset();
    wf.compute();
    wf.compute();
    wf.expandRivers(50.0f, 1.0f);
    wf.write_to_greyscale_png("/home/konrad/PhDStuff/data/H2_mflow_eco.png");
    */
    std::cout << "Calculating moisture flow data for " << argv[1] << "..." << std::endl;

    //sim_info info;
    //data_importer::read_rainfall(ddir.clim_fname, info);
    //data_importer::read_soil_params(ddir.biome_fname, info);
    data_importer::common_data data(argv[2]);

    MoistureSim msim(img_data, mod, mod);
    auto rainfall = std::vector<float> (12);
    auto &r = rainfall;
    r[0] = 114.3;
    r[1] = 124.46;
    r[2] = 91.44;
    r[3] = 35.56;
    r[4] = 15.24;
    r[5] = 3.81;
    r[6] = 0.0f;
    r[7] = 0.0f;
    r[8] = 6.35;
    r[9] = 27.94;
    r[10] = 76.2;
    r[11] = 115.57;
    float slopethresh = 5.0f;
    //float slopemax = 55.0f;
    float slopemax = 75.0f;
    float evap = 0.25f;
    float runofflim = 100.0f;
    //float soilsat = 180.0f;
    float soilsat = 180.0f;
    float riverlevel = 5000.0f;
    std::vector< ValueGridMap<float> > dest;
    msim.simSoilCycle(data.rainfall,
                      data.soil_info.slopethresh,
                      data.soil_info.slopemax,
                      data.soil_info.evap,
                      data.soil_info.runofflim,
                      data.soil_info.soilsat,
                      data.soil_info.riverlevel,
                      dest);
    bool success;

    for (auto &gm : dest)
    {
        gm.setDimReal(rw, rh);
    }
    /*
    auto avgs = calc_average_vecs(dest);
    //write_to_greyscale_png("/home/konrad/PhDStuff/H2_mflow_eco.png", avgs, width, height);
    success = write_to_greyscale_png(argv[2], avgs, width, height);

    if (success)
        std::cout << "Wrote moisture flow data to " << argv[2] << std::endl;
    else
        std::cout << "Failed to write moisture flow data to " << argv[2] << std::endl;
    */

    //success = write_to_monthly_txt(monthly_file, dest, width, height);
    data_importer::write_monthly_map(ddir.wet_fname, dest);

	return 0;
}


/*
int main(int argc, char * argv [])
{
    if (argc != 5)
    {
        std::cout << "usage: ./main <elv_filename> <biome_filename> <climate_filename> <mflow_filename>" << std::endl;
        return 1;
    }

    mflow_info mflow = runsim(argv[1], argv[2], argv[3]);

    write_mflow(argv[4], mflow);

    return 0;
}
*/
