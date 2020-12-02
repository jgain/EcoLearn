// By K.P Kapp
// April/May 2019

#include "data_importer.h"
#include "basic_types.h"

#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <map>
#include <sstream>
#include <string>
#include <cassert>
#include <sqlite3.h>
const std::array<std::string, 12> months_arr = {"January",
                                        "February",
                                        "March",
                                        "April",
                                        "May",
                                        "June",
                                        "July",
                                        "August",
                                        "September",
                                        "October",
                                        "November",
                                        "December"};


std::map<std::string, int> make_monthmap()
{
    std::map<std::string, int> monthmap;
    for (int i = 0; i < months_arr.size(); i++)
    {
        monthmap[months_arr[i]] = i;
    }
    return monthmap;
}

const std::map<std::string, int> monthmap = make_monthmap();

/*
 * exactly like load_elv, except that we do not read the step and latitude values after width and height, and we do not account for feet to meters conversion
 */
basic_types::MapFloat data_importer::load_txt(std::string filename, int &width, int &height)
{
    using namespace std;
    using namespace basic_types;

    MapFloat retmap;

    int dx, dy;

    float val;
    ifstream infile;

    std::vector<std::string> file_contents;

    infile.open((char *) filename.c_str(), ios_base::in);
    if (infile.is_open())
    {
        int lnum = 0;
        while (infile.good())
        {
            std::string line;
            std::getline(infile, line);
            std::stringstream linestream(line);
            while (linestream.good())
            {
                std::string token;
                std::getline(linestream, token, ' ');
                auto token_end = token.end();
                auto token_begin = token.begin();
                //auto erase_iter = std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return iscntrl(ch) || isspace(ch); });
                auto erase_iter = std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch); });

                token.erase(erase_iter, token.end());
                if (token.size() > 0)
                    file_contents.push_back(token);
            }
        }
    }
    else
    {
        throw runtime_error("Error data_importer::load_txt: unable to open file " + filename);
    }

    width = atoi(file_contents[0].c_str());
    height = atoi(file_contents[1].c_str());

    if (file_contents.size() > width * height + 2)
    {
        std::vector<std::string> junk_tokens;
        junk_tokens.insert(junk_tokens.begin(), std::next(file_contents.begin(), width * height - 3), file_contents.end());

        std::string errstring = std::string("Error: txt file ") + filename + " contains " + std::to_string(junk_tokens.size()) + " residual elements. ";
        errstring += "The first " + std::to_string(std::min(5, (int)junk_tokens.size())) + " of which are the following: ";
        for (int i = 0; i < std::min(5, (int)junk_tokens.size()); i++)
        {
            errstring += junk_tokens[i] + " ";
        }
        throw runtime_error(errstring);
    }
    else if (file_contents.size() < width * height + 2)
    {
        throw runtime_error("Error: File " + filename + " does not contain the specified amount of elements. File probably corrupted.");
    }

    retmap.setDim(width, height);

    int idx = 0;
    for (auto iter = std::next(file_contents.begin(), 2); iter != file_contents.end(); advance(iter, 1), idx++)
    {
        int x = idx % width;
        int y = idx / width;
        retmap.set(x, y, atof(iter->c_str()));
    }

    return retmap;
}


using namespace basic_types;

MapFloat data_importer::load_elv(std::string filename, int &width, int &height)
{
    using namespace std;

    float step, lat;
    int dx, dy;

    float val;
    ifstream infile;

    MapFloat retmap;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> dx >> dy;
        infile >> step;
        infile >> lat;
        width = dx;
        height = dy;
        retmap.setDim(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                infile >> val;
                retmap.set(x, y, val * 0.3048f);
            }
        }
        infile.close();
    }
    else
    {
        throw runtime_error("Error data_importer::load_elv: unable to open file " + filename);
    }
    return retmap;
}

std::vector< MapFloat > data_importer::read_monthly_map(std::string filename, int &width, int &height)
{
    std::ifstream ifs(filename);

    std::vector< MapFloat > mmap(12);

    if (ifs.is_open())
    {
        ifs >> width >> height;
        if (width > 0 && width <= 10240 && height > 0 && height <= 10240)
        {
            std::for_each(mmap.begin(), mmap.end(), [&width, &height](MapFloat &mapf) { mapf.setDim(width, height);});
        }
        else
        {
            throw std::runtime_error("Size of imported monthly map is either negative, zero, too large (or file is corrupted)");
        }

        int m = 0, x = 0, y = 0;
        while (ifs.good())
        {
            float val;
            ifs >> val;
            //std::cout << "m , x, y, val: " << m << ", " << x << ", " << y << ", " << val << std::endl;
            //assert(m >= 0 && m < 12 && x >= 0 && x < width && y >= 0 && y < height);
            //ifs >> mmap[m][y * width + x];
            mmap[m].set(x, y, val);
            m++;
            if (m == 12)
            {
                m = 0;
                y++;
                if (y == height)
                {
                    y = 0;
                    x++;
                }
            }
            if ((x * height + y) * 12 + m >= width * height * 12)
            {
                break;
            }
        }
    }
    else
    {
        throw std::runtime_error("Could not open monthly map at " + filename);
    }
    return mmap;
}

MapFloat data_importer::average_mmap(std::vector< MapFloat > &mmap)
{
    if (mmap.size() != 12)
    {
        throw std::runtime_error("Monthly map does not have twelve maps");
    }
    int width = -1, height = -1;
    for (auto &m : mmap)
    {
        int curr_w, curr_h;
        m.getDim(curr_w, curr_h);
        if (width == -1 || (curr_w == width && curr_h == height))
        {
            width = curr_w, height = curr_h;
        }
        else
        {
            throw std::runtime_error("monthly map does not contain maps of the same size");
        }
    }

    MapFloat avgmap;
    avgmap.setDim(width, height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int m = 0; m < 12; m++)
                avgmap.set(x, y, avgmap.get(x, y) + mmap[m].get(x, y));
            avgmap.set(x, y, avgmap.get(x, y) / 12.0f);
        }
    }

    return avgmap;
}

MapFloat data_importer::get_temperature_map(MapFloat &heightmap, float basetemp, float reduce_per_meter)
{
    int width, height;
    heightmap.getDim(width, height);
    MapFloat retmap;
    retmap.setDim(width, height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            retmap.set(x, y, basetemp - reduce_per_meter * heightmap.get(x, y));
        }
    }
    return retmap;
}

void data_importer::normalize_data(MapFloat &data)
{
    int width, height;
    data.getDim(width, height);
    float mean = 0.0f;
    std::for_each(data.begin(), data.end(), [&mean](float &val) { mean += val; });
    mean /= (width * height);
    float stddev = 0.0f;
    std::for_each(data.begin(), data.end(), [&stddev, &mean](float &val) { stddev += (val - mean) * (val - mean); });
    stddev = sqrt(stddev / (width * height));

    std::cout << "mean, stddev: " << mean << ", " << stddev << std::endl;

    float minval = *std::min_element(data.begin(), data.end());
    float maxval = *std::max_element(data.begin(), data.end(), [&stddev, &mean](float &val1, float &val2) { if (fabs(val2 - mean) > stddev * 3) return false; else return val2 > val1; });
    if (maxval == minval)
    {
        std::fill(data.begin(), data.end(), 0.0f);
        return;
    }
    maxval += (maxval - minval) * 0.001f;
    std::for_each(data.begin(), data.end(), [&maxval](float &val) { if (val >= maxval) val = maxval;});

    std::for_each(data.begin(), data.end(), [&minval, &maxval](float &val) { val = (val - minval) / (maxval - minval); });
}

void data_importer::eliminate_outliers(MapFloat &data)
{
    int width, height;
    data.getDim(width, height);
    float mean = std::accumulate(data.begin(), data.end(), 0.0f);
    mean = mean / (width * height);

    float stddev = std::accumulate(data.begin(), data.end(), 0.0f, [&mean](float &sum, float &val) { return sum + (val - mean) * (val - mean); });
    stddev = sqrt(stddev / (width * height));

    std::for_each(data.begin(), data.end(), [&mean, &stddev](float &val) { if (val > mean + 3 * stddev) val = mean; });
}

bool data_importer::read_pdb(std::string filename, std::map<int, std::vector<MinimalPlant> > &retvec)
{
    //std::vector< std::vector<basic_plant> > retvec;
    std::ifstream infile;
    int numcat, skip;

    infile.open(filename, std::ios_base::in);
    if(infile.is_open())
    {
        // list of prioritized categories, not all of which are used in a particular sandbox
        infile >> numcat;
        //retvec.resize(numcat);
        for(int c = 0; c < numcat; c++)
        {
            float junk;
            int cat;
            int nplants;

            infile >> cat;
            for (int i = 0; i < 3; i++)
                infile >> junk;	// skip minheight, maxheight, and avgCanopyRadToHeightRatio

            infile >> nplants;
            retvec[cat].resize(nplants);
            for (int plnt_idx = 0; plnt_idx < nplants; plnt_idx++)
            {
                float x, y, z, radius, height;
                infile >> x >> y >> z;
                infile >> height;
                infile >> radius;
                MinimalPlant plnt = {x, y, height, radius, false, cat};
                retvec[cat][plnt_idx] = plnt;
                /*
                int height_bin = get_height_bin(height, height_bins);
                if (height_bin < 0)
                    continue;
                else if (height_bin == height_bins.size() - 1)
                {
                    std::cerr << "Warning: height is higher than supposed maximum height. Assigning to maximum height bin" << std::endl;
                }
                */
            }
        }
        std::cerr << std::endl;


        infile.close();
        return true;
    }
    else
        return false;
}



std::vector<int> data_importer::get_nonzero_idxes(MapFloat &data)
{
    int width, height;
    data.getDim(width, height);
    std::vector<int> nonzero_idxes;
    nonzero_idxes.reserve(width * height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (data.get(x, y) > 0)
                nonzero_idxes.push_back(data.flatten(x, y));
        }
    }
}

/*
std::vector<species_params> data_importer::read_species_params_before_specassign(std::string params_filename)
{
    std::ifstream ifs(params_filename);

    std::vector<species_params> all_params;

    if (ifs.is_open())
    {
        while (ifs.good())
        {
            float a, b;
            std::string line;
            std::getline(ifs, line);
            std::stringstream sstr(line);
            int i = 0;
            for (; i < 2 && sstr.good(); i++)
            {
                std::string token;
                std::getline(sstr, token, ' ');
                switch(i)
                {
                    case (0):
                        a = atof(token.c_str());
                        break;
                    case (1):
                        b = atof(token.c_str());
                        break;
                }
            }
            if (i < 2)	// there were less than the desired amount of values on the line we just read - therefore skip these values
            {
                continue;
            }
            all_params.emplace_back(a, b);
        }
    }
    return all_params;
}
*/

static std::string get_name_from_line(std::string line)
{
    std::stringstream sstr(line);
    std::string name;
    std::getline(sstr, name, ' ');
    return name;
}

static species_params parse_species_line(std::string line, std::string name)
{
    std::vector<float> locs, width_scales;
    std::stringstream sstr = std::stringstream(line);
    std::string astr, bstr, percstr;
    std::getline(sstr, astr, ' ');
    std::getline(sstr, bstr, ' ');
    std::getline(sstr, percstr, ' ');
    float a, b, perc;
    a = std::stof(astr);
    b = std::stof(bstr);
    perc = std::stof(percstr);

    int i = 0;
    for (; sstr.good(); i++)
    {
        float location, width_scale;
        std::string token;
        std::getline(sstr, token, ' ');
        int mod = i % 2;
        switch(mod)
        {
        case (0):
            location  = atof(token.c_str());
            locs.push_back(location);
            break;
        case (1):
            width_scale = atof(token.c_str());
            width_scales.push_back(width_scale);
            break;
        default:
            break;
        }
    }
    while (locs.size() > width_scales.size())
    {
        locs.pop_back();
    }
    while (width_scales.size() > locs.size())
    {
        width_scales.pop_back();
    }

    return species_params(name, a, b, locs, width_scales, perc);
}

std::vector<species_params> data_importer::read_species_params(std::string params_filename)
{
    std::ifstream ifs(params_filename);

    std::vector<species_params> all_params;
    std::vector< std::string > all_lines;

    if (ifs.is_open())
    {
        while (ifs.good())
        {
            std::string line;
            std::getline(ifs, line);
            all_lines.push_back(line);
        }
    }
    else
    {
        std::string errstr = "Cannot open file at ";
        errstr += params_filename + " ";
        errstr += "in function data_importer::read_species_params";
        throw std::runtime_error(errstr);
    }

    int nspecies;
    nspecies = std::stoi(all_lines[1]);
    ifs >> nspecies;
    for (int curr_spc = 0, curr_line = 4; curr_spc < nspecies && curr_line < all_lines.size(); curr_spc++, curr_line += 3)
    {
        std::string line = all_lines[curr_line];
        std::string nameline = all_lines[curr_line - 2];
        std::string name = get_name_from_line(nameline);
        species_params params = parse_species_line(line, name);
        all_params.push_back(params);
    }
    return all_params;
}

void data_importer::write_species_params_after_specassign(std::string params_filename, std::vector<species_params> all_params)
{
    std::vector< std::string > file_lines;
    std::ifstream ifs(params_filename);

    if (ifs.is_open())
    {
        while (ifs.good())
        {
            std::string line;
            std::getline(ifs, line);
            file_lines.push_back(line);
        }
    }
    else
    {
        std::string errstr = "";
        errstr += "File ";
        errstr += params_filename + " could not be opened in function data_importer::write_species_params_after_specassign";
        throw std::runtime_error(errstr);
    }

    while (file_lines.back().size() == 0)
        file_lines.pop_back();

    int nspecies = std::stoi(file_lines[1]);
    for (int linenum = 4, idx = 0; linenum < file_lines.size() && idx < nspecies; linenum += 3, idx++)
    {
        std::string name_line = file_lines[linenum - 2];
        std::string params_line = file_lines[linenum];

        std::string name = get_name_from_line(name_line);
        species_params file_params = parse_species_line(params_line, name);

        auto params = all_params[idx];

        float actual_perc = params.percentage;
        params.percentage = file_params.percentage;

        assert(file_params.name == params.name && abs(file_params.a - params.a) < 1e-5 && abs(file_params.b - params.b) < 1e-5);

        file_lines[linenum] = params.to_string(true);

        params.percentage = actual_perc;
    }

    ifs.close();

    std::ofstream ofs(params_filename);

    if (ofs.is_open())
    {
        for (auto &line : file_lines)
        {
            std::string outstring = line + "\n";
            ofs << outstring;
        }
    }
    else
    {
        std::string errstr = std::string("Cannot write to file ") + params_filename + " in function data_importer::write_species_params_after_specassign";
        throw std::runtime_error(errstr.c_str());
    }
    /*
    if (ofs.is_open())
    {
        for (auto &params : all_params)
        {
            ofs << params.a << " " << params.b << " ";
            for (int i = 0; i < params.locs.size(); i++)
            {
                ofs << params.locs[i] << " " << params.width_scales[i] << " ";
            }
            ofs << params.percentage << std::endl;
        }
    }
    else
    {
        std::string errstr = std::string("Cannot write to file ") + params_filename;
        throw std::runtime_error(errstr.c_str());
    }
    */
}

void data_importer::read_rainfall(std::string filename, sim_info &info)
{
    float elv, val;
    std::ifstream infile;


    infile.open((char *) filename.c_str(), std::ios_base::in);
    if(infile.is_open())
    {
        info.rainfall.resize(12);

        /*
        std::string line;
        std::getline(infile, line);
        std::getline(infile, line);
        std::getline(infile, line);
        */

        infile >> elv;

        std::string junk;
        for (int i = 0; i < 24; i++)
        {
            infile >> junk;
        }

        // rainfall
        for(int m = 0; m < 12; m++)
        {
            infile >> val;
            info.rainfall[m] = val;
        }

        infile.close();
    }
    else
    {
        throw std::runtime_error("Error data_importer::read_rainfall: unable to open file");
    }
}

void data_importer::read_soil_params(std::string filename, sim_info &info)
{
    int nb, nc;
    std::ifstream infile;

    infile.open((char *) filename.c_str(), std::ios_base::in);
    if(infile.is_open())
    {
        std::string name;
        infile >> name;

        // plant functional types
        infile >> nb; // number of pft
        for(int t = 0; t < nb; t++)
        {
            std::string shapestr;
            std::string junk;

            std::getline(infile, junk);
            std::getline(infile, junk);
            std::getline(infile, junk);

            //infile >> pft.code >> pft.basecol[0] >> pft.basecol[1] >> pft.basecol[2] >> pft.draw_hght >> pft.draw_radius >> pft.draw_box1 >> pft.draw_box2;
            //infile >> shapestr;

            /*
            for (int i = 0; i < 9; i++)
            {
                infile >> junk;
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    infile >> junk;
                }
            }
            */

            /*
            // viability response values
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.sun.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.wet.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.temp.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.slope.setValues(o1, o2, z1, z2, i1, i2);
            */

            // -------------------------------------------------------- BELOW NOT RELEVANT??

            //infile >> pft.alpha;
            //infile >> pft.maxage;

            //< growth parameters
            /*
            infile >> pft.grow_months;
            infile >> pft.grow_m >> pft.grow_c1 >> pft.grow_c2;

            //< allometry parameters
            infile >> pft.alm_m >> pft.alm_c1;
            infile >> pft.alm_rootmult;*/
        }

        // category names
        infile >> nc; // number of categories
        for(int c = 0; c < nc; c++)
        {
            std::string str;
            //infile >> str;
            //catTable.push_back(str);
            std::getline(infile, str);
        }

        // soil moisture parameters
        infile >> info.slopethresh;
        infile >> info.slopemax;
        infile >> info.evap;
        infile >> info.runofflim;
        infile >> info.soilsat;
        infile >> info.riverlevel;

        infile.close();
    }
    else
    {
        throw std::runtime_error("Error data_importer::read_soil_params: unable to open file");
    }
}

std::vector<float> data_importer::read_temperature(std::string filename)
{
    float elv, val;
    std::ifstream infile;


    infile.open((char *) filename.c_str(), std::ios_base::in);
    if(infile.is_open())
    {
        //info.rainfall.resize(12);
        std::vector<float> temperature(12);

        /*
        std::string line;
        std::getline(infile, line);
        std::getline(infile, line);
        std::getline(infile, line);
        */

        infile >> elv;

        // temperature
        for(int m = 0; m < 12; m++)
        {
            infile >> val;
            temperature[m] = val;
        }

        infile.close();

        return temperature;
    }
    else
    {
        throw std::runtime_error("Error data_importer::read_temperature: unable to open file");
    }

}

static void assign_viability(data_importer::viability &sp_vb, const data_importer::viability &vb)
{
    sp_vb.cmax = vb.cmax;
    sp_vb.cmin = vb.cmin;
    sp_vb.c = (vb.cmax + vb.cmin) / 2.0f;
    sp_vb.r = vb.cmax - vb.cmin;
}

static int sql_callback_common_data_monthlies(void *write_info, int argc, char **argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;
    int monthidx;
    float rainfall, temperature, cloudiness;

    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Month_ID")
        {
            monthidx = std::stoi(valstr);
        }
        else if (colstr == "Rainfall")
        {
            rainfall = std::stof(valstr);
        }
        else if (colstr == "Temperature")
        {
            temperature = std::stof(valstr);
        }
        else if (colstr == "Cloudiness")
        {
            cloudiness = std::stof(valstr);
        }
    }
    common->rainfall[monthidx] = rainfall;
    common->temperature[monthidx] = temperature;
    common->cloudiness[monthidx] = cloudiness;

    return 0;
}

static int sql_callback_common_data_biome_stats(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    std::string statname;
    float value;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Biome_stat_name")
        {
            statname = valstr;
        }
        else if (colstr == "Value")
        {
            value = std::stof(valstr);
        }
    }
    if (statname == "Slope threshold")
    {
        common->soil_info.slopethresh = value;
    }
    else if (statname == "Slope maximum")
    {
        common->soil_info.slopemax = value;
    }
    else if (statname == "Evaporation rate")
    {
        common->soil_info.evap = value;
    }
    else if (statname == "Runoff limit")
    {
        common->soil_info.runofflim = value;
    }
    else if (statname == "Soil saturation limit")
    {
        common->soil_info.soilsat = value;
    }
    else if (statname == "Riverlevel")
    {
        common->soil_info.riverlevel = value;
    }
    else if (statname == "Latitude")
    {
        common->latitude = value;
    }
    else if (statname == "Temp lapse rate")
    {
        common->temp_lapse_rate = value;
    }
    return 0;

}

static int sql_callback_common_data_subbiomes(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    data_importer::sub_biome sb;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Sub_biome_ID")
        {
            sb.idx = std::stoi(valstr);
        }
        else if (colstr == "Value")
        {
            sb.name = valstr;
        }
    }
    common->subbiomes[sb.idx] = sb;
    common->subbiomes_all_species[sb.idx] = sb;
    return 0;
}

static int sql_callback_common_data_subbiomes_species(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int sb_idx, tree_idx;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Sub_biome_ID")
        {
            sb_idx = std::stoi(valstr);
        }
        else if (colstr == "Tree_ID")
        {
            tree_idx = std::stoi(valstr);
        }
    }
    data_importer::sub_biome &sb = common->subbiomes[sb_idx];
    data_importer::species_encoded sp;
    sp.idx = tree_idx;
    sb.species.insert(sp);
    return 0;
}

static int sql_callback_common_data_subbiomes_all_species(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int sb_idx, tree_idx;
    bool canopy;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Sub_biome_ID")
        {
            sb_idx = std::stoi(valstr);
        }
        else if (colstr == "Tree_ID")
        {
            tree_idx = std::stoi(valstr);
        }
        else if (colstr == "Canopy")
        {
            int canopyval = std::stoi(valstr);
            if (canopyval)
            {
                std::cout << "Encountered canopy species" << std::endl;
                canopy = true;
            }
            else
                canopy = false;
        }
    }
    data_importer::sub_biome &sb = common->subbiomes_all_species[sb_idx];
    data_importer::species_encoded sp;
    sp.idx = tree_idx;
    sp.canopy = canopy;
    auto insert_result = sb.species.insert(sp);
    if (!insert_result.second)
    {
        auto sp_iter = sb.species.find(sp);

        assert(canopy || sp_iter->canopy);

        std::cout << "Inserting canopy species" << std::endl;

        data_importer::species_encoded sp = *sp_iter;
        sb.species.erase(sp_iter);
        sp.canopy = true;
        sb.species.insert(sp);

    }
    return 0;
}

static int sql_callback_common_data_species(void *write_info, int argc, char ** argv, char **colnames)
{
    std::cout << "Processing row in sql_callback_common_data_species" << std::endl;

    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int tree_idx;
    float a, b;
    data_importer::viability sun, moisture, temp, slope;
    float trunkrad;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Tree_ID")
        {
            tree_idx = std::stoi(valstr);
        }
        else if (colstr == "canopy")
        {
            if (valstr != "Y")
            {
                return 0;	// we only look at canopy species. If not a canopy species, return, not writing
                            // the current row to the data struct
            }
        }
        else if (colstr == "a")
        {
            a = std::stof(valstr);
        }
        else if (colstr == "b")
        {
            b = std::stof(valstr);
        }
        else if (colstr == "shadeval")
        {
            sun.cmin = std::stof(valstr);
        }
        else if (colstr == "sunval")
        {
            sun.cmax = std::stof(valstr);
        }
        else if (colstr == "droughtval")
        {
            moisture.cmin = std::stof(valstr);
        }
        else if (colstr == "floodval")
        {
            float val = std::stof(valstr);
            moisture.cmax = std::stof(valstr);
        }
        else if (colstr == "coldval")
        {
            temp.cmin = std::stof(valstr);
            temp.cmax = 35.0f;
        }
        else if (colstr == "slopeval")
        {
            slope.cmax = std::stof(valstr);
            slope.cmin = 0.0f;
        }
        else if (colstr == "Max_trunk_radius")
        {
            trunkrad = std::stof(valstr);
        }
    }
    data_importer::species sp;
    sp.idx = tree_idx;
    sp.a = a;
    sp.b = b;
    sp.max_trunk_radius = trunkrad;
    assign_viability(sp.sun, sun);
    assign_viability(sp.wet, moisture);
    assign_viability(sp.temp, temp);
    assign_viability(sp.slope, slope);
    auto result = common->all_species.insert({sp.idx, sp});
    assert(result.second);		// Each species should only be inserted once. If it already exists in the map
                                // there is a bug
    return 0;
}

static int sql_callback_common_data_all_species(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int tree_idx;
    float a, b;
    float alpha, maxage, maxheight;
    char grow_period;
    int grow_start, grow_end;
    float grow_m, grow_c1, grow_c2;
    float draw_color [4];
    float draw_height;
    float draw_radius, draw_box1, draw_box2;
    std::string name;
    data_importer::treeshape draw_shape;
    data_importer::viability sun, temp, moisture, slope;
    float trunkrad;
    for (int i = 0; i < argc; i++)
    {
        std::string valstr;
        std::string colstr;
        if (argv[i])
        {
            valstr = argv[i];
        }
        else
        {
            valstr = "NULL";
        }
        colstr = colnames[i];

        if (colstr == "Tree_ID")
        {
            tree_idx = std::stoi(valstr);
        }
        /*
        else if (colstr == "canopy")
        {
            if (valstr != "Y")
            {
                return 0;	// we only look at canopy species. If not a canopy species, return, not writing
                            // the current row to the data struct		NOT IN THIS CASE
            }
        }
        */
        else if (colstr == "a")
        {
            a = std::stof(valstr);
        }
        else if (colstr == "b")
        {
            b = std::stof(valstr);
        }
        else if (colstr == "maxage")
        {
            maxage = std::stof(valstr);
        }
        else if (colstr == "maxheight")
        {
            maxheight = std::stof(valstr);
        }
        else if (colstr == "alpha")
        {
            alpha = std::stof(valstr);
        }
        else if (colstr == "Growth_ID")
        {
            grow_period = valstr[0];
        }
        else if (colstr == "Start_month")
        {
            grow_start = std::stoi(valstr);
        }
        else if (colstr == "End_month")
        {
            grow_end = std::stoi(valstr);
        }
        else if (colstr == "grow_m")
        {
            grow_m = std::stof(valstr);
        }
        else if (colstr == "grow_c1")
        {
            grow_c1 = std::stof(valstr);
        }
        else if (colstr == "grow_c2")
        {
            grow_c2 = std::stof(valstr);
        }
        else if (colstr == "base_col_red")
        {
            draw_color[0] = std::stof(valstr);
        }
        else if (colstr == "base_col_green")
        {
            draw_color[1] = std::stof(valstr);
        }
        else if (colstr == "base_col_blue")
        {
            draw_color[2] = std::stof(valstr);
        }
        else if (colstr == "draw_height")
        {
            draw_height = std::stof(valstr);
        }
        else if (colstr == "draw_radius")
        {
            draw_radius = std::stof(valstr);
        }
        else if (colstr == "draw_box1")
        {
            draw_box1 = std::stof(valstr);
        }
        else if (colstr == "draw_box2")
        {
            draw_box2 = std::stof(valstr);
        }
        else if (colstr == "draw_shape")
        {
            if (valstr == "CONE")
            {
                draw_shape = data_importer::treeshape::CONE;
            }
            else if (valstr == "SPHR")
            {
                draw_shape = data_importer::treeshape::SPHR;
            }
            else if (valstr == "BOX")
            {
                draw_shape = data_importer::treeshape::BOX;
            }
        }
        else if (colstr == "common_name")
        {
            name = valstr;
        }
        else if (colstr == "shadeval")
        {
            sun.cmin = std::stof(valstr);
        }
        else if (colstr == "sunval")
        {
            sun.cmax = std::stof(valstr);
        }
        else if (colstr == "droughtval")
        {
            moisture.cmin = std::stof(valstr);
        }
        else if (colstr == "floodval")
        {
            moisture.cmax = std::stof(valstr);
        }
        else if (colstr == "coldval")
        {
            temp.cmin = std::stof(valstr);
            temp.cmax = 35.0f;
        }
        else if (colstr == "slopeval")
        {
            slope.cmax = std::stof(valstr);
            slope.cmin = 0.0f;
        }
        else if (colstr == "Max_trunk_radius")
        {
            trunkrad = std::stof(valstr);
        }
    }
    data_importer::species sp;
    sp.idx = tree_idx;
    sp.a = a;
    sp.b = b;
    sp.maxage = maxage;
    sp.maxhght = maxheight;
    sp.alpha = alpha;
    sp.growth_period = grow_period;
    sp.grow_start = grow_start;
    sp.grow_end = grow_end;
    if (grow_start > grow_end)
        sp.grow_months = grow_end + 12 - grow_start + 1;
    else
        sp.grow_months = grow_end - grow_start + 1;
    sp.grow_m = grow_m;
    sp.grow_c1 = grow_c1;
    sp.grow_c2 = grow_c2;
    sp.basecol[0] = draw_color[0], sp.basecol[1] = draw_color[1], sp.basecol[2] = sp.basecol[2];
    sp.basecol[3] = 1.0f;
    sp.draw_hght = draw_height;
    sp.draw_radius = draw_radius;
    sp.draw_box1 = draw_box1;
    sp.draw_box2 = draw_box2;
    sp.name = name;
    sp.max_trunk_radius = trunkrad;

    assign_viability(sp.slope, slope);
    assign_viability(sp.sun, sun);
    assign_viability(sp.temp, temp);
    assign_viability(sp.wet, moisture);

    auto result = common->canopy_and_under_species.insert({sp.idx, sp});
    assert(result.second);		// Each species should only be inserted once. If it already exists in the map
                                // there is a bug
    return 0;
}

static int sql_callback_common_data_check_tables(void *junk, int argc, char ** argv, char **colnames)
{
    for (int i = 0; i < argc; i++)
    {
        std::string value;
        if (argv[i])
            value = argv[i];
        else
            value = "NULL";
        std::cout << colnames[i] << ": " << value << std::endl;
    }
    std::cout << std::endl;
}

static void sql_err_handler(sqlite3 *db, int errcode, char * errmsg)
{
    if (errcode != SQLITE_OK)
    {
        std::cout << "SQL error: " << errmsg << std::endl;
        sqlite3_free(errmsg);
        sqlite3_close(db);
        throw std::runtime_error("SQL SELECT statment error");
    }
}

data_importer::common_data::common_data(std::string db_filename)
{
    sqlite3 *db;
    int errcode;
    std::cout << "Opening database file at " << db_filename << std::endl;
    errcode = sqlite3_open(db_filename.c_str(), &db);
    if (errcode)
    {
        std::string errstr = std::string("Cannot open database file at ") + db_filename;
        sqlite3_close(db);
        throw std::runtime_error(errstr.c_str());
    }
    char * errmsg;
    //errcode = sqlite3_exec(db, "SELECT * FROM sqlite_master WHERE type='table'", sql_callback_common_data_check_tables, this, &errmsg);
    //sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT Tree_ID, \
                                    a, \
                                    b, \
                                    Canopy, \
                                    shadeTolLow.value as shadeval, \
                                    sunTolUpper.value as sunval, \
                                    droughtTolLow.value as droughtval, \
                                    floodTolUpper.value as floodval, \
                                    coldTolLow.value as coldval, \
                                    slopeTolUpper.value as slopeval \
                               FROM species INNER JOIN allometry ON species.Allometry_ID = allometry.Allometry_ID \
                                INNER JOIN shadeTolLow ON species.shade_tol_lower = shadeTolLow.shade_tol_lower \
                                INNER JOIN slopeTolUpper ON species.slope_tol_upper = slopeTolUpper.slope_tol_upper \
                                INNER JOIN coldTolLow ON species.cold_tol_lower = coldTolLow.cold_tol_lower \
                                INNER JOIN sunTolUpper ON species.sun_tol_upper = sunTolUpper.sun_tol_upper \
                                INNER JOIN droughtTolLow ON species.drought_tol_lower = droughtTolLow.drought_tol_lower \
                                INNER JOIN floodTolUpper ON species.flood_tol_upper = floodTolUpper.flood_tol_upper",
                          sql_callback_common_data_species,
                          this,
                          &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT Tree_ID, \
                                    common_name, \
                                    scientific_name, \
                                    form, \
                                    canopy, \
                                    maxage, \
                                    maxheight, \
                                    alpha, \
                                    species.Growth_ID as Growth_ID, \
                                    species.Allometry_ID as Allometry_ID, \
                                    grow_m, \
                                    grow_c1, \
                                    grow_c2, \
                                    base_col_red, \
                                    base_col_green, \
                                    base_col_blue, \
                                    draw_height, \
                                    draw_radius, \
                                    draw_box1, \
                                    draw_box2, \
                                    draw_shape, \
                                    a, \
                                    b, \
                                    Start_month, \
                                    End_month, \
                                    shadeTolLow.value as shadeval, \
                                    sunTolUpper.value as sunval, \
                                    droughtTolLow.value as droughtval, \
                                    floodTolUpper.value as floodval, \
                                    coldTolLow.value as coldval, \
                                    slopeTolUpper.value as slopeval \
                               FROM species INNER JOIN allometry ON species.Allometry_ID = allometry.Allometry_ID \
                               INNER JOIN growth ON species.Growth_ID = growth.Growth_ID \
                                INNER JOIN shadeTolLow ON species.shade_tol_lower = shadeTolLow.shade_tol_lower \
                                INNER JOIN sunTolUpper ON species.sun_tol_upper = sunTolUpper.sun_tol_upper \
                                INNER JOIN droughtTolLow ON species.drought_tol_lower = droughtTolLow.drought_tol_lower \
                                INNER JOIN floodTolUpper ON species.flood_tol_upper = floodTolUpper.flood_tol_upper \
                                INNER JOIN coldTolLow ON species.cold_tol_lower = coldTolLow.cold_tol_lower \
                                INNER JOIN slopeTolUpper ON species.slope_tol_upper = slopeTolUpper.slope_tol_upper",
                          sql_callback_common_data_all_species,
                          this,
                          &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT * FROM monthlies", sql_callback_common_data_monthlies, this, &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT * FROM biome_stats", sql_callback_common_data_biome_stats, this, &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT * FROM subBiomes", sql_callback_common_data_subbiomes, this, &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT Sub_biome_ID, Tree_ID FROM subBiomesMapping WHERE Canopy = 1",
                           sql_callback_common_data_subbiomes_species,
                           this,
                           &errmsg);
    sql_err_handler(db, errcode, errmsg);

    errcode = sqlite3_exec(db, "SELECT * FROM subBiomesMapping",
                           sql_callback_common_data_subbiomes_all_species,
                           this,
                           &errmsg);
    sql_err_handler(db, errcode, errmsg);

    sqlite3_close(db);

    for (auto &subb : subbiomes)
    {
        int subcode = subb.first;
        sub_biome sbiome = subb.second;
        for (auto spec_enc : sbiome.species)
        {
            int speccode = spec_enc.idx;
            canopyspec_to_subbiome[speccode] = subcode;
        }
    }
}


std::vector<basic_tree> data_importer::minimal_to_basic(const std::map<int, std::vector<MinimalPlant> > &plants)
{
    std::vector<basic_tree> trees;

    for (auto &speccls : plants)
    {
        int spec = speccls.first;
        const std::vector<MinimalPlant> &specplants = speccls.second;
        for (const auto &ctree : specplants)
        {
            float x, y, radius, height;
            x = ctree.x;
            y = ctree.y;
            radius = ctree.r;
            basic_tree newtree(x, y, radius, ctree.h);
            newtree.species = spec;
            trees.push_back(newtree);
        }
    }
    return trees;
}

std::map<std::string, data_importer::grass_viability> data_importer::read_grass_viability(std::string filename)
{
    auto get_viability = [](std::stringstream &sstr, grass_viability &v) {
        sstr >> v.absmin;
        sstr >> v.innermin;
        sstr >> v.innermax;
        sstr >> v.absmax;
    };

    std::ifstream ifs(filename);

    if (!ifs.is_open())
    {
        throw std::invalid_argument("Could not open grass viability parameters file at " + filename);
    }

    std::string line;

    std::map<std::string, grass_viability> vs;

    bool moisture_good = false, sun_good = false, temp_good = false;

    for (int i = 0; i < 3; i++)
    {
        std::getline(ifs, line);
        std::stringstream sstr(line);
        std::string token;
        sstr >> token;
        for (auto &ch : token)
            ch = tolower(ch);
        if (token == "moisture")
        {
            moisture_good = true;
        }
        else if (token == "sunlight")
        {
            sun_good = true;
        }
        else if (token == "temperature")
        {
            temp_good = true;
        }
        else
        {
            throw std::invalid_argument("Error parsing grass viability file at " + filename);
        }
        get_viability(sstr, vs[token]);
    }

    if (!(temp_good && sun_good && moisture_good))
    {
        throw std::invalid_argument("Error importing grass viability file at " + filename);
    }

    return vs;
}
