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


#include <data_importer/data_importer.h>
#include <common/basic_types.h>

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

void data_importer::modelset::add_model(treemodel model)
{
    int count = 0;
    for (auto &m : models)
    {
        if (model.hmin < m.hmin)
        {
            models.insert(std::next(models.begin(), count), model);
            break;
            //return;
        }
        count++;
    }
    //models.push_back(model);		// if we return, instead of break above, use this code

    // if we break, instead of return above, use this code
    if (count == models.size())
    {
        models.push_back(model);
    }

    vueid_to_ratio[model.vueid] = model.whratio / 2.0f;

    //setup_ranges();		// if we want the object ready after sql import, this has to happen...
}

void data_importer::modelset::setup_ranges()
{
    ranges.clear();
    selections.clear();
    samplemap.clear();
    for (int i = 0; i < models.size(); i++)
    {
        add_to_ranges(i);
    }
    setup_selections();

    // setup samplemap

    for (int i = 1; i < ranges.size(); i++)
    {
        float cand;
        if ((cand = ranges[i] - ranges[i - 1]) < minrange)
        {
            minrange = cand;
        }
    }
    // let's make binsize 1.0f - if
    // we can enforce that range borders are integers, we will not have bins that stride
    // range borders
    //binsize = 1.0f;


    binsize = minrange;
    nbins = std::ceil(ranges.back() - ranges.front()) / binsize;

    //float binsize = minrange / 5.0f;
    //int nbins = std::ceil((ranges.back() - ranges.front()) / binsize);
    /*
    if (nbins > 200)
    {
        nbins = 200;
        binsize = (ranges.back() - ranges.front()) / nbins;
    }
    */
    samplemap.resize(nbins);

    int curridx = 0;
    samplemap.at(0) = curridx;
    for (int i = 1; i < nbins; i++)
    {
        if (i * binsize >= ranges.at(curridx + 1) - 1e-4f)		// curridx + 1, because ranges[curr_idx + 1] corresponds to selections[curr_idx]
        {
            curridx++;
            if (i * binsize >= ranges.at(curridx + 1) - 1e-4)
            {
                throw std::runtime_error("binsize too big in modelset::setup_ranges");
            }
        }
        samplemap.at(i) = curridx;
    }
}

float data_importer::modelset::sample_rh_ratio(float height, int *vuemodel)
{
    int vm = sample_selection_robust(height);
    if (vuemodel)
    {
        *vuemodel = vm;
    }
    return vueid_to_ratio.at(vm);
}

int data_importer::modelset::sample_selection_robust(float height)
{
    std::vector<int> *sel;
    //binsize = (ranges.back() - ranges.front()) / nbins;
    int idx = height / binsize;
    if (idx >= samplemap.size() || height > ranges.back())
    {
        throw std::runtime_error("height out of range in modelset::sample_selection");
    }
    int selidx = -1;
    if (idx < samplemap.size() - 1)
    {
        if (samplemap.at(idx) != samplemap.at(idx + 1))
        {
            if (height > ranges.at(samplemap.at(idx + 1)))
            {
                selidx = samplemap.at(idx + 1);
            }
        }
        if (selidx == -1)
            selidx = samplemap.at(idx);
    }
    else
        selidx = samplemap.back();

    sel = &selections.at(selidx);

    // std::cout << "selection size: " << sel->size() << std::endl;
    int randidx = rand() % sel->size();
    return sel->at(randidx);
}

// this function makes the assumption that each bin has size 1.0f
int data_importer::modelset::sample_selection_fast(float height)
{
    throw std::runtime_error("modelset::sample_selection_fast not implemented");

    std::vector<int> *sel;
    //float binsize = (ranges.back() - ranges.front()) / nbins;
    int idx = height - ranges.front();
    if (idx >= samplemap.size() || height > ranges.back())
    {
        throw std::runtime_error("height out of range in modelset::sample_selection");
    }
    int selidx = samplemap.at(idx);

    sel = &selections.at(selidx);

    int randidx = rand() % sel->size();
    return sel->at(randidx);
}

int data_importer::modelset::sample_selection_simple(float height)
{
    throw std::runtime_error("modelset::sample_selection_simple not implemented");

    int selidx = -1;
    for (int i = 1; i < ranges.size(); i++)
    {
        if (height < ranges.at(i))
        {
            selidx = i;
            break;
        }
    }
    if (selidx == -1 || height < ranges.at(0))
    {
        throw std::runtime_error("height out of range in modelset::sample_selection_simple");
    }
    auto &sel = selections.at(selidx);
    int randidx = rand() % sel.size();
    return sel.at(randidx);
}

void data_importer::modelset::add_to_ranges(int midx)
{
    auto &m = models.at(midx);
    int minidx = -1;
    for (int i = 0; i < ranges.size(); i++)
    {
        if (fabs(ranges[i] - m.hmin) < 1e-4f)
        {
            // this model's hmin already coincides with another range border. So we don't add this division
            minidx = i;
            break;
        }
        else if (m.hmin < ranges[i])
        {
            // the division at ranges[i] is the smallest div bigger than this m.hmin. Insert before it
            minidx = i;
            ranges.insert(std::next(ranges.begin(), i), m.hmin);
            break;
        }
    }
    if (minidx == -1)
    {
        //if (ranges.size() > 0)
        //    throw std::runtime_error("minimum value leaves a gap in ranges, in data_importer::modelset::setup_ranges");
        ranges.push_back(m.hmin);
        ranges.push_back(m.hmax);
        //selections.push_back({midx});
    }
    else
    {
       int maxidx = -1;
       for (int i = minidx + 1; i < ranges.size(); i++)
       {
           bool found = false;
            if (fabs(ranges[i] - m.hmax) < 1e-4f)
            {
                found = true;
            }
            else if (m.hmax < ranges[i])
            {
                ranges.insert(std::next(ranges.begin(), i), m.hmax);
                //selections.insert(std::next(selections.begin(), i - 1), {});
                found = true;
            }
            //selections.at(i - 1).push_back(midx);
            if (found)
            {
                maxidx = i;
                break;
            }
       }
       if (maxidx == -1)
       {
           ranges.push_back(m.hmax);
           //selections.push_back({midx});
       }
    }
}

void data_importer::modelset::setup_selections()
{
    selections.clear();
    for (int i = 0; i < ranges.size() - 1; i++)
    {
        selections.push_back({});
        float min = ranges[i];
        float max = ranges[i + 1];
        for (treemodel &m : models)
        {
            if (m.hmin + 1e-2f < max && m.hmax - 1e-2f > min)   // allowing for quite a large margin of error here...
            {
                selections.back().push_back(m.vueid);
            }
        }
    }
}


using namespace basic_types;



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

std::vector<basic_tree> data_importer::read_pdb(std::string filename)
{
    std::map<int, std::vector<MinimalPlant> > retvec;
    if (!read_pdb(filename, retvec))
    {
        throw std::runtime_error("File " + filename + " not found in data_importer::read_pdb");
    }
    else
    {
        return data_importer::minimal_to_basic(retvec);
    }
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
            }
        }
        std::cerr << std::endl;


        infile.close();
        return true;
    }
    else
        return false;
}


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
            sb.id = std::stoi(valstr);
        }
        else if (colstr == "Value")
        {
            sb.name = valstr;
        }
    }
    common->subbiomes[sb.id] = sb;
    common->subbiomes_all_species[sb.id] = sb;
    return 0;
}

static int sql_callback_common_data_subbiomes_species(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int sb_id, tree_id;
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
            sb_id = std::stoi(valstr);
        }
        else if (colstr == "Tree_ID")
        {
            tree_id = std::stoi(valstr);
        }
    }
    data_importer::sub_biome &sb = common->subbiomes[sb_id];
    data_importer::species_encoded sp;
    sp.id = tree_id;
    sb.species.insert(sp);
    return 0;
}

static int sql_callback_common_data_subbiomes_all_species(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int sb_id, tree_id;
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
            sb_id = std::stoi(valstr);
        }
        else if (colstr == "Tree_ID")
        {
            tree_id = std::stoi(valstr);
        }
        else if (colstr == "Canopy")
        {
            int canopyval = std::stoi(valstr);
            if (canopyval)
            {
                //std::cout << "Encountered canopy species" << std::endl;
                canopy = true;
            }
            else
                canopy = false;
        }
    }
    data_importer::sub_biome &sb = common->subbiomes_all_species[sb_id];
    data_importer::species_encoded sp;
    sp.id = tree_id;
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

static int sql_callback_common_data_models(void *write_info, int argc, char ** argv, char **colnames)
{
    data_importer::common_data *common = (data_importer::common_data *)write_info;

    int tree_id;
    data_importer::treemodel model;

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

        if (colstr == "vueid")
        {
            model.vueid = std::stoi(valstr);
        }
        else if (colstr == "Tree_ID")
        {
            tree_id = std::stoi(valstr);
        }
        else if (colstr == "hmin")
        {
            model.hmin = std::stof(valstr);
        }
        else if (colstr == "hmax")
        {
            model.hmax = std::stof(valstr);
        }
        else if (colstr == "prob")
        {
            model.prob = std::stof(valstr);
        }
        else if (colstr == "modheight")
        {
            model.modheight = std::stof(valstr);
        }
        else if (colstr == "whratio")
        {
            model.whratio = std::stof(valstr);
        }
        else if (colstr == "modname")
        {
            model.modname = valstr;
        }
    }

    if (common->canopy_and_under_species.at(tree_id).maxhght < model.hmax)
    {
        std::cout << "WARNING: maximum height (" << model.hmax << ") for model " << model.vueid << " is higher than maximum height for species " << tree_id << ", which is " << common->canopy_and_under_species.at(tree_id).maxhght << std::endl;
    }
    common->modelsamplers[tree_id].add_model(model);

}

static int sql_callback_common_data_species(void *write_info, int argc, char ** argv, char **colnames)
{
    //std::cout << "Processing row in sql_callback_common_data_species" << std::endl;

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
    bool iscanopy;
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
            else if (valstr == "INVCONE")
            {
                draw_shape = data_importer::treeshape::INVCONE;
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
        else if (colstr == "canopy")
        {
            if (valstr == "Y")
                iscanopy = true;
            else
                iscanopy = false;
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
    sp.basecol[0] = draw_color[0], sp.basecol[1] = draw_color[1], sp.basecol[2] = draw_color[2];
    sp.basecol[3] = 1.0f;
    sp.draw_hght = draw_height;
    sp.draw_radius = draw_radius;
    sp.draw_box1 = draw_box1;
    sp.draw_box2 = draw_box2;
    sp.name = name;
    sp.max_trunk_radius = trunkrad;
    sp.shapetype = draw_shape;

    assign_viability(sp.slope, slope);
    assign_viability(sp.sun, sun);
    assign_viability(sp.temp, temp);
    assign_viability(sp.wet, moisture);

    auto result = common->canopy_and_under_species.insert({sp.idx, sp});
    assert(result.second);		// Each species should only be inserted once. If it already exists in the map
                                // there is a bug

    if (iscanopy)		// insert into the canopy species-only map also
    {
        result = common->all_species.insert({sp.idx, sp});
        assert(result.second);
    }

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

    errcode = sqlite3_exec(db, "SELECT modelMapping.vueid as vueid, \
                                modelMapping.Tree_ID as Tree_ID, \
                                hmin, hmax, prob, modheight, modname, whratio \
                                FROM modelDetails INNER JOIN modelMapping ON modelDetails.vueid = modelMapping.vueid",
                                sql_callback_common_data_models, this, &errmsg);
    sql_err_handler(db, errcode, errmsg);
    for (auto &p : modelsamplers)
        p.second.setup_ranges();		// can actually call setup_selections here, instead of this

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
            int speccode = spec_enc.id;
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
