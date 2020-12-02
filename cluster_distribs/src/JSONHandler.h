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


#include <rapidjson/reader.h>
#include <vector>
#include <map>
#include "common.h"

struct raw_grid_dist_info
{
    std::vector< std::vector < std::vector< std::pair<int, int> > > > species_interactions;
    std::vector< std::vector< std::vector< std::vector<common_types::decimal> > > > histograms;
    std::vector<std::vector<int> > plant_classes;
    std::vector<std::vector<int> > planttype_counts;
    int width, height;
    int block_width, block_height;
    int nblock_rows, nblock_cols;
    int hist_nbins;
    std::vector<common_types::decimal> height_bins;
    std::map<int, std::pair<int, int > > class_height_map;
};

class JSONHandler : public raw_grid_dist_info
{
public:

    enum class input_type
    {
        NONE,
        WIDTH,
        HEIGHT,
        BLOCK_WIDTH,
        BLOCK_HEIGHT,
        NBLOCK_ROWS,
        NBLOCK_COLS,
        HIST_NBINS,
        HEIGHT_BINS,
        SPECIES_INTERACT,
        PLANT_CLASSES,
        PLANT_COUNTS,
        HISTOGRAM,
        CLASS_HEIGHT_MAP
    };

    enum class input_types_int
    {
        NONE,
        WIDTH,
        HEIGHT,
        BLOCK_WIDTH,
        BLOCK_HEIGHT,
        NBLOCK_ROWS,
        NBLOCK_COLS,
        HIST_NBINS,
        HEIGHT_BINS,
        SPECIES_INTERACT,
        PLANT_CLASSES
    };

    enum class input_types_float
    {
        NONE,
        HISTOGRAM,
        HEIGHT_BINS
    };

    bool Null() {return true;}
    bool Bool(bool b) {return true;}
    bool Int(int i)
    {
        switch (curr_input)
        {
            case input_type::WIDTH:
                width = i;
                break;
            case input_type::HEIGHT:
                height = i;
                break;
            case input_type::BLOCK_WIDTH:
                block_width = i;
                break;
            case input_type::BLOCK_HEIGHT:
                block_height = i;
                break;
            case input_type::NBLOCK_ROWS:
                nblock_rows = i;
                break;
            case input_type::NBLOCK_COLS:
                nblock_cols = i;
                break;
            case input_type::HIST_NBINS:
                hist_nbins = i;
                break;
            case input_type::HEIGHT_BINS:
                height_bins.push_back(i);
                break;
            case input_type::SPECIES_INTERACT:
                if (species_interactions.back().back().back().first == -1)
                    species_interactions.back().back().back().first = i;
                else
                    species_interactions.back().back().back().second = i;
                break;
            case input_type::PLANT_CLASSES:
                plant_classes.back().push_back(i);
                break;
            case input_type::PLANT_COUNTS:
                planttype_counts.back().push_back(i);
                break;
            case input_type::CLASS_HEIGHT_MAP:
                //class_height_map.push_back(i);
                class_height_array.push_back(i);
                break;
        }
        return true;
    }

    bool Uint(unsigned i)
    {
        if (i >= 0)
        {
            return Int(i);
        }
        else
        {
            return true;
        }
    }

    bool Int64(int64_t i)
    {
        return Int(i);
    }

    bool Uint64(uint64_t i)
    {
        if (i >= 0)
        {
            return Int(i);
        }
        else
        {
            return true;
        }
    }

    bool Double(double d)
    {
        switch (curr_input)
        {
            case input_type::HISTOGRAM:
                if (array_depth == 4)
                    histograms.back().back().back().push_back(d);
                break;
            case input_type::HEIGHT_BINS:
                height_bins.push_back(d);
                break;
        }
        return true;
    }

    bool Key(const char *ch_str, size_t length, bool copy)
    {
        std::string str(ch_str);
        try
        {
            curr_input = input_map.at(str);
            return true;
        }
        catch (std::out_of_range &e)
        {
            curr_input = input_type::NONE;
        }

        if (str != "Sandbox distributions")
        {
            curr_array_string = str;
        }
        else
        {
            distribs_started = true;
        }

        return true;
    }

    bool StartObject()
    {
        if (distribs_started)
        {
            plant_classes.push_back(std::vector<int>());
            histograms.push_back(std::vector<std::vector<std::vector<common_types::decimal> > >());
            species_interactions.push_back(std::vector<std::vector<std::pair<int, int> > >());
            planttype_counts.push_back(std::vector<int>());
        }
        return true;
    }

    bool EndObject(size_t length)
    {
        curr_block_idx++;
        return true;
    }

    bool RawNumber(const char* str, size_t length, bool copy)
    {
        return true;
    }

    bool String(const char *str, size_t length, bool copy)
    {
        return true;
    }

    bool StartArray()
    {
        array_depth++;
        if (curr_input == input_type::HISTOGRAM)
        {
            if (array_depth == 2)
            {
                //XXX: we add the histogram when we enter the object
            }
            else if (array_depth == 3)
            {
                histograms.back().push_back(std::vector< std::vector<common_types::decimal> > ());
            }
            else if (array_depth == 4)
            {
                histograms.back().back().push_back(std::vector<common_types::decimal> ());
            }
            else if (array_depth == 5)
            {
                std::cout << "Array depth 5 should not be possible for histogram matrix" << std::endl;
                return false;
            }
        }
        else if (curr_input == input_type::SPECIES_INTERACT)
        {
            switch (array_depth)
            {
                case 2:
                    // XXX: species interactions matrix will be added when the object is entered
                    break;
                case 3:
                    species_interactions.back().push_back(std::vector<std::pair<int, int> >());
                    break;
                case 4:
                    species_interactions.back().back().push_back({-1, -1});
                    break;
                case 5:
                    std::cout << "Array depth 5 should not be possible for species interactions matrix" << std::endl;
                    return false;
                    break;

            }
        }
        else if (curr_input == input_type::PLANT_CLASSES)
        {
            //curr_plant_classes = &plant_classes.back();
        }
        else if (curr_input == input_type::CLASS_HEIGHT_MAP)
        {
            // don't need to do anything - we already allocated the array for the association
        }
        return true;
    }

    bool EndArray(size_t length)
    {
        if (array_depth == 4)
        {
            if (curr_hist)
                curr_hist = nullptr;
        }
        else if (array_depth == 3)
        {
            if (curr_hist_row)
                curr_hist_row = nullptr;
        }
        else if (array_depth == 2)
        {
            if (curr_hist_matrix)
                curr_hist_matrix = nullptr;
            if (curr_plant_classes)
                curr_plant_classes = nullptr;
        }

        if (curr_input == input_type::CLASS_HEIGHT_MAP)
        {
            for (int i = 0; i < class_height_array.size(); i++)
            {
                class_height_map[class_height_array[i]] = {height_bins[i], height_bins[i + 1]};
            }
        }

        array_depth--;
        return true;
    }

    std::pair<int, int> *current_active_pair;
    std::vector<common_types::decimal> *current_active_hist;
    std::vector<std::vector<std::vector<common_types::decimal> > > *curr_hist_matrix;
    std::vector<std::vector<common_types::decimal> > *curr_hist_row;
    std::vector<common_types::decimal> *curr_hist;

    std::vector<std::vector<std::pair<int, int> > > *curr_species_matrix;
    std::vector<std::pair<int, int> > *curr_species_row;
    std::pair<int, int> *curr_species_pair;

    std::vector<int> *curr_plant_classes;

    std::vector<int> class_height_array;

    bool distribs_started = false;


    input_type curr_input = input_type::NONE;
    int curr_block_idx = 0;
    int array_depth = 0;
    std::string curr_array_string;

    std::map<std::string, input_type> input_map
    = {{"Height", input_type::HEIGHT},
       {"Width", input_type::WIDTH},
       {"Block width", input_type::BLOCK_WIDTH},
       {"Block height", input_type::BLOCK_HEIGHT},
       {"Block rows", input_type::NBLOCK_ROWS},
       {"Block columns", input_type::NBLOCK_COLS},
       {"Height bins", input_type::HEIGHT_BINS},
       {"Histogram number of bins", input_type::HIST_NBINS},
       {"Matrix", input_type::HISTOGRAM},
       {"Height bins", input_type::HEIGHT_BINS},
       {"Species interactions", input_type::SPECIES_INTERACT},
       {"Plant classes", input_type::PLANT_CLASSES},
       {"Number of plants", input_type::PLANT_COUNTS},
       {"Class-height bin map", input_type::CLASS_HEIGHT_MAP}};
};
