#ifndef DATA_IMPORTER_H
#define DATA_IMPORTER_H

// By K.P. Kapp
// April/May 2019

//#include "MinimalPlant.h"
//#include "MapFloat.h"
//#include "canopy_placement/canopy_placer.h"
#include "basic_types.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <array>
#include <set>
#include <cassert>
#include <cmath>

//#include <boost/filesystem.hpp>
//#include <boost/regex.hpp>

namespace data_importer
{
        struct viability
        {
            float c, r;
            float cmin, cmax;	// codes for upper, lower bounds: L, M, H
        };

        enum treeshape
        {
            SPHR,
            BOX,
            CONE
        };

        struct grass_viability
        {
            float absmin, innermin, innermax, absmax;
        };

        struct species
        {
            std::string name;

            int idx;
            float a;
            float b;

            float basecol[4];  //< base colour for the PFT, individual plants will vary
            float draw_hght;    //< canopy height scaling
            float draw_radius;  //< canopy radius scaling
            float draw_box1;    //< box aspect ratio scaling
            float draw_box2;    //< box aspect ration scaling
            treeshape shapetype; //< shape for canopy: sphere, box, cone

            viability sun, wet, temp, slope; //< viability functions for different environmental factors
            float alpha;        //< sunlight attenuation multiplier
            int maxage;         //< expected maximum age of the species in months
            float maxhght;      //< expected maximum height in meters
            float max_trunk_radius;	//< expected maximum trunk radius in meters

            //< growth parameters
            char growth_period; //< long (L) or short (S) growing season
            int grow_months;    //< number of months of the year in the plants growing season
            int grow_start, grow_end; //< start and stop months for growth (in range 0 to 11)
            float grow_m, grow_c1, grow_c2; // terms in growth equation: m * (c1 + exp(c2))

            //< allometry parameters
            char allometry_code; //< allometry patterns: A, B, C, or D
            //float alm_a, alm_b; //< terms in allometry equation to convert height to canopy radius: r = e ** (a + b ln(h))
            float alm_rootmult = 1.0f; //< multiplier to convert canopy radius to root radius
        };

        struct species_encoded
        {
            int idx;
            float percentage;
            bool canopy;

            bool operator < (const species_encoded &other) const
            {
                bool result = idx < other.idx;
                return result;
            }
        };

        struct sub_biome
        {
            int idx;
            std::string name;
            float percentage;
            std::set<species_encoded> species;
        };

        struct common_data
        {
            common_data(std::string db_filename);

            std::array<float, 12> rainfall;
            std::array<float, 12> cloudiness;
            std::array<float, 12> temperature;
            sim_info soil_info;
            float latitude;
            float temp_lapse_rate;
            std::map<int, sub_biome> subbiomes;
            std::map<int, sub_biome> subbiomes_all_species;
            std::map<int, species> all_species;
            std::map<int, species> canopy_and_under_species;
            std::map<int, int> canopyspec_to_subbiome;
        };

        struct data_dir
        {
            data_dir(std::string dirname_arg, int nsims)
                : dirname(dirname_arg)
            {
                init_filenames(dirname_arg);
                init_required_simulations(nsims);
            }

            data_dir(std::string dirname_arg, const common_data &simdata)
                : dirname(dirname_arg)
            {
                init_filenames(dirname_arg);
                init_required_simulations(simdata);
                //init_required_simulations(4);

                /*
                filesystem::path dirpath(dirname);
                if (!filesystem::exists(dirpath))
                {
                    std::string errstr = "Directory ";
                    errstr += dirname + " does not exist";
                    throw std::runtime_error(errstr);
                }

                //boost::regex expr{std::string("^") + dataset_name + "_biome"};
                std::string start_str = dataset_name + "_biome";
                filesystem::directory_iterator end_iter;
                for (filesystem::directory_iterator iter(dirpath); iter != end_iter; iter++)
                {
                    auto path = iter->path();
                    std::string filename = path.filename().string();
                    std::string beginning = filename.substr(0, start_str.size());
                    std::string end = filename.substr(start_str.size(), filename.size() - start_str.size());
                    size_t dotpos = end.find_last_of('.');
                    std::string numstr = end.substr(0, dotpos);
                    int biome_num;
                    try
                    {
                        biome_num = std::stoi(numstr);
                    }
                    catch (std::invalid_argument &e)
                    {
                        std::string errstr = "Biome files must be in the format <DATASET NAME>_biome<INTEGER>.txt (ignoring the angled brackets)\n";
                        errstr += "Example: datasetname_biome2.txt";
                        throw std::invalid_argument(errstr);
                    }
                    if (start_str == beginning)
                    {
                        biome_fnames.push_back(path.string());
                        //canopy_fnames.push_back(dirname + "/" + dataset_name + "_canopy" + std::to_string(biome_num) + ".txt");
                        canopy_fnames.push_back(get_canopy_filename(biome_fnames.back()));
                        species_fnames.push_back(get_species_filename(biome_fnames.back()));
                    }
                }
                */

            }

            data_dir(std::string dirname_arg)
            {
                init_filenames(dirname_arg);
            }


            void init_filenames(std::string dirname_arg)
            {
                dirname = dirname_arg;
                while (dirname[dirname.size() - 1] == '/')
                {
                        dirname.pop_back();
                }
                size_t fwsl_pos = dirname.find_last_of('/');
                dataset_name = this->dirname.substr(fwsl_pos + 1);

                simspec_name = dirname + "/" + dataset_name + "_simspec.txt";
                chm_fname = dirname + "/" + dataset_name + ".chm";
                dem_fname = dirname + "/" + dataset_name + ".elv";
                cdm_fname = dirname + "/" + dataset_name + ".cdm";
                //species_fname = dirname + "/" + dataset_name + "_species.txt";
                species_params_fname = dirname + "/" + dataset_name + "_species_params.txt";
                //biome_fname = dirname + "/" + dataset_name + "_biome.txt";
                clim_fname = dirname + "/" + dataset_name + "_clim.txt";
                sun_fname = dirname + "/" + dataset_name + "_sun_landscape.txt";
                sun_tree_fname = dirname + "/" + dataset_name + "_sun.txt";
                wet_fname = dirname + "/" + dataset_name + "_wet.txt";
                slope_fname = dirname + "/" + dataset_name + "_slope.txt";
                temp_fname = dirname + "/" + dataset_name + "_temp.txt";
                grass_fname = dirname + "/" + dataset_name + "_grass.txt";
                grass_params_fname = dirname + "/" + dataset_name + "_grass_params.txt";
                //canopy_fname = dirname + "/" + dataset_name + "_canopy.txt";

            }

            void init_required_simulations(int nsims)
            {
                for (int i = 0; i < nsims; i++)
                {
                    canopy_fnames.push_back(dirname + "/" + dataset_name + "_canopy" + std::to_string(i) + ".pdb");
                    species_fnames.push_back(dirname + "/" + dataset_name + "_species" + std::to_string(i) + ".txt");
                    undergrowth_fnames.push_back(dirname + "/" + dataset_name + "_undergrowth" + std::to_string(i) + ".pdb");
                    circ_count_fnames.push_back(dirname + "/" + dataset_name + "_circ_count" + std::to_string(i) + ".txt");
                    canopy_texture_fnames.push_back(generate_canopy_texture_fname(i));
                }
            }

            void init_required_simulations(const common_data &simdata)
            {
                required_simulations = get_required_simulations(simspec_name, simdata.subbiomes);
                for (int i = 0; i < required_simulations.size(); i++)
                {
                    canopy_fnames.push_back(dirname + "/" + dataset_name + "_canopy" + std::to_string(i) + ".pdb");
                    species_fnames.push_back(dirname + "/" + dataset_name + "_species" + std::to_string(i) + ".txt");
                    undergrowth_fnames.push_back(dirname + "/" + dataset_name + "_undergrowth" + std::to_string(i) + ".pdb");
                    circ_count_fnames.push_back(dirname + "/" + dataset_name + "_circ_count" + std::to_string(i) + ".txt");
                    canopy_texture_fnames.push_back(generate_canopy_texture_fname(i));
                }
            }

            std::string generate_species_fname(int idx)
            {
                return dirname + "/" + dataset_name + "_species" + std::to_string(idx) + ".txt";
            }

            std::string generate_canopy_texture_fname(int idx)
            {
                return dirname + "/" + dataset_name + "_canopy_texture" + std::to_string(idx) + ".txt";
            }

            /*
            int get_biome_number(std::string full_filename)
            {
                boost::filesystem::path path(full_filename);
                std::string filename = path.string();
                std::string start_str = dataset_name + "_biome";
                std::string beginning = filename.substr(0, start_str.size());
                // first check that the filename is a valid biome file
                if (start_str != beginning)
                {
                    return -1;
                }

                std::string end = filename.substr(start_str.size(), filename.size() - start_str.size());
                size_t dotpos = end.find_last_of('.');
                std::string numstr = end.substr(0, dotpos);
                int biome_num;
                try
                {
                    biome_num = std::stoi(numstr);
                }
                catch (std::invalid_argument &e)
                {
                    std::string errstr = "Biome files must be in the format <DATASET NAME>_biome<INTEGER>.txt (ignoring the angled brackets)\n";
                    errstr += "Example: datasetname_biome2.txt";
                    throw std::invalid_argument(errstr);
                }
            }

            std::string get_canopy_filename(std::string full_biome_filename)
            {
                int number = get_biome_number(full_biome_filename);
                if (number < 0)
                {
                    std::string errstr = "Biome files must be in the format <DATASET NAME>_biome<INTEGER>.txt (ignoring the angled brackets)\n";
                    errstr += "Example: datasetname_biome2.txt";
                    throw std::invalid_argument(errstr);
                }
                std::string full_filename = dirname + "/" + dataset_name + "_canopy" + std::to_string(number) + ".txt";
                return full_filename;
            }

            std::string get_species_filename(std::string full_biome_filename)
            {
                int number = get_biome_number(full_biome_filename);
                if (number < 0)
                {
                    std::string errstr = "Biome files must be in the format <DATASET NAME>_biome<INTEGER>.txt (ignoring the angled brackets)\n";
                    errstr += "Example: datasetname_biome2.txt";
                    throw std::invalid_argument(errstr);
                }
                std::string full_filename = dirname + "/" + dataset_name + "_species" + std::to_string(number) + ".txt";
                return full_filename;
            }
            */

            static void trim_string(std::string &str)
            {
                while (str.size() > 0 && str.front() == ' ')
                        str = std::string(std::next(str.begin(), 1), str.end());
                while (str.size() > 0 && str.back() == ' ')
                        str = std::string(str.begin(), std::next(str.end(), -1));
            }

            std::vector<std::map<int, sub_biome> > get_required_simulations(std::string filename,
                                                                            const std::map<int, sub_biome> &benchmark)
            {
                using namespace data_importer;
                std::ifstream ifs(filename);

                std::vector< std::map<int, sub_biome> > sim_sbiomes;
                if (ifs.is_open())
                {
                    while (ifs.good())
                    {
                        int nspecies = 0;
                        std::map<int, sub_biome> sbiomes;
                        std::string line;
                        std::getline(ifs, line);
                        trim_string(line);
                        if (line.size() == 0)
                            continue;
                        std::cout << "Line: " << line << std::endl;
                        std::stringstream sstr(line);
                        while (sstr.good())
                        {
                            sub_biome sb;
                            sstr >> sb.idx >> sb.percentage;
                            sbiomes[sb.idx] = sb;
                            //nspecies += benchmark[sb.idx].species.size();
                            nspecies += benchmark.at(sb.idx).species.size();
                        }
                        sim_sbiomes.push_back(sbiomes);
                        for (int i = 0; i < nspecies && ifs.good(); i++)
                        {
                            int sb_idx;
                            species_encoded spe;
                            line.clear();
                            std::getline(ifs, line);
                            trim_string(line);
                            if (line.size() == 0)
                                continue;
                            std::stringstream sstr_species(line);
                            sstr_species >> sb_idx >> spe.idx >> spe.percentage;
                            sim_sbiomes.back()[sb_idx].species.insert(spe);
                        }
                    }
                }
                return sim_sbiomes;
            }

            static void test_biome_and_canopy_fnames(std::string dirname_arg)
            {
                data_dir ddir(dirname_arg);

                std::cout << "Biome filenames: " << std::endl;
                for (auto &fname : ddir.biome_fnames)
                {
                    std::cout << fname << std::endl;
                }
                std::cout << std::endl;

                std::cout << "Canopy filenames: " << std::endl;
                for (auto &fname : ddir.canopy_fnames)
                {
                    std::cout << fname << std::endl;
                }
            }

            std::string dirname;
            std::string dataset_name;

            std::string simspec_name;
            std::string chm_fname;
            std::string dem_fname;
            std::string cdm_fname;
            std::string species_params_fname;
            std::string clim_fname;
            std::string sun_fname;
            std::string sun_tree_fname;
            std::string wet_fname;
            std::string slope_fname;
            std::string temp_fname;
            std::string grass_fname;
            std::string grass_params_fname;

            std::vector<std::string> species_fnames;
            std::vector<std::string> biome_fnames;
            std::vector<std::string> canopy_fnames;
            std::vector<std::string> canopy_texture_fnames;
            std::vector<std::string> undergrowth_fnames;
            std::vector<std::string> circ_count_fnames;
            std::vector<std::map<int, sub_biome> > required_simulations;
        };
}

namespace data_importer
{
        /*
         * exactly like load_elv, except that we do not read the step and latitude values after width and height, and we do not account for feet to meters conversion
         */
        basic_types::MapFloat load_txt(std::string filename, int &width, int &height);

        basic_types::MapFloat load_elv(std::string filename, int &width, int &height);

        std::vector<basic_types::MapFloat> read_monthly_map(std::string filename, int &width, int &height);

        basic_types::MapFloat average_mmap(std::vector<basic_types::MapFloat> &mmap);

        basic_types::MapFloat get_temperature_map(basic_types::MapFloat &heightmap, float basetemp, float reduce_per_meter);

        void normalize_data(basic_types::MapFloat &data);

        void eliminate_outliers(basic_types::MapFloat &data);

        std::vector<basic_tree> minimal_to_basic(const std::map<int, std::vector<MinimalPlant> > &plants);

        bool read_pdb(std::string filename, std::map<int, std::vector<MinimalPlant> > &retvec);
        void read_rainfall(std::string filename, sim_info &info);
        void read_soil_params(std::string filename, sim_info &info);
        std::vector<float> read_temperature(std::string filename);

        std::vector<int> get_nonzero_idxes(basic_types::MapFloat &data);

        std::vector<species_params> read_species_params_before_specassign(std::string params_filename);
        std::vector<species_params> read_species_params(std::string params_filename);
        void write_species_params_after_specassign(std::string params_filename, std::vector<species_params> all_params);

        std::vector<std::map<int, sub_biome> > get_required_simulations(std::string filename, std::map<int, sub_biome> &benchmark);

        std::map<std::string, grass_viability> read_grass_viability(std::string filename);
}

namespace data_importer
{
    template<typename T>
    void write_txt(std::string filename, T *data, float step_size = -1, float latitude = 0.0f)
    {
        int width, height;
        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

        if (ofs.good())
        {
            ofs << data->width() << " " << data->height() << " " << step_size << " " << latitude << std::endl;
            for (int y = 0; y < data->height(); y++)
            {
                for (int x = 0; x < data->width(); x++)
                {
                    ofs << data->get(x, y) << " ";
                }
                ofs << std::endl;
            }
        }
        else
        {
            std::cout << "WARNING: could not write to file " << filename << std::endl;
        }
    }

    template<typename T>
    T load_elv(std::string filename)
    {
        using namespace std;

        float step, lat;
        int dx, dy;
        int width, height;
        int real_w, real_h;

        float val;
        ifstream infile;

        T retmap;

        infile.open((char *) filename.c_str(), ios_base::in);
        if(infile.is_open())
        {
            infile >> dx >> dy;
            infile >> step;
            infile >> lat;
            width = dx;
            height = dy;
            real_w = int(ceil(width * step));
            real_h = int(ceil(height * step));

            retmap.setDimReal(real_w, real_h);
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

    inline void write_pdb(std::string filepath, const std::vector<output_tree> &trees)
    {
        using namespace std;

        struct species_info
        {
            int species_id;
            float max_height;
            float min_height;
            float average_height;
            float average_radius;
            int nplants;
        };

        std::cout << "Opening file " << filepath << "..." << std::endl;
        ofstream ofs(filepath);

        if (!ofs.is_open())
        {
            std::cout << "Can't write to file " << filepath << std::endl;
        }

        int num_species = 1;
        float min_height = 100000.0f, max_height = 0.0f;
        int species_id = 0;
        int num_plants_of_species = trees.size();	// because we assume that all trees are of one species - this will probably change
        float avg_radius = 0;
        float avg_height = 0;
        float avg_radius_over_height;

        std::map< int, std::vector<const output_tree *> > trees_map;
        std::map< int, species_info > species_info_map;

        for (auto &tree  : trees)
        {
            std::vector< const output_tree *> tempvec;
            tempvec.push_back(&tree);
            auto result = trees_map.insert({(int)tree.species, tempvec });
            if (!result.second)		// if an element with that key already existed
            {
                result.first->second.push_back(&tree);
            }
        }

        num_species = trees_map.size();

        for (auto &keyval : trees_map)
        {
            species_info info;
            info.max_height = -std::numeric_limits<float>::max();
            info.min_height = std::numeric_limits<float>::max();
            info.average_height = 0.0f;
            info.average_radius = 0.0f;
            info.nplants = 0;
            for (auto &tree_ptr : keyval.second)
            {
                if (tree_ptr->height > info.max_height)
                    info.max_height = tree_ptr->height;
                if (tree_ptr->height < info.min_height)
                    info.min_height = tree_ptr->height;
                info.average_height += tree_ptr->height;
                info.average_radius += tree_ptr->radius;
                info.nplants++;
                info.species_id = tree_ptr->species;
                if (info.species_id > 50)
                {
                    std::cout << "invalid species id: " << info.species_id << std::endl;
                }
            }
            auto result = species_info_map.insert({info.species_id, info});
            assert(result.second);		// each must only be inserted once, since we are iterating over each species only once
        }

        if (ofs.is_open())
        {
            ofs << num_species << endl;
            for (auto &keyval : trees_map)
            {
                int species_id = keyval.first;
                species_info &info = species_info_map[species_id];
                ofs << info.species_id << " " << info.min_height << " " << info.max_height << " " << info.average_radius / info.average_height << std::endl;
                ofs << info.nplants << std::endl;
                for (auto &tree_ptr : keyval.second)
                {
                    int radius;
                    if (tree_ptr->radius < 1) radius = 1;
                    else
                        radius = tree_ptr->radius;
                    ofs << tree_ptr->x << " " << tree_ptr->y << " " << tree_ptr->z << " " << tree_ptr->height << " " << radius << std::endl;
                }
            }

        }

    }

    template<typename T>
    inline void write_pdb(std::string filepath, const float *heightmap, T *trees_begin, T *trees_end, int width, int height)
    {
        using namespace std;

        vector< output_tree > tree_list;

        float m_per_foot = 0.3048;
        float f_per_pixel_unit = 3;

        int i = 0;
        for (T *tree_ptr = trees_begin; tree_ptr != trees_end; tree_ptr++)
        {
            int idx = tree_ptr->y * width + tree_ptr->x;
            float x = tree_ptr->x, y = tree_ptr->y, z = heightmap[idx];
            x *= m_per_foot * f_per_pixel_unit;
            y *= m_per_foot * f_per_pixel_unit;
            z *= m_per_foot;

            output_tree tree(x, y, z, tree_ptr->radius * (f_per_pixel_unit * m_per_foot), tree_ptr->height);
            tree.species = tree_ptr->species;
            tree_list.push_back(tree);
            i++;
        }

        write_pdb(filepath, tree_list);


    }

    template<typename T>
    inline void write_pdb(std::string filepath, T *trees_begin, T *trees_end)
    {
        using namespace std;

        vector< output_tree > tree_list;

        float m_per_foot = 0.3048;
        float f_per_pixel_unit = 3;

        int i = 0;
        for (T *tree_ptr = trees_begin; tree_ptr != trees_end; tree_ptr++)
        {
            float z = 0.0f;
            float x = tree_ptr->x, y = tree_ptr->y;

            output_tree tree(x, y, z, tree_ptr->radius, tree_ptr->height);
            tree.species = tree_ptr->species;
            tree_list.push_back(tree);
            i++;
        }

        write_pdb(filepath, tree_list);


    }


    template<typename T>
    void minimaltree_to_othertree(const std::map<int, std::vector<MinimalPlant> > &minplants, std::map<int, std::vector<T> > &otherplants)
    {
        otherplants.clear();
        for (auto &specplants : minplants)
        {
            for (auto &minplnt : specplants.second)
            {
                T newplant;
                newplant.x = minplnt.x;
                newplant.y = minplnt.y;
                newplant.radius = minplnt.r;
                newplant.height = minplnt.h;
                newplant.species = specplants.first;
                otherplants[specplants.first].push_back(newplant);
            }
        }
    }

    template<typename T>
    T load_txt(std::string filename)
    {
        using namespace std;

        T retmap;

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

        int width = atoi(file_contents[0].c_str());
        int height = atoi(file_contents[1].c_str());

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
            if (std::is_same<T, double>::value)
                retmap.set(x, y, std::stod(*iter));
            else if (std::is_same<T, float>::value)
                retmap.set(x, y, std::stof(*iter));
            else if (std::is_same<T, int>::value)
                retmap.set(x, y, std::stoi(*iter));
            else if (std::is_same<T, long>::value)
                retmap.set(x, y, std::stol(*iter));
            else if (std::is_same<T, long long>::value)
                retmap.set(x, y, std::stoll(*iter));
            else if (std::is_same<T, long double>::value)
                retmap.set(x, y, std::stold(*iter));
            else if (std::is_same<T, short>::value)
                retmap.set(x, y, std::stoi(*iter));
            else if (std::is_same<T, unsigned int>::value)
                retmap.set(x, y, std::stoul(*iter));
            else if (std::is_same<T, unsigned long>::value)
                retmap.set(x, y, std::stoul(*iter));
            else if (std::is_same<T, unsigned long long>::value)
                retmap.set(x, y, std::stoull(*iter));
            else
                retmap.set(x, y, std::stod(*iter));
        }

        return retmap;
    }

    template <typename T>
    std::vector< T > read_monthly_map(std::string filename)
    {
        std::ifstream ifs(filename);

        std::vector< T > mmap(12);

        int width, height;

        if (ifs.is_open())
        {
            ifs >> width >> height;
            if (width > 0 && width <= 10240 && height > 0 && height <= 10240)
            {
                std::for_each(mmap.begin(), mmap.end(), [&width, &height](T &mapf) { mapf.setDim(width, height);});
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
                if (m == 12)		// all contiguous values for current month have been read...
                {
                    m = 0;
                    x++;			// move to next location, in the x-direction
                    if (x == width)		// if we have finished the row...
                    {
                        x = 0;		// move to start of next row
                        y++;		// increment y, move to next row
                    }
                }
                if ((y * width + x) * 12 + m >= width * height * 12)
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

    template<typename T>
    void write_monthly_map(std::string filename, const std::vector< T > &months)
    {
        int width, height;
        bool valid = true;
        if (months.size() == 0)
        {
            throw std::invalid_argument("there must be at least one map for the monthly map write to txt in data_importer::write_monthly_map");
        }
        months[0].getDim(width, height);
        std::for_each(months.begin(), months.end(), [&valid, &width, &height](const T &map) {
            int prevw = width;
            int prevh = height;
            map.getDim(width, height);
            if (prevw != width || prevh != height)
            {
                valid = false;
            }
        });
        if (!valid)
        {
            throw std::invalid_argument("Each map in the argument to data_importer::write_monthly_map must have the same dimensions");
        }

        std::ofstream ofs(filename);

        if (ofs.is_open() && ofs.good())
        {
            ofs << width << " " << height << "\n";
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int m = 0; m < 12; m++)
                    {
                        //ofs << sunmap_monthly[m][y * ter.get_width() + x] << " ";
                        ofs << months[m].get(x, y) << " ";
                    }
                }
            }
        }
    }

    template<typename T, typename U>
    T average_mmap(const std::vector< U > &mmap)
    {
        if (mmap.size() != 12)
        {
            throw std::runtime_error("Monthly map does not have twelve maps");
        }
        int width = -1, height = -1;
        for (const auto &m : mmap)
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

        T avgmap;
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

}

#endif //DATA_IMPORTER_H
