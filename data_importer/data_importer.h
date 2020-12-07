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


#ifndef DATA_IMPORTER_H
#define DATA_IMPORTER_H

#include <common/basic_types.h>
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
#include <unordered_map>

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
            CONE,
            INVCONE
        };

        struct grass_viability
        {
            float absmin, innermin, innermax, absmax;
        };

        struct treemodel
        {
            int vueid;
            float hmin, hmax, prob, modheight, whratio;
            std::string modname;
        };

        struct modelset
        {
            std::vector<treemodel> models;
            std::vector<float> ranges;
            std::vector< std::vector<int> > selections;
            std::vector<int> samplemap;
            std::unordered_map<int, float> vueid_to_ratio;	// radius to height ratio, that is
            float minrange = std::numeric_limits<float>::max();
            int nbins;
            float binsize;

            void add_model(treemodel model);
            void add_to_ranges(int midx);
            void setup_ranges();

            int sample_selection_robust(float height);
            int sample_selection_fast(float height);
            int sample_selection_simple(float height);
            void setup_selections();

        public:
            float sample_rh_ratio(float height, int *vuemodel = nullptr);
        };

        inline std::map<int, float> get_whratios(const std::map<int, modelset> &samplers)
        {
            std::map<int, float> ratios;
            for (const std::pair<int, modelset> &s : samplers)
            {
                for (const treemodel &tm : s.second.models)
                {
                    ratios[tm.vueid] = tm.whratio;
                }
            }
            return ratios;
        }

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
            int id;
            float percentage;
            bool canopy;

            bool operator < (const species_encoded &other) const
            {
                bool result = id < other.id;
                return result;
            }
        };

        struct sub_biome
        {
            int id;
            std::string name;
            float percentage;
            std::set<species_encoded> species;

            std::vector<int> get_specids() const
            {
                std::vector<int> specids;
                for (auto &sp : species)
                {
                    specids.push_back(sp.id);
                }
                return specids;
            }
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
            std::map<int, data_importer::modelset> modelsamplers;
        };

        /*
         * Structure that generates required filenames for import, given a directory
         * name 'dirname_arg'
         */
        struct data_dir
        {
            data_dir(std::string dirname_arg)
                : data_dir(dirname_arg, 1)
            {}

            data_dir(std::string dirname_arg, int nsims)
                : dirname(dirname_arg)
            {
                init_filenames(dirname_arg);
                init_required_simulations(nsims);
            }

            /*
             * Checks if a file is in our binary format by checking for suffix .bin
             */
            static bool is_binary(std::string fname)
            {
                if (fname.substr(fname.size() - 3, 3) == "bin")
                    return true;
                else
                    return false;
            }

            /*
             * Checks if a binary file exists. If not, then remove .bin suffix and set suffix to .txt,
             * which is the original text-based import format
             */
            void check_binary(std::string &fname)
            {
                std::ifstream fcheck(fname);

                if (!fcheck.good())
                {
                    fname.erase(fname.size() - 3, 3);
                    fname += "txt";
                }
            }

            /*
             * Initialize filenames, based on directory 'dirname_arg'.
             */
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
                species_params_fname = dirname + "/" + dataset_name + "_species_params.txt";
                clim_fname = dirname + "/" + dataset_name + "_clim.txt";
                sun_fname = dirname + "/" + dataset_name + "_sun_landscape.bin";
                sun_tree_fname = dirname + "/" + dataset_name + "_sun.bin";
                wet_fname = dirname + "/" + dataset_name + "_wet.bin";
                temp_fname = dirname + "/" + dataset_name + "_temp.bin";
                slope_fname = dirname + "/" + dataset_name + "_slope.txt";
                grass_fname = dirname + "/" + dataset_name + "_grass.txt";
                grass_params_fname = dirname + "/" + dataset_name + "_grass_params.txt";

                check_binary(sun_fname);
                check_binary(sun_tree_fname);
                check_binary(wet_fname);
                check_binary(temp_fname);
            }

            /*
             * If more than one undergrowth simulation is to be done, for example, for multiple sets of canopy trees
             * on the same landscape, then prepare all input/output filenames for each simulation
             */
            void init_required_simulations(int nsims)
            {
                for (int i = 0; i < nsims; i++)
                {
                    canopy_fnames.push_back(dirname + "/" + dataset_name + "_canopy" + std::to_string(i) + ".pdb");
                    species_fnames.push_back(dirname + "/" + dataset_name + "_species" + std::to_string(i) + ".txt");
                    undergrowth_fnames.push_back(dirname + "/" + dataset_name + "_undergrowth" + std::to_string(i) + ".pdb");
                    circ_count_fnames.push_back(dirname + "/" + dataset_name + "_circ_count" + std::to_string(i) + ".txt");
                    canopy_texture_fnames.push_back(generate_canopy_texture_fname(i));
                    canopydensity_fnames.push_back(dirname + "/" + dataset_name + "_canopydensity" + std::to_string(i) + ".txt");
                    underdensity_fnames.push_back(dirname + "/" + dataset_name + "_underdensity" + std::to_string(i) + ".txt");
                    seedchance_fnames.push_back(dirname + "/" + dataset_name + "_seedchance" + std::to_string(i) + ".txt");
                    rendertexture_fnames.push_back(generate_rendertexture_fname(i));
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

            std::string generate_rendertexture_fname(int idx)
            {
                return dirname + "/" + dataset_name + "_rendertexture" + std::to_string(idx) + ".txt";
            }

            static void trim_string(std::string &str)
            {
                while (str.size() > 0 && str.front() == ' ')
                        str = std::string(std::next(str.begin(), 1), str.end());
                while (str.size() > 0 && str.back() == ' ')
                        str = std::string(str.begin(), std::next(str.end(), -1));
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
            std::vector<std::string> canopydensity_fnames;
            std::vector<std::string> underdensity_fnames;
            std::vector<std::string> rendertexture_fnames;
            std::vector<std::string> undergrowth_fnames;
            std::vector<std::string> circ_count_fnames;
            std::vector<std::string> seedchance_fnames;
            std::vector<std::map<int, sub_biome> > required_simulations;
        };

}

/*
 * Declarations for functions defined in data_importer.cpp
 */
namespace data_importer
{
        void eliminate_outliers(basic_types::MapFloat &data);
        std::vector<basic_tree> minimal_to_basic(const std::map<int, std::vector<MinimalPlant> > &plants);
        bool read_pdb(std::string filename, std::map<int, std::vector<MinimalPlant> > &retvec);
        std::vector<basic_tree> read_pdb(std::string filename);
        std::map<std::string, grass_viability> read_grass_viability(std::string filename);
}

namespace data_importer
{


    /*
     * Class to do compile time checks if a type has a step parameter when writing to output file
     */
    template<class Type>
    class type_has_realdim
    {
    private:
        template<typename T, T> struct typecheck;

        typedef char yes;
        typedef int no;

        template <typename T> struct GetDimReal
        {
            typedef void (T::*fptr) (float &, float &) const;
        };

        template <typename T> static yes TypeHasRealdim(typecheck< typename GetDimReal<T>::fptr, &T::getDimReal > *);
        template <typename T> static no  TypeHasRealdim(...);
    public:
        const static bool value = sizeof(TypeHasRealdim<Type>(0)) == sizeof(yes);
    };

    template<typename T>
    void get_realdim(const ValueGridMap<T> &map, float &w, float &h)
    {
        map.getDimReal(w, h);
    }

    template<typename T>
    void get_realdim(const T &map, float &w, float &h)
    {
        w = 0; h = 0;
    }

    template<typename T>
    void set_realdim(ValueGridMap<T> &map, float w, float h)
    {
        map.setDimReal(w, h);
    }

    template<typename T>
    void set_realdim(T &map, float w, float h)
    {
    }

    /*
     * Write to text-based .txt grid format, with a required step size parameter. This function is still used a
     * few places, but with some refactoring could probably be removed and the other write_txt used in its place
     */
    template<typename T>
    void write_txt(std::string filename, T *data, float step_size)
    {
        int width, height;
        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

        if (ofs.good())
        {
            ofs << data->width() << " " << data->height() << " " << step_size << std::endl;
            for (int y = 0; y < data->height(); y++)
            {
                for (int x = 0; x < data->width(); x++)
                {
                    ofs << data->get(x, y) << " ";
                }
                ofs << std::endl;
            }
        }
    }

    /*
     * Write to text-based .txt grid format. This function will automatically check if type T has and supports
     * a step size for each grid cell
     */
    template<typename T>
    void write_txt(std::string filename, const T *data)
    {
        int width, height;
        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

        bool has_realdim = type_has_realdim<T>::value;
        float step;
        if (has_realdim)
        {
            float rw, rh;
            int gw, gh;
            get_realdim(*data, rw, rh);
            /*
             * // dont think this should be implemented...add an if statement?
            assert(rw > 1e-5 && rh > 1e-5);
            if (fabs(rw) < 1e-5 && fabs(rh) < 1e-5)
            {
                throw std::runtime_error("rw and rh must both be ")
            }
            */
            //data->getDimReal(rw, rh);
            gw = data->width();
            gh = data->height();
            step = rw / gw;		// let's assume that the proportion is equal for width and height...add an assert for this...?
        }

        if (ofs.good())
        {
            ofs << data->width() << " " << data->height();
            if (has_realdim)
            {
                ofs << " " << step;
            }
            ofs << std::endl;
            for (int y = 0; y < data->height(); y++)
            {
                for (int x = 0; x < data->width(); x++)
                {
                    ofs << data->get(x, y) << " ";
                }
                ofs << std::endl;
            }
        }
    }

    /*
     * Load an elevation file from 'filename' and return as type T.
     * XXX: This function can probably be merged quite easily with load_txt since the formats are so similar
     */
    template<typename T>
    T load_elv(std::string filename)
    {
        using namespace std;

        float step, lat;
        int dx, dy;
        int width, height;
        float real_w, real_h;

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
            real_w = width * step;
            real_h = height * step;

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

    template<typename T>
    void write_elv(std::string filename, const T &data)
    {
        std::ofstream ofs(filename, std::ios_base::out | std::ios_base::trunc);

        float step;
        {
            float rw, rh;
            int gw, gh;
            get_realdim(*data, rw, rh);
            gw = data->width();
            gh = data->height();
            step = rw / gw;		// let's assume that the proportion is equal for width and height...add an assert for this...?
        }

        if (ofs.good())
        {
            ofs << data->width() << " " << data->height() << " " << step;
            ofs << " " << 0.0f;		// add dummy value for latitude - only keeping it for compatibility
            ofs << std::endl;
            for (int y = 0; y < data->height(); y++)
            {
                for (int x = 0; x < data->width(); x++)
                {
                    ofs << data->get(x, y) << " ";
                }
                ofs << std::endl;
            }
        }
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
                    float radius = tree_ptr->radius;
                    /*
                    if (tree_ptr->radius < 1) radius = 1;
                    else
                        radius = tree_ptr->radius;
                    */
                    ofs << tree_ptr->x << " " << tree_ptr->y << " " << tree_ptr->z << " " << tree_ptr->height << " " << radius << std::endl;
                }
            }

        }

    }

    /*
     * Write tree array of type T to pdb file at 'filepath'. Type T must have at least
     * float x
     * float y
     * float radius
     * float height
     * int species
     */
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

    /*
     * Write array of trees of type T to pdb file at 'filepath'. Type T must have at least
     * float x
     * float y
     * float radius
     * float height
     * int species
     */
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


    /*
     * Converts MinimalPlant object to type T, as long as type T as attributes
     * float x
     * float y
     * float radius
     * float height
     * int species
     */
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



    /*
     * Load a grid-based data file from file 'filename'. File must be in format
     * width height [stride]
     * x1 x2 x3 ... xn
     * where width and height are the number of columns and rows in the grid, respectively, and
     * stride is an optional value indicating cell size to get a corresponding real-world size for the
     * grid. xi is the ith data element, with n = width * height
     */
    template<typename T>
    T load_txt(std::string filename)
    {
        using namespace std;

        T retmap;

        int dx, dy;

        float val;
        ifstream infile;

        std::vector<std::string> file_contents;

        bool hasstep = false;
        infile.open((char *) filename.c_str(), ios_base::in);
        if (infile.is_open())
        {
            int lnum = 0;
            while (infile.good())
            {
                std::string line;
                std::getline(infile, line);
                std::stringstream linestream(line);
                int ntokens = 0;
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
                    {
                        file_contents.push_back(token);
                        ntokens++;
                    }
                }
                if (lnum == 0 && (ntokens == 3 || ntokens == 4))
                {
                    hasstep = true;
                    if (ntokens == 4)	// we disregard the latitude info in the case of txt files
                        file_contents.pop_back();
                }
                lnum++;
            }
        }
        else
        {
            throw runtime_error("Error data_importer::load_txt: unable to open file " + filename);
            abort();
        }

        int width = atoi(file_contents[0].c_str());
        int height = atoi(file_contents[1].c_str());
        float step;
        if (hasstep)
            step = atof(file_contents[2].c_str());

        int start_idx = 2;
        if (hasstep)
            start_idx = 3;

        // check if file has too many elements...
        if (file_contents.size() > width * height + start_idx)
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
        // check if file has too few elements...
        else if (file_contents.size() < width * height + start_idx)
        {
            throw runtime_error("Error: File " + filename + " does not contain the specified amount of elements. File probably corrupted.");
            abort();
        }

        // file has the right amount of elements. Proceed to put each element into the map type at the appropriate x, y location...

        retmap.setDim(width, height);

        float realw, realh;
        if (hasstep)
        {
            realw = width * step;
            realh = height * step;
            // we can also force the user of the function to pass a type T that has the setDimReal function, but let's be nice. For now.
            if (type_has_realdim<T>::value)
            {
                //retmap.setDimReal(realw, realh);
                set_realdim(retmap, realw, realh);
            }
        }


        int idx = 0;	// the for loop below increments the idx variable...
        for (auto iter = std::next(file_contents.begin(), start_idx); iter != file_contents.end(); advance(iter, 1), idx++)
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



    /*
     * Read a text-based monthly map of grid data. Format of file must be
     * width height [stride]
     * x(1) x(2) ... x(n * 12)
     * Where n = width * height. Months change the fastest, then x locations, then y locations.
     */
    template <typename T>
    std::vector< T > read_monthly_map(std::string filename)
    {
        std::ifstream ifs(filename);

        std::vector< T > mmap(12);

        int width, height;
        float step = -1.0f;

        int totel = 0;
        if (ifs.is_open())
        {
            std::string firstline;
            std::getline(ifs, firstline);
            std::stringstream ss_first(firstline);
            ss_first >> width >> height;
            if (width > 0 && width <= 10240 && height > 0 && height <= 10240)
            {
                std::for_each(mmap.begin(), mmap.end(), [&width, &height](T &mapf) { mapf.setDim(width, height);});
            }
            else
            {
                throw std::runtime_error("Size of imported monthly map is either negative, zero, too large (or file is corrupted)");
            }
            if (ss_first.good())
                ss_first >> step;
            if (step > 0.0f)
            {
                float rw = width * step, rh = height * step;
                std::for_each(mmap.begin(), mmap.end(), [&rw, &rh](T &mapf) { set_realdim(mapf, rw, rh);});
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
                totel++;
                m++;
                if (m == 12)
                {
                    m = 0;
                    x++;
                    if (x == width)
                    {
                        x = 0;
                        y++;
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

        if (totel != width * height * 12)
        {
            throw std::runtime_error("Monthly map " + filename + " contains " + std::to_string(totel) + " elements, while the required number is " + std::to_string(width * height * 12));
        }

        return mmap;
    }

    /*
     * Read monthly map from binary format, rather than text-based format. Much faster than text-based
     */
    template<typename T>
    std::vector< T > read_monthly_map_binary(std::string filename)
    {
        std::streampos filesize;
        std::ifstream ifs(filename, std::ios::binary);

        ifs.seekg(0, std::ios::end);
        filesize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        int wh[2];
        float step;

        ifs.read((char *)wh, sizeof(int) / sizeof(char) * 2);
        ifs.read((char *)&step, sizeof(float) / sizeof(char));

        int width = wh[0];
        int height = wh[1];
        std::vector<T> indata(12);

        for (int i = 0; i < 12; i++)
        {
            indata.at(i).setDim(width, height);
            indata.at(i).setDimReal(width * step, height * step);
            ifs.read((char *) indata.at(i).data(), sizeof(float) / sizeof(char) * width * height);
        }

        return indata;
    }

    /*
     * Write monthly map to binary format
     */
    template<typename T>
    void write_monthly_map_binary(std::string filename, const std::vector< T > &months)
    {
        std::vector<float> alldata;

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

        alldata.resize(width * height * 12);

        int midx = 0;
        for (const auto &m : months)
        {
            memcpy(alldata.data() + midx * width * height, m.data(), sizeof(float) * width * height);
            midx++;
        }


        float rw, rh;
        months[0].getDimReal(rw, rh);
        float step = rw / width;

        int floatstep = sizeof(float) / sizeof(char);
        int intstep = sizeof(int) / sizeof(char);
        int nchar_el = floatstep * (alldata.size() + 1) + intstep * 2;
        char *writearr = new char[nchar_el];

        memcpy(writearr, &width, sizeof(int));
        memcpy(writearr + intstep, &height, sizeof(int));
        memcpy(writearr + 2 * intstep, &step, sizeof(float));
        memcpy(writearr + 2 * intstep + floatstep, alldata.data(), sizeof(float) * alldata.size());

        std::ofstream ofs(filename, std::ofstream::binary);
        ofs.write(writearr, sizeof(char) * nchar_el);

        delete writearr;
    }

    template<class, template<class, class...> class >
    struct is_instance : public std::false_type {};

    template<class...Ts, template<class, class...> class U>
    struct is_instance<U<Ts...>, U> : public std::true_type {};

    /*
     * Write monthly map to text-based format
     */
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
            ofs << width << " " << height << " ";
            float rw, rh;
            get_realdim(months.at(0), rw, rh);
            if (rw > 0.0f && rh > 0.0f)
            {
                ofs << rw / width;
            }
            ofs << "\n";

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

    /*
     * Average a monthly map to a single map.
     * template type T is the output type, while type U is the input type for a single month
     */
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

        float rw = 0.0f, rh = 0.0f;
        /*
        if (is_instance<T, ValueGridMap>{} && is_instance<U, ValueGridMap>{})
        {
            mmap.at(0).getDimReal(rw, rh);
            avgmap.setDimReal(rw, rh);
        }
        */

        get_realdim(mmap.at(0), rw, rh);
        set_realdim(avgmap, rw, rh);

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

    /*
     * Load a monthly map from file, then average and return the averaged map
     */
    template<typename T>
    T load_average_mmap(std::string filename)
    {
        auto mmap = read_monthly_map<T>(filename);
        return average_mmap<T, T>(mmap);
    }

}

#endif //DATA_IMPORTER_H
