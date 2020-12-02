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


#include <iostream>
#include <experimental/filesystem>
//#include "distribution.h"
//#include "GridDistribs.h"
#include "dice.h"
#include "ClusterMatrices.h"
#include "ClusterDistribDerivator.h"
#include "data_importer/AbioticMapper.h"
#include "kmeans.h"

namespace fs = std::experimental::filesystem;

ValueGridMap<float> calc_adaptsun(const std::vector<basic_tree> &trees, const ValueGridMap<float> &landsun, const data_importer::common_data &cdata)
{
    ValueGridMap<float> adaptsun;
    adaptsun.setDim(landsun);

    int dx, dy;
    landsun.getDim(dx, dy);
    float rw, rh;
    landsun.getDimReal(rw, rh);

    float *datbegin = adaptsun.data();
    memcpy(datbegin, landsun.data(), sizeof(float) * dx * dy);

    auto trim = [](int &v, int dim) { if (v < 0) v = 0; if (v >= dim) v = dim - 1; };

    for (auto &t : trees)
    {
        float sxr = (t.x - t.radius);
        float exr = (t.x + t.radius);
        float syr = (t.y - t.radius);
        float eyr = (t.y + t.radius);
        xy<int> scoords = landsun.togrid(sxr, syr);
        xy<int> ecoords = landsun.togrid(exr, eyr);
        int sx = scoords.x, sy = scoords.y;
        int ex = ecoords.x, ey = ecoords.y;

        trim(sx, dx);
        trim(ex, dx);
        trim(sy, dy);
        trim(ey, dy);

        xy<int> tgrid = landsun.togrid(t.x, t.y);

        xy<int> maxradc = landsun.togrid(t.radius, t.radius);
        int maxradsq = maxradc.x * maxradc.x;
        for (int y = sy; y <= ey; y++)
        {
            for (int x = sx; x < ex; x++)
            {
                int d = (tgrid.x - x) * (tgrid.x - x) + (tgrid.y - y) * (tgrid.y - y);
                if (d <= maxradsq)
                {
                    float alpha = cdata.canopy_and_under_species.at(t.species).alpha;
                    adaptsun.set(x, y, adaptsun.get(x, y) * (1.0f - alpha));
                }
            }
        }
    }

    return adaptsun;
}

int main(int argc, char * argv [])
{
    //std::vector<std::string> suffixes = {"0_0", "256_0", "256_256"};
    //std::vector<std::string> suffixes = {"11", "12", "21", "22"};
    //std::vector<std::string> suffixes = {"128_1", "128_2", "128_3"};
    //std::vector<std::string> suffixes = {"_1", "_2"};

    if (argc != 6)
    {
        std::cout << "Usage: main <data dir> <database filename> <kmeans number of clusters> <kmeans number of iterations> <output dir>" << std::endl;
        return 1;
    }

    // NOTE, TODO: Due to format of this main function, haven't actually made do anything with argv[1] yet...

    std::string dirname = argv[1];
    std::string db_filename = argv[2];
    int nmeans = std::stoi(argv[3]);
    int niters = std::stoi(argv[4]);
    std::string outdir = argv[5];

    fs::path p(dirname);

    if (!fs::exists(p))
    {
        char errstr[256];
        sprintf(errstr, "Directory %s does not exist", dirname.c_str());
        throw std::runtime_error(errstr);
    }

    data_importer::common_data cdata(db_filename);

    std::vector<std::string> datadirs;

    for (auto &el : fs::directory_iterator(p))
    {
        std::string curr_dirname = el.path().c_str();
        if (!fs::is_directory(curr_dirname))
        {
            continue;
        }
        data_importer::data_dir ddir(curr_dirname, 1);
        std::string missingstr = "";
        if (!fs::exists(ddir.canopy_fnames.at(0)))
            missingstr += "canopy pdb file\n";

        if (!fs::exists(ddir.undergrowth_fnames.at(0)))
            missingstr += "undergrowth pdb file\n";
        if (!fs::exists(ddir.wet_fname))
            missingstr += "Moisture file\n";
        //if (!fs::exists(ddir.sun_fname))
        //    missingstr += "Sunlight landscape file\n";
        if (!fs::exists(ddir.sun_tree_fname))
            missingstr += "Sunlight canopy file\n";
        if (!fs::exists(ddir.slope_fname))
            missingstr += "Slope file\n";
        if (!fs::exists(ddir.temp_fname))
            missingstr += "Temperature file\n";
        if (missingstr.size() > 0)
        {
            std::cout << "Directory " << curr_dirname << " has the following required files missing. Skipping it: " << std::endl << missingstr << std::endl;
            continue;
        }
        else
        {
            while (curr_dirname.back() == '/')
                curr_dirname.pop_back();
            datadirs.push_back(curr_dirname);
        }
    }

    std::cout << "Directories found in " << dirname << ":" << std::endl;
    for (auto &d : datadirs)
    {
        std::cout << d << std::endl;
    }


    if (fs::exists(outdir) && !fs::is_directory(outdir))
    {
        char errstr[256];
        sprintf(errstr, "The required output directory %s already exists, but not as a directory. Aborting.", errstr);
        throw std::runtime_error(errstr);
    }
    else if (!fs::exists(outdir))
    {
        fs::create_directories(outdir);
    }

    ClusterDistribDerivator derivator(datadirs, cdata);
    derivator.do_kmeans(nmeans, niters, cdata);
    auto all_clusterdistribs = derivator.deriveHistogramMatricesSeparate(cdata);

    if (all_clusterdistribs.size() != datadirs.size())
    {
        throw std::runtime_error("number of distribution files to be written not equal to number of valid directories found. Aborting");
    }

    std::list<ClusterMatrices>::iterator cliter = all_clusterdistribs.begin();
    for (int i = 0; i < all_clusterdistribs.size(); i++, std::advance(cliter, 1))
    {
        auto targetpath = fs::path(datadirs[i]);
        auto outdir_path = fs::path(outdir);
        std::string dataset_name = targetpath.filename().c_str();
        dataset_name += "_distribs.clm";
        std::string target_out = (outdir_path / dataset_name).c_str();

        std::cout << "Writing to file " << target_out << "..." << std::endl;
        cliter->write_clusters(target_out);
        std::cout << "Done with target number " << i + 1 << " of " << all_clusterdistribs.size() << std::endl;
    }

    /*
    int target_num = 0;
    for (auto &target_in : all)
    {
        auto targetpath = fs::path(target_in);
        auto outdir_path = fs::path(outdir);
        std::string dataset_name = targetpath.filename().c_str();
        dataset_name += "_distribs.clm";

        std::string target_out = (outdir_path / dataset_name).c_str();
        auto clusterdistribs = ClusterMatrices::CreateClusterMatrices(target_in, minmax_ranges, clusters, "/home/konrad/EcoSynth/ecodata/sonoma.db");
        clusterdistribs->check_plants_validity();
        std::cout << "All plants valid" << std::endl;
        std::cout << "Writing to file " << target_out << "..." << std::endl;
        clusterdistribs->write_clusters(target_out);
        std::cout << "Done with target number " << target_num + 1 << " of " << all.size() << std::endl;
        target_num++;

        clusterdistribs->write_region_sizes("/home/konrad/Desktop/deriveSizes.txt");
    }
    */

	return 0;
}

        /*
        auto clusterdistribs = ClusterMatrices::CreateClusterMatrices(target_in,
            all,
             "/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db", div_method::EQUAL_SPLIT,
            4, niters);
        */
