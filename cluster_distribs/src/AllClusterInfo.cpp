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


#include "AllClusterInfo.h"
#include "ClusterData.h"


AllClusterInfo::AllClusterInfo(std::vector<std::string> filenames, const data_importer::common_data &cdata)
{

    std::vector<std::array<float, kMeans::ndim> > prev_kclusters;
    std::array<std::pair<float, float>, kMeans::ndim> prev_minmax_ranges;

    bool firstfile = true;

    // go through each file and import clusterdata for it
    for (auto &fname : filenames)
    {
        std::ifstream ifs(fname);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Could not open cluster file at " + fname);
        }
        minmax_ranges = import_minmax_ranges(ifs);
        kclusters = import_clusters(ifs);
        int nmeans = kclusters.size();

        // if it's NOT the first file we are importing from, then we need to check if the minmax ranges
        // and cluster means for each clusterfile is the same. If not, throw error, because having clusterdata
        // from more than one file only makes sense if clustermeans and minmax ranges are the same (another part
        // the codebase, the ClusterDistribDerivator class, ensures that these are equal when deriving clusterfiles
        // simultaneously from more than one landscape, which will produce a clusterfile for each landscape)
        if (!firstfile)
        {

            // check if minmax ranges are equal
            bool valid = true;
            if (minmax_ranges.size() != prev_minmax_ranges.size())
            {
                valid = false;
            }
            if (valid)
                for (int i = 0; i < minmax_ranges.size(); i++)
                {
                    if (minmax_ranges.at(i) != prev_minmax_ranges.at(i))
                    {
                        valid = false;
                        break;
                    }
                }
            if (!valid)
            {
                std::string errstr = "minmax ranges is not the same for the following clusterfiles: \n";
                for (auto &otherfname : filenames)
                {
                    errstr += otherfname + "\n";
                }
                errstr += "Current clusterfile: " + fname + "\n";
                throw std::runtime_error(errstr.c_str());
            }
            prev_minmax_ranges = minmax_ranges;

            // check if cluster means are equal
            if (kclusters.size() != prev_kclusters.size())
            {
                valid = false;
            }
            if (valid)
                for (int i = 0; i < kclusters.size(); i++)
                {
                    if (kclusters.at(i) != prev_kclusters.at(i))
                    {
                        valid = false;
                        break;
                    }
                }
            if (!valid)
            {
                std::string errstr = "cluster means not the same for the following clusterfiles: \n";
                for (auto &otherfname : filenames)
                {
                    errstr += otherfname + "\n";
                }
                errstr += "Current clusterfile: " + fname + "\n";
                throw std::runtime_error(errstr.c_str());
            }
        }
        firstfile = false;
        prev_minmax_ranges = minmax_ranges;
        prev_kclusters = kclusters;

        // total number of clusters is not the same as number of means, because there is a cluster set
        // for each subbiome combination also. Number of subbiome combinations is 2 ^ N, where N is
        // number of subbiomes
        nclusters = nmeans * std::pow(2, cdata.subbiomes.size());

        int count = 0;
        all_distribs.push_back(std::map<int, HistogramMatrix>());
        all_sizemtxes.push_back(std::map<int, std::map<int, HistogramDistrib> >());
        all_specratios.push_back(std::unordered_map<int, std::unordered_map<int, float> >());
        all_densities.push_back(std::map<int, float>());
        while (ifs.good() && count < nclusters)
        {
            // density gets read directly
            float density;
            ifs >> density;
            all_densities.back().emplace(count, density);

            // we delegate the reading of the spatial and size distributions, and species proportions,
            // to these functions below
            all_distribs.back().emplace(count, HistogramMatrix::read_matrix(ifs));
            std::map<int, HistogramDistrib> distribs = read_sizedistribs(ifs, cdata);
            all_sizemtxes.back().emplace(count, distribs);
            read_species_props(ifs, count, all_specratios.back());
            /*
            try {
                info.all_distribs.back().emplace(count, HistogramMatrix::read_matrix(ifs));
                std::map<int, HistogramDistrib> distribs = read_sizedistribs(ifs);
                info.all_sizemtxes.back().emplace(count, distribs);
            } catch (std::runtime_error &e)
            {
                break;
            }
            */
            count++;
        }
        if (count != nclusters)
        {
            throw std::runtime_error("Number of distribution matrices not equal to number of clusters in clusterfile " +
                                     fname);
        }
    }

    // Check if number of clusters is the same in all clusterfile data
    // FIXME: this just checks the number of clusters via spatial distribution structure,
    // 		  perhaps there's a better way?
    for (auto iter = std::next(all_distribs.begin(), 1); iter != all_distribs.end(); advance(iter, 1))
    {
        if (iter->size() != all_distribs.front().size())
        {
            std::string errstr = "Number of distributions (clusters) not the same in the following clusterfiles: \n";
            for (auto &otherfname : filenames)
            {
                errstr += otherfname + "\n";
            }
            throw std::runtime_error(errstr.c_str());
            //return std::unique_ptr<ClusterMatrices>(nullptr);
        }
    }
}

ClusterData AllClusterInfo::sample_cluster(int clusteridx, const data_importer::common_data &cdata)
{
    int dataset_idx = unif(gen) * this->all_distribs.size();
    return ClusterData(all_distribs.at(dataset_idx).at(clusteridx), all_sizemtxes.at(dataset_idx).at(clusteridx),
                       all_densities.at(dataset_idx).at(clusteridx), all_specratios.at(dataset_idx).at(clusteridx),
                       cdata);
}

void AllClusterInfo::print_kmeans_clusters()
{
    std::cout << "Number of clusters: " << kclusters.size() << ": " << std::endl;
    std::cout << "------------------------------------" << std::endl;

    for (auto &arr : kclusters)
    {
        for (auto &el : arr)
        {
            std::cout << el << " ";
        }
        std::cout << std::endl;
    }
}

void AllClusterInfo::print_densities()
{
    int count = 1;
    for (const std::map<int, float> &fd : all_densities)
    {
        std::cout << "Densities for file " << count << ": " << std::endl;
        std::cout << "------------------------" << std::endl;
        for (const auto &p : fd)
        {
            std::cout << "Cluster " << p.first << " density: " << p.second << std::endl;
        }
        count++;
    }
}



/* Static helper functions
 * ------------------------------------------------------
 */

std::vector<std::array<float, kMeans::ndim> > AllClusterInfo::import_clusters(std::ifstream &ifs)
{
    int nmeans;
    std::vector<std::array<float, kMeans::ndim> > clusters;
    std::string numstr;

    std::string mean_linestring;
    std::getline(ifs, mean_linestring);
    std::stringstream meanstream(mean_linestring);

    meanstream >> nmeans;
    if (!meanstream.eof())		// check if we are reading file correctly
    {
        throw std::runtime_error("Line that contains number of means has more than one value in import_clusters. File is invalid");
    }
    if (meanstream.fail())
    {
        throw std::runtime_error("Could not read number of means in import_clusters. File is invalid");
    }

    for (int i = 0; i < nmeans; i++)
    {
        std::array<float, kMeans::ndim> clmean;
        for (int i = 0; i < kMeans::ndim; i++)
        {
            ifs >> numstr;
            clmean.at(i) = std::stof(numstr);
        }
        clusters.push_back(clmean);
        if (!ifs.good())
        {
            throw std::runtime_error("Faulty input file. Mistake encountered at line " + std::to_string(i));
        }
    }
    return clusters;
}

std::array<std::pair<float, float>, kMeans::ndim> AllClusterInfo::import_minmax_ranges(std::ifstream &ifs)
{
    std::array<std::pair<float, float>, kMeans::ndim> minmax_ranges;
    for (int i = 0; i < kMeans::ndim; i++)
    {
        std::string line;
        std::getline(ifs, line);
        std::stringstream sstr(line);
        sstr >> minmax_ranges.at(i).first;
        sstr >> minmax_ranges.at(i).second;
        if (sstr.fail())		// we do this check here, to see if we could read both values. If not, we throw exception
        {
            throw std::runtime_error("Could not read two values for dimension " + std::to_string(i) +
                                     " (zero-indexed) in import_minmax_ranges. File is invalid");
        }
        //if (std::getline(sstr, line))		// if end of line has not been reached here yet, throw exception
        if (!sstr.eof())		// if end of line has not been reached here yet, throw exception
        {
            throw std::runtime_error("minmax range for dimension " + std::to_string(i) +
                                     " (zero-indexed) has more than two values in import_minmax_ranges. File is invalid");
        }
        //ifs >> minmax_ranges.at(i).first;
        //ifs >> minmax_ranges.at(i).second;
    }
    return minmax_ranges;
}

std::map<int, HistogramDistrib> AllClusterInfo::read_sizedistribs(std::ifstream &ifs,
                                                                  const data_importer::common_data &cdata)
{
    int mapsize, nbins;
    float maxsize, minsize;

    ifs >> mapsize;

    std::map<int, HistogramDistrib> distribmap;

    for (int i = 0; i < mapsize; i++)
    {
        int specid;
        ifs >> specid;
        ifs >> nbins;
        ifs >> minsize;
        ifs >> maxsize;
        std::vector<common_types::decimal> bins(nbins, -1.0f);
        auto &sp = cdata.canopy_and_under_species.at(specid);
        float maxheight = cdata.canopy_and_under_species.at(specid).maxhght;
        if (maxheight > 5)
            maxheight = 5;

        for (int i = 0; i < nbins; i++)
        {
            ifs >> bins[i];
        }

        distribmap.emplace(specid, HistogramDistrib(nbins, 0, minsize, maxsize, bins));
        distribmap.at(specid).buildrnddistrib(100, std::chrono::steady_clock::now().time_since_epoch().count());
    }
    return distribmap;
}

void AllClusterInfo::read_species_props(std::ifstream &ifs, int clusterid,
                                        std::unordered_map<int, std::unordered_map<int, float> > &specratios)
{
    int nspec;
    ifs >> nspec;

    for (int i = 0; i < nspec; i++)
    {
        int specid;
        float ratio;
        ifs >> specid >> ratio;
        specratios[clusterid][specid] = ratio;
    }
}

/*
 * the previous way in which we chose random datasets for each cluster. Had some thorough tests that we might want to include in this class
 *
void ClusterMatrices::select_random_clusters(const ClusterMatrices::AllClusterInfo &info,
                                             std::map<int, HistogramMatrix> &distribs,
                                             std::map<int, std::map<int, HistogramDistrib> > &sizemtxes,
                                             std::unordered_map<int, std::unordered_map<int, float> > &specratios)
{
    std::default_random_engine gen(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> unif_int(0, info.all_distribs.size() - 1);

    int nclusters = info.nclusters;

    // select random distribution matrix from all landscapes given (for each cluster)
    // for each cluster i, we select a distribution from the idx-th file (or landscape/dataset)
    for (int i = 0; i < nclusters; i++)
    {
        /// FIXME: what happens if clusterfile idx, does not have info on cluster i? Or do we have info for each cluster, on all clusterfiles?
        int idx = unif_int(gen);

        //std::cout << "Using distribution matrix from clusterfile " << idx << std::endl;

        distribs.emplace(i, info.all_distribs[idx].at(i));
        sizemtxes.emplace(i, info.all_sizemtxes[idx].at(i));
        specratios.emplace(i, info.all_specratios[idx].at(i));
    }

    bool firstfile = true;
    std::vector<int> prev_plantids, prev_canopyids;
    for (auto &d : distribs)
    {
        std::vector<int> curr_plantids, curr_canopyids;
        curr_plantids = d.second.get_plant_ids(), curr_canopyids = d.second.get_canopy_ids();
        if (!firstfile)
        {
            if (curr_plantids.size() != prev_plantids.size())
            {
                throw std::runtime_error("number of plantids not identical in some clusterfile imports");
            }
            if (curr_canopyids.size() != prev_canopyids.size())
            {
                throw std::runtime_error("number of canopyids not identical in some clusterfile imports");
            }
            for (int i = 0; i < curr_plantids.size(); i++)
            {
                if (curr_plantids.at(i) != prev_plantids.at(i))
                {
                    throw std::runtime_error("plantids not identical in some clusterfile imports");
                }
            }
            for (int i = 0; i < curr_canopyids.size(); i++)
            {
                if (curr_canopyids.at(i) != prev_canopyids.at(i))
                {
                    throw std::runtime_error("canopyids not identical in some clusterfile imports");
                }
            }
        }
        prev_plantids = curr_plantids;
        prev_canopyids = curr_canopyids;
        firstfile = false;
    }

}
*/
