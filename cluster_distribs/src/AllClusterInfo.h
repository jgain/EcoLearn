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


#ifndef ALLCLUSTERINFO_H
#define ALLCLUSTERINFO_H

#include "EcoSynth/kmeans/src/kmeans.h"

#include <unordered_map>
#include <map>
#include <random>

// forward declarations
class HistogramDistrib;
class HistogramMatrix;
class ClusterData;
namespace data_importer
{
    struct common_data;
}


/*
 * This class imports cluster data from a file. It also allows random clusters to be sampled if imported
 * from more than one file, which yields a ClusterData object for each cluster
 */

struct AllClusterInfo
{
    AllClusterInfo(std::vector<std::string> filenames, const data_importer::common_data &cdata);

    std::vector<std::array<float, kMeans::ndim> > kclusters;
    std::array<std::pair<float, float>, kMeans::ndim> minmax_ranges;
    std::vector<std::map<int, HistogramMatrix> > all_distribs;
    std::vector<std::map<int, std::map<int, HistogramDistrib> > > all_sizemtxes;
    std::vector<std::unordered_map<int, std::unordered_map<int, float> > > all_specratios;
    std::vector<std::map<int, float> > all_densities;

    int nmeans;
    int nclusters;		// the actual number of clusters (i.e. means * (pow(2, nsubbiomes) - 1))
    std::default_random_engine gen;
    std::uniform_real_distribution<float> unif;

    void print_kmeans_clusters();
    void print_densities();

    ClusterData sample_cluster(int clusteridx, const data_importer::common_data &cdata);

protected:
    static std::array<std::pair<float, float>, kMeans::ndim> import_minmax_ranges(std::ifstream &ifs);
    static std::vector<std::array<float, kMeans::ndim> > import_clusters(std::ifstream &ifs);

    static std::map<int, HistogramDistrib> read_sizedistribs(std::ifstream &ifs,
                                                             const data_importer::common_data &cdata);

    static void read_species_props(std::ifstream &ifs, int clusterid,
                                   std::unordered_map<int, std::unordered_map<int, float> > &specratios);
};

#endif  	// ALLCLUSTERINFO_H
