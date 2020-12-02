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


#include "ClusterMatrices.h"
#include "AllClusterInfo.h"
#include "common.h"
#include "common/constants.h"
#include "EcoSynth/kmeans/src/kmeans.h"

#include <data_importer/data_importer.h>
#include <map>
#include <list>
#include <functional>
#include <random>
#include <iomanip>


// this def is for debugging add/remove of plant effects on histograms
#define ADDREMOVE_DEBUG
#undef ADDREMOVE_DEBUG

ClusterMatrices::ClusterMatrices(AllClusterInfo &clusterinfo, const data_importer::common_data &cdata)
    : classign(clusterinfo), cdata(cdata)
{

    //this->select_random_clusters(clusterinfo, distribs, sizemtxes, this->species_props);

     for (int clusteridx = 0; clusteridx < clusterinfo.nclusters; clusteridx++)
     {
         clusters_data.emplace(clusteridx, clusterinfo.sample_cluster(clusteridx, cdata));
     }


    // TODO: find a better way to assign plant and canopy ids? Don't really want to depend on hmatrices index zero being available at all times
    // XXX: the standard way to assign plant and canopy ids could be the one in ClusterMatrices::ClusterMatrices(const data_importer::common_data &cdata)?
    plant_ids = clusters_data.at(0).get_plant_ids();
    canopy_ids = clusters_data.at(0).get_canopy_ids();
}


ClusterMatrices::ClusterMatrices(AllClusterInfo &&clusterinfo, const data_importer::common_data &cdata)
    : ClusterMatrices(clusterinfo, cdata)
{

}


ClusterMatrices::ClusterMatrices(const data_importer::common_data &cdata)
    : cdata(cdata)
{
    for (auto &canopyspecpair : this->cdata.all_species)
    {
        canopy_ids.push_back(canopyspecpair.first);
    }
    for (auto &underspecpair : this->cdata.canopy_and_under_species)
    {
        plant_ids.push_back(underspecpair.first);
    }
}


ClusterMatrices::ClusterMatrices(std::vector<std::string> clusterfilenames, const data_importer::common_data &cdata)
    : ClusterMatrices(AllClusterInfo(clusterfilenames, cdata), cdata)
{
}


HistogramMatrix &ClusterMatrices::get_locmatrix(int idx)
{
    return get_cluster(idx).get_locmatrix();
}


const HistogramMatrix &ClusterMatrices::get_locmatrix_nocheck(int idx) const
{
    return get_cluster_nocheck(idx).get_locmatrix();
}


std::map<int, HistogramDistrib> &ClusterMatrices::get_sizematrix(int idx)
{
    return get_cluster(idx).get_sizematrix();
}


float ClusterMatrices::get_region_density(int idx)
{
    return get_cluster(idx).get_density();
}


float ClusterMatrices::get_region_density(int idx) const
{
    if (clusterdata_exists(idx))
        return get_cluster_nocheck(idx).get_density();
    else
        return -1.0f;
}


ClusterData &ClusterMatrices::get_cluster(int idx)
{
    if (clusters_data.count(idx))
        return clusters_data.at(idx);
    else
    {
        /* if no cluster data exists for cluster at index idx, then create a new structure that can hold this data,
         * then return newly-created ClusterData structure
         */
        clusters_data.emplace(idx, ClusterData(std::set<int>(plant_ids.begin(), plant_ids.end()),
                                         std::set<int>(canopy_ids.begin(), canopy_ids.end()),
                                         HistogramMatrix::global_under_metadata,
                                         HistogramMatrix::global_canopy_metadata,
                                         cdata));
        return clusters_data.at(idx);
    }
}


const ClusterData &ClusterMatrices::get_cluster_nocheck(int idx) const
{
    return clusters_data.at(idx);
}


bool ClusterMatrices::clusterdata_exists(int idx) const
{
    return clusters_data.count(idx);
}


int ClusterMatrices::get_nclusters() const
{
    return clusters_data.size();
}


void ClusterMatrices::check_matrices()
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> unif;
    constexpr float prob = 0.1f;

    std::map<int, std::vector<std::pair<int, int> > > zeros;
    std::map<int, std::vector<std::pair<int, int> > > noteq_one;
    for (std::pair<const int, ClusterData> &clp : clusters_data)
    {
        ClusterData &cldata = clp.second;
        auto &mtx = cldata.get_locmatrix();
        if (mtx.has_active_distribs())
        {
            int id = clp.first;
            noteq_one[id] = mtx.get_noteqone_active();
            zeros[id] = mtx.get_zero_active();
        }
    }
    std::cout << "Zero distribs: ";
    for (auto &p : zeros)
    {
        int id = p.first;
        std::cout << "For cluster: " << id << std::endl;
        auto &vec = p.second;
        for (auto &rowcol : vec)
        {
            std::cout << "(" << rowcol.first << ", " << rowcol.second << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "Not equal to one distribs: ";
    for (auto &p : noteq_one)
    {
        int id = p.first;
        std::cout << "For cluster: " << id << std::endl;
        auto &vec = p.second;
        for (auto &rowcol : vec)
        {
            std::cout << "(" << rowcol.first << ", " << rowcol.second << ") ";
        }
        std::cout << std::endl;
    }
}


void ClusterMatrices::updateHistogramLocMatrix(const basic_tree &reftree, const std::vector<basic_tree> &othertrees, const ClusterMaps &clmaps, ClusterMatrices *benchmark)
{
    int clusteridx = clmaps.get_cluster_idx(reftree.x, reftree.y);
    if (clusteridx < 0)
    {
        return;
    }
    float width = clmaps.get_width();
    float height = clmaps.get_height();

    // if 'benchmark' pointer is not null, it indicates that we want to derive the distributions
    // in accordance with some benchmark model. So, if a certain distribution is inactive in the benchmark model,
    // we ignore that distribution when deriving our model here
    HistogramMatrix *benchmtx = nullptr;
    if (benchmark)
        benchmtx = &benchmark->get_locmatrix(clusteridx);

    const auto &canopytrees = clmaps.get_canopytrees();

    // refmtx is the distribution matrix to which we will add effects (from the point of view of undergrowth plant
    // 'refplnt'.
    HistogramMatrix &refmtx = get_locmatrix(clusteridx);

    // add effects of undergrowth plants relative to undergrowth plant 'reftree'
    refmtx.addPlantsUndergrowth(reftree, othertrees, width, height, benchmtx);
    // add effects of canopy trees relative to undergrowth plant 'reftree'
    refmtx.addPlantsCanopy(reftree, canopytrees, width, height, benchmtx);
}


void ClusterMatrices::updateHistogramSizeMatrix(const basic_tree &reftree, const ClusterMaps &clmaps)
{
    int clusteridx = clmaps.get_cluster_idx(reftree.x, reftree.y);
    if (clusteridx < 0)
    {
        return;
    }

    std::map<int, HistogramDistrib> &sizemtx = get_sizematrix(clusteridx);

    sizemtx.at(reftree.species).add_dist(reftree.height, normalize_method::COMPLETE);
}


void ClusterMatrices::check_maxdists()
{
    int nugcheck = 0;
    int ncanopycheck = 0;
    for (std::pair<const int, ClusterData> &clpair : clusters_data)
    {
        HistogramMatrix &mtx = clpair.second.get_locmatrix();
        auto canopyspecs = cdata.all_species;
        auto allspecs = cdata.canopy_and_under_species;
        for (auto &p : canopyspecs)
        {
            for (auto &p2 : allspecs)
            {
                float maxdist = mtx.get_distrib_canopy(p.first, p2.first).getmax();
                ncanopycheck++;
                if (maxdist != HistogramMatrix::global_canopy_metadata.maxdist)
                {
                    std::cout << "number of undergrowth distribs checked: " << nugcheck << std::endl;
                    std::cout << "number of canopy distribs checked: " << ncanopycheck << std::endl;
                    throw std::runtime_error("maxdist for canopy species mismatch");
                }
            }
        }
        for (auto &p : allspecs)
        {
            for (auto &p2 : allspecs)
            {
                float maxdist = mtx.get_distrib_undergrowth(p.first, p2.first).getmax();
                nugcheck++;
                if (maxdist != HistogramMatrix::global_under_metadata.maxdist)
                {
                    std::cout << "number of undergrowth distribs checked: " << nugcheck << std::endl;
                    std::cout << "number of canopy distribs checked: " << ncanopycheck << std::endl;
                    throw std::runtime_error("maxdist for undergrowth species mismatch");
                }
            }
        }
    }
    std::cout << "number of undergrowth distribs checked: " << nugcheck << std::endl;
    std::cout << "number of canopy distribs checked: " << ncanopycheck << std::endl;
}


float ClusterMatrices::canopymtx_diff(const ClusterMatrices *other) const
{
    float sum = 0.0f;
    int all_nactive = 0;
    for (auto &hp : clusters_data)
    {
        int nactive;
        std::cout << "Differencing cluster id " << hp.first << std::endl;
        if (!other->clusters_data.count(hp.first))		// check that both have this cluster
        {
            throw std::out_of_range("in ClusterMatrices::canopymtx_diff, other does not have same cluster indices as this");
        }
        const ClusterData &cldata = hp.second;
        if (other->clusterdata_exists(hp.first))
            sum += cldata.get_locmatrix().canopymtx_diff(other->get_locmatrix_nocheck(hp.first), nactive);
        all_nactive += nactive;
    }

    return sum / all_nactive;
}


void ClusterMatrices::write_sizedistribs(std::ofstream &ofs, const std::map<int, HistogramDistrib> &distribs)
{

    // start with number of species
    ofs << distribs.size() << "\n";
    for (const auto &sizepair : distribs)
    {
        int specid = sizepair.first;
        const HistogramDistrib &sizedistrib = sizepair.second;
        int nbins = sizedistrib.getnumbins();
        float minsize = sizedistrib.getmin();
        float maxsize = sizedistrib.getmax();
        // metadata for the distribution
        ofs << specid << " ";
        ofs << nbins << " " << minsize << " " << maxsize << std::endl;
        const auto &bins = sizedistrib.getbins();
        // write out bins separated by whitespace (on their own line)
        for (int i = 0; i < bins.size(); i++)
        {
            auto bval = bins.at(i);
            ofs << bval;
            if (i < bins.size() - 1)
                ofs << " ";
        }
        ofs << "\n";
    }
}




void ClusterMatrices::select_random_clusters(const AllClusterInfo &info,
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
        // select from random file with index 'idx'
        int idx = unif_int(gen);

        distribs.emplace(i, info.all_distribs[idx].at(i));
        sizemtxes.emplace(i, info.all_sizemtxes[idx].at(i));
        specratios.emplace(i, info.all_specratios[idx].at(i));
    }

    // check that canopy and undergrowth plant species ids are the same in all files
    // XXX: test can be more extensive? I.e. test more things? Is this test even necessary?
    // XXX: perhaps do this test in a more appropriate place?
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

void ClusterMatrices::set_plant_ids(std::set<int> plant_ids)
{
    this->plant_ids.clear();
    for (auto &id : plant_ids)
        this->plant_ids.push_back(id);
}

void ClusterMatrices::set_canopy_ids(std::set<int> canopy_ids)
{
    this->canopy_ids.clear();
    for (auto &id : canopy_ids)
        this->canopy_ids.push_back(id);
}

const std::vector<int> &ClusterMatrices::get_canopy_ids() const
{
    return canopy_ids;
}

const std::vector<int> &ClusterMatrices::get_plant_ids() const
{
    return plant_ids;
}

bool ClusterMatrices::is_equal(const ClusterMatrices &other) const
{
    if (this->clusters_data.size() != other.clusters_data.size())
        return false;

    for (const auto &p : this->clusters_data)
    {
        if (!other.clusters_data.count(p.first))
            return false;
    }
    for (const auto &p : other.clusters_data)
    {
        if (!this->clusters_data.count(p.first))
            return false;
    }

    // go through all clusters of both models and check if spatial distribution
    // matrices are equal
    bool equal = true;
    for (const auto &p : this->clusters_data)
    {
        const ClusterData &otherdata = other.clusters_data.at(p.first);
        const ClusterData &thisdata = p.second;
        if (!thisdata.get_locmatrix().is_equal(otherdata.get_locmatrix()))
            equal = false;
    }
    return equal;
}

HistogramDistrib::Metadata ClusterMatrices::get_undergrowth_meta() const
{
    return HistogramMatrix::global_under_metadata;
}

HistogramDistrib::Metadata ClusterMatrices::get_canopy_meta() const
{
    return HistogramMatrix::global_canopy_metadata;
}

// perhaps define this id to idx mapping in this class also? Then these two functions below can be const
// on the other hand, it's probably better to have this class not 'know' about the mapping, but only the
// ids. It's the job of the HistogramMatrix class to know about the id to idx mapping
std::vector<int> ClusterMatrices::get_under_to_idx()
{
    return get_locmatrix(0).get_underid_to_idx();
}

std::vector<int> ClusterMatrices::get_canopy_to_idx()
{
    return get_locmatrix(0).get_canopyid_to_idx();
}

void ClusterMatrices::show_region_densities()
{
    for (auto &p : clusters_data)
    {
        std::cout << "Region " << p.first << " density: " << p.second.get_density() << std::endl;
    }
}

void ClusterMatrices::write_clusters(std::string out_filename)
{
    std::ofstream ofs(out_filename);

    const auto &minmax_ranges = classign.get_minmax_ranges();

    for (int i = 0; i < kMeans::ndim; i++)
    {
        ofs << minmax_ranges.at(i).first << " ";
        ofs << minmax_ranges.at(i).second << "\n";
    }

    ofs << classign.get_nmeans();
    ofs << "\n";

    const auto &kclusters = classign.get_means();

    for (auto &clmean : kclusters)
    {
        for (int i = 0; i < clmean.size(); i++)
        {
            ofs << clmean.at(i);
            if (i != clmean.size() - 1)
            {
                ofs << " ";
            }
        }
        ofs << "\n";
    }

    int nclusters = classign.get_nmeans() * std::pow(2, cdata.subbiomes.size());

    for (int i = 0; i < nclusters; i++)
    {
        HistogramMatrix &mtx = get_locmatrix(i);
        // write region density on its own line
        ofs << get_region_density(i) << "\n";
        // delegate writing of spatial distribution matrix to HistogramMatrix::write_matrix function
        mtx.write_matrix(ofs);
        // delegate writing of size distributions to write_sizedistribs function
        write_sizedistribs(ofs, get_sizematrix(i));
        // delegate writing of species proportions to write_species_props function
        write_species_props(ofs, i);
    }
}

void ClusterMatrices::write_species_props(std::ofstream &ofs, int clusterid)
{
    ofs << plant_ids.size() << "\n";
    for (auto &id : plant_ids)
    {
        ofs << id << " " << get_cluster(clusterid).get_species_ratio(id) << "\n";
    }
}



void ClusterMatrices::set_cdata(const data_importer::common_data &cdata_arg)
{
    cdata = data_importer::common_data(cdata_arg);
}

void ClusterMatrices::set_cluster_params(const std::vector<std::array<float, kMeans::ndim> > &kclusters, const std::array<std::pair<float, float>, kMeans::ndim> &minmax_pairs)
{
    set_cluster_params(ClusterAssign(kclusters, minmax_pairs));
}

void ClusterMatrices::set_cluster_params(const ClusterAssign &classign)
{
    this->classign = classign;
}

void ClusterMatrices::fill_empty(int nmeans)
{
    /*
     * LAMBDA: Compute hamming distance
     */
    auto hdist = [](int n1, int n2)
    {
        int x = n1 ^ n2;
        int setbits = 0;

        while (x > 0) {
            setbits += x & 1;
            x >>= 1;
        }

        return setbits;
    };

    // stride by nmeans to get same subbiome combo
    int stride = nmeans;

    std::vector<bool> prevempty(clusters_data.size(), false);

    for (auto &mp : clusters_data) {
        int clidx = mp.first;
        auto &cldata = mp.second;
        int sbc = clidx % stride;

        int closest_d = 100000;
        int closest_c = -1;

        // if cluster is empty, plant density will be negative.
        // If empty, we proceed to fill it with cluster/region/segment with smallest
        // hamming distance for bitstring created from active subbiomes.
        // For example, if we have 4 subbiomes, and subbiomes 1 and 3 are active, the bistring is 1010
        if (cldata.get_density() < 0.0f) {
            std::cout << "Filling in cluster index " << clidx << std::endl;

            // Mark this cluster as being previously empty, so that we do not accidentally
            // fill another empty cluster with this cluster. We only fill empty clusters with
            // originally filled ones
            prevempty.at(clidx) = true;

            // we stride over the clusters/segments, so that we only work with segments that have the
            // same cluster mean as the one to be filled (but obviously with different bitstrings for active subbiomes)
            // We then find the closest one in terms of hamming distance and record it
            int curridx = clidx;
            for (; curridx >= 0; curridx -= stride) {
                if (clusters_data.count(curridx) && clusters_data.at(curridx).get_density() >= 0.0f && !prevempty.at(curridx)) {
                    int d = hdist(curridx / stride + 1, clidx / stride + 1);
                    if (d < closest_d) {
                        closest_d = d;
                        closest_c = curridx;
                    }
                }
            }
            for (curridx = clidx; curridx < clusters_data.size() ; curridx += stride) {
                if (clusters_data.count(curridx) && clusters_data.at(curridx).get_density() >= 0.0f && !prevempty.at(curridx)) {
                    int d = hdist(curridx / stride, clidx / stride);
                    if (d < closest_d) {
                        closest_d = d;
                        closest_c = curridx;
                    }
                }
            }
            if (closest_c == -1) {
                std::cout << "Could not find a cluster for index " + std::to_string(clidx) << std::endl;
                //throw std::runtime_error("Could not find a cluster for index " + std::to_string(clidx));
            }
            else {
                std::cout << "Assigning cluster " << closest_c << " to empty cluster " << clidx << std::endl;
                mp.second = clusters_data.at(closest_c);
            }
        }
    }
}

void ClusterMatrices::normalizeAll(normalize_method nmeth)
{
    for (std::pair<const int, ClusterData> &p : clusters_data)
    {
        p.second.get_locmatrix().normalizeAll(nmeth);
    }
}

void ClusterMatrices::unnormalizeAll(normalize_method nmeth)
{
    for (std::pair<const int, ClusterData> &p : clusters_data)
    {
        p.second.get_locmatrix().unnormalizeAll(nmeth);
    }

}

const data_importer::common_data &ClusterMatrices::get_cdata() const
{
    return cdata;
}

void ClusterMatrices::set_allspecies_maxheights()
{
    int nclusters = get_nclusters();
    for (int clidx = 0; clidx < nclusters; clidx++)
    {
        auto &cluster = get_cluster(clidx);
        for (auto &specpair : cdata.canopy_and_under_species)
        {
            int specid = specpair.first;
            const HistogramDistrib &distrib = cluster.get_sizedistrib(specid);
            const std::vector<common_types::decimal> &bins = distrib.getbins();
            float dmin = distrib.getmin();
            float dmax = distrib.getmax();
            float incr = (dmax - dmin) / bins.size();

            // find the last non empty bin in the histogram
            int lastfound = -1;
            for (int i = 0; i < bins.size(); i++)
            {
                if (bins.at(i) > 1e-6)
                {
                    lastfound = i;
                }
            }

            // set maximum height for undergrowth species to upper range of highest non empty bin
            float dmaxheight = incr * (lastfound + 1);
            if (allspecies_maxheights.count(specid) == 0)
            {
                allspecies_maxheights[specid] = dmaxheight;
            }
            else
            {
                if (dmaxheight > allspecies_maxheights.at(specid))
                {
                    allspecies_maxheights.at(specid) = dmaxheight;
                }
            }
        }
    }
}


float ClusterMatrices::diff_other(ClusterMatrices *other)
{
    float totdiff = 0.0f;
    for (auto &p : clusters_data)
    {
        int clusterid = p.first;
        HistogramMatrix &thismtx = get_locmatrix(clusterid);
        HistogramMatrix &othermtx = other->get_locmatrix(clusterid);
        bool showdiff = false;
        totdiff += thismtx.diff(othermtx, showdiff);
    }
    return totdiff;
}

float ClusterMatrices::diff_other(ClusterMatrices *other, const std::unordered_map<int, std::set<std::pair<int, int> > > &rowcol_check)
{
    float totdiff = 0.0f;
    for (auto &p : clusters_data)
    {
        int clusterid = p.first;
        HistogramMatrix &thismtx = get_locmatrix(clusterid);
        HistogramMatrix &othermtx = other->get_locmatrix(clusterid);
        bool showdiff = false;
        try
        {
            const auto &rccheck = rowcol_check.at(clusterid);
            totdiff += thismtx.diff(othermtx, rccheck);
        }
        catch (std::out_of_range &e)
        {
            //throw std::out_of_range("in ClusterMatrices::diff_other: clusterid " + std::to_string(clusterid) + " does not exist in rowcol_check arg");
            continue;
        }
    }
    return totdiff;
}
