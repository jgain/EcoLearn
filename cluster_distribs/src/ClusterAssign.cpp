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


#include "ClusterAssign.h"
#include "AllClusterInfo.h"
#include "ClusterMatrices.h"
#include "data_importer/AbioticMapper.h"

#include <functional>

struct uninitialized_object : public std::exception
{
    uninitialized_object(std::string msg)
        : msg(msg)
    {
    }

    const char * what () const throw ()
    {
        return msg.c_str();
    }

    std::string msg;
};

ClusterAssign::ClusterAssign()
{
}

void ClusterAssign::do_kmeans(const std::vector<std::string> &datadirs, int nmeans, int niters,
                              const data_importer::common_data &cdata)
{
    std::vector<abiotic_maps_package> all_amaps;
    for (auto &dirname : datadirs)
    {
        all_amaps.push_back(abiotic_maps_package(dirname, abiotic_maps_package::suntype::CANOPY,
                                                 abiotic_maps_package::aggr_type::AVERAGE));
    }

    do_kmeans(all_amaps, nmeans, niters, cdata);
}

void ClusterAssign::do_kmeans(const std::vector<abiotic_maps_package> &all_amaps, int nmeans, int niters,
                              const data_importer::common_data &cdata)
{
    means.clear();

    // create a vector of maps for each abiotic condition
    std::vector< ValueGridMap<float> > wetvec, sunvec, tempvec, slopevec;
    for (const auto &amaps : all_amaps)
    {
        wetvec.push_back(amaps.wet);
        sunvec.push_back(amaps.sun);
        tempvec.push_back(amaps.temp);
        slopevec.push_back(amaps.slope);
    }

    kMeans km(niters);

    int gw, gh;
    float rw, rh;
    wetvec.at(0).getDim(gw, gh);
    wetvec.at(0).getDimReal(rw, rh);
    assert(rw > 0.0f && rh > 0.0f);

    // make sure we have equal amounts of each abiotic map
    assert(wetvec.size() == sunvec.size());
    assert(sunvec.size() == tempvec.size());
    assert(tempvec.size() == slopevec.size());

    std::vector<std::array<float, 4> > features;
    std::vector<int> assigns;
    std::array<std::pair<float, float>, 4> ranges;
    for (auto &p : ranges)
    {
        p.first = std::numeric_limits<float>::max();
        p.second = -std::numeric_limits<float>::max();
    }

    // determine min and max values for each abiotic factor over all terrains
    for (int i = 0; i < wetvec.size(); i++)
    {
        int nelements = wetvec.at(i).nelements();

        // make sure abiotic maps are of equal size for terrain
        assert(nelements == sunvec.at(i).nelements());
        assert(nelements == tempvec.at(i).nelements());
        assert(nelements == slopevec.at(i).nelements());

        // iterate over all elements of current map
        for (int j = 0; j < nelements; j++)
        {
            std::array<float, kMeans::ndim> feature;
            feature.at(0) = wetvec.at(i).get(j) > 250.0f ? 250.0f : wetvec.at(i).get(j);
            feature.at(1) = sunvec.at(i).get(j);
            feature.at(2) = slopevec.at(i).get(j);
            feature.at(3) = tempvec.at(i).get(j);

            features.push_back(feature);

            // update min max for each abiotic factor, if necessary
            for (int i = 0; i < kMeans::ndim; i++)
            {
                if (features.back().at(i) < ranges.at(i).first)
                {
                    ranges.at(i).first = features.back().at(i);
                }
                if (features.back().at(i) > ranges.at(i).second)
                {
                    ranges.at(i).second = features.back().at(i);
                }
            }
        }
    }

    minmax_ranges = ranges;
    for (auto &f : features)
    {
        for (int i = 0; i < 4; i++)
        {
            // check if feature has a constant value - if not, then scale current value. If yes, assign zero
            if (fabs(ranges.at(i).second - ranges.at(i).first) > 1e-5f)
                f.at(i) = (f.at(i) - ranges.at(i).first) / (ranges.at(i).second - ranges.at(i).first);
            else
                f.at(i) = 0.0f;
        }
    }

    // do clustering, which assigns to 'means' member variable
    km.cluster(features, nmeans, assigns, means);
}

ClusterAssign::ClusterAssign(const std::vector<std::array<float, 4> > &means,
                             const std::array<std::pair<float, float>, 4> &minmax_ranges)
    : means(means), minmax_ranges(minmax_ranges)
{}

ClusterAssign::ClusterAssign(const AllClusterInfo clusterinfo)
    : means(clusterinfo.kclusters), minmax_ranges(clusterinfo.minmax_ranges)
{}

int ClusterAssign::assign(float moisture, float sun, float slope, float temp) const
{
    if (!has_model())
    {
        throw uninitialized_object("ClusterAssign::assign called before object is initialized with a trained kmeans model");
    }

    // scale abiotic values according to min/max values obtained from model
    float wet = (moisture - minmax_ranges.at(0).first) / (minmax_ranges.at(0).second - minmax_ranges.at(0).first);
    float sunl = (sun - minmax_ranges.at(1).first) / (minmax_ranges.at(1).second - minmax_ranges.at(1).first);
    float sl = (slope - minmax_ranges.at(2).first) / (minmax_ranges.at(2).second - minmax_ranges.at(2).first);
    float t = (temp - minmax_ranges.at(3).first) / (minmax_ranges.at(3).second - minmax_ranges.at(3).first);

    float closest_dist = std::numeric_limits<float>::max();
    int closest_idx = -1;
    for (int i = 0; i < means.size(); i++)
    {
        // obtain means vector (one scalar for each abiotic condition) for cluster i
        auto &clvec = means.at(i);

        // compute distance to cluster i
        float distsq = (clvec.at(0) - wet) * (clvec.at(0) - wet);
        distsq += (clvec.at(1) - sunl) * (clvec.at(1) - sunl);
        distsq += (clvec.at(2) - sl) * (clvec.at(2) - sl);
        distsq += (clvec.at(3) - t) * (clvec.at(3) - t);

        // update closest distance and cluster index if necessary
        if (distsq < closest_dist)
        {
            closest_idx = i;
            closest_dist = distsq;
        }
    }
    return closest_idx;
}

int ClusterAssign::get_nmeans() const
{
    return means.size();
}

bool ClusterAssign::has_model() const
{
    return means.size() > 0;
}

const std::vector<std::array<float, 4> > &ClusterAssign::get_means() const
{
    return means;
}

const std::array<std::pair<float, float>, 4> &ClusterAssign::get_minmax_ranges() const
{
    return minmax_ranges;
}
