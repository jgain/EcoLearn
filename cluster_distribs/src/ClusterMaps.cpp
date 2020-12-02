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
#include "ClusterMaps.h"
#include "data_importer/AbioticMapper.h"
#include "PlantSpatialHashmap.h"
#include "common.h"
#include "EcoSynth/kmeans/src/kmeans.h"
#include <common/constants.h>

#include <data_importer/data_importer.h>
#include <map>
#include <list>
#include <functional>
#include <random>
#include <iomanip>


// this def is for debugging add/remove of plant effects on histograms
#define ADDREMOVE_DEBUG
#undef ADDREMOVE_DEBUG


ClusterMaps::ClusterMaps(const abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata)
    : canopytrees(canopytrees), cdata(cdata), amaps(amaps), maps_set({{"sun", true}, {"moisture", true}, {"slope", true}, {"temp", true}}),
      width(amaps.rw), height(amaps.rh)
{
}

ClusterMaps::ClusterMaps(const ClusterAssign &classign, const abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata)
    : ClusterMaps(amaps, canopytrees, cdata)
{
    this->classign = classign;

    //compute_maps();
}

void ClusterMaps::set_maps(const abiotic_maps_package &amaps)
{
    set_moisturemap(amaps.wet);
    set_sunmap(amaps.sun);
    set_slopemap(amaps.slope);
    set_tempmap(amaps.temp);
}

const abiotic_maps_package &ClusterMaps::get_maps() const
{
    return amaps;
}
abiotic_maps_package &ClusterMaps::get_maps()
{
    return amaps;
}

const ValueGridMap<int> &ClusterMaps::get_clustermap() const
{
    if (!maps_computed)
    {
        throw std::runtime_error("Clustermap not computed. Call ClusterMaps::compute_maps first");
    }
    return cluster_assigns;
}

void ClusterMaps::erase_outofbounds_plants(std::list<basic_tree> &plnts)
{
    std::vector<std::list<basic_tree>::iterator> remove_iters;
    for (auto iter = plnts.begin(); iter != plnts.end(); advance(iter, 1))
    {
        if (iter->x >= width || iter->x < 0 || iter->y >= height || iter->y < 0)
            remove_iters.push_back(iter);
    }
    for (auto &iter : remove_iters)
    {
        plnts.erase(iter);
    }
}



// XXX: Check these functions for implementation hints for new organisation of classes, in previous commits (or in ClusterMatricesCopy.cpp)
bool ClusterMaps::remove_canopytree(const basic_tree &tree)
{
    for (auto iter = canopytrees.begin(); iter != canopytrees.end(); advance(iter, 1))
    {
        if (*iter == tree)
        {
            canopytrees.erase(iter);
            return true;
        }
    }
    return false;
}

void ClusterMaps::add_canopytree(const basic_tree &tree)
{
    canopytrees.push_back(tree);
}


void ClusterMaps::do_kmeans_separate(std::vector<string> targetdirs, int nmeans, int niters, std::vector<std::array<float, kMeans::ndim> > &clusters, std::array<std::pair<float, float>, kMeans::ndim> &minmax_ranges, float undersim_sample_mult, const data_importer::common_data &cdata)
{
    clusters.clear();

    std::vector< ValueGridMap<float> > wetvec, sunvec, tempvec, slopevec;
    std::vector< std::map<int, ValueGridMap<unsigned char> > > sbvec;

    for (auto &d : targetdirs)
    {
        abiotic_maps_package amaps(d, abiotic_maps_package::suntype::CANOPY, abiotic_maps_package::aggr_type::AVERAGE);
        std::vector<basic_tree> trees = data_importer::read_pdb(data_importer::data_dir(d, 1).canopy_fnames.at(0));

        std::map<int, ValueGridMap<unsigned char> > sbmap = create_subbiome_map(trees, amaps.gw, amaps.gh, amaps.rw, amaps.rh, undersim_sample_mult, cdata);
        //auto adaptsun = calc_adaptsun(trees, amaps.sun, cdata);
        auto adaptsun = amaps.sun;
        wetvec.push_back(amaps.wet);
        sunvec.push_back(adaptsun);
        tempvec.push_back(amaps.temp);
        slopevec.push_back(amaps.slope);
        sbvec.push_back(sbmap);
    }

    if (targetdirs.size() == 0)
        return;

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

    std::vector<std::array<float, kMeans::ndim> > features;
    //std::vector<std::array<float, 4> > clusters;
    std::vector<int> assigns;
    std::array<std::pair<float, float>, kMeans::ndim> ranges;
    for (auto &p : ranges)
    {
        p.first = std::numeric_limits<float>::max();
        p.second = -std::numeric_limits<float>::max();
    }

    // iterate over all terrains
    for (int i = 0; i < wetvec.size(); i++)
    {
        int nelements = wetvec.at(i).nelements();
        // make sure abiotic maps are of equal size for terrain
        assert(nelements == sunvec.at(i).nelements());
        assert(nelements == tempvec.at(i).nelements());
        assert(nelements == slopevec.at(i).nelements());
        for (int j = 0; j < nelements; j++)
        {
            std::array<float, kMeans::ndim> feature;
            feature.at(0) = wetvec.at(i).get(j) > 250.0f ? 250.0f : wetvec.at(i).get(j);
            feature.at(1) = sunvec.at(i).get(j);
            feature.at(2) = slopevec.at(i).get(j);
            feature.at(3) = tempvec.at(i).get(j);

            features.push_back(feature);

            // update min max for each abiotic factor
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
        //for (int i = 0; i < kMeans::ndim; i++)
        for (int i = 0; i < 4; i++)		// XXX: only doing [0, 1] normalization for 4 abiotic conditions - subbiomes are left as-is since we assign more importance to them
        {
            if (fabs(ranges.at(i).second - ranges.at(i).first) > 1e-5f)
                f.at(i) = (f.at(i) - ranges.at(i).first) / (ranges.at(i).second - ranges.at(i).first);
            else		// this feature has a constant value - assign zero
                f.at(i) = 0.0f;
        }
    }
    km.cluster(features, nmeans, assigns, clusters);

}


void ClusterMaps::compute_sampleprob_map(ValueGridMap<float> &map, const std::vector<basic_tree> &trees,
                                         float sample_mult, float seedprob)
{
    float rw, rh;
    int gw, gh;
    map.getDim(gw, gh);
    map.getDimReal(rw, rh);
    map.fill((float)0.0f);
    ValueGridMap<unsigned char> trunks(gw, gh, rw, rh);
    trunks.fill((unsigned char)0);

    // assume the cell width and height are the same
    float cellsize = map.toreal(1, 1).x;
    for (auto &t : trees)
    {
        // the maximum distance, squared, so that we don't have to worry about square root
        float maxdistsq = t.radius * t.radius * sample_mult * sample_mult;

        // this is just a variable used for a sanity check later (in ifndef NDEBUG block below)
        float maxdistsq_error = 2.0f * (t.radius + cellsize) * (t.radius + cellsize) * sample_mult * sample_mult;

        xy<int> tgrid = map.togrid(t.x, t.y);
        trunks.set(tgrid.x, tgrid.y, 1);
        xy<int> start = map.togrid(t.x - t.radius * sample_mult, t.y - t.radius * sample_mult);
        xy<int> end = map.togrid(t.x + t.radius * sample_mult, t.y + t.radius * sample_mult);
        start.trim(0, gw - 1, 0, gh - 1);
        end.trim(0, gw - 1, 0, gh - 1);
        for (int y = start.y; y <= end.y; y++)
            for (int x = start.x; x <= end.x; x++)
            {
                // squared distance calculation
                xy<float> rxy = map.toreal(x, y);
                float distsq = (rxy.x - t.x) * (rxy.x - t.x) + (rxy.y - t.y) * (rxy.y - t.y);
#ifndef NDEBUG
                if (distsq > maxdistsq_error)
                {
                    // just a quick way to have a breakpoint
                    std::cout << "distsq too big" << std::endl;
                }
                assert(distsq <= maxdistsq_error);
#endif
                // if outside the radius * sample_mult of the tree, skip
                if (distsq > maxdistsq)
                {
                    distsq = maxdistsq;		// ?: this is basically a no op
                    continue;
                }

                // if inside the tree trunk, skip
                if (trunks.get(x, y))
                    continue;

                map.set(x, y, map.get(x, y) + seedprob);
            }
    }
}

void ClusterMaps::create_clustermap()
{
    int gw, gh;
    float rw, rh;

    int cgw, cgh;
    float crw, crh;

    amaps.wet.getDim(gw, gh);
    amaps.wet.getDimReal(rw, rh);
    assert(rw > 0.0f && rh > 0.0f);

    cluster_assigns.getDim(cgw, cgh);
    cluster_assigns.getDimReal(crw, crh);
    // only resize if necessary
    if (cgw != gw || cgh != gh)
        cluster_assigns.setDim(gw, gh);
    if (crw != rw || crh != rh)
        cluster_assigns.setDimReal(rw, rh);

    if (!cluster_assigns.eqdim(sampleprob_map))
    {
        throw std::runtime_error("cluster_assigns ValueGridMap object must have same dimensions as sampleprob_map");
    }

    cluster_assigns.fill((int)-1);
    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            if (sampleprob_map.get(x, y) > 1e-6)
            {
                // get each abiotic factor value at this location and cap moisture at 250
                float moisture, sun, temp, slope;
                moisture = this->amaps.wet.get(x, y) > 250.0f ? 250.0f : this->amaps.wet.get(x, y);
                sun = this->amaps.sun.get(x, y);
                temp = this->amaps.temp.get(x, y);
                slope = this->amaps.slope.get(x, y);
                int count = 0;

                int subhash = get_subbiome_hash_fromgrid(x, y);
                if (subhash == 0)
                {
                    throw std::runtime_error("In ClusterMaps::create_clustermap: Subbiome hash must be nonzero and positive (loc: " + std::to_string(x) + ", " + std::to_string(y) + ")");
                }

                int clidx = classign.assign(moisture, sun, slope, temp);

                // for each subbiome combination (given by subhash), we have all of the cluster means repeated
                clidx += (subhash - 1) * classign.get_nmeans();
                cluster_assigns.set(x, y, clidx);

            }
        }
    }
}

void ClusterMaps::set_clustermap(const ValueGridMap<int> &clmap)
{
    cluster_assigns = clmap;
}


void ClusterMaps::set_maps(std::string target_dirname, std::string db_filename)
{
    data_importer::data_dir targetdir(target_dirname, 1);
    data_importer::common_data cdata(db_filename);

    dem = data_importer::load_elv<ValueGridMap<float> >(targetdir.dem_fname);
    int gw, gh;
    float rw, rh;
    dem.getDimReal(rw, rh);
    dem.getDim(gw, gh);

    width = rw;
    height = rh;

    amaps = abiotic_maps_package(targetdir, abiotic_maps_package::suntype::CANOPY, abiotic_maps_package::aggr_type::AVERAGE);
}

void ClusterMaps::get_real_size(float &width, float &height) const
{
    //dem.getDimReal(width, height);
    width = this->width;
    height = this->height;
}

// TODO: implement iterator for PlantHashMap, so that one could iterate over all plant without regarding cells. Use that iterator here
const std::vector<basic_tree> &ClusterMaps::get_canopytrees() const
{
    return canopytrees;
}

float ClusterMaps::get_width() const
{
    return width;
}

float ClusterMaps::get_height() const
{
    return height;
}

void ClusterMaps::set_moisturemap(const ValueGridMap<float> &moisture)
{
    this->amaps.wet = moisture;
    maps_set.at("moisture") = true;
}

void ClusterMaps::set_sunmap(const ValueGridMap<float> &sun)
{
    this->amaps.sun = sun;
    maps_set.at("sun") = true;
}

void ClusterMaps::set_slopemap(const ValueGridMap<float> &slope)
{
    this->amaps.slope = slope;
    maps_set.at("slope") = true;
}

void ClusterMaps::set_tempmap(const ValueGridMap<float> &temp)
{
    this->amaps.temp = temp;
    maps_set.at("temp") = true;
}

void ClusterMaps::set_seedprob_map(const ValueGridMap<float> &probmap)
{
    if (!this->sampleprob_map.eqdim(probmap))
    {
        throw std::invalid_argument("in ClusterMaps::set_seedprob_map, sampleprob map given as argument must have same dimensions as existing one");
    }

    this->sampleprob_map = probmap;
}

void ClusterMaps::fill_seedprob_map(float value)
{
    sampleprob_map.fill((float)value);
}

const ValueGridMap<float> &ClusterMaps::get_seedprob_map() const
{
    if (!maps_computed)
    {
        throw std::runtime_error("Seedprob map not computed. Call ClusterMaps::compute_maps first");
    }
    return sampleprob_map;
}

const ValueGridMap<float> &ClusterMaps::get_tempmap() const
{
    return amaps.temp;
}

const ValueGridMap<float> &ClusterMaps::get_sunmap() const
{
    return amaps.sun;
}

const ValueGridMap<float> &ClusterMaps::get_slopemap() const
{
    return amaps.slope;
}

const ValueGridMap<float> &ClusterMaps::get_moisturemap() const
{
    return amaps.wet;
}

std::map<int, ValueGridMap<unsigned char> > ClusterMaps::create_subbiome_map(const std::vector<basic_tree> &canopytrees, int gw, int gh, float rw, float rh, float undersim_sample_mult, const data_importer::common_data &cdata)
{
    std::map<int, std::vector<const basic_tree *> > subbiome_plants;
    std::map<int, ValueGridMap<unsigned char> > subbiome_map;

    std::cout << "Initializing subbiome plants map ..." << std::endl;
    for (auto &tree : canopytrees)
    {
        int species = tree.species;
        int sb = cdata.canopyspec_to_subbiome.at(species);
        std::vector<const basic_tree *> &sbtrees = subbiome_plants[sb];
        sbtrees.push_back(&tree);
    }

    for (auto &sb : cdata.subbiomes)
    {
        subbiome_map[sb.first].setDim(gw, gh);
        subbiome_map[sb.first].setDimReal(rw, rh);
        subbiome_map[sb.first].fill((unsigned char)0);
    }

    auto trim = [](int &val, int min, int max) {
        if (max < min) std::swap(min, max);
        if (val > max) val = max;
        if (val < min) val = min;
    };

    //float undersim_sample_mult = 6.0f;

    std::cout << "Setting subbiome maps based on plant locations..." << std::endl;
    for (auto &sbpair : subbiome_plants)
    {
        std::vector<const basic_tree *> &plants = sbpair.second;
        std::cout << "Setting subbiome map for subbiome " << sbpair.first << "(" << plants.size() << " plants)" << std::endl;
        if (!subbiome_map.count(sbpair.first))
        {
            throw std::runtime_error("Unknown subbiome " + std::to_string(sbpair.first) + " encountered in ClusterMaps::create_subbiome_map");
        }
        for (auto &plnt : plants)
        {
            float radius = plnt->radius;
            float sr = radius * undersim_sample_mult;
            float srsq = sr * sr;
            float sxr = plnt->x - radius * undersim_sample_mult;
            float exr = plnt->x + radius * undersim_sample_mult;
            float syr = plnt->y - radius * undersim_sample_mult;
            float eyr = plnt->y + radius * undersim_sample_mult;
            /*
            trim(sxr, 0.0f, (float)rw);
            trim(exr, 0.0f, (float)rw);
            trim(syr, 0.0f, (float)rh);
            trim(eyr, 0.0f, (float)rh);
            */
            int sx = subbiome_map[sbpair.first].togrid(sxr, 0.0f).x;
            int ex = subbiome_map[sbpair.first].togrid(exr, 0.0f).x;
            int sy = subbiome_map[sbpair.first].togrid(0.0f, syr).y;
            int ey = subbiome_map[sbpair.first].togrid(0.0f, eyr).y;
            trim(sx, 0, gw - 1);
            trim(ex, 0, gw - 1);
            trim(sy, 0, gh - 1);
            trim(ey, 0, gh - 1);
            //std::cout << "Iterating over rectangle (" << sxr << ", " << syr << ") to (" << exr << ", " << eyr << ")" << std::endl;
            for (int y = sy; y <= ey; y++)
                for (int x = sx; x <= ex; x++)
                {
                    xy<float> realxy = subbiome_map[sbpair.first].toreal(x,y);
                    float dx = realxy.x - plnt->x;
                    float dy = realxy.y - plnt->y;
                    float rsq = dx * dx + dy * dy;
                    //float rdist = sqrt(rsq);
                    if (rsq <= srsq)
                    {
                        subbiome_map[sbpair.first].set(x, y, 1);
                    }

                }
        }
    }

    std::cout << "Done creating subbiome map" << std::endl;

    return subbiome_map;

}


// XXX: test this once done refactoring
void ClusterMaps::create_subbiome_map()
{

    // LAMBDA: compute subbiome hash based on indicator vector for each subbiome
    auto compute_sbhash = [](const std::vector<unsigned char> &vals) {
        int sum = 0;
        int i = 0;
        for (auto &v : vals)
        {
            if (v)
            {
                sum += std::pow(2, i);
            }
            i++;
        }
        return sum;
    };

    int gw, gh;
    amaps.wet.getDim(gw, gh);
    auto subbiome_map = create_subbiome_map(get_canopytrees(), gw, gh, width, height, common_constants::undersim_sample_mult, cdata);

    sbvals.setDim(gw, gh);
    sbvals.setDimReal(amaps.wet);

    std::vector<unsigned char> vals;
    vals.resize(subbiome_map.size());
    std::fill(vals.begin(), vals.end(), 0);

    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            // collect all indicators for subbiomes at this location and put them in the 'vals' vector
            int i = 0;
            for (std::pair<const int, ValueGridMap<unsigned char> > &mpair : subbiome_map)
            {
                ValueGridMap<unsigned char> &m = mpair.second;
                vals.at(i) = m.get(x, y);
                i++;
            }
            // compute and assign subbiome hash
            int h = compute_sbhash(vals);
            sbvals.set(x, y, h);
        }
    }
}



int ClusterMaps::get_subbiome_hash(float rx, float ry)
{
    if (!maps_computed)
    {
        throw std::runtime_error("Subbiome map not computed yet. Call ClusterMaps::compute_maps first");
    }
    return sbvals.get_fromreal(rx, ry);
}

std::map<int, float> ClusterMaps::get_region_sizes()
{
    return region_size;
}

int ClusterMaps::get_nclusters()
{
    return classign.get_nmeans() * (std::pow(2, cdata.subbiomes.size()) - 1);
}

int ClusterMaps::get_nmeans()
{
    return classign.get_nmeans();
}

int ClusterMaps::get_subbiome_hash_fromgrid(int gx, int gy)
{
    /*
    if (!maps_computed)
    {
        throw std::runtime_error("Subbiome map not computed yet. Call ClusterMaps::compute_maps first");
    }
    */
    return sbvals.get(gx, gy);
}

float ClusterMaps::get_moisture(float x, float y) const
{
    return amaps.wet.get_fromreal(x, y);
}

float ClusterMaps::get_sun(float x, float y) const
{
    return amaps.sun.get_fromreal(x, y);
}

float ClusterMaps::get_temp(float x, float y) const
{
    return amaps.temp.get_fromreal(x, y);
}

float ClusterMaps::get_slope(float x, float y) const
{
    return amaps.slope.get_fromreal(x, y);
}

int ClusterMaps::get_cluster_idx(float x, float y) const
{
    if (!maps_computed)
    {
        throw std::runtime_error("Clustermap not computed yet. Call ClusterMaps::compute_maps first");
    }
    return cluster_assigns.get_fromreal(x, y);
}



void ClusterMaps::compute_region_sizes()
{
    region_size.clear();

    int gw, gh;
    cluster_assigns.getDim(gw, gh);

    if (gw == 0 || gh == 0)
    {
        throw std::runtime_error("cluster_assigns must be set before computing region sizes (in ClusterMaps::compute_region_sizes)");
    }

    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            int idx = cluster_assigns.get(x, y);
            // some parts of the landscape out of range of canopy trees will not have a region assigned to them
            // - skip in this case
            if (idx > -1)
            {
                //float samplprob = sampleprob_map.get_fromreal(realxy.x, realxy.y);
                // we cancel this assert for now, because this function also gets used when removing a canopy tree
                //assert(samplprob > 0.0f);		// because regions only get assigned to cells with sample probability greater than zero
                if (region_size.count(idx) == 0)
                {
                    //region_size[idx] = samplprob;
                    region_size[idx] = 1.0f;
                }
                else
                {
                    //region_size.at(idx) += samplprob;
                    region_size[idx] += 1.0f;
                }
            }
        }
    }
}


void ClusterMaps::compute_sampleprob_map()
{
    int gw, gh;
    //sampleprob_map.getDim(gw, gh);
    //if (gw == 0 || gh == 0)
    //    gw = std::ceil(width), gh = std::ceil(height);
    //sampleprob_map = ValueGridMap<float>(moisture);
    sampleprob_map.setDim(amaps.wet);
    sampleprob_map.setDimReal(amaps.wet);
    float seedprob = 0.00004f;
    compute_sampleprob_map(sampleprob_map, get_canopytrees(), common_constants::undersim_sample_mult, common_constants::sampleprob);
}

void ClusterMaps::compute_maps()
{
    auto bt = std::chrono::steady_clock::now().time_since_epoch();

    compute_sampleprob_map();
    auto sp_et = std::chrono::steady_clock::now().time_since_epoch();
    create_subbiome_map();
    auto sb_et = std::chrono::steady_clock::now().time_since_epoch();
    std::cout << "Creating clustermap..." << std::endl;
    create_clustermap();
    auto clmap_et = std::chrono::steady_clock::now().time_since_epoch();
    std::cout << "Computing region sizes..." << std::endl;
    compute_region_sizes();
    auto regsize_et = std::chrono::steady_clock::now().time_since_epoch();
    auto et = std::chrono::steady_clock::now().time_since_epoch();

    std::cout << "Sampleprobmap time: " << std::chrono::duration_cast<std::chrono::milliseconds>(sp_et - bt).count() << std::endl;
    std::cout << "sbmap time: " << std::chrono::duration_cast<std::chrono::milliseconds>(sb_et - sp_et).count() << std::endl;
    std::cout << "clmap time: " << std::chrono::duration_cast<std::chrono::milliseconds>(clmap_et - sb_et).count() << std::endl;
    std::cout << "regsize time: " << std::chrono::duration_cast<std::chrono::milliseconds>(regsize_et - clmap_et).count() << std::endl;

    maps_computed = true;
}

std::map<int, int> ClusterMaps::compute_region_plantcounts(const std::vector<basic_tree> &undergrowth)
{
    if (!maps_computed)
    {
        compute_maps();
    }

    std::map<int, int> region_plantcount;

    for (auto &plnt : undergrowth)
    {
        // get region index for current plant location, then increment
        // that region's plant count by 1 (and init to 1 if no previous
        // plants recorded for this region)
        int region_idx = get_cluster_idx(plnt.x, plnt.y);
        if (region_plantcount.count(region_idx) == 0)
        {
            region_plantcount[region_idx] = 1;
        }
        else
        {
            region_plantcount.at(region_idx)++;
        }
    }
    return region_plantcount;
}

int ClusterMaps::compute_overall_target_count(const ClusterMatrices &model)
{
    auto targetcounts = compute_region_target_counts(model);

    int sum = 0;
    for (auto &pair : targetcounts)
    {
        if (pair.second >= 0)
            sum += pair.second;
    }
    return sum;
}

std::unordered_map<int, int> ClusterMaps::compute_region_target_counts(const ClusterMatrices &model)
{
    if (!maps_computed)
    {
        compute_maps();
    }
    std::unordered_map<int, int> region_target_counts;
    for (int i = 0; i < model.get_nclusters(); i++)
        region_target_counts[i] = 0;
    for (auto &sizepair : region_size)
    {
        int regionid = sizepair.first;
        float regionsize = sizepair.second;
        int targetcount;
        float density = model.get_region_density(regionid);
        if (density >= 0.0f)
        {
            targetcount = density * regionsize;
        }
        else
        {
            targetcount = -1;
        }
        region_target_counts[regionid] = targetcount;
        //region_target_counts[regionid] = 10000;		// just for debugging of GPU initial sampling
    }
    return region_target_counts;
}

std::unordered_map<int, std::unordered_map<int, int> > ClusterMaps::calc_species_counts(const PlantSpatialHashmap &undergrowth)
{
    // if maps have not been computed yet, compute them
    if (!maps_computed)
    {
        compute_maps();
    }

    std::unordered_map<int, std::unordered_map<int, int> > species_counts;

    // LAMBDA: for a given cluster index and species id, increment the corresponding count
    auto incr_species_count = [&species_counts](int clid, int species) {
        if (species_counts.count(clid))
        {
            if (species_counts.at(clid).count(species))
            {
                species_counts.at(clid).at(species)++;
            }
            else
            {
                species_counts.at(clid)[species] = 1;
            }
        }
        else
        {
            species_counts[clid][species] = 1;
        }
    };

    // go over each plant, get its cluter id (based on location) and its species, and pass those to lambda defined above
    auto plants = undergrowth.get_all_plants();
    for (auto &plnt : plants)
    {
        int clid = get_cluster_idx(plnt.x, plnt.y);
        incr_species_count(clid, plnt.species);
    }

    return species_counts;
}

void ClusterMaps::set_cdata(const data_importer::common_data &cdata_arg)
{
    cdata = cdata_arg;
}

const data_importer::common_data &ClusterMaps::get_cdata() const
{
    return cdata;
}

float ClusterMaps::viability(float val, float c, float r)
{
    float s = log(0.2f) / pow(r/2.0, 4.5f);
    float v = pow(M_E, s * pow(fabs(val-c), 4.5f));
    // adjust to include negative viability for stress purposes
    v *= 1.2f;
    v -= 0.2f;
    return v;

}

void ClusterMaps::set_classign(const ClusterAssign &classign)
{
    this->classign = classign;
    compute_maps();
}

void ClusterMaps::update_canopytrees(const std::vector<basic_tree> canopytrees, const ValueGridMap<float> &canopysun)
{
    amaps.sun = canopysun;
    update_canopytrees(canopytrees);
}

void ClusterMaps::update_canopytrees(const std::vector<basic_tree> canopytrees)
{
    this->canopytrees = canopytrees;

    compute_maps();
}

ClusterAssign ClusterMaps::get_classign()
{
    return classign;
}

float ClusterMaps::calcviability(float x, float y, int specid)
{
    // get viability function for 'specid'
    const data_importer::species &spinfo = cdata.canopy_and_under_species.at(specid);
    const auto &wetv = spinfo.wet;
    const auto &sunv = spinfo.sun;
    const auto &slopev = spinfo.slope;
    const auto &tempv = spinfo.temp;

    // get each abiotic factor's value
    float wv = amaps.wet.get_fromreal(x, y);
    float sv = amaps.sun.get_fromreal(x, y);
    float tv = amaps.temp.get_fromreal(x, y);
    float slv = amaps.slope.get_fromreal(x, y);

    // get adaptability values for each abiotic condition
    float wadapt = std::max(0.0f, viability(wv, wetv.c, wetv.r));
    float sunadapt = std::max(0.0f, viability(sv, sunv.c, sunv.r));
    float tempadapt = std::max(0.0f, viability(tv, tempv.c, tempv.r));
    float slopeadapt = std::max(0.0f, viability(slv, slopev.c, slopev.r));
    // take minimum of all adaptability values
    float adapt = std::min(wadapt, std::min(sunadapt, std::min(tempadapt, slopeadapt)));
    return adapt;
}
