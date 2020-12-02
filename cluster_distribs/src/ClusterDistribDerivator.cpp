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


#include "ClusterDistribDerivator.h"
#include <data_importer/data_importer.h>
#include "data_importer/AbioticMapper.h"
#include <common/constants.h>
#include <algorithm>

ClusterDistribDerivator::ClusterDistribDerivator(std::vector<std::string> datasets_paths, const data_importer::common_data &cdata)
{
    for (auto &dpath : datasets_paths)
    {
        data_importer::data_dir ddir(dpath, 1);
        abiotic_maps_package amaps(ddir, abiotic_maps_package::suntype::CANOPY, abiotic_maps_package::aggr_type::AVERAGE);
        auto canopytrees = data_importer::read_pdb(ddir.canopy_fnames.at(0));
        auto undergrowth = data_importer::read_pdb(ddir.undergrowth_fnames.at(0));
        allocd_maps.emplace_back(amaps, canopytrees, cdata);
        add_dataset(allocd_maps.back(), undergrowth);
    }
}

ClusterDistribDerivator::ClusterDistribDerivator(ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth)
{
    add_dataset(clmaps, undergrowth);
}

void ClusterDistribDerivator::add_dataset(ClusterMaps &clmaps, const PlantSpatialHashmap &plantmap)
{
    datasets.push_back({&clmaps, plantmap});
}

void ClusterDistribDerivator::add_dataset(ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth)
{
    float cellsize = PlantSpatialHashmap::calculate_required_cellsize(HistogramMatrix::global_under_metadata.maxdist, undergrowth);		// assuming we will not add more undergrowth plants. TODO: perhaps use a global maximum radius for undergrowth plants?
    PlantSpatialHashmap phash(cellsize, cellsize, clmaps.get_width(), clmaps.get_height());
    phash.addplants(undergrowth);
    add_dataset(clmaps, phash);
}

bool ClusterDistribDerivator::remove_dataset(int idx)
{
    if (idx < 0 || idx >= datasets.size())
        return false;
    else
    {
        datasets.erase(std::next(datasets.begin(), idx));
        return true;
    }
}

bool ClusterDistribDerivator::remove_dataset(const ClusterMaps &clmaps, const std::vector<basic_tree> &undergrowth)
{
    for (auto dsiter = datasets.begin(); dsiter != datasets.end(); std::advance(dsiter, 1))
    {
        auto &ds = *dsiter;
        if (ds.clmaps == &clmaps)
        {
            auto &pmap = ds.plantmap;
            std::vector<basic_tree> allplants = pmap.get_all_plants();
            if (allplants.size() == undergrowth.size())
            {
                bool go_next = false;
                for (const auto &plnt : undergrowth)
                {
                    if (std::find(allplants.begin(), allplants.end(), plnt) == allplants.end())
                    {
                        go_next = true;
                        break;
                    }
                }
                if (go_next)
                    break;
                else
                {
                    // if we get here, it means all plants in undergrowth argument equals plants in plantmap. We have therefore found our dataset
                    datasets.erase(dsiter);
                    return true;
                }
            }
        }
    }
    // if we get here, we could not find a dataset equaling the arguments we passed
    return false;
}


void ClusterDistribDerivator::do_kmeans(int nmeans, int niters, const data_importer::common_data &cdata)
{
    ClusterAssign classign;

    std::vector<abiotic_maps_package> all_amaps;

    for (auto &ds : datasets)
    {
        all_amaps.push_back(ds.clmaps->get_maps());
    }

    classign.do_kmeans(all_amaps, nmeans, niters, cdata);

    std::cout << "Setting cluster assigns..." << std::endl;
    for (auto &ds : datasets)
    {
        ds.clmaps->set_classign(classign);
        //ds.clmaps->compute_maps();
    }
    std::cout << "Done" << std::endl;
}

std::map<int, float> ClusterDistribDerivator::compute_region_densities(const Dataset &ds, const std::vector<basic_tree> &undergrowth, bool allow_zero_region)
{
    std::map<int, float> region_density;

    auto region_plantcount = ds.clmaps->compute_region_plantcounts(undergrowth);
    auto region_size = ds.clmaps->get_region_sizes();

    //auto region_plantcount = compute_region_plantcounts(undergrowth);

    std::cout << "Computing region densities..." << std::endl;
    for (auto &regpair : region_size)
    {
        int idx = regpair.first;
        float size = regpair.second;
        int count = 0;
        if (region_plantcount.count(idx))
            count = region_plantcount.at(idx);
        if (count > 0 && fabs(size) < 1e-6f)
        {
            throw std::domain_error("size for region " + std::to_string(idx) + " is zero, but it contains " + std::to_string(count) + " plants. How?");
        }
        if (fabs(size) < 1e-6f && !allow_zero_region)
        {
            throw std::domain_error("size for region " + std::to_string(idx) + " is zero");
        }
        if (fabs(size) > 1e-6f)
            region_density[idx] = count / (float)size;
        else
            region_density[idx] = 0.0f;
        std::cout << "For region " << idx << ":" << std::endl;
        std::cout << "Number of plants: " << count << std::endl;
        std::cout << "Size: " << size << std::endl;
        std::cout << "Density: " << region_density[idx] << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    return region_density;
}

std::list<ClusterMatrices> ClusterDistribDerivator::deriveHistogramMatricesSeparate(const data_importer::common_data &cdata)
{
    //throw std::out_of_range("ClusterDistribDerivator::deriveHistogramMatricesSeparate not implemented");

    std::list<ClusterMatrices> clmatrices;

    int datasetnum = 0;
    for (auto &ds : datasets)
    {
        clmatrices.push_back(deriveHistogramMatrices(ds, cdata, nullptr));
        datasetnum++;
        std::cout << "Done with dataset " << datasetnum << std::endl;
    }

    return clmatrices;
}


// two deriveHistogramMatrices functions, can actually be static functions
ClusterMatrices ClusterDistribDerivator::deriveHistogramMatrices(const Dataset &ds, const data_importer::common_data &cdata, ClusterMatrices *benchmark)
{
    ClusterMatrices clmatrices(cdata);

    int nrows, ncols;
    ds.plantmap.get_griddim(nrows, ncols);
    int ncells = nrows * ncols;

    // First compute spatial and size distributions for each cluster
    for (int i = 0; i < ncells; i++)
    {
        std::cout << "Calculating distance distribution for cell " << i << std::endl;
        std::vector<int> surr_flatidxes = ds.plantmap.get_surr_flatidxes(i);
        const auto &cellplants = ds.plantmap.get_cell_direct(i);
        for (const auto &refplnt : cellplants)
        {
            for (auto &surrflat : surr_flatidxes)
            {
                const auto &otherplants = ds.plantmap.get_cell_direct(surrflat);
                clmatrices.updateHistogramLocMatrix(refplnt, otherplants, *ds.clmaps, benchmark);
            }
            clmatrices.updateHistogramSizeMatrix(refplnt, *ds.clmaps);
        }
    }

    auto all_undergrowth = ds.plantmap.get_all_plants();

    // compute, then set region densities in for loop below
    auto region_densities = compute_region_densities(ds, all_undergrowth, true);

    int nclusters = ds.clmaps->get_nclusters();
    std::cout << "Setting densities for " << nclusters << " clusters" << std::endl;
    for (int clidx = 0; clidx < nclusters; clidx++)
    {
        if (!region_densities.count(clidx))
        {
            // set it to -1.0f to indicate missing data, otherwise we could mistake it for a region without any plants
            clmatrices.get_cluster(clidx).set_density(-1.0f);
        }
        else
        {
            clmatrices.get_cluster(clidx).set_density(region_densities.at(clidx));
        }
    }
    std::cout << "Done setting densities" << std::endl;

    //compute, and set species proportions for each region in for loop below

    std::map<int, int> plantcounts = ds.clmaps->compute_region_plantcounts(all_undergrowth);
    std::unordered_map<int, std::unordered_map<int, int> > species_counts = ds.clmaps->calc_species_counts(ds.plantmap);

    std::cout << "Setting species proportions for " << nclusters << " clusters" << std::endl;
    for (int clidx = 0; clidx < nclusters; clidx++)
    {
        std::unordered_map<int, float> species_ratios;
        if (!species_counts.count(clidx) || plantcounts.at(clidx) == 0)
        {
            clmatrices.get_cluster(clidx).set_species_ratios(species_ratios);
        }
        else
        {
            for (auto &speccount_pair : species_counts.at(clidx))
            {
                int specid = speccount_pair.first;
                int speccount = speccount_pair.second;
                int plantcount = plantcounts.at(clidx);
                species_ratios[specid] = static_cast<float>(speccount) / plantcount;	// no need to worry about plantcount == 0, we already take care of this case above, in the if condition
            }
            clmatrices.get_cluster(clidx).set_species_ratios(species_ratios);
        }
    }

    clmatrices.set_cluster_params(ds.clmaps->get_classign());

    clmatrices.fill_empty(ds.clmaps->get_nmeans());

    std::cout << "Done. Returning clmatrices object from deriveHistogramMatrices" << std::endl;

    return clmatrices;
}

ClusterMatrices ClusterDistribDerivator::deriveHistogramMatrices(ClusterMaps &clmaps, std::vector<basic_tree> undergrowth, std::vector<basic_tree> canopy, data_importer::common_data &cdata, ClusterMatrices *benchmark)
{
    float cellsize = PlantSpatialHashmap::calculate_required_cellsize(HistogramMatrix::global_under_metadata.maxdist, undergrowth);
    PlantSpatialHashmap phash(cellsize, cellsize, clmaps.get_width(), clmaps.get_height());
    Dataset ds = {&clmaps, phash};
    return deriveHistogramMatrices(ds, cdata, benchmark);
}
