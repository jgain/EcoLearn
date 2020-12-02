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


#include "ClusterData.h"
#include "common/constants.h"


ClusterData::ClusterData(std::set<int> plant_ids, std::set<int> canopy_ids,
                         HistogramDistrib::Metadata undergrowth_meta, HistogramDistrib::Metadata canopy_meta,
                         const data_importer::common_data &cdata)
    : plant_ids(plant_ids), canopy_ids(canopy_ids),
      spatial_data(std::vector<int>(plant_ids.begin(), plant_ids.end()),
                   std::vector<int>(canopy_ids.begin(), canopy_ids.end()), undergrowth_meta, canopy_meta),
      cdata(cdata), density(-1.0f)
{
    check_plant_and_canopy_ids();

    for (auto &plntid : plant_ids)
    {
        const auto &spec = cdata.canopy_and_under_species.at(plntid);
        size_data.emplace(plntid,
                          HistogramDistrib(
                              common_constants::NUMSIZEBINS,
                              0,
                              common_constants::MINHEIGHT,
                              std::min(spec.maxhght, common_constants::MAX_UNDERGROWTH_HEIGHT)));
        species_ratios.emplace(plntid, 0.0f);		// assign -1.0f to indicate non-existence?
    }
}

ClusterData::ClusterData(const HistogramMatrix &spatial_data, const std::map<int, HistogramDistrib> &size_data,
                         float density, std::unordered_map<int, float> species_ratios,
                         const data_importer::common_data &cdata)
    : spatial_data(spatial_data), size_data(size_data), species_ratios(species_ratios), cdata(cdata), density(density)
{
    auto &plntids = this->spatial_data.get_plant_ids();
    auto &canopyids = this->spatial_data.get_canopy_ids();
    plant_ids = std::set<int>(plntids.begin(), plntids.end());
    canopy_ids = std::set<int>(canopyids.begin(), canopyids.end());

    check_plant_and_canopy_ids();

    fix_species_ratios();

    for (auto &canopyid : canopy_ids)
    {
        subbiome_species_ratios.emplace(canopyid, subbiome_clusters_type(cdata, canopyid, species_ratios));
    }
}

float ClusterData::get_subspecies_ratio(int canopyspec_id, int underspec_id) const
{
    const subbiome_clusters_type &sbtype = subbiome_species_ratios.at(canopyspec_id);
    if (sbtype.species_ratios.count(underspec_id))
        return sbtype.species_ratios.at(underspec_id);
    else
        return -1.0f;
}

const std::map<int, float> &ClusterData::get_subspecies_ratios(int canopyspec_id) const
{
    return subbiome_species_ratios.at(canopyspec_id).species_ratios;
}

void ClusterData::fix_species_ratios()
{
    for (auto &pid : plant_ids)
    {
        if (!species_ratios.count(pid))
        {
            // make this zero, or a negative number to indicate the data is missing...?
            species_ratios[pid] = 0.0f;
        }
    }
}

void ClusterData::check_plant_and_canopy_ids() const
{
    /*
     * For simplicity, we enforce the rule that all plant and canopy ids in the database have to be present in the
     * spatial_data argument and vice versa.
     * However, this rule can be removed or altered by removing/altering these checks
     */
    check_plantids();
    check_canopyids();

}

void ClusterData::check_plantids() const
{
    for (auto &specpair : cdata.canopy_and_under_species)
    {
        if (!plant_ids.count(specpair.first))
        {
            throw std::invalid_argument("Plant id " + std::to_string(specpair.first) +
                                        ", present in database, is not present in plant_ids member of ClusterData");
        }
    }
    for (auto &pid : plant_ids)
    {
        if (!cdata.canopy_and_under_species.count(pid))
        {
            throw std::invalid_argument("Plant id " + std::to_string(pid) +
                                        ", present in plant_ids member of ClusterData, is not present in database");
        }
    }
}

void ClusterData::check_canopyids() const
{
    for (auto &specpair : cdata.all_species)
    {
        if (!canopy_ids.count(specpair.first))
        {
            throw std::invalid_argument("Canopy id " + std::to_string(specpair.first) +
                                        ", present in database, is not present in canopy_ids member of ClusterData");
        }
    }
    for (auto &pid : canopy_ids)
    {
        if (!cdata.all_species.count(pid))
        {
            throw std::invalid_argument("Canopy id " + std::to_string(pid) +
                                        ", present in canopy_ids member of ClusterData, is not present in database");
        }
    }
}

float ClusterData::get_density() const
{
    return density;
}

std::map<int, HistogramDistrib> &ClusterData::get_sizematrix()
{
    return size_data;
}

const std::map<int, HistogramDistrib> &ClusterData::get_sizematrix() const
{
    return size_data;
}

HistogramMatrix &ClusterData::get_locmatrix()
{
    return spatial_data;
}

const HistogramMatrix &ClusterData::get_locmatrix() const
{
    return spatial_data;
}

HistogramDistrib &ClusterData::get_sizedistrib(int specid)
{
    return size_data.at(specid);
}

const HistogramDistrib &ClusterData::get_sizedistrib(int specid) const
{
    return size_data.at(specid);
}

const std::unordered_map<int, float> &ClusterData::get_species_ratios()
{
    return species_ratios;
}

float ClusterData::get_species_ratio(int specid)
{
    if (species_ratios.count(specid))
        return species_ratios.at(specid);
    else
        // we don't initialize a ratio, as with when a requested distribution is missing, since updating a ratio
        // doesn't really make sense. It has to be computed all at once
        return -1.0f;
}

std::vector<int> ClusterData::get_underid_to_idx() const
{
    return spatial_data.get_underid_to_idx();
}

std::vector<int> ClusterData::get_canopyid_to_idx() const
{
    return spatial_data.get_canopyid_to_idx();
}

std::vector<int> ClusterData::get_canopy_ids() const
{
    return std::vector<int>(canopy_ids.begin(), canopy_ids.end());
}

std::vector<int> ClusterData::get_plant_ids() const
{
    return std::vector<int>(plant_ids.begin(), plant_ids.end());
}

void ClusterData::set_species_ratios(const std::unordered_map<int, float> &ratios)
{
    this->species_ratios = ratios;
    fix_species_ratios();	// assign ratio of zero to species for which we don't have a ratio
}

void ClusterData::set_species_ratio(int specid, float ratio)
{
    species_ratios[specid] = ratio;
}

void ClusterData::set_density(float density)
{
    this->density = density;
}


ClusterData::subbiome_clusters_type::subbiome_clusters_type(const data_importer::common_data &cdata, int canopyspecies,
                                                            const std::unordered_map<int, float> &species_props)
{
    int subbiome_id = cdata.canopyspec_to_subbiome.at(canopyspecies);
    data_importer::sub_biome sb = cdata.subbiomes_all_species.at(subbiome_id);
    std::vector<int> subspecies;
#ifndef RW_SELECTION_DEBUG
    // collect all ids of species in this subbiome
    for (auto &spenc : sb.species)
    {
        int specid = spenc.id;
        subspecies.push_back(specid);
    }
#else
     // to debug the roulette wheel selection of species in sample_from_tree
    for (auto &specpair : cdata.canopy_and_under_species)
    {
        subspecies.push_back(specpair.first);
    }
#endif 	// RW_SELECTION_DEBUG
    float sum = 0.0f;

    // for each species in this subbiome, we get the proportion of the total plants in the cluster it makes up,
    // then add that proportion to the probability map for this cluster (species_ratios)
    for (auto &specid : subspecies)
    {
        float prop;
        if (species_props.count(specid))
            prop = species_props.at(specid);
        else
            prop = 0.0f;
        species_ratios.insert({specid, prop});
        sum += prop;
    }
    if (sum > 1e-5f)
    {
        // if at least one species from the subbiome occurs in this cluster, we proceed to normalize the percentages
        for (auto &proppair : species_ratios)
        {
            proppair.second /= sum;
        }
    }
    else
    {
        // this cluster contains no species from this subbiome, so we clear the map
        species_ratios.clear();
    }

}
