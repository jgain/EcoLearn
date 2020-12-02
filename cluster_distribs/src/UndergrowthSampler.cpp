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


#include "UndergrowthSampler.h"
#include "ClusterMatrices.h"
#include "ClusterMaps.h"
#include "common/constants.h"
#include <gpusample/src/gpusample.h>



UndergrowthSampler::UndergrowthSampler(std::vector<std::string> cluster_filenames,
                                       abiotic_maps_package amaps,
                                       std::vector<basic_tree> canopytrees,
                                       data_importer::common_data cdata)
    : model(cluster_filenames, cdata),
      clmaps(AllClusterInfo(cluster_filenames, cdata), amaps, canopytrees,
             cdata),
      region_target_counts(clmaps.compute_region_target_counts(model)),
      sunbackup(amaps.sun)
{

}

void UndergrowthSampler::init_synthcount()
{
    for (int regionidx = 0; regionidx < model.get_nclusters(); regionidx++)
    {
        region_synthcount[regionidx] = 0;
    }
}

void UndergrowthSampler::update_sunmap(const ValueGridMap<float> &sunmap)
{
    clmaps.set_sunmap(sunmap);
}

void UndergrowthSampler::set_progress_callback(std::function<void (int)> callback)
{
    progress_callback = callback;
}

void UndergrowthSampler::update_canopytrees(const std::vector<basic_tree> &canopytrees,
                                            const ValueGridMap<float> &sunmap)
{
    clmaps.update_canopytrees(canopytrees, sunmap);
    region_target_counts = clmaps.compute_region_target_counts(model);
}

void UndergrowthSampler::init_occgrid()
{
    occgrid.setDim(int(clmaps.get_maps().rw * 10), int(clmaps.get_maps().rh * 10));
    occgrid.setDimReal(clmaps.get_maps().rw, clmaps.get_maps().rh);
    occgrid.fill(false);

    for (const auto &tree : clmaps.get_canopytrees())
    {
        float sxr = tree.x - 1.0f;
        float exr = tree.x + 1.0f;
        float syr = tree.y - 1.0f;
        float eyr = tree.y + 1.0f;

        xy<int> sg = occgrid.togrid_safe(sxr, syr);
        xy<int> eg = occgrid.togrid_safe(exr, eyr);

        for (int y = sg.y; y <= eg.y; y++)
        {
            for (int x = sg.x; x <= eg.x; x++)
            {
                occgrid.set(x, y, true);
            }
        }
    }
}

bool UndergrowthSampler::sample_from_tree(const basic_tree *ptr,
                                          basic_tree &new_underplant,
                                          ValueGridMap<bool> *occgrid)
{

    auto cdata = model.get_cdata();

    auto in_landscape = [this](float x, float y) {
        return x > 1e-5f &&
                x < clmaps.get_maps().rw - 1e-5f &&
                y > 1e-5f &&
                y < clmaps.get_maps().rh - 1e-5f;
    };

    int attempt_count = 0;

    int region_saturated = 0;
    int nospecies = 0;
    int occupied = 0;
    bool abiotic_ok = false;
    float newx, newy, newradius, newheight;
    int newspecies;
    int region_idx;
    int occgw, occgh;
    if (occgrid)
        occgrid->getDim(occgw, occgh);
    do {
        abiotic_ok = false;

        //std::cout << "Start sampling..." << std::endl;

        // check, and sample until new point is inside landscape
        // ----------------------------------------------------
        do {
            float angle = unif(gen) * 2 * M_PI;
            float rad = unif(gen) * ptr->radius * common_constants::undersim_sample_mult;
            float dx = cos(angle) * rad;
            float dy = sin(angle) * rad;
            newx = ptr->x + dx, newy = ptr->y + dy;
        } while (!in_landscape(newx, newy));
        //std::cout << "End sampling" << std::endl;


        // duplicate safety check, if requested, by passing a valid, non-null pointer occgrid to this function
        if (occgrid)
        {
            xy<int> gc = occgrid->togrid(newx, newy);
            // if no other undergrowth plants are very close ('very close' defined by cell size of occgrid) then
            // we sample and set surrounding cells as occupied. Else, we count this as a failed sample and try again
            if (!occgrid->get(gc.x, gc.y))
            {
                int sx = gc.x - 1;
                int ex = gc.x + 1;
                int sy = gc.y - 1;
                int ey = gc.y + 1;
                common_funcs::trim(0, occgw - 1, sx);
                common_funcs::trim(0, occgw - 1, ex);
                common_funcs::trim(0, occgh - 1, sy);
                common_funcs::trim(0, occgh - 1, ey);
                for (int cy = sy; cy <= ey; cy++)
                {
                    for (int cx = sx; cx <= ex; cx++)
                    {
                        occgrid->set(cx, cy, true);
                    }
                }
            }
            else
            {
                occupied++;

                attempt_count++;
                continue;
            }
        }

        // check if density of region is still good after sampling this plant
        // ----------------------------------------------------
        region_idx =  clmaps.get_cluster_idx(newx, newy);
        if (region_idx < 0)
        {
            attempt_count++;
            continue;
        }
        if (region_synthcount.at(region_idx) + 1 > region_target_counts.at(region_idx))
        {
            region_saturated++;

            attempt_count++;
            continue;
        }

        const auto &specratios = model.get_cluster(region_idx).get_subspecies_ratios(ptr->species);
        if (specratios.size() == 0)
        {
            nospecies++;

            // none of the species in this canopytree's subbiome are present in this cluster - skip
            attempt_count++;
            continue;
        }

        float rndselect = unif(gen);

        float rndsum = 0.0f;
        //for (auto &plntid : plant_ids)
        int lastspec;
        for (const auto &proppair : specratios)
        {
            //rndsum += species_props.at(region_idx).at(plntid);
            rndsum += proppair.second;
            int plntid = proppair.first;		// candidate plant id
            if (rndsum > rndselect)
            {
                newspecies = plntid;
                abiotic_ok = true;

                newheight = model.get_cluster(region_idx).get_sizedistrib(newspecies).rndgen();
                newradius = cdata.modelsamplers.at(newspecies).sample_rh_ratio(newheight) * newheight;

                break;
            }
            lastspec = plntid;
        }
        if (!abiotic_ok)
        {
            newspecies = lastspec;
            abiotic_ok = true;

            newheight = model.get_cluster(region_idx).get_sizedistrib(newspecies).rndgen();
            newradius = cdata.modelsamplers.at(newspecies).sample_rh_ratio(newheight) * newheight;

        }

        // XXX: we are not checking for abiotic conditions,
        // because we are assuming that the clusters already sufficiently summarise the abiotic conditions
    } while (!abiotic_ok && attempt_count <= 10);


    // if all is good with new plant, add to plant population
    // --------------------------------------------
    if (abiotic_ok) {
        new_underplant = basic_tree(newx, newy, newradius, newheight);
        new_underplant.species = newspecies;
        region_synthcount[region_idx]++;
        return true;
    }
    else
    {
        return false;
    }
}


// TODO: let overall target count for undergrowth plants also determine
// 		 stopping condition
std::vector<basic_tree> UndergrowthSampler::sample_undergrowth()
{
    init_occgrid();
    init_synthcount();

    std::vector<basic_tree> undergrowth;
    std::vector<bool> sample_from(clmaps.get_canopytrees().size(), true);

    int nsampled_round;
    do
    {
        nsampled_round = 0;
        int idx = 0;
        for (const auto &canopytree : clmaps.get_canopytrees())
        {
            if (sample_from.at(idx))
            {
                basic_tree new_underplant;
                bool sampled = sample_from_tree(&canopytree, new_underplant, &occgrid);
                if (sampled)
                {
                    nsampled_round++;
                    undergrowth.push_back(new_underplant);
                }
                else
                {
                    sample_from.at(idx) = false;
                }
            }
            idx++;
        }
    }
    while (nsampled_round > 0);

    return undergrowth;
}

std::vector<basic_tree> UndergrowthSampler::sample_undergrowth_gpu(const std::vector<basic_tree> &canopytrees)
{
    clmaps.set_sunmap(sunbackup);
    return compute_and_sample_plants(clmaps, model, canopytrees, progress_callback);
}

int UndergrowthSampler::get_overall_target_count()
{
    auto targetcounts = clmaps.compute_region_target_counts(model);

    int sum = 0;
    for (auto &countpair : targetcounts)
    {
        if (countpair.second >= 0)
            sum += countpair.second;
    }
    return sum;
}

const ClusterMaps &UndergrowthSampler::get_clustermaps() const
{
    return clmaps;
}

ClusterMaps &UndergrowthSampler::get_clustermaps()
{
    return clmaps;
}

const ClusterMatrices &UndergrowthSampler::get_model() const
{
    return model;
}
