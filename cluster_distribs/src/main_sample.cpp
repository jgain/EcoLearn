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
#include <chrono>
#include <iostream>
//#include "UndergrowthSampler.h"
#include "UndergrowthRefiner.h"

int main(int argc, char * argv [])
{
    /*
    int nsample = 5000;
    validcheck checks = validcheck::CANOPY;
    std::string checkstr;

    switch (checks)
    {
        case validcheck::BOTH:
            checkstr = "both";
            break;
        case validcheck::CANOPY:
            checkstr = "canopy";
            break;
        case validcheck::UNDERGROWTH:
            checkstr = "undergrowth";
            break;
        default:
            assert(false);
            break;
    }

    std::vector<std::string> cluster_filenames = {"/home/konrad/PhDStuff/clusters1024/S4500-4500-1024_distribs.clm", "/home/konrad/PhDStuff/clusters1024/S4500-4500-1024-1_distribs.clm"};

    //std::string targetdir = "/home/konrad/PhDStuff/data/datasets128/sonoma128_" + targetnum;
    std::string targetdir = "/home/konrad/PhDStuff/abioticfixed/S4500-4500-1024";
    std::string dbfile = "/home/konrad/EcoSynth/data_preproc/common_data/sonoma.db";

    data_importer::data_dir ddir(targetdir, 1);

    std::map<int, std::vector<MinimalPlant> > mplants;
    data_importer::read_pdb(ddir.canopy_fnames.at(0), mplants);
    std::vector<basic_tree> trees = data_importer::minimal_to_basic(mplants);

    int nrounds = 0;
    int nattempts = 0;

    std::vector<basic_tree> undergrowth = ClusterMatrices::SynthFromCanopy(cluster_filenames,
            targetdir, dbfile, trees, nsample, nrounds, nattempts);

    data_importer::write_pdb("/home/konrad/canopysample_test.pdb", undergrowth.data(), undergrowth.data() + undergrowth.size());
    */

    UndergrowthRefiner::test_encode_decode();

    std::cout.setstate(std::ios_base::failbit);

    data_importer::common_data cdata("/home/konrad/EcoSynth/ecodata/sonoma.db");

    //std::vector<std::string> filenames = {"/home/konrad/PhDStuff/clusterdata/toSimulateRedwood3-test/S2000-2256-256-2_distribs.clm"};
    std::vector<std::string> filenames = {"/home/konrad/PhDStuff/clusterdata/S2000-2256-256-2/S2000-2256-256-2_distribs.clm"};
    AllClusterInfo clusterinfo(filenames, cdata);

    std::cout << "Clustermeans: " << std::endl;
    for (auto &cl : clusterinfo.kclusters)
    {
        for (int i = 0; i < kMeans::ndim; i++)
            std::cout << cl[i] << " ";
        std::cout << std::endl;
    }

    //std::cout << "Press enter to continue..." << std::endl;
    //std::cin.get();

    for (auto &regdensity : clusterinfo.all_densities.at(0))
    {
        if (regdensity.second >= 0.0f)
            std::cout << "Density for region " << regdensity.first << ": " << regdensity.second << std::endl;
    }

    //std::cout << "Press enter to continue..." << std::endl;
    //std::cin.get();

    for (auto &specratios: clusterinfo.all_specratios.at(0))
    {
        int region_idx = specratios.first;
        if (clusterinfo.all_densities.at(0).at(region_idx) >= 0.0f)
        {
            std::cout << "For region " << region_idx << ": " << std::endl;
            for (auto &specpair : specratios.second)
            {
                std::cout << "Species " << specpair.first << " ratio: " << specpair.second << std::endl;
            }
        }
    }
    std::cout << "Total number of clustermeans: " << clusterinfo.kclusters.size() << std::endl;

    auto canopytrees = data_importer::read_pdb("/home/konrad/PhDStuff/abioticfixed/S2000-2256-256/UNDERSIM_RESULT/S2000-2256-256-2/S2000-2256-256-2_canopy0.pdb");

    UndergrowthRefiner sampler(filenames, abiotic_maps_package(data_importer::data_dir("/home/konrad/PhDStuff/abioticfixed/S2000-2256-256/UNDERSIM_RESULT/S2000-2256-256-2", 1), abiotic_maps_package::suntype::CANOPY, abiotic_maps_package::aggr_type::AVERAGE), canopytrees, cdata);
    auto undergrowth = sampler.sample_undergrowth();
    data_importer::write_pdb("/home/konrad/initundergrowth.pdb", undergrowth.data(), undergrowth.data() + undergrowth.size());
    int overall_target = sampler.get_overall_target_count();

    std::cout.clear();
    std::cout << undergrowth.size() << " plants sampled" << std::endl;
    std::cout << "Target: " << overall_target << std::endl;
    std::cout.setstate(std::ios_base::failbit);

    //data_importer::write_pdb("/home/konrad/PhDStuff/undergrowth_refactored_sampled.pdb", undergrowth.data(), undergrowth.data() + undergrowth.size());
    sampler.set_undergrowth(undergrowth);
    std::cout.clear();
    sampler.refine();

    undergrowth = sampler.get_undergrowth();

    data_importer::write_pdb("/home/konrad/synthundergrowth.pdb", undergrowth.data(), undergrowth.data() + undergrowth.size());

    return 0;
}
