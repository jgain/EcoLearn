#include "ClusterMatrices.h"
#include "data_importer.h"

int main(int argc, char * argv [])
{
    /*
    auto canopytrees = data_importer::read_pdb("/home/konrad/PhDStuff/abioticfixed/S4500-4500-1024/S4500-4500-1024_canopy0.pdb");
    abiotic_maps_package amaps(data_importer::data_dir("/home/konrad/PhDStuff/abioticfixed/S4500-4500-1024/"), abiotic_maps_package::suntype::LANDSCAPE_ONLY, abiotic_maps_package::aggr_type::APRIL);
    std::unique_ptr<ClusterMatrices> clptr = ClusterMatrices::CreateClusterMatrices({"/home/konrad/PhDStuff/data/clusters_test_1024.clm"}, "/home/konrad/PhDStuff/abioticfixed/S4500-4500-1024/", amaps, "/home/konrad/EcoSynth/ecodata/sonoma.db", &canopytrees);
    auto plnthashmap = clptr->sample_from_probmap(nullptr);

    auto plnts = clptr->get_sampled_plants();

    data_importer::write_pdb("/home/konrad/PhDStuff/undergrowth_test1.pdb", plnts.data(), plnts.data() + plnts.size());
    */
}
