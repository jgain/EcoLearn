#include "data_importer.h"
#include "ClusterMatrices.h"

int main(int argc, char * argv [])
{
    ClusterMatrices::AllClusterInfo clinfo = ClusterMatrices::import_clusterinfo({"/home/konrad/PhDStuff/data/clusters1024.clm"}, "/home/konrad/EcoSynth/ecodata/sonoma.db");

    std::cout << "Number of clusters in all_distribs vector: " << clinfo.all_distribs.at(0).size() << std::endl;
    clinfo.print_densities();

    return 0;
}
