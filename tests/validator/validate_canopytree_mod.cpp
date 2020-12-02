#include "ClusterMatrices.h"

int main(int argc, char * argv [])
{
    std::string targetdir, db_pathname, pdb_pathname;
    if (argc < 3 || argc > 4)
    {
        std::cout << "Usage: validate_addremove <targetdir> <db pathname> [pdb pathname]" << std::endl;
        return 1;
    }
    else if (argc >= 3)
    {
        targetdir = argv[1];
        db_pathname = argv[2];
        if (argc == 4)
        {
            pdb_pathname = argv[3];
        }
    }

    abiotic_maps_package amaps(targetdir, abiotic_maps_package::suntype::CANOPY, abiotic_maps_package::aggr_type::APRIL);
    data_importer::data_dir ddir(targetdir, 1);

    std::vector<basic_tree> canopytrees, underplants;
    canopytrees = data_importer::read_pdb(ddir.canopy_fnames.at(0));

    if (pdb_pathname.size() > 0)
    {
        underplants = data_importer::read_pdb(pdb_pathname);
    }
    else
    {
        underplants = data_importer::read_pdb(ddir.undergrowth_fnames.at(0));
    }

    ClusterMatrices::test_canopytree_add_remove({"/home/konrad/PhDStuff/clusters1024/S4500-4500-1024-1_distribs.clm"}, targetdir, amaps, db_pathname, canopytrees, underplants, ClusterMatrices::layerspec::CANOPY);

    return 0;
}
