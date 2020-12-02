// By K.P. Kapp 
// July 2019

#include "grass_sim.h"
#include "data_importer.h"

#include <iostream>
#include <map>
#include <experimental/filesystem>

using fspath = std::experimental::filesystem::path;

int main(int argc, char * argv []) {
    bool use_treeshade = false;
    std::string canopyfname;
    if (argc < 3 || argc > 5)
    {
        std::cout << "Usage: grass_sim <data directory> <outfile> [canopy filename] [--use_treeshade]" << std::endl;
        return 1;
    }
    if (argc == 5)
    {
        if (strcmp(argv[4], "--use_treeshade"))
        {
            std::cout << "Usage: grass_sim <data directory> <outfile> [canopy filename] [--use_treeshade]" << std::endl;
            return 1;
        }
        use_treeshade = true;
        canopyfname = argv[3];
    }
    if (argc == 4)
    {
        if (!strcmp(argv[3], "--use_treeshade"))
                use_treeshade = true;
        else
            canopyfname = argv[3];
    }
    std::string out_filename = argv[2];
    data_importer::data_dir ddir(argv[1], 1);
    std::string shade_filename;
    if (use_treeshade)
        shade_filename = ddir.sun_tree_fname;
    else
        shade_filename = ddir.sun_fname;
    ValueGridMap<float> ter = data_importer::load_elv<ValueGridMap<float> >(ddir.dem_fname);
    std::map<int, std::vector<MinimalPlant> > minplants;
    if (canopyfname.size() == 0)
        canopyfname = ddir.canopy_fnames.at(0);

    std::cout << "data dir: " << argv[1] << std::endl;
    std::cout << "Out filename: " << out_filename << std::endl;
    std::cout << "Canopy filename: " << canopyfname << std::endl;
    std::cout << "Use treeshade? " << use_treeshade << std::endl;

    fspath outpath(out_filename);
    fspath noext = outpath.replace_extension();
    std::string litfall_out = std::string(noext.c_str());
    litfall_out += "_litfall.txt";
    std::cout << "Litterfall output file: " << litfall_out << std::endl;

    data_importer::read_pdb(canopyfname, minplants);
    std::vector<basic_tree> trees = data_importer::minimal_to_basic(minplants);
    std::vector<basic_tree *> treeptrs;
    for (auto &tr : trees)
        treeptrs.push_back(&tr);

    std::vector<MapFloat> moisture, sunlight, temperature;
    moisture = data_importer::read_monthly_map<MapFloat>(ddir.wet_fname);
    sunlight = data_importer::read_monthly_map<MapFloat>(shade_filename);
    temperature = data_importer::read_monthly_map<MapFloat>(ddir.temp_fname);

    /*
    std::cout << "Computing moisture average map..." << std::endl;
    MapFloat moisture_avg = data_importer::average_mmap<MapFloat, MapFloat>(moisture);
    std::cout << "Computing sunlight average map..." << std::endl;
    MapFloat sunlight_avg = data_importer::average_mmap<MapFloat, MapFloat>(sunlight);
    std::cout << "Computing temperature average map..." << std::endl;
    MapFloat temp_avg = data_importer::average_mmap<MapFloat, MapFloat>(temperature);
    */
    MapFloat moisture_avg = moisture.at(3);
    MapFloat sunlight_avg = sunlight.at(3);
    MapFloat temp_avg = temperature.at(3);

    auto viability_params = data_importer::read_grass_viability(ddir.grass_params_fname);

    std::cout << "Constructing grass sim object..." << std::endl;
    GrassSim gsim(ter, 1);
    //gsim.setConditions(&moisture[5], &sunlight[5], &temperature[5]);
    gsim.setConditions(&moisture_avg, &sunlight_avg, &temp_avg);
    gsim.set_viability_params(viability_params);
    gsim.set_commondata("/home/konrad/EcoSynth/ecodata/sonoma.db");		// TODO: pass db pathname as parameter
    std::cout << "Running grass simulation..." << std::endl;
    gsim.grow(ter, treeptrs);
    std::cout << "All done" << std::endl;

    std::cout << "Writing grass file to " << out_filename << std::endl;
    gsim.write(out_filename);
    std::cout << "Writing litterfall file to " << litfall_out << std::endl;
    gsim.write_litterfall(litfall_out);

    //std::cout << "Hello, World!" << std::endl;
    return 0;
}
