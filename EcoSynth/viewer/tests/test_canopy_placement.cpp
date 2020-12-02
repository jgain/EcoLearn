#include "canopy_placement/gpu_procs.h"
#include "data_importer.h"
#include <string>
#include "canopy_placement/basic_types.h"
#include <sqlite3.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

void test_local_maxima_find()
{
    std::cerr << "test_find_local_maxima_gpu: " << std::endl;
    if (test_find_local_maxima_gpu(verbosity::ALL))
    {
        std::cerr << "PASS" << std::endl;
    }
    else
    {
        std::cerr << "FAIL" << std::endl;
    }
}

void remove_files_containing(std::string containing)
{

}

std::string get_next_filename(std::string basename, std::string directory)
{
    using namespace boost::filesystem;

    path p(directory);

    if (!boost::filesystem::is_directory(p))
    {
        throw std::invalid_argument("Directory must be passed to get_next_filename");
    }

    int nexist = 0;
    for (auto &entry : boost::make_iterator_range(directory_iterator(p), {}))
    {
        std::string exist_filename = entry.path().filename().string();
        if (exist_filename.find(basename) != std::string::npos)
        {
            nexist++;
        }
        //std::cout << entry.path().string() << std::endl;
        //std::cout << newp.string() << std::endl;
    }

    std::string full_basename = basename + std::to_string(nexist) + ".txt";
    full_basename = (p / full_basename).string();

    return full_basename;
}

void write_array_to_file(std::string out_file, const std::vector<uint32_t> &values, int width, int height)
{
    ValueMap<uint32_t> outmap;
    outmap.setDim(width, height);
    memcpy(outmap.data(), values.data(), sizeof(uint32_t) * width * height);

    data_importer::write_txt(out_file, &outmap);
}

int main(int argc, char * argv [])
{
    using namespace basic_types;

    if (argc != 3)
    {
        std::cout << "Usage: canopy_placement <dataset directory> <database filename>" << std::endl;
        return 1;
    }

    std::string test_texture_out_filename_base = "texture_test_out";

    data_importer::common_data simdata(argv[2]);
    data_importer::data_dir ddir(argv[1], simdata);
    int nsims = ddir.required_simulations.size();
    std::map<int, data_importer::species> all_species = simdata.all_species;

    int width, height;
    MapFloat chm = data_importer::load_txt< MapFloat >(ddir.chm_fname);
    MapFloat dem = data_importer::load_elv< MapFloat >(ddir.dem_fname);
    //for (auto &biome_fname : ddir.biome_fnames)
    for (int i = 0; i < 1; i++)
    {
        std::string test_texture_out_filename = get_next_filename(test_texture_out_filename_base, argv[1]);

        std::string canopy_fname = ddir.canopy_fnames[i];
        std::string species_fname = ddir.species_fnames[i];

        ValueMap<int> species = data_importer::load_txt< ValueMap<int> >(species_fname);
        //std::vector<species_params> all_params = data_importer::read_species_params(biome_fname);
        canopy_placer placer(&chm, &species, all_species);
        //placer.optimise(20);

        chm.getDim(width, height);

        placer.init_optim();
        for (int i = 0; i < 50; i++)
        {
            placer.iteration();
            std::vector<uint32_t> out_values;
            //placer.get_chm_rendered_texture(out_values);
            //write_array_to_file(test_texture_out_filename, out_values, width, height);
            placer.check_duplicates();
        }
        //placer.final_adjustments_gpu();
        placer.write_chm_data_to_file(test_texture_out_filename);
        placer.save_to_file(canopy_fname, dem.data());
    }
    return 0;

}

