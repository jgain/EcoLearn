#include <canopy_placement/gl_wrapper.h>
#include <data_importer.h>
#include <canopy_placement/basic_types.h>
#include <canopy_placement/gpu_procs.h>
#include <chrono>
#include <random>

int main(int argc, char * argv [])
{
    const int w = 128;
    const int h = 128;

    std::default_random_engine generator(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> unif_w(0, w - 1);
    std::uniform_int_distribution<int> unif_h(0, h - 1);
    std::uniform_int_distribution<int> unif_treeh(5, 50);
    std::uniform_int_distribution<int> unif_species(0, 2);

    std::vector<int> default_species = {5, 7, 9};

    const int ntrees = 400;

    std::vector<float> chmdata(128 * 128, 100.0f / 0.3048f);

    gl_wrapper gl_object(chmdata.data(), 128, 128);

    std::vector<mosaic_tree> trees;

    std::map<int, std::vector<mosaic_tree> > treemap;


    if (argc == 1)
    {
        for (int i = 0; i < ntrees; i++)
        {
            int x = unif_w(generator);
            int y = unif_h(generator);
            int h = unif_treeh(generator);
            int r = h * 0.2f;
            r = r > 0 ? r : 1;
            auto newtree = mosaic_tree(x, y, r, h, true);
            int species = default_species[unif_species(generator)];
            newtree.species = species;
            treemap[species].push_back(newtree);
        }
    }
    else
    {
        std::string pdb_filename = argv[1];
        std::map<int, std::vector<MinimalPlant> > plantmap;
        data_importer::read_pdb(pdb_filename, plantmap);
        std::map<int, std::vector<mosaic_tree> > treemap;
        data_importer::minimaltree_to_othertree(plantmap, treemap);
        for (auto &spectrees : treemap)
        {
            for (auto &tree : spectrees.second)
            {
                trees.push_back(mosaic_tree(tree.x, tree.y, tree.radius, tree.height, true));
                trees.back().species = spectrees.first;
            }
        }
    }

    int r = 10;
    int g = 0;
    int b = 0;

    std::vector < std::vector<int> > colors;
    colors = { {255, 0, 0}, {255, 255, 0}, {255, 0, 255}, {0, 255, 0}, {0, 0, 255}, {0, 255, 255} };
    int curr_col_idx = 0;
    std::set<int> colors_assigned;

    for (auto &spectrees : treemap)
    {
        auto result = colors_assigned.insert(spectrees.first);
        for (auto &tree : spectrees.second)
        {
            auto &currcol = colors[curr_col_idx];
            trees.push_back(tree);
            trees.back().r = currcol[0];
            trees.back().g = currcol[1];
            trees.back().b = currcol[2];
            trees.back().a = 255;
            tree.b = 1;
            tree.r = tree.g = 0;
            tree.a = 255;
        }
        if (result.second)
        {
            curr_col_idx++;
        }
    }


    gl_object.build_chm_instanced(trees.begin(), trees.end());

    auto colordata = gl_object.get_color_id_data();

    ValueMap<uint32_t> writemap;
    writemap.setDim(128, 128);
    memcpy(writemap.data(), colordata.data(), sizeof(uint32_t) * w * h);

    data_importer::write_txt("/home/konrad/undersample_map_all.txt", &writemap);

    for (auto &spectrees : treemap)
    {
        int species = spectrees.first;
        gl_object.build_chm_instanced(spectrees.second.begin(), spectrees.second.end());
        colordata = gl_object.get_color_id_data();
        memcpy(writemap.data(), colordata.data(), sizeof(uint32_t) * w * h);
        ValueMap<float> writefloatmap;
        writefloatmap.setDim(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                writefloatmap.set(x, y, (float)writemap.get(x, y));
            }
        }
        ValueMap<float> smoothmap;
        smoothmap.setDim(w, h);
        for (auto &val : writefloatmap)
        {
            val -= 255;
        }
        data_importer::write_txt("/home/konrad/undersample_map_species" + std::to_string(species) + ".txt", &writemap);
        smooth_uniform_radial(15, writefloatmap.data(), smoothmap.data(), w, h);
        data_importer::write_txt("/home/konrad/undersample_map_species" + std::to_string(species) + "_smoothed.txt", &smoothmap);
    }

    std::cout << colordata[50 * 128 + 50] << std::endl;

	return 0;
}
