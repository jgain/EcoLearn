#include "canopy_placement/basic_types.h"
#include "canopy_placement/species_assign.h"
#include "canopy_placement/misc.h"
#include "canopy_placement/extract_png.h"

#include <functional>
#include <algorithm>

int main(int argc, char * argv [])
{
    int width, height;
    int mwidth, mheight;
    std::vector<basic_tree> trees;
    read_pdb("/home/konrad/PhDStuff/spacing_pipeline_out.pdb", trees);
    std::vector<float> chm_data = get_image_data_8bit("/home/konrad/PhDStuff/chm_pipeline_out.png", width, height)[0];
    std::vector<float> mflow_data = get_image_data_48bit("/home/konrad/PhDStuff/H10mflow.png", mwidth, mheight)[0];

    for (auto &v : mflow_data)
    {
        if (v > 5000)
        {
            v = 5000;
        }
    }

    if (width != mwidth || height != mheight)
    {
        throw std::runtime_error("Width and/or height of CHM and mflow data don't match. Aborting.");
    }


    std::vector<basic_tree*> tree_ptrs(trees.size());
    int i = 0;
    for (auto &tree : trees)
    {
        tree_ptrs[i] = &tree;
        i++;
    }
    std::vector<float *> adapt_maps_data = {chm_data.data(), mflow_data.data()};
    std::vector<bool> adapt_maps_ownership = {false, false};
    std::vector<bool> row_major = {true, true};

    std::vector< std::vector<std::function<float(float)> > > adapt_maps_per_species = {
    { [](float val) { return val/30; }, [](float val) { return 1000/val;} },
    { [](float val) { return 30/val; }, [](float val){ return val/1000; }},
    { [](float val) { return 30/val; }, [](float val){ return 1000/val; }}
    };

    int nspecies = 3;

    auto specassign = species_assign(tree_ptrs, nspecies, width, height, adapt_maps_data, adapt_maps_ownership, row_major, adapt_maps_per_species);
    specassign.assign_species();

    std::vector< std::vector<basic_tree *> > trees_sorted(specassign.get_nspecies());
    for (auto &tptr : tree_ptrs)
    {
        trees_sorted[tptr->species].push_back(tptr);
    }

    write_pdb("/home/konrad/PhDStuff/H10pipeline_assigntest.pdb", trees_sorted);

    return 0;
}
