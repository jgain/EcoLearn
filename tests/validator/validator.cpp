#include "validator.h"
#include "canopy_placement/basic_types.h"
#include "data_importer.h"

validator::validator(std::string pdbfilename, std::string datadir)
    : validator(data_importer::read_pdb(pdbfilename), datadir)
{

}

validator::validator(const std::vector<basic_tree> &plants, std::string datadir)
    : plants(plants), bounds_valid(-1), water_valid(-1), intersect_valid(-1), slope_valid(-1)
{
    data_importer::data_dir ddir(datadir, 1);

    moisture = data_importer::read_monthly_map<ValueGridMap<float> >(ddir.wet_fname);
    sun = data_importer::read_monthly_map<ValueGridMap<float> >(ddir.sun_fname);
    temp = data_importer::read_monthly_map<ValueGridMap<float> >(ddir.temp_fname);
    dem = data_importer::load_elv<ValueGridMap<float> >(ddir.dem_fname);
    chm = data_importer::load_txt<ValueGridMap<float> >(ddir.chm_fname);

    moisture_avg = data_importer::average_mmap<ValueGridMap<float>, ValueGridMap<float> >(moisture);
    sun_avg = data_importer::average_mmap<ValueGridMap<float>, ValueGridMap<float> >(sun);
    temp_avg = data_importer::average_mmap<ValueGridMap<float>, ValueGridMap<float> >(temp);
}

int validator::validate_bounds()
{
    float width, height;
    dem.getDimReal(width, height);

    bounds_valid = 1;

    for (auto &plnt : plants)
    {
        if (!is_within_landscape(plnt.x, plnt.y))
        {
            bounds_valid = 0;
            return 0;
        }
    }
    return bounds_valid;
}

int validator::validate_water(bool monthly_validate)
{
    water_valid = 1;
    for (auto &plnt : plants)
    {
        if (monthly_validate)
        {
            for (auto &vgmap : moisture)
            {
                if (vgmap.get_fromreal(plnt.x, plnt.y) > 1999.0f)		// 2000.0f is supposed to be standing water (check waterfill program, in the MoistureSim::simSoilCycle function)
                {
                    water_valid = 0;
                    return 0;
                }
            }
        }
        else
        {
            if (moisture_avg.get_fromreal(plnt.x, plnt.y) > 1999.0f)
            {
                water_valid = 0;
                return 0;
            }
        }
    }
    return water_valid;
}

int validator::validate_intersect()
{
    intersect_valid = 1;
    float sq_trunksize = 0.2f * 0.2f;
    for (int i = 0; i < plants.size(); i++)
    {
        for (int j = i + 1; j < plants.size(); j++)
        {
            const basic_tree &plnt1 = plants.at(i);
            const basic_tree &plnt2 = plants.at(j);
            float dsq = (plnt1.x - plnt2.x) * (plnt1.x - plnt2.x) + (plnt1.y - plnt2.y) * (plnt1.y - plnt2.y);
            if (dsq < sq_trunksize)
            {
                intersect_valid = 0;
                return 0;
            }
        }
    }
    return intersect_valid;
}

int validator::water_isvalid()
{
    return water_valid;
}

int validator::bounds_isvalid()
{
    return bounds_valid;
}

std::vector<basic_tree> validator::analyse_boundsfailure()
{
    std::vector<basic_tree> outplants;

    for (auto &plnt : plants)
    {
        if (!is_within_landscape(plnt.x, plnt.y))
        {
            outplants.push_back(plnt);
        }
    }

    float w, h;
    dem.getDimReal(w, h);

    std::cout << "Out of bound plants for landscape " << w << " x " << h << ": " << std::endl;
    for (auto &op : outplants)
    {
        std::cout << op.x << ", " << op.y << ", " << op.radius << std::endl;
    }
    std::cout << "Number of out of bound plants: " << outplants.size() << std::endl;

    return outplants;
}

int validator::intersect_isvalid()
{
    return intersect_valid;
}

bool validator::is_within(float min, float max, float val)
{
    return val >= min && val <= max;
}

bool validator::is_within(float min1, float min2, float max1, float max2, float val1, float val2)
{
    return is_within(min1, max1, val1) && is_within(min2, max2, val2);
}

bool validator::is_within_landscape(float x, float y)
{
    float width, height;
    dem.getDimReal(width, height);
    return is_within(0.0f, 0.0f, width, height, x, y);
}
