#include <vector>
#include <string>
#include "canopy_placement/basic_types.h"

class validator
{
public:
	validator(const std::vector<basic_tree> &plants, std::string datadir);
    validator(std::string pdbfilename, std::string datadir);

    int validate_water(bool monthly_validate);
    int validate_slope();
    int validate_intersect();
    int validate_bounds();

    int water_isvalid();
    int slope_isvalid();
    int intersect_isvalid();
    int bounds_isvalid();

    std::vector<basic_tree> analyse_boundsfailure();

private:
    int water_valid;
    int slope_valid;
    int intersect_valid;
    int bounds_valid;

    std::vector<basic_tree> plants;

    ValueGridMap<float> moisture_avg, sun_avg, temp_avg, slope;
    ValueGridMap<float> dem, chm;

    std::vector<ValueGridMap<float> > moisture, sun, temp;

private:
    static bool is_within(float min, float max, float val);
    static bool is_within(float min1, float min2, float max1, float max2, float val1, float val2);
    bool is_within_landscape(float x, float y);
};
