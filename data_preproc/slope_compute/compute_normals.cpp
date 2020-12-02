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
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/



#define GLM_ENABLE_EXPERIMENTAL

#include "common/basic_types.h"
#include "data_importer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/string_cast.hpp>


glm::vec3 get_normal(int x, int y, const ValueMap<float> &heights, float xunit, float yunit)
{
    int width, height;
    heights.getDim(width, height);

    int xsub = -1, xadd = 1;
    int ysub = -1, yadd = 1;
    if (x == 0)
    {
        xsub = 0;
    }
    else if (x == width - 1)
    {
        xadd = 0;
    }
    if (y == 0)
    {
        ysub = 0;
    }
    else if (y == height - 1)
    {
        yadd = 0;
    }

    float zh0 = heights.get(x + xsub, y);
    float zh1 = heights.get(x + xadd, y);

    float zv0 = heights.get(x, y + ysub);
    float zv1 = heights.get(x, y + yadd);

    glm::vec3 hcross = glm::vec3(2 * xunit, 0.0f, zh1 - zh0);
    glm::vec3 vcross = glm::vec3(0.0f, 2 * yunit, zv1 - zv0);

    return glm::normalize(glm::cross(hcross, vcross));
}

ValueGridMap<float> compute_slopes(const ValueGridMap<float> &heights, float xunit, float yunit)
{
    int width, height;
    heights.getDim(width, height);

    ValueGridMap<float> slopes;
    slopes.setDim(heights);
    slopes.setDimReal(heights);

    glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            glm::vec3 normal = get_normal(x, y, heights, xunit, yunit);
            assert(normal[2] >= 0.0f);
            float dotprod = glm::dot(up, normal);
            float slope = std::acos(dotprod);
            float slope_degr = slope / (2 * M_PI) * 360.0f;
            slopes.set(x, y, slope_degr);
        }
    }
    return slopes;
}

int main(int argc, char * argv [])
{
    if (argc != 2)
    {
        std::cout << "Usage: slope_compute <data directory>" << std::endl;
        return 1;
    }

    data_importer::data_dir ddir(argv[1]);

    const float xunit = 0.3048 * 3;
    const float yunit = 0.3048 * 3;

    ValueGridMap<float> heights = data_importer::load_elv<ValueGridMap<float> >(ddir.dem_fname);
    ValueGridMap<float> slopes = compute_slopes(heights, xunit, yunit);

    data_importer::write_txt(ddir.slope_fname, &slopes);

    return 0;
}
