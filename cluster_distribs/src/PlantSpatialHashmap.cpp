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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include "PlantSpatialHashmap.h"
#include "data_importer/AbioticMapper.h"

#include <numeric>

PlantSpatialHashmap::PlantSpatialHashmap(float cellw, float cellh, float real_width, float real_height)
    : cellw(cellw), cellh(cellh), real_width(real_width), real_height(real_height)
{
    /*
     * round up to get number of rows and columns based on division by cell height and width
     */
    nrows = static_cast<int>(std::ceil(real_height / cellh) + 1e-5f);
    ncols = static_cast<int>(std::ceil(real_width / cellw) + 1e-5f);

    /*
     * Pre-allocate the required size for the hashmap
     */
    hashmap.resize(nrows);
    for (auto &row : hashmap)
    {
        row.resize(ncols);
    }
}

xy<int> PlantSpatialHashmap::get_gridcoord(float x, float y) const
{
    xy<int> loc;
    loc.x = x / cellw;
    loc.y = y / cellh;
    return loc;
}

void PlantSpatialHashmap::get_griddim(int &nrows, int &ncols) const
{
    nrows = this->nrows;
    ncols = this->ncols;
}

std::vector<basic_tree> PlantSpatialHashmap::get_all_plants() const
{
    std::vector<basic_tree> allplants;

    for (int y = 0; y < nrows; y++)
    {
        for (int x = 0; x < ncols; x++)
        {
            const auto &cellplants = hashmap.at(y).at(x);
            allplants.insert(allplants.end(), cellplants.begin(), cellplants.end());
        }
    }
    return allplants;
}


void PlantSpatialHashmap::addplant(const basic_tree &plnt)
{
    if (plnt.x < 0.0f || plnt.x > real_width - 1e-5f || plnt.y < 0.0f || plnt.y > real_height - 1e-5f)
    {
        std::cout << "WARNING: plant at " << plnt.x << ", " << plnt.y << " is out of bounds. Not adding it to spatial hashmap" << std::endl;
        return;
    }

    const xy<int> &loc = get_gridcoord(plnt.x, plnt.y);
    if (loc.x >= 0 && loc.x < ncols && loc.y >= 0 && loc.y < nrows)
        hashmap.at(loc.y).at(loc.x).push_back(plnt);
    else
        std::cout << "WARNING: plant at " << plnt.x << ", " << plnt.y << " is out of bounds. Not adding it to spatial hashmap" << std::endl;
}

void PlantSpatialHashmap::removeplant(const basic_tree &plnt)
{
    const xy<int> &loc = get_gridcoord(plnt.x, plnt.y);
    if (loc.x >= 0 && loc.x < ncols && loc.y >= 0 && loc.y < nrows)
    {
        std::vector<basic_tree> &cellplants = hashmap.at(loc.y).at(loc.x);
        for (auto iter = cellplants.begin(); iter != cellplants.end(); std::advance(iter, 1))
        {
            if (*iter == plnt)
            {
                cellplants.erase(iter);
                return;
            }
        }
    }
}

std::vector< std::vector< basic_tree> *> PlantSpatialHashmap::get_relevant_cells(float x, float y, float radius)
{
    // get starting and ending locations based on a box centered around location x, y,
    // and make adjustments if outside bounds
    float sx = x - radius;
    if (sx < 0.0f) sx = 0.0f;
    float sy = y - radius;
    if (sy < 0.0f) sy = 0.0f;
    float ex = x + radius;
    if (ex >= real_width - 1e-3f) ex = real_width - 1e-3f;
    float ey = y + radius;
    if (ey >= real_height - 1e-3f) ey = real_height - 1e-3f;

    std::vector< std::vector< basic_tree> *> relcells;

    // Get cell index range based on starting and ending real-world locations
    int sxc = sx / cellw;
    int exc = ex / cellw;
    int syc = sy / cellh;
    int eyc = ey / cellh;

    for (int yc = syc; yc <= eyc; yc++)
        for (int xc = sxc; xc <= exc; xc++)
        {
            relcells.push_back(&hashmap.at(yc).at(xc));
        }

    return relcells;
}

void PlantSpatialHashmap::addplants(const std::vector<basic_tree> &plants)
{
    for (const auto &plnt : plants)
    {
        addplant(plnt);
    }
}

const std::vector<basic_tree> &PlantSpatialHashmap::get_cell_direct(int flatidx) const
{
    int x = flatidx % ncols;
    int y = flatidx / ncols;
    return get_cell_direct(x, y);
}

const std::vector<basic_tree> &PlantSpatialHashmap::get_cell_direct(int x, int y) const
{
    return hashmap.at(y).at(x);
}

std::vector<basic_tree> &PlantSpatialHashmap::get_cell_direct(int x, int y)
{
    return hashmap.at(y).at(x);
}

const std::vector<basic_tree> &PlantSpatialHashmap::get_cell_fromreal(float x, float y) const
{
    const auto &loc = get_gridcoord(x, y);
    return get_cell_direct(loc.x, loc.y);
}

std::vector<int> PlantSpatialHashmap::get_surr_flatidxes(int x, int y) const
{
    std::vector<int> flatidxes;
    for (int cx = x - 1; cx <= x + 1; cx++)
    {
        for (int cy = y - 1; cy <= y + 1; cy++)
        {
            if (cx >= 0 && cy >= 0 && cx < ncols && cy < nrows)
            {
                flatidxes.push_back(cy * ncols + cx);
            }
        }
    }
    return flatidxes;
}

std::vector<int> PlantSpatialHashmap::get_surr_flatidxes(int flatidx) const
{
    int x = flatidx % ncols;
    int y = flatidx / ncols;
    return get_surr_flatidxes(x, y);
}

bool PlantSpatialHashmap::test_add()
{
    basic_tree tree(real_width - real_width / 2, real_height - real_height / 2, 10.0f, 10.0f);

    int cx = tree.x / cellw;
    int cy = tree.y / cellh;

    addplant(tree);

    const auto &plants = get_cell_direct(cx, cy);

    for (const auto &plnt : plants)
    {
        float xd = abs(plnt.x - tree.x);
        float yd = abs(plnt.y - tree.y);
        float rd = abs(plnt.radius - tree.radius);
        float hd = abs(plnt.height - tree.height);
        if (xd < 1e-5f && yd < 1e-5f && rd < 1e-5f && hd < 1e-5f)
        {
            return true;
        }
    }

    std::cout << "PlantSpatialHashMap::test_add failed" << std::endl;

    return false;
}

void PlantSpatialHashmap::runtests()
{
    PlantSpatialHashmap map(10.0f, 10.0f, 90.0f, 90.0f);
    if (map.test_add())
    {
        std::cout << "First PlantSpatialHashmap::test_add in PlantSpatialHashmap::runtests successful" << std::endl;
    }

    PlantSpatialHashmap map2(10.0f, 10.0f, 100.0f, 100.0f);
    if (map2.test_add())
    {
        std::cout << "Second PlantSpatialHashmap::test_add in PlantSpatialHashmap::runtests successful" << std::endl;
    }
}

float PlantSpatialHashmap::calculate_required_cellsize(float maxdist, const std::vector<basic_tree> &plants)
{
    float maxrad = std::accumulate(plants.begin(), plants.end(), 0.0f, [](const float &acc, const basic_tree &currplant) {
        if (currplant.radius > acc)
            return currplant.radius;
        else
            return acc;
    });
    return maxdist + maxrad * 2.0f;
}
