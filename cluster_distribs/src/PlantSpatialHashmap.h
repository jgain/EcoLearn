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


#ifndef PLANTSPATIALHASHMAP_H
#define PLANTSPATIALHASHMAP_H

#include <common/basic_types.h>
#include <vector>

/*
 * Spatial hashmap for plants. Most important functionality is to get relevant cells around a plant,
 * based on a radius that includes such plants
 */

class PlantSpatialHashmap
{
public:
    PlantSpatialHashmap(float cellw, float cellh, float real_width, float real_height);

    void addplant(const basic_tree &plnt);
    void removeplant(const basic_tree &plnt);

    /*
     * Add multiple plants to the hashmap, as opposed to just adding one as above
     */
    void addplants(const std::vector<basic_tree> &plants);

    /*
     * Get cell at cell location at row y, column x. Note that this is CELL location,
     * not the world location on the landscape
     */
    std::vector<basic_tree> &get_cell_direct(int x, int y);

    /*
     * Const version of above
     */
    const std::vector<basic_tree> &get_cell_direct(int x, int y) const;

    /*
     * Flattened index version of above (also const)
     */
    const std::vector<basic_tree> &get_cell_direct(int flatidx) const;

    /*
     * Get cell that coincides with world location x, y
     */
    const std::vector<basic_tree> &get_cell_fromreal(float x, float y) const;

    /*
     * Get flattened indices of cells surrounding and including the cell at row y, column x
     */
    std::vector<int> get_surr_flatidxes(int x, int y) const;

    /*
     * Flattened index version of above
     */
    std::vector<int> get_surr_flatidxes(int flatidx) const;

    /*
     * Get grid cell grid coordinate of real world location x, y
     */
    xy<int> get_gridcoord(float x, float y) const;

    /*
     * Get the grid dimensions of the cells and assign them to nrows, ncols arguments
     */
    void get_griddim(int &nrows, int &ncols) const;

    // XXX: make a custom iterator instead of this
    std::vector<basic_tree> get_all_plants() const;

    /*
     * Test function for addition of new plants
     */
    bool test_add();

    /*
     * Run written tests
     */
    static void runtests();

    /*
     * Calculate minimum required cellsize if we wish to consider all plants within radius 'maxdist' of each
     * reference plant, such that only the surrounding cells of the reference plant's cell will be relevant
     */
    static float calculate_required_cellsize(float maxdist, const std::vector<basic_tree> &plants);

    /*
     * Get relevant cells around reference location x, y (world coordinates) that contain plants within radius 'radius' of
     * reference location
     */
    std::vector< std::vector<basic_tree> *> get_relevant_cells(float x, float y, float radius);
private:
    float cellw, cellh;
    float real_width, real_height;
    int nrows, ncols;
    using HashCell = std::vector<basic_tree>;
    using HashRow = std::vector<HashCell>;
    using SpatialHash = std::vector<HashRow>;
    SpatialHash hashmap;
};

#endif // PLANTSPATIALHASHMAP_H
