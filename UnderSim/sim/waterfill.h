/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
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


#ifndef WATERFILL_H
#define WATERFILL_H

#include <vector>
#include <QImage>
#include "terrain.h"
// class Terrain;

class WaterFill
{
public:
    WaterFill()
    {
        precipitation = 0.1f; // m per year
        river_width_constant = 0.00178f; //  y^(1/2)m^(-1/2)
    }

    void setTerrain(Terrain*);
    void setAbsorbsion(uint a){absorbsion = a;}

    void compute(uint step = (uint)(-1));

    void expandRivers(float max_moisture_factor, float slope_effect);

    std::vector<uint> getRiverMoisture() {return river_side;}

    bool isFlowingWater(uint x, uint y);
    float riverMoisture(uint x, uint y);

    void reset();
    void addWaterInflow(uint x, uint y, int delta);
    void smartWaterInflow(uint x, uint y);


private:
    std::vector<uint> flow, river_side, inflow;
    std::vector<double> lakes;
    Terrain* terrain;

    uint absorbsion;
    float precipitation;
    float river_width_constant;
};

#endif // WATERFILL_H
