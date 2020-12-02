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


#ifndef MOISTURE_H
#define MOISTURE_H

#include "common/basic_types.h"

#include <eigen3/Eigen/Core>

#include <vector>

//#define SAVANNAHBIOME
// #define MEDBIOME
// #define CANYONBIOME
#define CANYONWETBIOME
//#define ALPINEBIOME

/// Wrapper for Waterfill river simulator and for more sophisticated calculation of soil moisture retention
class MoistureSim
{
public:
    MoistureSim(const ValueMap<float> &ter, float tw_mod, float th_mod)
        : ter(ter)
    {
        ter.getDim(gw, gh);
        tw = gw * tw_mod;
        th = gw * th_mod;
        calcSlope();
    }

private:

    /**
     * @brief slopeImpact Adjust runoff according to piecewise linear function of slope
     * @param slope         current slope value on terrain
     * @param slopethresh   slope below which all water up to runofflimit is absorbed
     * @param slopemax      slope above which all water runs off due to steepness
     * @param runofflimit   maximum amount of runoff
     * @return revised runofflimit
     */
    float slopeImpact(float slope, float slopethresh, float slopemax, float runofflimit);

    ValueMap<float> ter;
    ValueMap<float> terSlope;
    int tw, th, gw, gh;

    void getNormal(int x, int y, Eigen::Vector3f &vec);
    void calcSlope();
public:
    /**
     * @brief simSoilCycle Simulate one year of rainfall on the terrain, including river and floodplain formation
     * @param ter               underlying terrain undergoing simulation
     * @param terslope          slope of landscape undergoing simulation
     * @param precipitation     monthly rainfall in mm
     * @param slopethresh       slope at which runoff starts increasing linearly
     * @param slopemax          slope at which water stops being absorbed altogether
     * @param evaporation       proportion of rainfall that is evaporated
     * @param runofflevel       cap on mm per month that can be asorbed by the soil before runoff
     * @param soilsaturation    cap in mm on amount of water that soil can hold
     * @param waterlevel        surface water level above which a location is marked as a river
     * @param wsh               maps to hold soil moisture values
     */
    void simSoilCycle(std::array<float, 12> precipitation, float slopethresh, float slopemax, float evaporation, float runofflimit, float soilsaturation, float riverlevel, std::vector<ValueGridMap<float> > &wsh);

    /// Check that with a constant rainfall below threshold, the mositure values are also constant
    //bool constantValidation(MapFloat * terslope, std::vector<MapFloat> &wsh, int precipitation);
};

#endif // MOISTURE_H
