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


#ifndef ABIOTICMAPPER_H
#define ABIOTICMAPPER_H

#include "common/basic_types.h"

namespace data_importer
{
    struct data_dir;
}

/*
 * This struct packages abiotic maps into one convenient structure, with consistency checks, aggregation, etc.
 * Each abiotic map can be accessed directly by name
 */

struct abiotic_maps_package
{
    enum class suntype
    {
        CANOPY,
        LANDSCAPE_ONLY
    };

    enum class aggr_type
    {
        JANUARY,
        FEBRUARY,
        MARCH,
        APRIL,
        MAY,
        JUNE,
        JULY,
        AUGUST,
        SEPTEMBER,
        OCTOBER,
        NOVEMBER,
        DECEMBER,
        AVERAGE
    };

    abiotic_maps_package(const ValueGridMap<float> &wet,
                 const ValueGridMap<float> &sun,
                 const ValueGridMap<float> &temp,
                 const ValueGridMap<float> &slope);

    // TODO: let the caller optionally give a monthly, or average filename for moisture and sunlight
    abiotic_maps_package(std::string wet_fname, std::string sun_fname, std::string temp_fname, std::string slope_fname, aggr_type aggr);

    abiotic_maps_package(data_importer::data_dir targetdir, suntype sun, aggr_type aggr);

    void validate_maps();

    ValueGridMap<float> wet, sun, temp, slope;
    int gw, gh;
    float rw, rh;
};

#endif 		// ABIOTICMAPPER_H
