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


#ifndef COMMON_H
#define COMMON_H

namespace common_types
{
        using decimal = double;
}

//#include "MinimalPlant.h"
#include <common/basic_types.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>


namespace common_types
{
        struct basic_plant
        {
            common_types::decimal x, y, z, height, radius;
        };

        struct rectangle
        {
            int xmin, xmax, ymin, ymax;
        };
}

static common_types::decimal pluszero_var = 1e-5;

class generic_rng;

namespace common_vars
{
    extern generic_rng unif_rng;
    extern unsigned float_round;
    extern unsigned double_round;
}

class HistogramDistrib;

namespace common_funcs
{
        using namespace common_types;
        bool read_pdb(std::string filename, std::map<int, std::vector<MinimalPlant> > &retvec);
        bool write_pdb(std::string filename, std::vector<std::vector<MinimalPlant> > &plants);
        void categorise_to_heights(std::vector<MinimalPlant> &plants, const std::vector<common_types::decimal> &height_bins, std::map<int, std::vector<MinimalPlant> > &plant_map, std::map<int, std::pair<int, int> > &class_height_map);
        std::vector<int> get_surr_idxes(int idx, int w, int h, int extent_x, int extent_y, bool include_self);
        std::vector<int> get_surr_idxes(int x, int y, int w, int h, int extent_x, int extent_y, bool include_self);
        float get_radius_from_height(float h, common_types::decimal a = -1.8185, common_types::decimal b = 1.1471);
        float get_height_from_radius(float radius, common_types::decimal a, common_types::decimal b);
        common_types::decimal get_radius_from_height_convert(common_types::decimal height, common_types::decimal a = -1.8185, common_types::decimal b = 1.1471);
        bool is_this_higher_priority(int thistype, int othertype);
        common_types::decimal calc_ring_area(MinimalPlant &refplnt, MinimalPlant &cmp, HistogramDistrib &hst, const rectangle &valid_area);
        common_types::decimal calc_ring_area(MinimalPlant &refplnt, int bin, HistogramDistrib &hst, const rectangle &valid_area);
        common_types::decimal calc_disk_area(const MinimalPlant & refplnt, common_types::decimal r, const rectangle &area);
        common_types::decimal calc_quadrant_area(common_types::decimal dv, common_types::decimal dh, common_types::decimal r);
        common_types::decimal dist_code(MinimalPlant & refplnt, MinimalPlant & cmpplnt, bool &xsect);

        bool files_are_equal(std::string filename1, std::string filename2, bool show_unequal, bool verbose);

        template<int Num, unsigned Exp>
        struct compile_time_pow_type
        {
            const static long val = (long)Num * compile_time_pow_type<Num, Exp - 1>::val;
        };

        template<int Num>
        struct compile_time_pow_type<Num, 1>
        {
            const static long val = 1;
        };

        template<int Num>
        struct compile_time_pow_type<Num, 0>
        {
            const static long val = 1;
        };

        template<int Num, unsigned Exp>
        int compile_time_pow()
        {
            return compile_time_pow_type<Num, Exp>::val;
        }

        template<typename T>
        void round_for_type(T &var)
        {
            var *= 1e0;
        }

        template<>
        inline void round_for_type(float &var)
        {
            const long mult = compile_time_pow<10, 6>();
            var = std::trunc(var * mult) / mult;
        }

        template<>
        inline void round_for_type(double &var)
        {
            const long mult = compile_time_pow<10, 8>();
            var = std::trunc(var * mult) / mult;
        }

        template<typename T>
        inline void trim(T min, T max, T &target)
        {
            if (target < min)
                target = min;
            if (target > max)
                target = max;
        }

        template<typename T>
        inline T trimret(T min, T max, T target)
        {
            if (target < min)
                target = min;
            if (target > max)
                target = max;
            return target;
        }
}

#endif
