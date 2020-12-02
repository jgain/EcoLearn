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


#ifndef GENERIC_RNG_H
#define GENERIC_RNG_H

#include <random>
#include "common.h"

class generic_rng
{
public:
    generic_rng(common_types::decimal min, common_types::decimal max, int seed = std::default_random_engine::default_seed);
    generic_rng();

    common_types::decimal get_min();
    common_types::decimal get_max();
    int get_seed();
    void set_seed(int seed_value);
    common_types::decimal operator() ();


    common_types::decimal min, max;
    int seed;
    std::default_random_engine generator;
    std::uniform_real_distribution<common_types::decimal> distrib;
};

#endif
