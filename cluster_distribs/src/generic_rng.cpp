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


#include "generic_rng.h"
#include "common.h"

generic_rng::generic_rng()
    : generic_rng(0.0, 1.0f)
{}

generic_rng::generic_rng(common_types::decimal min, common_types::decimal max, int seed)
    : min(min), max(max), seed(seed), generator(seed), distrib(min, max)
{
}

common_types::decimal generic_rng::get_min()
{
	return min;
}

common_types::decimal generic_rng::get_max()
{
	return max;
}

int generic_rng::get_seed()
{
    return seed;
}

void generic_rng::set_seed(int seed_value)
{
    seed = seed_value;
    generator.seed(seed);
}

common_types::decimal generic_rng::operator () ()
{
    return distrib(generator);
}
