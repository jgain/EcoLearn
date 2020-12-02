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


#include "dice_roller.h"

DiceRoller::DiceRoller(int from, int to) :
    generator(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count())),
    distribution(std::uniform_int_distribution<int>(from,to))
{
}

DiceRoller::~DiceRoller()
{
//    delete generator;
//    delete distribution;
}

int DiceRoller::generate()
{
    return distribution.operator()(generator);
}
