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


#include "dice.h"

Dice::Dice(int from, int to, long initSeed) :
    generator(std::default_random_engine(initSeed ? initSeed : std::chrono::system_clock::now().time_since_epoch().count())),
    distribution(std::uniform_int_distribution<int>(from,to))
{
	seed = initSeed;
}

Dice::~Dice()
{
//    delete generator;
//    delete distribution;
}

void Dice::init(int from, int to, long initSeed)
{
	setSeed(initSeed);
    // generator = std::default_random_engine(1); // TO DO! Change Back
    distribution = std::uniform_int_distribution<int>(from,to);
}

void Dice::setSeed(long seed)
{
	this->seed = seed;
	generator = std::default_random_engine(seed ? seed : std::chrono::system_clock::now().time_since_epoch().count());
}

long Dice::getSeed()
{
	return seed;
}

int Dice::gen()
{
    return distribution.operator()(generator);
}

