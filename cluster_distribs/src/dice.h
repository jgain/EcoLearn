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


#ifndef DICE_H
#define DICE_H

#include <chrono>
#include <random>

class Dice
{
private:
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution;
	
	long seed;

public:
    Dice(int from, int to, long init_seed=0);
    Dice(){}
    ~Dice();

    void init(int from, int to, long init_seed=0);
    int gen();
	void setSeed(long new_seed=0);
	long getSeed();
};

#endif //DICE_ROLLER_H

