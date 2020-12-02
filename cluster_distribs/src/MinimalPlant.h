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


#ifndef MINIMALPLANT_H
#define MINIMALPLANT_H

#include <iostream>

struct MinimalPlant
{
    int x; //< x-position in cm
    int y; //< y-position in cm
    int h;	// height in cm
    int r; //< radius in cm
    bool s; //< shaded status for individual plant

        void print() { std::cerr << "x: " << x << ", y: " << y << ", r: " << r << ", s: " << s << std::endl; }

        bool operator==(const MinimalPlant &other) { return x == other.x && y == other.y && r == other.r && s == other.s; }
        bool operator!=(const MinimalPlant &other) { return !(*this == other); }
};

#endif
