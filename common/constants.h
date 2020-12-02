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


#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace common_constants
{
    static const float undersim_sample_mult = 12.0f;
    static const float sampleprob = 0.00004f;
    static const float NUMSIZEBINS = 50;
    static const float MINHEIGHT = 0.05f;
    static const float MAX_UNDERGROWTH_HEIGHT = 5.0f;
    static const int SAMPLE_RESOLUTION = 200;
}

#endif 	// CONSTANTS_H
