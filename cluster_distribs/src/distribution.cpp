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


// distribution.cpp: reimplementation of HL analysis and synthesis of distributions
// author: James Gain
// date: 16 March 2017

#include "distribution.h"
#include "common.h"
//#include "interp.h"   XXX: uncomment if can't compile

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <list>
#include <algorithm>
#include <math.h>
#include <rapidjson/reader.h>
#include <map>
#include <iomanip>

/*XXX: remove this include*/
#include <exception>

using namespace std;

// XXX: remove this later. Just for debuggin purposes
static bool inModifyHistogram = false;
static bool inFullAnalyse = false;


// Mediterrainean Biome PFT colours
/*
GLcommon_types::decimal M12[] = {0.173f, 0.290f, 0.055f, 1.0f}; // Black Pine (ID 12)
GLcommon_types::decimal M13[] = {0.498f, 0.258f, 0.094f, 1.0f}; // European Beech (ID 13)   {0.498f, 0.208f, 0.094f, 1.0f};
GLcommon_types::decimal M14[] = {0.573f, 0.600f, 0.467f, 1.0f}; // Silver Birch (ID 14)
GLcommon_types::decimal M15[] = {0.376f, 0.443f, 0.302f, 1.0f}; // Holly Oak (ID 15)
GLcommon_types::decimal M16[] = {0.164f, 0.164f, 0.09f, 1.0f}; // Kermes Oak Shrub (ID 16)
GLcommon_types::decimal M17[] = {0.678f, 0.624f, 0.133f, 1.0f}; // Juniper Shrub (ID 17)
GLcommon_types::decimal M18[] = {0.561f, 0.267f, 0.376f, 1.0f}; // Trea of Heaven invader species (ID 18)
*/




//// HistogramDistrib ////


// TODO: make sure refplnt_prev and refplnt_new are not in vector of MinimalPlants for PlantTypeDistrib object
/*
*/


//// PlantTypeDistrib ////


//// SandboxDistrib ////
