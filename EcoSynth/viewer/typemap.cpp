/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp (konrad.p.kapp@gmail.com)
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

//
// TypeMap
//

#include "data_importer/data_importer.h"

#include "palette.h"
#include "typemap.h"
#include "vecpnt.h"
#include "outimage.h"
#include "grass.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <QFileInfo>
#include <QLabel>
#include <QImage>
#include <QRgb>

/*
Perceptually uniform colourmaps from:
http://peterkovesi.com/projects/colourmaps/
*/

using namespace std;

// Sonoma County Colours
float hardwood[] = {0.749f, 0.815f, 0.611f, 1.0f};
float conifer[] = {0.812f, 0.789f, 0.55f, 1.0f};
float mixed[] = {0.552f, 0.662f, 0.533f, 1.0f};
float riparian[] = {0.4f, 0.6f, 0.6f, 1.0f};
float nonnative[] = {0.7, 0.6, 0.4, 1.0f};
float sliver[] = {0.652f, 0.762f, 0.633f, 1.0f};
float shrubery[] = {0.882f, 0.843f, 0.713f, 1.0f};
float ripshrub[] = {0.509f, 0.67f, 0.584f, 1.0f};
float herb[] = {0.75f, 0.7f, 0.7f, 1.0f};
float herbwet[] = {0.623f, 0.741f, 0.825f, 1.0f};
float aquatic[] = {0.537f, 0.623f, 0.752f, 1.0f};
float salt[] = {0.727f, 0.763f, 0.534f, 1.0f};
float barrenland[] = {0.818f, 0.801f, 0.723f, 1.0f};
float agriculture[] = {0.894f, 0.913f, 0.639f, 1.0f};
float wet[] = {0.737f, 0.823f, 0.952f, 1.0f};
float developed[] = {0.5f, 0.4f, 0.5f, 1.0f};

// Patterson's USGS Natural Map Colours

/*
// video colours
float barren[] = {0.818f, 0.801f, 0.723f, 1.0f};            // 1
float ravine[] = {0.755f, 0.645f, 0.538f, 1.0f};            // 2
float canyon[] = {0.771f, 0.431f, 0.351f, 1.0f};              // 3
float grassland[] = {0.75f, 0.7f, 0.7f, 1.0f};              // 4
float pasture[] = {0.894f, 0.913f, 0.639f, 1.0f};           // 5
float foldhills[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float orchard[] = {0.749f, 0.815f, 0.611f, 1.0f};           // 7
float evergreenforest[] = {0.812f, 0.789f, 0.55f, 1.0f};    // 8
float otherforest[] = {0.552f, 0.662f, 0.533f, 1.0f};       // 9
float woodywetland[] = {0.509f, 0.67f, 0.584f, 1.0f};       // 10
float herbwetland[] = {0.623f, 0.741f, 0.825f, 1.0f};       // 11
float frillbank[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12
float shrub[] = {0.882f, 0.843f, 0.713f, 1.0f};             // 13
float flatinterest[] = {0.300f, 0.515f, 0.0f, 1.0f};       // 14
float water[] = {0.737f, 0.823f, 0.952f, 1.0f};            // 15
float special[] = {0.4f, 0.4f, 0.4f, 1.0f};                 // 16
float extra[] = {0.5f, 0.4f, 0.5f, 1.0f};                   // 17
float realwater[] = {0.537f, 0.623f, 0.752f, 1.0f};         // 18
float boulders[] = {0.671f, 0.331f, 0.221f, 1.0f};          // 19
*/

/*
// quilt fig colours
float a1[] = {0.627f, 0.627f, 0.666f, 1.0f};         // 18 - 160 160 170
float a2[] = {0.771f, 0.431f, 0.351f, 1.0f};         // 3 - 196 110 90
float a3[] = {0.458, 0.518f, 0.439f, 1.0f};          // 7 mod - 117 132 112
float a4[] = {0.486, 0.415f, 0.317f, 1.0f};          // 17 mod - 124 106 81
float a5[] = {0.755f, 0.645f, 0.538f, 1.0f};         // 2 - 193 164 137
float a6[] = {0.812f, 0.789f, 0.55f, 1.0f};          // 14 - 207 201 140
float a7[] = {0.552f, 0.662f, 0.533f, 1.0f};         // 9 - 141 169 136
float a8[] = {0.305f, 0.541f, 0.384f, 1.0f};         // 8 mod - 78 138 98
float a9[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12 - 102 153 153
float q1[] = {0.818f, 0.801f, 0.723f, 1.0f};         // 1
float q2[] = {0.75f, 0.7f, 0.7f, 1.0f};              // 4
float q3[] = {0.894f, 0.913f, 0.639f, 1.0f};         // 5
float q4[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float q5[] = {0.509f, 0.67f, 0.584f, 1.0f};          // 10
float q6[] = {0.623f, 0.741f, 0.825f, 1.0f};         // 11
float q7[] = {0.882f, 0.843f, 0.713f, 1.0f};         // 13
float q8[] = {0.737f, 0.823f, 0.952f, 1.0f};         // 15
float q9[] = {0.4f, 0.4f, 0.4f, 1.0f};               // 16
float q10[] = {0.671f, 0.331f, 0.221f, 1.0f};        // 19
*/

/*
// replication fig colours
float a1[] = {0.627f, 0.627f, 0.666f, 1.0f};         // 18 - 160 160 170
float a2[] = {0.771f, 0.431f, 0.351f, 1.0f};         // 3 - 196 110 90
float a3[] = {0.458, 0.518f, 0.439f, 1.0f};          // 7 mod - 117 132 112
float a4[] = {0.486, 0.415f, 0.317f, 1.0f};          // 17 mod - 124 106 81
float a5[] = {0.755f, 0.645f, 0.538f, 1.0f};         // 2 - 193 164 137
float a6[] = {0.812f, 0.789f, 0.55f, 1.0f};          // 14 - 207 201 140
float a7[] = {0.588f, 0.521f, 0.443f, 1.0f};         // 9 - (GC walls) 150 133 113
float a8[] = {0.305f, 0.541f, 0.384f, 1.0f};         // 8 mod - 78 138 98
float a9[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12 - 102 153 153
float q1[] = {0.823f, 0.639f, 0.56f, 1.0f};         // 1 - (GC tops) 175 136 119
float q2[] = {0.75f, 0.7f, 0.7f, 1.0f};              // 4
float q3[] = {0.894f, 0.913f, 0.639f, 1.0f};         // 5
float q4[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float q5[] = {0.509f, 0.67f, 0.584f, 1.0f};          // 10
float q6[] = {0.623f, 0.741f, 0.825f, 1.0f};         // 11
float q7[] = {0.882f, 0.843f, 0.713f, 1.0f};         // 13
float q8[] = {0.737f, 0.823f, 0.952f, 1.0f};         // 15
float q9[] = {0.4f, 0.4f, 0.4f, 1.0f};               // 16
float q10[] = {0.671f, 0.331f, 0.221f, 1.0f};        // 19
*/

/*
// coherence fig colours
float a1[] = {0.627f, 0.627f, 0.666f, 1.0f};         // 18 - 160 160 170
float a2[] = {0.771f, 0.431f, 0.351f, 1.0f};         // 3 - 196 110 90
float a3[] = {0.458, 0.518f, 0.439f, 1.0f};          // 7 mod - 117 132 112
float a4[] = {0.486, 0.415f, 0.317f, 1.0f};          // 17 mod - 124 106 81
float a5[] = {0.755f, 0.645f, 0.538f, 1.0f};         // 2 - 193 164 137
float a6[] = {0.812f, 0.789f, 0.55f, 1.0f};          // 14 - 207 201 140
float a7[] = {0.588f, 0.521f, 0.443f, 1.0f};         // 9 - (GC walls) 150 133 113
float a8[] = {0.305f, 0.541f, 0.384f, 1.0f};         // 8 mod - 78 138 98
float a9[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12 - 102 153 153
float q1[] = {0.823f, 0.639f, 0.56f, 1.0f};         // 1 - (GC tops) 175 136 119
float q2[] = {0.75f, 0.7f, 0.7f, 1.0f};              // 4
float q3[] = {0.772f, 0.702f, 0.345f, 1.0f};         // (vegas gold) 197 179 88
float q4[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float q5[] = {0.509f, 0.67f, 0.584f, 1.0f};          // 10
float q6[] = {0.623f, 0.741f, 0.825f, 1.0f};         // 11
float q7[] = {0.882f, 0.843f, 0.713f, 1.0f};         // 13
float q8[] = {0.737f, 0.823f, 0.952f, 1.0f};         // 15
float q9[] = {0.552f, 0.662f, 0.533f, 1.0f};         // (formerly greyland)
float q10[] = {0.429f, 0.498f, 0.602f, 1.0f};        // (real water)
*/
/*
// banner fig colours
float a1[] = {0.812f, 0.789f, 0.55f, 1.0f};         // 18 - 160 176 178
float a2[] = {0.771f, 0.431f, 0.351f, 1.0f};         // 3 - 196 110 90
float a3[] = {0.458, 0.518f, 0.439f, 1.0f};          // 7 mod - 117 132 112
float a4[] = {0.486, 0.415f, 0.317f, 1.0f};          // 17 mod - 124 106 81
float a5[] = {0.755f, 0.645f, 0.538f, 1.0f};         // 2 - 193 164 137
float a6[] = {0.627f, 0.69f, 0.7f, 1.0f};        // 14 - 207 201 140
float a7[] = {0.588f, 0.521f, 0.443f, 1.0f};         // 9 - (GC walls) 150 133 113
float a8[] = {0.305f, 0.541f, 0.384f, 1.0f};         // 8 mod - 78 138 98
float a9[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12 - 102 153 153
float q1[] = {0.823f, 0.639f, 0.56f, 1.0f};         // 1 - (GC tops) 175 136 119
float q2[] = {0.757f, 0.701f, 0.423f, 1.0f};           // 193 179 108
float q3[] = {0.772f, 0.702f, 0.345f, 1.0f};         // (vegas gold) 197 179 88
float q4[] = {0.618f, 0.649f, 0.454f, 1.0f};         // (darken foldhills)
float q5[] = {0.737f, 0.823f, 0.952f, 1.0f};        // 10
float q6[] = {0.623f, 0.741f, 0.825f, 1.0f};         // 11
float q7[] = {0.882f, 0.843f, 0.713f, 1.0f};         // 13
float q8[] = {0.509f, 0.67f, 0.584f, 1.0f};        // 15
float q9[] = {0.552f, 0.662f, 0.533f, 1.0f};         // (formerly greyland)
float q10[] = {0.429f, 0.498f, 0.602f, 1.0f};        // (real water)
*/

/*
// default colours
float barren[] = {0.818f, 0.801f, 0.723f, 1.0f};            // 1
float ravine[] = {0.755f, 0.645f, 0.538f, 1.0f};            // 2
float canyon[] = {0.771f, 0.431f, 0.351f, 1.0f};            // 3
float grassland[] = {0.75f, 0.7f, 0.7f, 1.0f};              // 4
float pasture[] = {0.894f, 0.913f, 0.639f, 1.0f};           // 5
float foldhills[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float orchard[] = {0.749f, 0.815f, 0.611f, 1.0f};           // 7
float evergreenforest[] = {0.300f, 0.515f, 0.0f, 1.0f};     // 8
float otherforest[] = {0.552f, 0.662f, 0.533f, 1.0f};       // 9
float woodywetland[] = {0.509f, 0.67f, 0.584f, 1.0f};       // 10
float herbwetland[] = {0.623f, 0.741f, 0.825f, 1.0f};       // 11
float frillbank[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12
float shrub[] = {0.882f, 0.843f, 0.713f, 1.0f};             // 13
float flatinterest[] = {0.812f, 0.789f, 0.55f, 1.0f};      // 14
float water[] = {0.737f, 0.823f, 0.952f, 1.0f};            // 15
float special[] = {0.4f, 0.4f, 0.4f, 1.0f};                 // 16
float extra[] = {0.5f, 0.4f, 0.5f, 1.0f};                   // 17
float realwater[] = {0.537f, 0.623f, 0.752f, 1.0f};         // 18
float boulders[] = {0.671f, 0.331f, 0.221f, 1.0f};          // 19
*/

// palette colours

float freecol[] = {0.755f, 0.645f, 0.538f, 1.0f};
float sparseshrub[] = {0.814f, 0.853f, 0.969f, 1.0f};
float sparsemed[] = {0.727f, 0.763f, 0.834f, 1.0f};
float sparsetall[] = {0.537f, 0.623f, 0.752f, 1.0f};
float denseshrub[] = {0.749f, 0.815f, 0.611f, 1.0f};
float densemed[] = {0.552f, 0.662f, 0.533f, 1.0f};
float densetall[] = {0.300f, 0.515f, 0.1f, 1.0f};

float basic_red[] = {1.0f, 0.0f, 0.0f, 1.0f};
float basic_green[] = {0.0f, 1.0f, 0.0f, 1.0f};
float basic_blue[] = {0.0f, 0.0f, 1.0f, 1.0f};

float denseveg[] = {freecol[0] * 0.25f + 0.0f, freecol[1] * 0.25f + 0.5f * 0.75f, freecol[2] * 0.25f + 0.0f, 1.0f};
float sparseveg[] = {freecol[0] * 0.65f + 0.0f, freecol[1] * 0.65f + 0.5f * 0.35f, freecol[2] * 0.65f + 0.0f, 1.0f};


/*
float rock[] = {0.4f, 0.4f, 0.4f, 1.0f};
float sand[] = {0.89f, 0.74f, 0.513f}; // {0.772f, 0.702f, 0.345f, 1.0f};
float grass[] = {0.552f, 0.662f, 0.533f, 1.0f};
float watery[] = {0.537f, 0.623f, 0.752f, 1.0f};
float blank[] = {1.0f, 1.0f, 1.0f, 1.0f};
// lilac
float younger[] = {0.797f, 0.703f, 0.785f}; // {0.818f, 0.801f, 0.723f, 1.0f};
float older[] = {0.647f, 0.553f, 0.635f}; // {0.572f, 0.560f, 0.506f, 1.0f};
// green shades
float sparser[] = {0.729f, 0.89f, 0.824f}; // {0.717f, 0.860f, 0.692f, 1.0f};
float denser[] = {0.583f, 0.712f, 0.659f}; // {0.552f, 0.662f, 0.533f, 1.0f};
// orange
float similar[] = {1.0f, 0.89f, 0.792f}; // {0.872f, 0.430f, 0.287f, 1.0f};
float variable[] = {0.8f, 0.712f, 0.634f}; // {0.671f, 0.331f, 0.221f, 1.0f};
float heal[] = {0.749f, 0.5f, 0.5f, 1.0f};
float distribution[] =  {0.849f, 0.915f, 0.711f, 1.0f}; //  {0.749f, 0.815f, 0.611f, 1.0f};
*/
// default colours
float barren[] = {0.818f, 0.801f, 0.723f, 1.0f};            // 1
float ravine[] = {0.755f, 0.645f, 0.538f, 1.0f};            // 2
float canyon[] = {0.771f, 0.431f, 0.351f, 1.0f};            // 3
float grassland[] = {0.552f, 0.662f, 0.533f, 1.0f};         // 4
float pasture[] = {0.894f, 0.913f, 0.639f, 1.0f};           // 5
float foldhills[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float orchard[] = {0.749f, 0.815f, 0.611f, 1.0f};           // 7
float evergreenforest[] = {0.300f, 0.515f, 0.0f, 1.0f};     // 8
float otherforest[] = {0.552f, 0.662f, 0.533f, 1.0f};       // 9
float woodywetland[] = {0.509f, 0.67f, 0.584f, 1.0f};       // 10
float herbwetland[] = {0.623f, 0.741f, 0.825f, 1.0f};       // 11
float frillbank[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12
float shrub[] = {0.882f, 0.843f, 0.713f, 1.0f};             // 13
float flatinterest[] = {0.812f, 0.789f, 0.55f, 1.0f};      // 14
float water[] = {0.737f, 0.823f, 0.952f, 1.0f};            // 15
float special[] = {0.4f, 0.4f, 0.4f, 1.0f};                 // 16
float extra[] = {0.5f, 0.4f, 0.5f, 1.0f};                   // 17
float realwater[] = {0.537f, 0.623f, 0.752f, 1.0f};         // 18
float boulders[] = {0.671f, 0.331f, 0.221f, 1.0f};          // 19

// cluster colours
float c0[] = {0.537f, 0.623f, 0.752f, 1.0f};      // water
float c1[] = {0.509f, 0.67f, 0.584f, 1.0f};       // high moisture
//float c0[] =  {0.755f, 0.645f, 0.538f, 1.0f};
//float c1[] = {0.89f, 0.74f, 0.513f};
float c2[] = {0.671f, 0.331f, 0.221f, 1.0f};      // rock for high slope
//float c3[] = {0.8f, 0.712f, 0.634f};
float c3[] = {0.647f, 0.553f, 0.635f};
//float c3[] = {0.894f, 0.913f, 0.639f, 1.0f};      // high sun
float c4[] = {0.552f, 0.662f, 0.533f, 1.0f};       // low sun
float c5[] = {0.771f, 0.431f, 0.351f, 1.0f};
//float c6[] = {0.749f, 0.815f, 0.611f, 1.0f};
// float c6[] = {0.4f, 0.6f, 0.6f, 1.0f};
float c6[] = {0.749f, 0.5f, 0.5f, 1.0f};
//float c7[] = {0.623f, 0.741f, 0.825f, 1.0f};
float c7[] = {0.812f, 0.789f, 0.55f, 1.0f};
float c8[] = {0.727f, 0.763f, 0.534f, 1.0f};
//float c9[] = {0.300f, 0.515f, 0.0f, 1.0f};
float c9[] = {0.8f, 0.712f, 0.634f};
float c10[] = {0.882f, 0.843f, 0.713f, 1.0f};
float c11[] = {0.755f, 0.645f, 0.538f, 1.0f};
float c12[] = {0.8f, 0.712f, 0.634f};
float c13[] = {0.647f, 0.553f, 0.635f};
float c14[] = {0.89f, 0.74f, 0.513f};

TypeMap::TypeMap(TypeMapType purpose)
{
    tmap = new MemMap<int>;
    setPurpose(purpose);
}

TypeMap::TypeMap(int w, int h, TypeMapType purpose)
{
    tmap = new MemMap<int>;
    matchDim(w, h);
    setPurpose(purpose);
}

TypeMap::~TypeMap()
{
    delete tmap;
    for(int i = 0; i < (int) colmap.size(); i++)
        delete [] colmap[i];
    colmap.clear();

    if (cdata_ptr)
        delete cdata_ptr;
}

void TypeMap::clear()
{
    tmap->fill(0);
}

void TypeMap::replace_value(int orig, int newval)
{
    int w = width();
    int h = height();
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int to = get(x, y);
            if (to == orig)
                set(x, y, newval);
        }
    }
}

void TypeMap::initPaletteColTable()
{
    GLfloat *col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    numSamples = 3;		//really? should be 3...?

    colmap[0] = freecol;
    colmap[1] = sparseveg;
    colmap[2] = denseveg;
}

void TypeMap::initSpeciesColTable(std::string dbname)
{
    data_importer::common_data cdata(dbname);

    GLfloat *col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    //numSamples = 3;
    numSamples = cdata.all_species.size();

    /*
    colmap[0] = sparsemed;
    colmap[1] = densemed;
    colmap[2] = sparseshrub;
    colmap[0] = basic_red;
    colmap[1] = basic_green;
    colmap[2] = basic_blue;
    */

    std::map<int, data_importer::species>::iterator speciter = cdata.all_species.begin();

    for (int i = 0; i < cdata.all_species.size(); i++, advance(speciter, 1))
    {
        assert(speciter->first < 32);
        for (int j = 0; j < 3; j++)
            colmap[speciter->first][j] =  speciter->second.basecol[j];
    }
}

void TypeMap::initCategoryColTable()
{
    GLfloat * col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // entry 0 is reserved as transparent
    numSamples = 16;
    colmap[1] = hardwood;
    colmap[2] = conifer;
    colmap[3] = mixed;
    colmap[4] = riparian;
    colmap[5] = nonnative;
    colmap[6] = sliver;
    colmap[7] = shrubery;
    colmap[8] = ripshrub;
    colmap[9] = herb;
    colmap[10] = herbwet;
    colmap[11] = aquatic;
    colmap[12] = salt;
    colmap[13] = barrenland;
    colmap[14] = agriculture;
    colmap[15] = wet;
    colmap[16] = developed;
}

void TypeMap::initNaturalColTable()
{
    GLfloat * col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // saturated prime colours and combos
    /*
     (colmap[1])[0] = 1.0f; // red
     (colmap[2])[1] = 1.0f; // green
     (colmap[3])[2] = 1.0f; // blue
     (colmap[4])[1] = 1.0f; (colmap[4])[2] = 1.0f; // cyan
     (colmap[5])[0] = 1.0f; (colmap[5])[1] = 1.0f; // yellow
     (colmap[6])[0] = 1.0f; (colmap[6])[2] = 1.0f; // magenta
     (colmap[7])[0] = 0.5f;  (colmap[7])[1] = 0.5f; (colmap[7])[2] = 0.5f; // grey
     (colmap[8])[1] = 0.5f; (colmap[8])[2] = 0.5f; // teal
     */

    numSamples = 20;

    // default
    //colmap[0] = c0;
    colmap[1] = barren;
    colmap[2] = ravine;
    colmap[3] = canyon;
    colmap[4] = grassland;
    colmap[5] = pasture;
    colmap[6] = foldhills;
    colmap[7] = orchard;
    colmap[8] = woodywetland;
    colmap[9] = otherforest;
    colmap[10] = woodywetland;
    colmap[11] = herbwetland;
    colmap[12] = frillbank;
    colmap[13] = shrub;
    colmap[14] = flatinterest;
    colmap[15] = water;
    colmap[16] = special;
    colmap[17] = extra;
    colmap[18] = realwater;
    colmap[19] = boulders;
}

void TypeMap::initPerceptualColTable(std::string colmapfile, int samples, float truncend)
{
    GLfloat *col;
    float r[256], g[256], b[256];
    ifstream infile;
    string valstr, line;
    int i, pos, step;

    if(samples < 3 || samples > 32)
        cerr << "Error: sampling of colour map must be in the range [3,32]" << endl;

    for(i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // input is a csv file, with 256 RGB entries, one on each line
    // note that this is not robust to format errors in the input file
    infile.open((char *) colmapfile.c_str(), ios_base::in);

    if(infile.is_open())
    {
        i = 0;
        while(std::getline(infile, line))
        {
            std::size_t prev = 0, pos;

            // red component
            pos = line.find_first_of(",", prev);
            valstr = line.substr(prev, pos-prev);
            istringstream isr(valstr);
            isr >> r[i];
            prev = pos+1;

            // green component
            pos = line.find_first_of(",", prev);
            valstr = line.substr(prev, pos-prev);
            istringstream isg(valstr);
            isg >> g[i];
            prev = pos+1;

            // blue component
            valstr = line.substr(prev, std::string::npos);
            istringstream isb(valstr);
            isb >> b[i];

            i++;
        }
        infile.close();
    }
    else
    {
        std::cout << "WARNING: Could not find perceptual colour table at " << colmapfile << std::endl;
    }

    // now sample the colour map at even intervals according to the number of samples
    // first and last samples map to the beginning and end of the scale
    step = (int) ((256.0f * truncend) / (float) (samples-1));
    pos = 0;
    for(i = 1; i <= samples; i++)
    {
        colmap[i][0] = (GLfloat) r[pos]; colmap[i][1] = (GLfloat) g[pos]; colmap[i][2] = (GLfloat) b[pos];
        pos += step;
    }
    numSamples = samples+1;
}

void TypeMap::clipRegion(Region &reg)
{
    if(reg.x0 < 0) reg.x0 = 0;
    if(reg.y0 < 0) reg.y0 = 0;
    if(reg.x1 > width()) reg.x1 = width();
    if(reg.y1 > height()) reg.y1 = height();
}

void TypeMap::matchDim(int w, int h)
{
    int mx, my;

    mx = tmap->width();
    my = tmap->height();

    // if dimensions don't match then reallocate
    if(w != mx || h != my)
    {
        dirtyreg = Region(0, 0, w, h);
        tmap->allocate(Region(0, 0, w, h));
        tmap->fill(0); // set to empty type
    }
}

void TypeMap::replaceMap(MemMap<int> * newmap)
{
    assert(tmap->width() == newmap->width());
    assert(tmap->height() == newmap->height());
    for (int y = 0; y < tmap->height(); y++)
        for (int x = 0; x < tmap->width(); x++)
            (* tmap)[y][x] = (* newmap)[y][x];
}

void TypeMap::bandCHMMap(MapFloat * chm, float mint, float maxt)
{
    int tp;
    float val;

    if(maxt > mint)
    {
        for (int x = 0; x < tmap->width(); x++)
            for (int y = 0; y < tmap->height(); y++)
            {
                val = chm->get(y, x);

                // discretise into ranges of height values
                if(val <= 0.0f) // transparent
                {
                    tp = 1;
                }
                else if(val <= mint) // black
                {
                    tp = 2;
                }
                else if(val >= maxt) // red
                {
                    tp = numSamples+2;
                }
                else // green range
                {
                    tp = (int) ((val-mint) / (maxt-mint+pluszero) * (numSamples-1))+2;
                }
                (* tmap)[y][x] = tp;
            }
    }
}

void TypeMap::bandCHMMapEric(MapFloat * chm, float mint, float maxt)
{
    int tp;
    float val;

    if(maxt > mint)
    {
        for (int x = 0; x < tmap->width(); x++)
            for (int y = 0; y < tmap->height(); y++)
            {
                val = chm->get(y, x);

                // discretise into ranges of height values
                if(val <= mint || val >= maxt) // transparent
                {
                    tp = 1;
                }
                else 
                {
                    tp = 2;
                }
                (* tmap)[y][x] = tp;
            }
    }
}

int TypeMap::load(const QImage &img, TypeMapType purpose)
{
    MemMap<mask_tag> mask;
    int tp, maxtp = 0; // mintp = 100;
    int width, height;
    ifstream infile;
    float val, maxval = 0.0f, range;

    width = img.width();
    height = img.height();

    matchDim(width, height);
    // convert to internal type map format

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            QColor col = img.pixelColor(x, y);
            int ival = col.red();

            switch(purpose)
            {
                case TypeMapType::EMPTY: // do nothing
                    break;
                case TypeMapType::CATEGORY:
                    tp = ival;
                    tp++;
                    break;
                case TypeMapType::SLOPE:
                    val = ival;
                    if(val > maxval)
                        maxval = val;

                     // discretise into ranges of slope values
                    range = 90.0f; // maximum slope is 90 degrees
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                    break;
                case TypeMapType::WATER:
                    val = ival;
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of water values
                    range = 100.0f;
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                    break;
                case TypeMapType::SUNLIGHT:
                    val = ival;
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    range = 12.0f; // hours of sunlight
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                    break;
                case TypeMapType::TEMPERATURE:
                    val = ival;
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of temperature values

                    range = 20.0f; //10
                    // clamp values to range, temperature is bidrectional
                    if(val < -range) val = -range;
                    if(val > range) val = range;
                    tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-1))+1;

                    break;
                case TypeMapType::CHM:
                    val = ival;
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of height values
                    range = 75.0f; // maximum tree height in feet
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                    break;
                case TypeMapType::CDM:
                   val = ival;
                   if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    range = 1.0f; // maximum density
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                    break;
                case TypeMapType::SUITABILITY:
                    break;
                case TypeMapType::PAINT:
                {
                    int sparsemin = 100, sparsemax = 150;
                    int densemin = 200;
                    if (img.depth() == 16)
                    {
                        sparsemin *= 256;
                        sparsemax *= 256;
                        densemin *= 256;
                    }
                    if (ival >= sparsemin && ival <= sparsemax)
                        tp = 1;
                    else if (ival >= densemin)
                        tp = 2;
                    else
                        tp = 0;
                    break;
                }
                default:
                    break;
            }
            (* tmap)[y][x] = tp;

            if(tp > maxtp)
                maxtp = tp;
            /*
            if(tp < mintp)
                mintp = tp;
            */

        }
    }
    infile.close();
    // cerr << "maxtp = " << maxtp << endl;
    // cerr << "mintp = " << mintp << endl;
    return maxtp;
}

int TypeMap::load(const uts::string &filename, TypeMapType purpose)
{
    MemMap<mask_tag> mask;
    int tp, maxtp = 0; // mintp = 100;
    int width, height;
    ifstream infile;
    float val, maxval = 0.0f, range;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> width >> height;
        // cerr << "width = " << width << " height = " << height << endl;
        matchDim(width, height);
        // convert to internal type map format

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                switch(purpose)
                {
                    case TypeMapType::EMPTY: // do nothing
                        break;
                    case TypeMapType::PAINT:
                        infile >> tp;
                        break;
                    case TypeMapType::CATEGORY:
                        infile >> tp;
                        tp++;
                        break;
                    case TypeMapType::SLOPE:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                         // discretise into ranges of slope values
                        range = 90.0f; // maximum slope is 90 degrees
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::WATER:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of water values
                        range = 100.0f;
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::SUNLIGHT:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of illumination values
                        range = 12.0f; // hours of sunlight
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::TEMPERATURE:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of temperature values

                        range = 20.0f; //10
                        // clamp values to range, temperature is bidrectional
                        if(val < -range) val = -range;
                        if(val > range) val = range;
                        tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-1))+1;

                        break;
                    case TypeMapType::CHM:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of height values
                        range = 75.0f; // maximum tree height in feet
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::CDM:
                       infile >> val;
                       if(val > maxval)
                            maxval = val;

                        // discretise into ranges of illumination values
                        range = 1.0f; // maximum density
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::SUITABILITY:
                        break;
                    default:
                        break;
                }
                (* tmap)[y][x] = tp;

                if(tp > maxtp)
                    maxtp = tp;
                /*
                if(tp < mintp)
                    mintp = tp;
                */

            }
        }
        infile.close();
        // cerr << "maxtp = " << maxtp << endl;
        // cerr << "mintp = " << mintp << endl;
    }
    else
    {
        cerr << "Error TypeMap::loadTxt: unable to open file" << filename << endl;
    }
    return maxtp;
}

bool TypeMap::loadCategoryImage(const uts::string &filename)
{
    int width, height;
    QImage img(QString::fromStdString(filename)); // load image from file

    QFileInfo check_file(QString::fromStdString(filename));

    if(!(check_file.exists() && check_file.isFile()))
        return false;

    // set internal storage dimensions
    width = img.width();
    height = img.height();
    matchDim(width, height);

    // convert to internal type map format
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            QColor col = img.pixelColor(x, y);
            int r, g, b;
            col.getRgb(&r, &g, &b); // all channels store the same info so just use red
            (* tmap)[y][x] = r - 100; // convert greyscale colour to category index
        }
    return true;
}

void TypeMap::setWater(MapFloat * wet, float wetthresh)
{
    int gx, gy;

    wet->getDim(gx, gy);
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            if(wet->get(x, y) >= wetthresh)
            {
                (* tmap)[y][x] = 0;
            }
        }
}

int TypeMap::convert(MapFloat * map, TypeMapType purpose, float range)
{
    int species_count = 0;
    int tp, maxtp = 0;
    int width, height;
    float val, maxval = 0.0f;

    map->getDim(width, height);
    matchDim(width, height);
    // convert to internal type map format
    int mincm, maxcm;
    mincm = 100; maxcm = -1;

    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            tp = 0;
            switch(purpose)
            {
                case TypeMapType::EMPTY: // do nothing
                    break;
                case TypeMapType::PAINT: // do nothing
                    break;
                case TypeMapType::CATEGORY: // do nothing, since categories are integers not floats
                    break;
                case TypeMapType::SLOPE:
                    val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::WATER:
                    val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of water values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::SUNLIGHT:
                     val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::TEMPERATURE:
                    val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of temperature values
                    // clamp values to range, temperature is bidrectional
                    if(val < -range) val = -range;
                    if(val > range) val = range;
                    tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-2)) + 1;

                    break;
                case TypeMapType::CHM:
                     val = map->get(y, x);
                     if(val > maxval)
                        maxval = val;

                    // discretise into ranges of tree height values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range)
                    {
                        val = range;
                        //std::cout << "clamping value to range (upper): " << range << std::endl;
                    }
                    //tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    tp = (int) (val / (range+pluszero) * 256) + 1;		// I am assuming we are not categorising this? Multiplying by 400 here will bring the heights back to their proper values in feet
                    if(tp < mincm)
                        mincm = tp;
                    if(tp > maxcm)
                        maxcm = tp;
                    break;         
                case TypeMapType::CDM:
                    val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of tree density values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    if(tp < mincm)
                        mincm = tp;
                    if(tp > maxcm)
                        maxcm = tp;
                    break;
                case TypeMapType::SUITABILITY:
                    val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;
                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;

                case TypeMapType::GRASS:
                    tp = (int)(map->get(x, y) / MAXGRASSHGHT * 255.0f);
                    break;
                case TypeMapType::PRETTY_PAINTED:
                    tp = (int)(map->get(x, y));
                    break;
                case TypeMapType::PRETTY:
                    tp = (int)(map->get(x, y));
                    break;
                case TypeMapType::SPECIES:
                    tp = (int)(map->get(x, y)) + 1;
                    if (tp >= 0)
                        species_count++;
                    break;
                case TypeMapType::CLUSTER:
                    tp = (int)(map->get(y, x));
                    break;
                case TypeMapType::CLUSTERDENSITY:
                    tp = (int)((map->get(y, x))) / 20.0f;
                    break;
                default:
                    break;
            }
            (* tmap)[y][x] = tp;

            if(tp > maxtp)
                maxtp = tp;
        }
    if(purpose == TypeMapType::CDM)
    {
        cerr << "Minimum colour value = " << mincm << endl;
        cerr << "Maxiumum colour value = " << maxcm << endl;
    }
    if (purpose == TypeMapType::SPECIES)
    {
        cerr << "Proportion of landscape assigned to species: " << species_count / ((float)width * height) << std::endl;
    }
    return maxtp;
}

void TypeMap::save(const uts::string &filename)
{
    ofstream outfile;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << width() << " " << height() << endl;

        // dimensions
        for (int x = 0; x < width(); x++)
            for (int y = 0; y < height(); y++)
            {
                outfile << get(x, y) << " ";
            }
        outfile.close();
    }
    else
    {
        cerr << "Error TypeMap::save: unable to write to file" << endl;
    }
}

void TypeMap::saveToBinaryImage(const uts::string &filename, int maskval)
{
    std::vector<float> mask;
    int i = 0;

    mask.resize(tmap->width()*tmap->height(), 0.0f);
    for (int x = 0; x < tmap->width(); x++)
        for (int y = 0; y < tmap->height(); y++)
        {       
            if((* tmap)[tmap->height()-y-1][x] == maskval)
                mask[i] = 1.0f;
            i++;
        }

    OutImage outimg;
    outimg.write(filename, tmap->width(), tmap->height(), mask);
}

/*
void TypeMap::saveToGreyscaleImage(const uts::string &filename, float maxrange)
{
    std::vector<float> mask;
    int i = 0;

    mask.resize(tmap->width()*tmap->height(), 0.0f);
    for (int x = 0; x < tmap->width(); x++)
        for (int y = 0; y < tmap->height(); y++)
        {
            mask[i] = (float) (* tmap)[tmap->height()-y-1][x] / maxrange;
            i++;
        }

    std::cout << "Writing CHM output image as: " << filename << std::endl;
    OutImage outimg;
    outimg.write(filename, tmap->width(), tmap->height(), mask);
}
*/
void TypeMap::saveToGreyscaleImage(const uts::string &filename, float maxrange, bool row_major)
{
    std::vector<unsigned char> mask;
    std::cout << "Size of greyscale image: " << tmap->width() << ", " << tmap->height() << std::endl;

    mask.resize(tmap->width()*tmap->height(), 0);
    for (int x = 0; x < tmap->width(); x++)
        for (int y = 0; y < tmap->height(); y++)
        {
            int i = row_major ? y * tmap->width() + x : x * tmap->height() + y;
            //std::cout << (* tmap)[tmap->height()-y-1][x] << " ";
            //mask[i] = (uint8_t)(((float) (* tmap)[tmap->height()-y-1][x] / maxrange) * 255.0f);
            mask[i] = (uint8_t)(((float) (* tmap)[y][x] / maxrange) * 255.0f);
        }

    std::cout << std::endl;
    std::cout << "Writing CHM output image as: " << filename << std::endl;
    QImage img;
    img = QImage(mask.data(), tmap->width(), tmap->height(), QImage::Format_Grayscale8);
    if (!img.save(QString::fromStdString(filename), "PNG", 100))
    {
        std::cout << "Failed to write CHM image" << std::endl;
    }

    //OutImage outimg;
    //outimg.write(filename, tmap->width(), tmap->height(), mask);
}

void TypeMap::saveToPaintImage(const uts::string &filename)
{
    unsigned char * mask = new unsigned char[tmap->width()*tmap->height()];
    int i = 0;

    cerr << "paint file: " << filename << endl;

    //mask.resize(tmap->width()*tmap->height(), 0.0f);
    for (int x = 0; x < tmap->width(); x++)
        for (int y = 0; y < tmap->height(); y++)
        {
            switch((*tmap)[x][y]) // check order
            {
            case 0:
                mask[i] = 0;
                break;
            case 1: // sparse (this was previously 1)
                mask[i] = 127;
                break;
            case 2: // dense (this was previously 2)
                mask[i] = 255;
                break;
            default:
                mask[i] = 0;
            }
            i++;
        }

    // use QT image save functions
    QImage img;
    img = QImage(mask, tmap->width(), tmap->height(), QImage::Format_Grayscale8);
    img.save(QString::fromStdString(filename), "PNG", 100);
    delete [] mask;
}

void TypeMap::setPurpose(TypeMapType purpose)
{
    std::string coldir = COLMAP_DIR;

    usage = purpose;
    switch(usage)
    {
        case TypeMapType::EMPTY:
            initPaletteColTable();
            break;
        case TypeMapType::PAINT:
            initPaletteColTable();
            break;
        case TypeMapType::CATEGORY:
            initCategoryColTable();
            break;
        case TypeMapType::SLOPE:
            initPerceptualColTable(coldir + "/linear_kry_5-95_c72_n256.csv", 10);
            break;
        case TypeMapType::WATER:
            initPerceptualColTable(coldir + "/linear_blue_95-50_c20_n256.csv", 10);
            break;
        case TypeMapType::SUNLIGHT:
            initPerceptualColTable(coldir + "/linear_kry_5-95_c72_n256.csv", 10);
            break;
        case TypeMapType::TEMPERATURE:
            initPerceptualColTable(coldir + "/diverging_bwr_55-98_c37_n256.csv", 10);
            break;
        case TypeMapType::CHM:
            // initPerceptualColTable(coldir + "/linear_ternary-green_0-46_c42_n256.csv", 20);
            initPerceptualColTable(coldir + "/linear_green_5-95_c69_n256.csv", 20);
            // replace 0 with natural terrain colour
            colmap[1][0] = 0.7f; colmap[1][1] = 0.6f; colmap[1][2] = 0.5f; // transparent
            colmap[2][0] = 0.0f; colmap[2][1] = 0.0f; colmap[2][2] = 1.0f; // black
            colmap[numSamples+2][0] = 1.0f; colmap[numSamples+2][1] = 0.0f; colmap[numSamples+2][2] = 0.0f; // red
            break;
        case TypeMapType::CDM:
            initPerceptualColTable(coldir + "/linear_green_5-95_c69_n256.csv", 20);
            // replace 0 with natural terrain colour
            colmap[1][0] = 0.7f; colmap[1][1] = 0.6f; colmap[1][2] = 0.5f;
            break;
        case TypeMapType::SUITABILITY:
            initPerceptualColTable(coldir + "/linear_gow_60-85_c27_n256.csv", 20, 0.8f);
            // initPerceptualColTable(coldir + "/isoluminant_cgo_70_c39_n256.csv", 10);
            break;
        case TypeMapType::SPECIES:
            initSpeciesColTable(std::string(PRJ_SRC_DIR) + "/ecodata/sonoma.db");
            break;
        default:
            break;
    }
}

void TypeMap::resetType(int ind)
{
    // wipe all previous occurrences of ind
    #pragma omp parallel for
    for(int j = 0; j < tmap->height(); j++)
        for(int i = 0; i < tmap->width(); i++)
            if((* tmap)[j][i] == ind)
                (* tmap)[j][i] = 0;
    dirtyreg.x0 = 0; dirtyreg.y0 = 0;
    dirtyreg.x1 = tmap->width(); dirtyreg.y1 = tmap->height();
}

void TypeMap::setColour(int ind, GLfloat * col)
{
    for(int i = 0; i < 4; i++)
        colmap[ind][i] = col[i];
}
