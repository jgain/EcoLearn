/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020 J.E. Gain (jgain@cs.uct.ac.za) and  K.P. Kapp  (konrad.p.kapp@gmail.com)
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

// eco.cpp: core classes for controlling ecosystems and plant placement
// author: James Gain and K.P. Kapp
// date: 27 February 2016

#include "eco.h"
#include "common/basic_types.h"
// #include "interp.h"
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <QDir>

// Mediterrainean Biome PFT colours
GLfloat MedMNEcol[] = {0.173f, 0.290f, 0.055f, 1.0f}; // Black Pine (ID 12)
GLfloat MedTBScol[] = {0.498f, 0.258f, 0.094f, 1.0f}; // European Beech (ID 13)   {0.498f, 0.208f, 0.094f, 1.0f};
GLfloat MedIBScol[] = {0.573f, 0.600f, 0.467f, 1.0f}; // Silver Birch (ID 14)
GLfloat MedTBEcol[] = {0.376f, 0.443f, 0.302f, 1.0f}; // Holly Oak (ID 15)
GLfloat MedMSEBcol[] = {0.164f, 0.164f, 0.09f, 1.0f}; // Kermes Oak Shrub (ID 16)
GLfloat MedMSENcol[] = {0.678f, 0.624f, 0.133f, 1.0f}; // Juniper Shrub (ID 17)
GLfloat MedITBScol[] = {0.561f, 0.267f, 0.376f, 1.0f}; // Trea of Heaven invader species (ID 18)

// Mediterrainean Biome PFT colours
GLfloat AlpTNEcol[] = {0.173f, 0.290f, 0.055f, 1.0f}; // Black Pine (ID 12)
GLfloat AlpTBScol[] = {0.498f, 0.258f, 0.094f, 1.0f}; // European Beech (ID 13)   {0.498f, 0.208f, 0.094f, 1.0f};
GLfloat AlpTBEcol[] = {0.376f, 0.443f, 0.302f, 1.0f}; // Holly Oak (ID 15)
GLfloat AlpTScol[] = {0.573f, 0.600f, 0.467f, 1.0f}; // Medlar
GLfloat AlpBNEcol[] = {0.514f, 0.682f, 0.588f, 1.0f}; // Scotch Pine
GLfloat AlpBNScol[] = {0.698f, 0.718f, 0.447f, 1.0f}; // Larch
GLfloat AlpBBScol[] = {0.573f, 0.600f, 0.467f, 1.0f}; // Silver Birch
GLfloat AlpBScol[] = {0.608f, 0.933f, 0.780f, 1.0f}; // Dogwood Shrub
// GLfloat AlpBScol[] = {0.778f, 0.624f, 0.133f, 1.0f}; // Dogwood Shrub 155 238 199

// Savannah Biome PFT colours
GLfloat SavPBEcol[] = {0.608f, 0.684f, 0.133f, 1.0f}; // Tamarind
GLfloat SavPBRcol[] = {0.473f, 0.500f, 0.427f, 1.0f}; // Acacia
GLfloat SavPBEScol[] = {0.164f, 0.164f, 0.09f, 1.0f}; // African Boxwood
GLfloat SavPBRScol[] = {0.476f, 0.543f, 0.402f, 1.0f}; // Arrow Poisin
GLfloat SavAEcol[] = {0.598f, 0.358f, 0.194f, 1.0f}; // Tree Aloe


/// PlantGrid

void PlantGrid::initSpeciesTable()
{
    speciesTable.clear();
#ifdef MEDBIOME
    std::vector<SubSpecies> submap;
    SubSpecies sub;

    // MNE
    sub.name = "AtlasCedar"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "LebanonCedar"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "SeaPine"; sub.chance = 300;
    submap.push_back(sub);
    sub.name = "StonePine"; sub.chance = 300;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // TBS
    sub.name = "EuropeanBeech"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "Chestnut"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "CutleafBeech"; sub.chance = 200;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // IBS
    sub.name = "GreyAlder"; sub.chance = 1000;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // TBE
    sub.name = "HolmOak"; sub.chance = 300;
    submap.push_back(sub);
    sub.name = "CorkOak"; sub.chance = 500;
    submap.push_back(sub);
    sub.name = "EnglishYew"; sub.chance = 200;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // MSEB
    sub.name = "Lentisk"; sub.chance = 500;
    submap.push_back(sub);
    sub.name = "Myrtle"; sub.chance = 500;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // MSEN
    sub.name = "Juniper"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "Broom"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "Tamarisk"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // **
    // ITBS
/*
    sub.name = "NONE"; sub.chance = 1000;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();
*/

    numSubSpecies = 16;

/*
    speciesTable.push_back("StonePine_S");
    speciesTable.push_back("StonePine_M");
    speciesTable.push_back("StonePine_L");
    speciesTable.push_back("EuropeanBeech_S");
    speciesTable.push_back("EuropeanBeech_M");
    speciesTable.push_back("EuropeanBeech_L");
    speciesTable.push_back("GreyAlder_S");
    speciesTable.push_back("GreyAlder_M");
    speciesTable.push_back("GreyAlder_L");
    speciesTable.push_back("CorkOak_S");
    speciesTable.push_back("CorkOak_M");
    speciesTable.push_back("CorkOak_L");
    speciesTable.push_back("Lentisk_S");
    speciesTable.push_back("Lentisk_M");
    speciesTable.push_back("Lentisk_L");
    speciesTable.push_back("SpanishBroom_S");
    speciesTable.push_back("SpanishBroom_M");
    speciesTable.push_back("SpanishBroom_L");
*/
#endif

#ifdef ALPINEBIOME
    std::vector<SubSpecies> submap;
    SubSpecies sub;

    // TNE
    sub.name = "AustrianPine"; sub.chance = 800;
    submap.push_back(sub);
    sub.name = "StonePine"; sub.chance = 200;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // TBS
    sub.name = "EuropeanBeech"; sub.chance = 600;
    submap.push_back(sub);
    sub.name = "Chestnut"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "CutleafBeech"; sub.chance = 200;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // NONE - needs to be skipped
    sub.name = "None"; sub.chance = 1000;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // TBE
    sub.name = "HolmOak"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "CorkOak"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "EnglishYew"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // TS
    sub.name = "SavinJuniper"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "Medlar"; sub.chance = 600;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // BNE
    sub.name = "ScotchPine"; sub.chance = 600;
    submap.push_back(sub);
    sub.name = "NorwaySpruce"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // BNS
    sub.name = "Larch"; sub.chance = 1000;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // BBS
    sub.name = "Ash"; sub.chance = 600;
    submap.push_back(sub);
    sub.name = "Birch"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // BS
    sub.name = "Dogwood"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "Hazel"; sub.chance = 300;
    submap.push_back(sub);
    sub.name = "Hawthorn"; sub.chance = 300;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    numSubSpecies = 18;
#endif

#ifdef SAVANNAHBIOME
    std::vector<SubSpecies> submap;
    SubSpecies sub;

    // PBE
    sub.name = "AfricanMahogany"; sub.chance = 300;
    submap.push_back(sub);
    sub.name = "WildPeach"; sub.chance = 300;
    submap.push_back(sub);
    sub.name = "Tamarind"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // PBR
    sub.name = "Acacia"; sub.chance = 600;
    submap.push_back(sub);
    sub.name = "Baobob"; sub.chance = 100;
    submap.push_back(sub);
    sub.name = "SandpiperFig"; sub.chance = 300;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // PBES
    sub.name = "Carissa"; sub.chance = 400;
    submap.push_back(sub);
    sub.name = "WildPear"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "AfricanBoxwood"; sub.chance = 400;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // PBRS
    sub.name = "ArrowPoisin"; sub.chance = 1000;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    // AE
    sub.name = "TreeAloe"; sub.chance = 500;
    submap.push_back(sub);
    sub.name = "DragonTree"; sub.chance = 200;
    submap.push_back(sub);
    sub.name = "BottleTree"; sub.chance = 300;
    submap.push_back(sub);
    speciesTable.push_back(submap);
    submap.clear();

    numSubSpecies = 13;
#endif
}

void PlantGrid::delGrid()
{
    int i, j;

    // clear out elements of the vector hierarchy
    for(i = 0; i < (int) pgrid.size(); i++)
        for(j = 0; j < (int) pgrid[i].pop.size(); j++)
            pgrid[i].pop[j].clear();
    for(i = 0; i < (int) pgrid.size(); i++)
    {
        pgrid[i].pop.clear();
        // sgrid[i].clear();
    }
    pgrid.clear();
    // sgrid.clear();
}

void PlantGrid::initGrid()
{
    int i, j, s;

    delGrid();

    // setup empty elements of the vector hierarchy according to the grid dimensions
    for(i = 0; i < gx; i++)
        for(j = 0; j < gy; j++)
        {
            PlantPopulation ppop;
            // std::vector<AnalysisPoint> apnt;
            for(s = 0; s < maxSpecies; s++)
            {
                std::vector<Plant> plnts;
                ppop.pop.push_back(plnts);
            }
            pgrid.push_back(ppop);
            // sgrid.push_back(apnt);
        }
}

bool PlantGrid::isEmpty()
{
    bool empty = true;
    int i, j, s, pos = 0;

    // setup empty elements of the vector hierarchy according to the grid dimensions
    for(i = 0; i < gx; i++)
        for(j = 0; j < gy; j++)
        {
            for(s = 0; s < maxSpecies; s++)
            {
                if(!pgrid[pos].pop[s].empty())
                    empty = false;
            }
            pos++;
        }
    return empty;
}

void PlantGrid::cellLocate(Terrain * ter, int mx, int my, int &cx, int &cy)
{
    int tx, ty;

    ter->getGridDim(tx, ty);

    // find grid bin for plant
    cx = (int) (((float) mx / (float) tx) * (float) gx);
    cy = (int) (((float) my / (float) ty) * (float) gy);
    if(cx >= gx)
        cx = gx-1;
    if(cy >= gy)
        cy = gy-1;
}


void PlantGrid::clearCell(int x, int y)
{
    int f = flatten(x, y);

    for(int s = 0; s < (int) pgrid[f].pop.size(); s++)
    {
        pgrid[f].pop[s].clear();
    }
}

void PlantGrid::placePlant(Terrain * ter, int species, Plant plant)
{
    int x, y, cx, cy;

    // find plant location on map
    ter->toGrid(plant.pos, x, y);
    cellLocate(ter, x, y, cx, cy);
    // cerr << "loc in " << cx << ", " << cy << " species " << species << endl;

    /*
    std::cout << "Adding xy tree " << plant.pos.z << ", " << plant.pos.x << " ";
    std::cout << "with species " << species << ", at flat grid loc " << flatten(cx, cy) << " ";
    std::cout << " at index " << pgrid[flatten(cx, cy)].pop[species].size() << std::endl;
    */

    // add plant to relevant population
    pgrid[flatten(cx, cy)].pop[species].push_back(plant);
}

void PlantGrid::placePlantExactly(Terrain * ter, int species, Plant plant, int x, int y)
{
     pgrid[flatten(x, y)].pop[species].push_back(plant);
}

void PlantGrid::clearRegion(Terrain * ter, Region region)
{
    int x, y, sx, sy, ex, ey;

    getRegionIndices(ter, region, sx, sy, ex, ey);

    for(x = sx; x <= ex; x++)
        for(y = sy; y <= ey; y++)
            clearCell(x, y); // clear a specific cell in the grid
}

void PlantGrid::clearAllPlants(Terrain *ter)
{
    int gw, gh;
    ter->getGridDim(gw, gh);
    Region wholeRegion(0, 0, gw - 1, gh - 1);
    clearRegion(ter, wholeRegion);
}

void PlantGrid::pickPlants(Terrain * ter, TypeMap * clusters, int niche, PlantGrid & outgrid)
{
    int x, y, s, p, mx, my, sx, sy, ex, ey, f;
    Plant plnt;
    Region region = clusters->getRegion();

    // map region to cells in the grid
    getRegionIndices(ter, region, sx, sy, ex, ey);
    // cerr << "region = " << region.x0 << ", " << region.y0 << " -> " << region.x1 << ", " << region.y1 << endl;
    // cerr << "s = " << sx << ", " << sy << " -> " << ex << ", " << ey << endl;

    for(x = sx; x <= ex; x++)
        for(y = sy; y <= ey; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < (int) pgrid[f].pop.size(); s++)
                for(p = 0; p < (int) pgrid[f].pop[s].size(); p++)
                {
                    plnt = pgrid[f].pop[s][p];

                    ter->toGrid(plnt.pos, mx, my); // map plant terrain location to cluster map
                    if((* clusters->getMap())[my][mx] == niche) // niche value on terrain matches the current plant distribution
                        outgrid.placePlantExactly(ter, s, plnt, x, y);
                }
        }
}

void PlantGrid::pickAllPlants(Terrain * ter, float offx, float offy, float scf, PlantGrid & outgrid)
{
    int x, y, s, p, f;
    Plant plnt;
    int cnt = 0;

    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < (int) pgrid[f].pop.size(); s++)
                for(p = 0; p < (int) pgrid[f].pop[s].size(); p++)
                {
                    plnt = pgrid[f].pop[s][p];
                    plnt.pos.x *= scf; plnt.pos.z *= scf;
                    plnt.height *= scf; plnt.canopy *= scf;
                    plnt.pos.x += offy; plnt.pos.z += offx; // allows more natural layout
                    outgrid.placePlant(ter, s, plnt);
                    cnt++;
                }
        }
    cerr << "Picked " << cnt << " plants" << endl;
}

void PlantGrid::vectoriseByPFT(int pft, std::vector<Plant> &pftPlnts)
{
    int x, y, p, s, f;
    Plant plnt;

    s = (int) pft;
    pftPlnts.clear();
    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            if(s < 0  || s > (int) pgrid[f].pop.size())
                cerr << "PlantGrid::vectoriseBySpecies: mismatch between requested species and available species" << endl;

            for(p = 0; p < (int) pgrid[f].pop[s].size(); p++)
            {
                plnt = pgrid[f].pop[s][p];
                pftPlnts.push_back(plnt);
            }
        }
}

/*
void PlantGrid::burnGrass(GrassSim * grass, Terrain * ter, float scale)
{
    int x, y, s, p, f;
    Plant plnt;
    float invscf, tx, ty;

    ter->getTerrainDim(tx, ty);
    invscf = scale / tx;

    int nburned = 0;

    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < (int) pgrid[f].pop.size(); s++)
                for(p = 0; p < (int) pgrid[f].pop[s].size(); p++)
                {
                    // get plant
                    plnt = pgrid[f].pop[s][p];

                    // apply radial burn of grass heights
                    // XXX: here, x and z were swapped, i.e. the call was previously grass->burnInPlant(plnt.pos.x*invscf, plnt.pos.z*invscf, 0.5f*plnt.canopy*invscf);
                    // this is because in the grass sim, xy locations of trees were swapped
                    grass->burnInPlant(plnt.pos.z*invscf, plnt.pos.x*invscf, 0.5f*plnt.canopy*invscf);

                    nburned++;
                }
        }

    std::cout << "Number of plants burned: " << nburned << std::endl;
}
*/

void PlantGrid::reportNumPlants()
{
    int i, j, s, plntcnt, speccnt;

    cerr << "grid dimensions = " << gx << " X " << gy << endl;
    for(i = 0; i < gx; i++)
        for(j = 0; j < gy; j++)
        {
            plntcnt = 0;
            for(s = 0; s < maxSpecies; s++)
            {
                speccnt = (int) pgrid[flatten(i,j)].pop[s].size();
                plntcnt += speccnt;
            }
            cerr << "count " << i << ", " << j << " = " << plntcnt << endl;
        }
}

void PlantGrid::setPopulation(int x, int y, PlantPopulation & pop)
{
    pgrid[flatten(x, y)] = pop;
}

PlantPopulation * PlantGrid::getPopulation(int x, int y)
{
    return & pgrid[flatten(x, y)];
}

/*
std::vector<AnalysisPoint> * PlantGrid::getSynthPoints(int x, int y)
{
    return &sgrid[flatten(x, y)];
}*/

void PlantGrid::getRegionIndices(Terrain * ter, Region region, int &sx, int &sy, int &ex, int &ey)
{
    cellLocate(ter, region.x0, region.y0, sx, sy);
    cellLocate(ter, region.x1, region.y1, ex, ey);

    // cerr << "map = " << region.x0 << ", " << region.y0 << " -> " << region.x1 << ", " << region.y1 << endl;
    // cerr << "grid = " << sx << ", " << sy << " -> " << ex << ", " << ey << endl;
}



bool PlantGrid::readPDB(string filename, Terrain * ter, float &maxtree)
{
    ifstream infile;
    int numSpecies;
    float rndoff;
    int currSpecies;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> numSpecies;

        for(int i = 0; i < numSpecies; i++)
        {
            int specName, numPlants;
            float canopyRatio, hghtMin, hghtMax;

            infile >> specName;
            // This should actually be a string but HL provides a number that depends on the plant database
            // hardcoded as first species in mediterranean list at 12
            currSpecies = specName - specOffset;

            infile >> hghtMin >> hghtMax >> canopyRatio;
            // cerr << "currSpecies = " << currSpecies << " hmin = " << hghtMin << " hmax = " << hghtMax << " cRatio = " << canopyRatio << endl;

            if(hghtMax > maxtree)
                maxtree = hghtMax;

           infile >> numPlants;
           for(int j = 0; j < numPlants; j++)
            {
                Plant p;
                float x, y, z, h, r;

                // terrain position and plant height
                infile >> x >> y >> z >> h >> r;

                // convert units to meters and drape onto terrain
                p.height = h;

                // supplied canopy ratio is actually radius to height (not diameter to height)
                p.canopy = r * 2.0f;

                rndoff = (float)(rand() % 100) / 100.0f * 0.6f;
                p.col = glm::vec4(-0.3f+rndoff, -0.3f+rndoff, -0.3f+rndoff, 1.0f); // randomly vary lightness of plant
                if(ter->drapePnt(vpPoint(x, z, y), p.pos)) // project plant onto the terrain
                {
                    placePlant(ter, currSpecies, p);
                    // cerr << "P" << j << ": " << p.pos.x << " " << p.pos.y << " " << p.pos.z << endl;
                    // cerr << "h = " << p.height << " c = " << p.canopy << endl;
                }
            }
        }
        infile.close();
    }
    else
    {
        cerr << "Error Mesh::readPDB: unable to open " << filename << endl;
        return false;
    }
    return true;
}

bool PlantGrid::writePDB(string filename)
{
    ofstream outfile;
    int writtenPlants = 0, numSeedlings = 0;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << numSubSpecies*3 << endl;

        for(int s = 0; s < maxSpecies; s++)
        {
            int numPlants;
            float canopyRatio, hghtMin, hghtMax;

            std::vector<Plant> lpop;
            lpop.clear();
            hghtMin = 1000000.0f;
            hghtMax = 0.0f;
            canopyRatio = 0.0f;

            // collect all species in this category
            for(int i = 0; i < gx; i++)
                for(int j = 0; j < gy; j++)
                {
                    for(int p = 0; p < (int) pgrid[flatten(i,j)].pop[s].size(); p++)
                    {
                        Plant plnt = pgrid[flatten(i,j)].pop[s][p];

                        if(plnt.height < hghtMin)
                            hghtMin = plnt.height;
                        if(plnt.height > hghtMax)
                            hghtMax = plnt.height;
                        canopyRatio += (plnt.canopy / plnt.height) * 0.5f ;
                        // average canopy to height ratio?
                        lpop.push_back(plnt);
                    }
                }
            // convert height range in meters to cm
            numPlants = (int) lpop.size();
            canopyRatio /= (float) numPlants;

            // now assign to subspecies
            int numSub = (int) speciesTable[s/3].size();
            cerr << "numSub = " << numSub << endl;
            for(int b = 0; b < numSub; b++)
            {
                std::vector<Plant> tpop; tpop.clear();
                std::vector<Plant> spop; spop.clear();

                if(b != numSub-1)
                {
                    for(auto p: lpop)
                    {
                        if(rand()%1000 < speciesTable[s/3][b].chance)
                            tpop.push_back(p);
                        else
                            spop.push_back(p);
                    }
                }
                else
                {
                    tpop = lpop;
                }

                // count only those plants over a height of 1
                numPlants = 0;
                for(int k = 0; k < (int) tpop.size(); k++)
                {
                    Plant plnt = tpop[k];
                    if(plnt.height > 0.01f)
                        numPlants++;
                    else
                        numSeedlings++;
                }

                std::string postfix;
                if(s%3 == 0)
                    postfix = "_S";
                else if(s%3 == 1)
                    postfix = "_M";
                else
                    postfix = "_L";
                outfile << speciesTable[s/3][b].name + postfix << endl;
                if(numPlants > 0)
                    outfile << hghtMin << " " << hghtMax << " " << canopyRatio << endl;
                else
                    outfile << 0 << " " << 0 << " " << 1.0f << endl;
                outfile << numPlants << endl;

                for(int k = 0; k < (int) tpop.size(); k++)
                {
                    Plant plnt = tpop[k];

                    // terrain position and plant height
                    float x = plnt.pos.x;
                    float y = plnt.pos.z;
                    float z = plnt.pos.y;
                    float h = plnt.height;
                    float r = 1.0f;
                    if(h > 0.01f)
                        outfile << x  << " " << y << " " << z << " " << h << " " << r << endl;
                }
                writtenPlants += numPlants;

                lpop.clear();
                lpop = spop;
            }
        }
        cerr << "num seedlings = " << numSeedlings << " vs. full plants = " << writtenPlants << endl;
        outfile.close();
    }
    else
    {
        cerr << "Error Mesh::writePDB: unable to open " << filename << endl;
        return false;
    }
    cerr << "num written plants = " << writtenPlants << endl;
    return true;
}


/// PlantShape

void ShapeGrid::genSpherePlant(float trunkheight, float trunkradius, Shape &shape)
{
    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkheight;

    // trunk - cylinder
    idt = glm::mat4(1.0f);
    tfm = glm::rotate(idt, glm::radians(-90.0f), rotx);
    // extra 0.1 to trunk is to ensure that trunk joins tree properly
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1, 3, 1, tfm, false);

    // canopy - sphere
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, trunkheight+canopyheight/2.0f, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::scale(tfm, glm::vec3(1.0, canopyheight, 1.0f)); // make sure tree fills 1.0f on a side bounding box
    tfm = glm::rotate(tfm, glm::radians(-90.0f), rotx);

#ifdef HIGHRES
    shape.genSphere(0.5f, 20, 20, tfm);
#endif
#ifdef LOWRES
    shape.genSphere(0.5f, 6, 6, tfm);
#endif
}

void ShapeGrid::genBoxPlant(float trunkheight, float trunkradius, float taper, float scale, Shape &shape)
{
    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkheight;

    // trunk - cylinder
    idt = glm::mat4(1.0f);
    tfm = glm::rotate(idt, glm::radians(-90.0f), rotx);
    // extra 0.1 to trunk is to ensure that trunk joins tree properly
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1, 3, 1, tfm, false);

    // canopy - tapered box
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, trunkheight, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::rotate(tfm, glm::radians(-90.0f), rotx);
    shape.genPyramid(1.0f*scale, taper*scale, canopyheight*scale, tfm);
}

void ShapeGrid::genConePlant(float trunkheight, float trunkradius, Shape &shape)
{
    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkheight;

    // trunk - cylinder
    idt = glm::mat4(1.0f);
    tfm = glm::rotate(idt, glm::radians(-90.0f), rotx);
    // extra 0.1 to trunk is to ensure that trunk joins tree properly
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1f, 3, 1, tfm, false);

    // canopy - cone
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, trunkheight, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::rotate(tfm, glm::radians(-90.0f), rotx);
#ifdef HIGHRES
    shape.genCappedCone(0.5f, canopyheight, 20, 1, tfm, false);
#endif
#ifdef LOWRES
    shape.genCappedCone(0.5f, canopyheight, 6, 1, tfm, false);
#endif

}

void ShapeGrid::genInvConePlant(float trunkheight, float trunkradius, Shape &shape)
{
    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkheight;

    // trunk - cylinder
    idt = glm::mat4(1.0f);
    tfm = glm::rotate(idt, glm::radians(-90.0f), rotx);
    // extra 0.1 to trunk is to ensure that trunk joins tree properly
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1f, 3, 1, tfm, false);

    // canopy - cone
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, 1.0f, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::rotate(tfm, glm::radians(-270.0f), rotx);
    //tfm = glm::translate(tfm, glm::vec3(0.0f, 0.0f, -canopyheight));
#ifdef HIGHRES
    shape.genCappedCone(0.5f, canopyheight, 20, 1, tfm, false);
#endif
#ifdef LOWRES
    shape.genCappedCone(0.5f, canopyheight, 6, 1, tfm, false);
#endif

}

void ShapeGrid::genUmbrellaPlant(float trunkheight, float trunkradius, Shape &shape)
{
    glm::mat4 idt, tfm;
    glm::vec3 trs, rotx;
    float canopyheight;

    rotx = glm::vec3(1.0f, 0.0f, 0.0f);
    canopyheight = 1.0f - trunkheight;

    // trunk - cylinder
    idt = glm::mat4(1.0f);
    tfm = glm::rotate(idt, glm::radians(-90.0f), rotx);
    // extra 0.1 to trunk is to ensure that trunk joins tree properly
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1f, 3, 1, tfm, false);

    // canopy - cone
    idt = glm::mat4(1.0f);
    trs = glm::vec3(0.0f, 1.0f, 0.0f);
    tfm = glm::translate(idt, trs);
    tfm = glm::rotate(tfm, glm::radians(90.0f), rotx);
#ifdef HIGHRES
    shape.genCappedCone(0.5f, canopyheight, 20, 1, tfm, false);
#endif
#ifdef LOWRES
    shape.genCappedCone(0.5f, canopyheight, 6, 1, tfm, false);
#endif

}

void ShapeGrid::delGrid()
{
    int i, j;

    // clear out elements of the vector hierarchy
    for(i = 0; i < (int) shapes.size(); i++)
        for(j = 0; j < (int) shapes[i].size(); j++)
            shapes[i][j].clear();
    for(i = 0; i < (int) shapes.size(); i++)
        shapes[i].clear();
    shapes.clear();
}

void ShapeGrid::initGrid()
{
    int i, j, s;

    // setup empty elements of the vector hierarchy according to the grid dimensions
    for(i = 0; i < gx; i++)
        for(j = 0; j < gy; j++)
        {
            std::vector<Shape> shapevec;
            for(s = 0; s < maxSpecies / 3; s++)
            {
                Shape shape;
                shapevec.push_back(shape);
            }
            shapes.push_back(shapevec);
        }
    if (SYMBOLIC_RENDERER)
        genPlants();
    else
        genPlants("");
}

void ShapeGrid::genOpenglTextures()
{
    for (int i = 0; i < shapes[0].size(); i++)
    {
        shapes[0][i].genOpenglTextures(10);		// starting ID will not really matter here
    }
}

void ShapeGrid::genPlants(std::string model_filename)
{
    float trunkheight, trunkradius;
    int x, y, s, f;
    //model_importer model("../../Data/AcerSaccharum/AcerSaccharum_LOD4.obj",
    //                         "../../Data/AcerSaccharum/");

    std::string models_basedir = "/home/konrad/PhDStuff/models3d/AcerSaccharum/";
    std::vector<std::string> models_files = {
        "AcerSaccharum/AcerSaccharum_LOD4.obj",
        //"QuercusRubraMature/QuercusRubraMature_LOD4.obj",
        "WhiteAsh/WhiteAsh_LOD4.obj"
        //"WhiteFirMature/WhiteFirMature_LOD4.obj"
    };
    //for (auto &fname :models_files)
    //    fname = models_basedir + fname;

    for(s = 0; s < biome->numPFTypes(); s++)
    {
        //model_importer model("/home/konrad/EcoSynth/Data/AcerSaccharum/AcerSaccharum_LOD4.obj");
        model_importer model(models_basedir + models_files[s % models_files.size()]);

        model.normalize_vertices_height();

        Shape currshape;
        PFType * pft = biome->getPFType(s);
        currshape.setColour(pft->basecol);
        trunkheight = pft->draw_hght; trunkradius = pft->draw_radius;
        /*
        switch(pft->shapetype)
        {
        case TreeShapeType::SPHR:
            genSpherePlant(trunkheight, trunkradius, currshape);
            break;
        case TreeShapeType::BOX:
            genBoxPlant(trunkheight, trunkradius, pft->draw_box1, pft->draw_box2, currshape);
            break;
        case TreeShapeType::CONE:
            genConePlant(trunkheight, trunkradius, currshape);
            break;
        default:
            break;
        }
        */
        std::cerr << "Importing model to currshape for grid xy, biome: " << x << ", " << y << ", " << s << std::endl;
        currshape.import_model(&model);
        std::cerr << "Done importing model to currshape" << std::endl;
        shapes[0][s] = currshape;
        std::cerr << "Done assigning currshape to shapes[0][s]" << std::endl;
    }
}

void ShapeGrid::genPlants()
{
    float trunkheight, trunkradius;
    int x, y, s, f;

    f = flatten(x, y);
    for(s = 0; s < biome->numPFTypes(); s++)
    {
        Shape currshape;
        PFType * pft = biome->getPFType(s);
        currshape.setColour(pft->basecol);
        trunkheight = pft->draw_hght; trunkradius = pft->draw_radius;
        //genSpherePlant(trunkheight, trunkradius, currshape);		// XXX: just for debugging. Remove later and uncomment below switch statement
        switch(pft->shapetype)
        {
        case TreeShapeType::SPHR:
            genSpherePlant(trunkheight, trunkradius, currshape);
            break;
        case TreeShapeType::BOX:
            genBoxPlant(trunkheight, trunkradius, pft->draw_box1, pft->draw_box2, currshape);
            break;
        case TreeShapeType::CONE:
            genConePlant(trunkheight, trunkradius, currshape);
            break;
        case TreeShapeType::INVCONE:
            genInvConePlant(trunkheight, trunkradius, currshape);
            break;
        default:
            break;
        }
        shapes[0][s] = currshape;
    }

}

/*
void ShapeGrid::genPlants()
{
    float trunkheight, trunkradius;
    int x, y, s, f;

    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < biome->numPFTypes(); s++)
            {
                Shape currshape;
                PFType * pft = biome->getPFType(s);
                currshape.setColour(pft->basecol);
                trunkheight = pft->draw_hght; trunkradius = pft->draw_radius;
                switch(pft->shapetype)
                {
                case TreeShapeType::SPHR:
                    genSpherePlant(trunkheight, trunkradius, currshape);
                    break;
                case TreeShapeType::BOX:
                    genBoxPlant(trunkheight, trunkradius, pft->draw_box1, pft->draw_box2, currshape);
                    break;
                case TreeShapeType::CONE:
                    genConePlant(trunkheight, trunkradius, currshape);
                    break;
                default:
                    break;
                }
                shapes[f][s] = currshape;
            }

        }

#ifdef ALPINEBIOME
                switch((FunctionalPlantType) s)
                {
                    case FunctionalPlantType::ALPTNE:     //< Alpine Temperate Needle-Leaved Evergreen
                        // sphere, med green, short trunk, h:w = 0.25

                        currshape.setColour(AlpTNEcol);
                        trunkheight = 0.1f; trunkradius = 0.07f;
                        genSpherePlant(trunkheight, trunkradius, currshape);
                        break;
                    case FunctionalPlantType::ALPTBS:     //< Alpine Temperate Broad-leaved Summergreen
                        // tapered box, autum red, short trunk
                        currshape.setColour(AlpTBScol);
                        trunkheight = 0.1f; trunkradius = 0.1f;
                        genBoxPlant(trunkheight, trunkradius, 0.75f, 0.8f, currshape);
                        break;
                    case FunctionalPlantType::NONE:
                        // SKIP
                        break;
                    case FunctionalPlantType::ALPTBE:     //< Alpine Temperate Broad-leaved Evergreen
                        // sphere, medium green, medium trunk
                        currshape.setColour(AlpTBEcol);
                        trunkheight = 0.1f; trunkradius = 0.07f;
                        genSpherePlant(trunkheight, trunkradius, currshape);
                        break;
                    case FunctionalPlantType::ALPTS:    //< Alpine Temperate Shrub
                        // squat cone, dark green, medium trunk
                        currshape.setColour(AlpTScol);
                        trunkheight = 0.15f; trunkradius = 0.1f;
                        genConePlant(trunkheight, trunkradius, currshape);
                        break;
                    case FunctionalPlantType::ALPBNE:    //< Alpine Boreal Needle-leaved Evergreen
                        // elongated sphere, dark green
                        currshape.setColour(AlpBNEcol);
                        trunkheight = 0.1f; trunkradius = 0.07f;
                        genSpherePlant(trunkheight, trunkradius, currshape);
                        break;
                    case FunctionalPlantType::ALPBNS:    //< Alpine Boreal Need-leaved Summergreen
                        // box, yellowy green
                        currshape.setColour(AlpBNScol);
                        trunkheight = 0.1f; trunkradius = 0.1f;
                        genBoxPlant(trunkheight, trunkradius, 0.75f, 1.0f, currshape);
                        break;
                    case FunctionalPlantType::ALPBBS:    //< Alpine Boreal Broad-leaved Summergreen
                        // sphere, silvery green
                        currshape.setColour(AlpBBScol);
                        trunkheight = 0.1f; trunkradius = 0.1f;
                        genSpherePlant(trunkheight, trunkradius, currshape);
                        break;
                    case FunctionalPlantType::ALPBS:    //< Alpine Boreal Shrub
                        // squat cone, light greeny yellow, short trunk
                        currshape.setColour(AlpBScol);
                        trunkheight = 0.05f; trunkradius = 0.07f;
                        genConePlant(trunkheight, trunkradius, currshape);
                        break;

                    default:
                        break;
                }
#endif
}
*/

void ShapeGrid::bindPlantsSimplified(Terrain *ter, PlantGrid *esys)
{
    std::default_random_engine generator_canopy;
    std::default_random_engine generator_under;
    std::uniform_real_distribution<GLfloat> rand_unif;
    int x, y, s, p, sx, sy, ex, ey, f;
    PlantPopulation * plnts;
    int bndplants = 0, culledplants = 0;
    int gwidth, gheight;

    ter->getGridDim(gwidth, gheight);
    Region wholeRegion = Region(0, 0, gwidth - 1, gheight - 1);
    esys->getRegionIndices(ter, wholeRegion, sx, sy, ex, ey);

    std::vector<std::vector<glm::mat4> > xforms; // transformation to be applied to each instance
    std::vector<std::vector<glm::vec4> > colvars; // colour variation to be applied to each instance


    for(x = sx; x <= ex; x++)
        for(y = sy; y <= ey; y++)
        {
            plnts = esys->getPopulation(x, y);

            // cerr << "cat 0, species 0, num = " << (int) plnts->pop[0].size() << endl;
            for(s = 0; s < (int) plnts->pop.size(); s+=3) // iterate over plant types
            {
                std::vector<glm::mat4> xform; // transformation to be applied to each instance
                std::vector<glm::vec4> colvar; // colour variation to be applied to each instance

                // cerr << "plants for " << s/3 << " visible" << endl;
                for(int a = 0; a < 3; a++)
                {
                    // cerr << "num plants = " << (int) plnts->pop[s+a].size() << endl;
                    for(p = 0; p < (int) plnts->pop[s+a].size(); p++) // iterate over plant specimens
                    {
                        if(plnts->pop[s+a][p].height > 0.01f) // only display reasonably sized plants
                        {
                            // setup transformation for individual plant, including scaling and translation
                            glm::mat4 idt, tfm;
                            glm::vec3 trs, sc, rotate_axis = glm::vec3(0.0f, 1.0f, 0.0f);
                            vpPoint loc = plnts->pop[s+a][p].pos;
                            GLfloat rotate_rad;
                            if (plnts->pop[s+a][p].iscanopy)	// we use a different generator depending on whether we have a canopy or undergrowth plant - keeps rotations the same whether we render undergrowth plants or not
                            {
                                rotate_rad = rand_unif(generator_canopy) * glm::pi<GLfloat>() * 2.0f;
                            }
                            else
                            {
                                rotate_rad = rand_unif(generator_under) * glm::pi<GLfloat>() * 2.0f;
                            }

                            idt = glm::mat4(1.0f);
                            trs = glm::vec3(loc.x, loc.y, loc.z);
                            tfm = glm::translate(idt, trs);
                            if (SYMBOLIC_RENDERER)
                                sc = glm::vec3(plnts->pop[s+a][p].canopy, plnts->pop[s+a][p].height, plnts->pop[s+a][p].canopy);		// XXX: use this for symbolic renderer
                            else
                                sc = glm::vec3(plnts->pop[s+a][p].height, plnts->pop[s+a][p].height, plnts->pop[s+a][p].height);		// XXX: use this for actual tree models
                            // cerr << loc.x << ", " << loc.y << ", " << loc.z << ", " << plnts->pop[s][p].canopy << ", " << plnts->pop[s][p].height << ", " << plnts->pop[s][s].col[0] << endl;
                            tfm = glm::scale(tfm, sc);
                            tfm = glm::rotate(tfm, rotate_rad, rotate_axis);
                            xform.push_back(tfm);

                            colvar.push_back(plnts->pop[s+a][p].col); // colour variation
                            // cerr << "col = " << plnts->pop[s+a][p].col.r << " " << plnts->pop[s+a][p].col.g << " " << plnts->pop[s+a][p].col.b << " " << plnts->pop[s+a][p].col.a << endl;
                            bndplants++;
                        }
                        else
                        {
                            culledplants++;
                        }
                    }
                    // cerr << "cat " << s+a << " has " << (int) plnts->pop[s+a].size() << " plants" << endl;
                }
                if (xforms.size() < s / 3 + 1)
                {
                    xforms.resize(s / 3 + 1);
                }
                if (colvars.size() < s / 3 + 1)
                {
                    colvars.resize(s / 3 + 1);
                }
                xforms[s / 3].insert(xforms[s / 3].end(), xform.begin(), xform.end());
                colvars[s / 3].insert(colvars[s / 3].end(), colvar.begin(), colvar.end());
                f = flatten(x, y);
                shapes[f][s / 3].removeAllInstances();

                /*
                // now bind buffers for the shape associated with this species
                f = flatten(x, y);
                // cerr << "bind to f = " << f << " s = " << s << " xform size = " << (int) xform.size() << endl;
                if((int) xform.size() > 0)
                {
                    // cerr << "xform size = " << (int) xform.size() << " colvar size = " << (int) colvar.size() << endl;
                    //std::cout << "Binding instance for index: " << s/3 << std::endl;
                    //shapes[f][s/3].bindInstances(nullptr, &xform, &colvar);
                    // {
                        // cerr << "BINDING FAILED" << endl;
                    // }
                }
                else
                {
                    //shapes[f][s/3].removeAllInstances();
                }
                */
            }
        }
    for (int i = 0; i < xforms.size(); i++)
    {
        shapes[0][i].removeAllInstances();
        shapes[0][i].bindInstances(nullptr, &xforms[i], &colvars[i]);
    }
    cerr << "num bound plants = " << bndplants << endl;
    cerr << "num culled plants = " << culledplants << endl;
}

void ShapeGrid::bindPlants(View * view, Terrain * ter, bool * plantvis, PlantGrid * esys, Region region)
{
    int x, y, s, p, sx, sy, ex, ey, f;
    PlantPopulation * plnts;
    int bndplants = 0, culledplants = 0;

    esys->getRegionIndices(ter, region, sx, sy, ex, ey);
    cerr << "bind: " << sx << ", " << sy << " -> " << ex << ", " << ey << endl;

    for(x = sx; x <= ex; x++)
        for(y = sy; y <= ey; y++)
        {
            plnts = esys->getPopulation(x, y);

            // cerr << "cat 0, species 0, num = " << (int) plnts->pop[0].size() << endl;
            for(s = 0; s < (int) plnts->pop.size(); s+=3) // iterate over plant types
            {
                std::vector<glm::mat4> xform; // transformation to be applied to each instance
                std::vector<glm::vec4> colvar; // colour variation to be applied to each instance

                xform.clear();
                colvar.clear();

                if(plantvis[s/3])
                {
                    // cerr << "plants for " << s/3 << " visible" << endl;
                    for(int a = 0; a < 3; a++)
                    {
                        // cerr << "num plants = " << (int) plnts->pop[s+a].size() << endl;
                        for(p = 0; p < (int) plnts->pop[s+a].size(); p++) // iterate over plant specimens
                        {
                            if(plnts->pop[s+a][p].height > 0.1f) // only display reasonably sized plants
                            {
                                // setup transformation for individual plant, including scaling and translation
                                glm::mat4 idt, tfm;
                                glm::vec3 trs, sc;
                                vpPoint loc = plnts->pop[s+a][p].pos;

                                idt = glm::mat4(1.0f);
                                trs = glm::vec3(loc.x, loc.y, loc.z);
                                tfm = glm::translate(idt, trs);
                                sc = glm::vec3(plnts->pop[s+a][p].canopy, plnts->pop[s+a][p].height, plnts->pop[s+a][p].canopy);
                                // cerr << loc.x << ", " << loc.y << ", " << loc.z << ", " << plnts->pop[s][p].canopy << ", " << plnts->pop[s][p].height << ", " << plnts->pop[s][s].col[0] << endl;
                                tfm = glm::scale(tfm, sc);
                                xform.push_back(tfm);

                                colvar.push_back(plnts->pop[s+a][p].col); // colour variation
                                // cerr << "col = " << plnts->pop[s+a][p].col.r << " " << plnts->pop[s+a][p].col.g << " " << plnts->pop[s+a][p].col.b << " " << plnts->pop[s+a][p].col.a << endl;
                                bndplants++;
                            }
                            else
                            {
                                culledplants++;
                            }
                        }
                        // cerr << "cat " << s+a << " has " << (int) plnts->pop[s+a].size() << " plants" << endl;
                    }
                }

                // now bind buffers for the shape associated with this species
                f = flatten(x, y);
                // cerr << "bind to f = " << f << " s = " << s << " xform size = " << (int) xform.size() << endl;
                if((int) xform.size() > 0)
                {
                    // cerr << "xform size = " << (int) xform.size() << " colvar size = " << (int) colvar.size() << endl;
                    shapes[f][s/3].bindInstances(view, &xform, &colvar);
                    // {
                        // cerr << "BINDING FAILED" << endl;
                    // }
                }
            }
        }
    cerr << "num bound plants = " << bndplants << endl;
    cerr << "num culled plants = " << culledplants << endl;
}

void ShapeGrid::drawPlants(std::vector<ShapeDrawData> &drawParams)
{
    int x, y, s, f;
    ShapeDrawData sdd;

    /*
    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < (int) shapes[f].size(); s++) // iterate over plant types
            {
                sdd = shapes[f][s].getDrawParameters();
                sdd.current = false;
                drawParams.push_back(sdd);
            }
        }
    */
    for(s = 0; s < (int) shapes[0].size(); s++) // iterate over plant types
    {
        sdd = shapes[0][s].getDrawParameters();
        sdd.current = false;
        drawParams.push_back(sdd);
    }

}


/// EcoSystem

EcoSystem::EcoSystem()
{
    biome = new Biome();
    init();
}

EcoSystem::EcoSystem(Biome * ecobiome)
{
    biome = ecobiome;
    init();
}

EcoSystem::~EcoSystem()
{
    esys.delGrid();
    eshapes.delGrid();
    for(int i = 0; i < (int) niches.size(); i++)
    {
        niches[i].delGrid();
    }
    niches.clear();
    // if(synthesizer != nullptr)
    //    delete synthesizer;
}

void EcoSystem::init()
{
    esys = PlantGrid(pgdim, pgdim);
    eshapes = ShapeGrid(pgdim, pgdim, biome);
    // cmap = ConditionsMap();

    for(int i = 0; i < maxNiches; i++)
    {
        PlantGrid pgrid(pgdim, pgdim);
        niches.push_back(pgrid);
    }
    clear();
    dirtyPlants = true;
    drawPlants = false;
    maxtreehght = 0.0f;
    srand (time(NULL));
    // synthesizer = nullptr;
}

void EcoSystem::clear()
{
    esys = PlantGrid(pgdim, pgdim);
    eshapes = ShapeGrid(pgdim, pgdim, biome);
    genOpenglTextures();
    for(int i = 0; i < (int) niches.size(); i++)
    {
        niches[i] = PlantGrid(pgdim, pgdim);
    }
}

void EcoSystem::genOpenglTextures()
{
    eshapes.genOpenglTextures();
}

bool EcoSystem::loadNichePDB(string filename, Terrain * ter, int niche)
{  
    bool success;

    success = niches[niche].readPDB(filename, ter, maxtreehght);
    if(success)
    {
        dirtyPlants = true; drawPlants = true;
        cerr << "plants loaded for Niche " << niche << endl;
    }
    return success;
}

bool EcoSystem::saveNichePDB(string filename, int niche)
{
    return niches[niche].writePDB(filename);
}

void EcoSystem::pickPlants(Terrain * ter, TypeMap * clusters)
{
    Region reg = clusters->getRegion();
    esys.clearRegion(ter, reg);
    for(int n = 0; n < (int) niches.size(); n++)
    {
        niches[n].pickPlants(ter, clusters, n, esys);
    }
    // esys.reportNumPlants();
    dirtyPlants = true;
}


void EcoSystem::placePlant(Terrain *ter, const Plant &plant, int species)
{
    esys.placePlant(ter, species, plant);
}

// TODO: consider letting the basic_tree have real-world coordinates, rather than grid coords?
// TODO: check if basic_tree still needs its height converted from feet to meters
void EcoSystem::placePlant(Terrain *ter, const basic_tree &tree, bool canopy)
{
    // these colours are just temporary for now
    //const GLfloat *colours[] = {MedMNEcol, MedTBScol, MedIBScol, MedITBScol, MedMSEBcol, MedMSENcol};
    const GLfloat colours [] [4] = { {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f} };
    //float h = ter->getHeight(tree.x, tree.y);
    float h = ter->getHeightFromReal(tree.x, tree.y);
    //vpPoint pos = ter->toWorld(tree.y, tree.x, h);	// h is not affected by this function
    vpPoint pos(tree.y, h, tree.x);
    //const GLfloat *coldata = colours[(int)tree.species];
    const GLfloat *coldata;
    if (canopy)
        coldata = biome->get_species_colour(tree.species);		// XXX: have to make sure that this is the species id, not the index...
    else
    {
        coldata = biome->get_species_colour(tree.species);
    }
    glm::vec4 colour(coldata[0], coldata[1], coldata[2], coldata[3]);

    /*
    if (canopy && (fmod(pos.x / 0.9144f, 0.5f) < 1e-4 || fmod(pos.z / 0.9144f, 0.5f) < 1e-4))
    {
        std::cout << "Middle of cell encountered " << std::endl;
    }
    if (canopy)
    {
        std::cout << "Adding canopytree xy: " << pos.x << ", " << pos.z << std::endl;
    }
    */


    Plant plnt = {pos, tree.height, tree.radius * 2, colour, canopy};	//XXX: not sure if I should multiply radius by 2 here - according to scaling info in the renderer, 'radius' is actually the diameter, as far as I can see (and visual results also imply this)
    esys.placePlant(ter, ((int)tree.species) * 3, plnt);		// FIXME, XXX: I don't think we should be multiplying by 3 here...

}

void EcoSystem::placeManyPlants(Terrain *ter, const std::vector<basic_tree> &trees, bool canopy)
{
    for (auto &tr : trees)
    {
        placePlant(ter, tr, canopy);
    }
}

void EcoSystem::clearAllPlants(Terrain *ter)
{
    esys.clearAllPlants(ter);
}

void EcoSystem::pickAllPlants(Terrain * ter)
{
    esys.clear();
    for(int n = 0; n < (int) niches.size(); n++)
    {
        niches[n].pickAllPlants(ter, 0.0f, 0.0f, 1.0f, esys);
        cerr << "for niche " << n << endl;
    }
    dirtyPlants = true;
}

void EcoSystem::bindPlantsSimplified(Terrain *ter, std::vector<ShapeDrawData> &drawParams)
{
    if(dirtyPlants) // plant positions have been updated since the last bindPlants
    {
        // t.start();
        drawPlants = true;
        dirtyPlants = false;
        //eshapes.bindPlants(nullptr, ter, plantvis, &esys, clusters->getRegion());
        eshapes.bindPlantsSimplified(ter, &esys);
    }

    if(drawPlants)
        eshapes.drawPlants(drawParams);

}

void EcoSystem::bindPlants(View * view, Terrain * ter, TypeMap * clusters, bool * plantvis, std::vector<ShapeDrawData> &drawParams)
{
    // Timer t;
    // bool timing;

    // timing = dirtyPlants;

    if(dirtyPlants) // plant positions have been updated since the last bindPlants
    {
        // t.start();
        drawPlants = true;
        dirtyPlants = false;
        eshapes.bindPlants(view, ter, plantvis, &esys, clusters->getRegion());
    }

    if(drawPlants)
        eshapes.drawPlants(drawParams);

    /*
    if(timing)
    {
        t.stop();
        cerr << "plant binding takes " << t.peek() << "s" << endl;
    }*/
}

