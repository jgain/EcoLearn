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
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include "grass_sim.h"
#include "../../viewer/canopy_placement/basic_types.h"
//#include "outimage.h"
//#include "dice_roller.h"
#include <canopy_placement/basic_types.h>

#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;

bool MapFloat::read(std::string filename)
{
    float val;
    ifstream infile;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> gx >> gy;
        initMap();

        for (int x = 0; x < gx; x++)
        {
            for (int y = 0; y < gy; y++)
            {
                infile >> val;
                set(x, y, val);
            }
        }
        infile.close();
        return true;
    }
    else
    {
        cerr << "Error TypeMap::loadTxt: unable to open file" << filename << endl;
        return false;
    }
}

void GrassSim::convertCoord(int x, int y, int &sx, int &sy, int &ex, int &ey)
{
    sx = cellmult * x; sy = cellmult * y;
    ex = sx+cellmult; ey = sy+cellmult;
}

GrassSim::GrassSim(ValueGridMap<float> &ter, int cellmult)
{
    this->cellmult = cellmult;
    //matchDim(ter, 1000.0f, 2);
    matchDim(ter, cellmult);
}

void GrassSim::matchDim(ValueGridMap<float> &ter, int cellmult)
{
    int gx, gy;
    float rx, ry;

    // get grid dimensions from terrain for simulation parameters
    //ter->getGridDim(gx, gy);
    ter.getDim(gx, gy);
    ter.getDimReal(rx, ry);

    // set number of grass cells
    grasshght.setDim(gx * cellmult, gy * cellmult);
    scx = rx / (float) (gx * cellmult); scy = ry / (float) (gy * cellmult);
    //hscx = 0.5f * scx; hscy = 0.5f * scy;
}

float GrassSim::suitability(float inval, float absmin, float innermin, float innermax, float absmax)
{
    if(inval >= absmin && inval <= absmax)
    {
        if(inval < innermin) // lead in interpolation
            return ((float) (inval - absmin)) / ((float) (innermin - absmin));
        if(inval > innermax) // lead out interpolation
            return 1.0f - ((float) (inval - innermax)) / ((float) (absmax - innermax));
        return 1.0f; // prime range
    }
    else // out of survival range
    {
        return 0.0f;
    }
}

void GrassSim::toGrassGrid(float x, float y, int &i, int &j)
{
    int gx, gy;
    grasshght.getDim(gx, gy);

    i = (int) (x / scx);
    j = (int) (y / scy);
    if(i < 0)
        i = 0;
    if(j < 0)
        j = 0;
    if(i >= gx)
        i = gx-1;
    if(j >= gy)
        j = gy-1;
}

void GrassSim::toTerrain(int i, int j, float &x, float &y)
{
    x = (float) i * scx + 0.5f * scx;
    y = (float) j * scy + 0.5f * scy;
}

void GrassSim::burnInPlant(float x, float y, float r)
{
    float nx = x-r, ny = y-r, fx = x+r, fy = y+r;
    //vpPoint tree, grass;

    //tree = vpPoint(x, y, 0.0f);

    // map corners to grass grid positions
    int ni, fi, nj, fj;
    toGrassGrid(nx, ny, ni, nj);
    toGrassGrid(fx, fy, fi, fj);

    // iterate top left corner to bottom right corner
    for(int i = ni; i <= fi; i++)
        for(int j = nj; j <= fj; j++)
        {
            float wght, d, tx, ty;
            toTerrain(i, j, tx, ty);
            //grass = vpPoint(tx, ty, 0.0f);
            //d = tree.dist(grass); // how far is the grass sample from the tree trunk
            float dx = tx - x, dy = ty - y;
            d = sqrt(dx * dx + dy * dy);
            if(d <= r) // linearly interpolate grass height from trunk to canopy radius
            {
                wght = d / r;
                //wght = 0.0f;
                grasshght.set(i, j, grasshght.get(i, j) * wght);
            }
        }
}

void GrassSim::smooth(int filterwidth, int passes, bool noblank)
{
    int gx, gy;
    float filterarea;
    MapFloat newgrasshght;

    grasshght.getDim(gx, gy);
    newgrasshght.setDim(gx, gy);
    filterarea = (float) ((filterwidth*2+1)*(filterwidth*2+1));

    for(int i = 0; i < passes; i++)
    {
        for(int x = 0; x < gx; x++)
            for(int y = 0; y < gy; y++)
            {
                if(noblank || grasshght.get(x,y) > 0.0f)
                {
                    float avg = 0.0f;

                    for(int cx = x-filterwidth; cx <= x+filterwidth; cx++)
                        for(int cy = y-filterwidth; cy <= y+filterwidth; cy++)
                        {
                            if(cx < 0 || cx >= gx || cy < 0 || cy >= gy)
                                avg += grasshght.get(x, y);
                            else
                                avg += grasshght.get(cx, cy);
                        }
                    newgrasshght.set(x,y, avg / filterarea);
                }
                else
                {
                    newgrasshght.set(x, y, 0.0f);
                }
            }

        for(int x = 0; x < gx; x++)
            for(int y = 0; y < gy; y++)
                grasshght.set(x, y, newgrasshght.get(x, y));
        if(i%25 == 0)
            cerr << i << " smoothing iterations" << endl;
    }
}

void GrassSim::grow(ValueGridMap<float> &ter, const std::vector<basic_tree *> &plnt_pointers)
{
    if (!conditions_set)
    {
        throw std::runtime_error("Error: set abiotic conditions first before running grass simulation");
    }
    if (!params_set)
    {
        throw std::runtime_error("Error: set grass viability parameters first before running grass simulation");
    }

    int dx, dy, gx, gy;
    bool bare = false;
    //DiceRoller roller(0,RAND_MAX);
    std::default_random_engine gen;
    std::uniform_int_distribution<int> unif_int(0, RAND_MAX);

    ter.getDim(dx, dy);
    grasshght.getDim(gx, gy);

    cerr << "grass-sim: initial grass height" << endl;
    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
        {
            int sx, sy, ex, ey;

            // rocky areas have zero grass heights
            /*
            bare = (* painted->getMap())[y][x] == (int) BrushType::ROCK; // TO DO - water type may be added later
            */

            // alpine settings
            /*
            float suitval = suitability(illumination.get(x,y), 1.0f, 6.5f, 10.0f, 15.0f); // illumination
            suitval = min(suitval, suitability(moisture.get(x,y), 20.0f, 100.0f, 10000.0f, 10000.0f)); // moisture
            suitval = min(suitval, suitability(temperature.get(x,y), 0.0f, 10.0f, 22.0f, 40.0f)); // temperature
            */

            // savannah settings
            /*
            float suitval = suitability(illumination.get(x,y), 1.0f, 6.5f, 10.0f, 15.0f); // illumination
            suitval = min(suitval, suitability(moisture.get(x,y), 20.0f, 200.0f, 10000.0f, 10000.0f)); // moisture
            suitval = min(suitval, suitability(temperature.get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature
            */

            // canyon settings
            /*
            float suitval = suitability(illumination.get(x,y), 1.0f, 6.5f, 10.0f, 15.0f); // illumination
            suitval = min(suitval, suitability(moisture.get(x,y), 50.0f, 200.0f, 10000.0f, 10000.0f)); // moisture
            suitval = min(suitval, suitability(temperature.get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature
            */

            // med settings
            //float suitval = suitability(illumination->get(x,y), 1.0f, 6.5f, 12.0f, 15.0f); // illumination	original
            //float suitval = suitability(illumination->get(x,y), 2000.0f, 2300.0f, 2500.0f, 2600.0f); // illumination
            //suitval = min(suitval, suitability(moisture->get(x,y), 20.0f, 100.0f, 10000.0f, 10000.0f)); // moisture	original
            //suitval = min(suitval, suitability(moisture->get(x,y), 20.0f, 100.0f, 1000.0f, 1500.0f)); // moisture
            //suitval = min(suitval, suitability(temperature->get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature   original
            //suitval = min(suitval, suitability(temperature->get(x,y), 2.5f, 15.0f, 22.0f, 40.0f)); // temperature

            auto &mv = viability_params["moisture"];
            auto &sv = viability_params["sunlight"];
            auto &tv = viability_params["temperature"];

            float moisture_suit = suitability(moisture->get(x,y), mv.absmin, mv.innermin, mv.innermax, mv.absmax);
            float temp_suit = suitability(temperature->get(x,y), tv.absmin, tv.innermin, tv.innermax, tv.absmax);
            float sun_suit = suitability(illumination->get(x,y), sv.absmin, sv.innermin, sv.innermax, sv.absmax);
            float suitval = min(moisture_suit, min(temp_suit, sun_suit));
            //if (suitval == 0)
            //    std::cout << "moisture, temp, sun: " << moisture_suit << ", " << temp_suit << ", " << sun_suit << std::endl;

            // if(x%100==0  && y%100==0)
            //    cerr << suitval << " ";

            // a terrain cell covers multiple grass cells, so calculate coverage
            convertCoord(x, y, sx, sy, ex, ey);
            // cerr << "sx, sy = " << sx << ", " << sy << " -> ex, ey = " << ex << ", " << ey << endl;
            for(int cx = sx; cx < ex; cx++)
                for(int cy = sy; cy < ey; cy++)
                {
                    float hght = 0.0f;
                    if(!bare) // lookup terrain conditions, value is minimum of all terrain condition ranges
                    {
                        hght = suitval * MAXGRASSHGHT;
                    }
                    grasshght.set(cx, cy, hght); // set grass height
                }

        }

    cerr << "grass-sim: smoothing" << endl;
    // smooth(2, 300, true); // alpine smooth
    smooth(2, 50, true); // med smooth
    // smooth(2, 50, true); // savannah smooth
    // smooth(2, 50, true); // canyon smooth

    cerr << "grass-sim: random variation" << endl;
    float rndvar;

    // rndvar = 0.1; // alpine variation
    // rndvar = 0.3; // savannah variation
    rndvar = 0.1; // canyon variation

    // one pass of random variation
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
            if(grasshght.get(x, y) > 0.0f)
            {
                float h = grasshght.get(x, y);
                //h *= (1.0f + rndvar * (float) (roller.generate()%2000-1000) / 1000.0f);
                h *= (1.0f + rndvar * (float) (unif_int(gen) % 2000 - 1000) / 1000.0f);
                if(h > MAXGRASSHGHT)
                    h = MAXGRASSHGHT;
                grasshght.set(x, y, h);
            }

    // a few passes to smooth out the noise and make it less local
    // smooth(2, 10, true);

    cerr << "grass-sim: water burn" << endl;

    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
        {
            int sx, sy, ex, ey;

            float river = 300.0f; // savannah

            if(moisture->get(x,y) > river) // river so remove grass
            {
                // a terrain cell covers multiple grass cells, so calculate coverage
                convertCoord(x, y, sx, sy, ex, ey);
                for(int cx = sx; cx < ex; cx++)
                    for(int cy = sy; cy < ey; cy++)
                    {
                        grasshght.set(cx, cy, 0.0f); // set grass height to zero
                    }
            }

        }

    // some more smoothing to blend water edges
    smooth(2, 50, false);

    cerr << "grass-sim: plant burn" << endl;
    // iterate over plants in the ecosystem, burning in their circles on the grass, depending on the plant alpha and distance from the plant center
    // TO DO - add plant alpha, test burn in fully
    //ecs->getPlants()->burnGrass(this, ter, scale);
    burnGrass(plnt_pointers);
    smooth(2, 2, false);
}

void GrassSim::burnGrass(const std::vector<basic_tree *> &plnt_pointers)
{
    for (const auto &plntptr : plnt_pointers)
    {
        burnInPlant(plntptr->x, plntptr->y, plntptr->radius);
    }
}

void GrassSim::set_viability_params(const std::map<string, data_importer::grass_viability> &viability_params)
{
    this->viability_params = viability_params;
    params_set = true;
}

void GrassSim::setConditions(MapFloat * wetfile, MapFloat * sunfile, MapFloat * tempfile)
{
    moisture = wetfile;
    illumination = sunfile;
    temperature = tempfile;
    conditions_set = true;
}

bool GrassSim::write(std::string filename)
{
    data_importer::write_txt<MapFloat>(filename, &grasshght);
}

/*
bool GrassSim::write(std::string filename)
{
    // convert grass simulation to greyscale in [0,1] range
    std::vector<float> normhght;
    float h;
    int gx, gy;

    grasshght.getDim(gx, gy);

    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            h = grasshght.get(x, gy-1-y) / MAXGRASSHGHT;
            if(h > 1.0f)
                h = 1.0f;
            if(h < 0.0f)
                h = 0.0f;
            normhght.push_back(h);
        }
    OutImage outimg;
    return outimg.write(filename, gx, gy, normhght);
}
 */
