/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
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

#include "grass.h"
#include "outimage.h"
#include "dice_roller.h"
#include "terrain.h"
#include "eco.h"
#include "data_importer/data_importer.h"

#include <fstream>
#include <iostream>


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

void GrassSim::convertCoord(Terrain * ter, int x, int y, int & sx, int & sy, int &ex, int &ey)
{
    int dx, dy, gx, gy;

    ter->getGridDim(dx, dy);
    grasshght.getDim(gx, gy);
    int cellx = gx / dx;
    int celly = gy / dy;
    sx = cellx * x; sy = celly * y;
    ex = sx+cellx; ey = sy+celly;
}

GrassSim::GrassSim(Terrain *ter)
{
    matchDim(ter, 1000.0f, 2);
}

void GrassSim::matchDim(Terrain *ter, float scale, int mult)
{
    int gx, gy;
    float tx, ty;

    // get grid dimensions from terrain for simulation parameters
    ter->getGridDim(gx, gy);
    ter->getTerrainDim(tx, ty);

    // set number of grass cells
    grasshght.setDim(gx * mult, gy * mult);
    scx = tx / (float) (gx * mult); scy = ty / (float) (gy * mult);
    hscx = 0.5f * scx; hscy = 0.5f * scy;
    cerr << "grass cell dimensions = " << scx << " X " << scy << endl;
    //cerr << "scale = " << scale << " mult = " << mult << endl;
    litterfall_density.setDim(gx * mult, gy * mult);
    litterfall_density.fill(0.0f);
}

bool GrassSim::write_litterfall(std::string filename)
{
    data_importer::write_txt<MapFloat>(filename, &litterfall_density, scx);
}

void GrassSim::set_viability_params(const std::map<string, data_importer::grass_viability> &viability_params)
{
    viability = viability_params;
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
    x = (float) i * scx + hscx;
    y = (float) j * scy + hscy;
}

void GrassSim::burnInPlant(float x, float y, float r, float alpha)
{
    float nx = x-r, ny = y-r, fx = x+r, fy = y+r;
    vpPoint tree, grass;

    tree = vpPoint(x, y, 0.0f);

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
           grass = vpPoint(tx, ty, 0.0f);
           d = tree.dist(grass); // how far is the grass sample from the tree trunk
           if(d <= r) // linearly interpolate grass height from trunk to canopy radius
           {
               wght = d / r;
               wght *= 1.0f - alpha;		// account for plant alpha
               //wght = 0.0f;
               grasshght.set(i, j, grasshght.get(i, j) * wght);
               //grasshght.set(i, j, MAXGRASSHGHT * 0.9f);
               float openground = 1.0f - litterfall_density.get(i, j);
               litterfall_density.set(i, j, 1.0f - wght * openground);
           }
       }
}

void GrassSim::smooth_general(int filterwidth, int passes, bool noblank, MapFloat &srcdest)
{
    int gx, gy;
    float filterarea;
    MapFloat newgrasshght;

    srcdest.getDim(gx, gy);
    newgrasshght.setDim(gx, gy);
    filterarea = (float) ((filterwidth*2+1)*(filterwidth*2+1));

    for(int i = 0; i < passes; i++)
    {
        for(int x = 0; x < gx; x++)
            for(int y = 0; y < gy; y++)
            {
                if(noblank || srcdest.get(x,y) > 0.0f)
                {
                    float avg = 0.0f;

                    for(int cx = x-filterwidth; cx <= x+filterwidth; cx++)
                        for(int cy = y-filterwidth; cy <= y+filterwidth; cy++)
                        {
                            if(cx < 0 || cx >= gx || cy < 0 || cy >= gy)
                                avg += srcdest.get(x, y);
                            else
                                avg += srcdest.get(cx, cy);
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
                srcdest.set(x, y, newgrasshght.get(x, y));
        if(i%25 == 0)
            cerr << i << " smoothing iterations" << endl;
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

void GrassSim::grow(Terrain * ter, std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float scale)
{
    if (has_backup)
    {
        grasshght = backup_grass;
    }
    else
    {
        int dx, dy, gx, gy;
        bool bare = false;
        DiceRoller roller(0,RAND_MAX);

        ter->getGridDim(dx, dy);
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

                /*
                // med settings
                float suitval = suitability(illumination->get(x,y), 2.0f, 6.5f, 12.0f, 16.0f); // illumination	original
                //float suitval = suitability(illumination->get(x,y), 2000.0f, 2300.0f, 2500.0f, 2600.0f); // illumination
                suitval = min(suitval, suitability(moisture->get(x,y), 20.0f, 100.0f, 10000.0f, 10000.0f)); // moisture	original
                //suitval = min(suitval, suitability(moisture->get(x,y), 5.0f, 100.0f, 1000.0f, 1500.0f)); // moisture
                //suitval = min(suitval, suitability(temperature->get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature   original
                suitval = min(suitval, suitability(temperature->get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature
                */

                auto sunv = viability["sunlight"];
                auto wetv = viability["moisture"];
                auto tempv = viability["temperature"];

                float suitval = suitability(illumination->get(x,y), sunv.absmin, sunv.innermin, sunv.innermax, sunv.absmax); // illumination	original
                //float suitval = suitability(illumination->get(x,y), 2000.0f, 2300.0f, 2500.0f, 2600.0f); // illumination
                suitval = min(suitval, suitability(moisture->get(x,y), wetv.absmin, wetv.innermin, wetv.innermax, wetv.absmax)); // moisture	original
                //suitval = min(suitval, suitability(moisture->get(x,y), 5.0f, 100.0f, 1000.0f, 1500.0f)); // moisture
                //suitval = min(suitval, suitability(temperature->get(x,y), 5.0f, 15.0f, 22.0f, 40.0f)); // temperature   original
                suitval = min(suitval, suitability(temperature->get(x,y), tempv.absmin, tempv.innermin, tempv.innermax, tempv.absmax)); // temperature

                // if(x%100==0  && y%100==0)
                //    cerr << suitval << " ";

                // a terrain cell covers multiple grass cells, so calculate coverage
                convertCoord(ter, x, y, sx, sy, ex, ey);
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
                    h *= (1.0f + rndvar * (float) (roller.generate()%2000-1000) / 1000.0f);
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
                    convertCoord(ter, x, y, sx, sy, ex, ey);
                    for(int cx = sx; cx < ex; cx++)
                        for(int cy = sy; cy < ey; cy++)
                        {
                            grasshght.set(cx, cy, 0.0f); // set grass height to zero
                        }
                }

            }

        // some more smoothing to blend water edges
        smooth(2, 50, false);

        backup_grass = grasshght;
        has_backup = true;
    }

    cerr << "grass-sim: plant burn" << endl;
    // iterate over plants in the ecosystem, burning in their circles on the grass, depending on the plant alpha and distance from the plant center
    // TO DO - add plant alpha, test burn in fully
    //ecs->getPlants()->burnGrass(this, ter, scale);
    burnGrass(ter, trees, cdata, scale);
    smooth(2, 2, false);
    smooth_general(5, 2, true, litterfall_density);
}

void GrassSim::burnGrass(Terrain *ter, const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float scale)
{
    float invscf, tx, ty;

    ter->getTerrainDim(tx, ty);
    invscf = scale / tx;

    int nburned = 0;

    litterfall_density.fill(0.0f);

    for (auto &tree : trees)
    {
        float alpha = cdata.canopy_and_under_species.at(tree.species).alpha;
        burnInPlant(tree.x * invscf, tree.y * invscf, tree.radius * invscf, alpha);
        nburned++;
    }

    std::cout << "Number of plants burned: " << nburned << std::endl;
}

void GrassSim::setConditions(MapFloat * wetfile, MapFloat * sunfile, MapFloat *landsun_file, MapFloat * tempfile)
{
    moisture = wetfile;
    illumination = sunfile;
    temperature = tempfile;
    landsun = landsun_file;
}

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
