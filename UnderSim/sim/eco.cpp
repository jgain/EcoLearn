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


// eco.cpp: core classes for controlling ecosystems and plant placement
// author: James Gain
// date: 27 February 2016

#include "eco.h"
// #include "interp.h"
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <QDir>

/// PlantGrid

void PlantGrid::initSpeciesTable()
{
    speciesTable.clear();
}

void PlantGrid::delGrid()
{
    int i, j;

    // clear out elements of the vector hierarchy
    for(i = 0; i < (int) pgrid.size(); i++)
        for(j = 0; j < (int) pgrid[i].pop.size(); j++)
            pgrid[i].pop[j].clear();
    for(i = 0; i < (int) pgrid.size(); i++)
        pgrid[i].pop.clear();
    pgrid.clear();
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
            for(s = 0; s < maxSpecies; s++)
            {
                std::vector<Plant> plnts;
                ppop.pop.push_back(plnts);
            }
            pgrid.push_back(ppop);
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
        pgrid[f].pop[s].clear();
}

void PlantGrid::placePlant(Terrain * ter, int species, Plant plant)
{
    int x, y, cx, cy;

    // find plant location on map
    ter->toGrid(plant.pos, x, y);
    cellLocate(ter, x, y, cx, cy);
    // cerr << "loc in " << cx << ", " << cy << " species " << species << endl;

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

void PlantGrid::pickPlants(Terrain * ter, TypeMap * clusters, int niche, PlantGrid & outgrid)
{
    int x, y, s, p, mx, my, sx, sy, ex, ey, f;
    Plant plnt;
    Region region = clusters->getRegion();

    // map region to cells in the grid
    getRegionIndices(ter, region, sx, sy, ex, ey);

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
                }
        }
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

void PlantGrid::getRegionIndices(Terrain * ter, Region region, int &sx, int &sy, int &ex, int &ey)
{
    cellLocate(ter, region.x0, region.y0, sx, sy);
    cellLocate(ter, region.x1, region.y1, ex, ey);

    // cerr << "map = " << region.x0 << ", " << region.y0 << " -> " << region.x1 << ", " << region.y1 << endl;
    // cerr << "grid = " << sx << ", " << sy << " -> " << ex << ", " << ey << endl;
}



bool PlantGrid::readPDB(string filename, Biome * biome, Terrain * ter, float &maxtree)
{
    ifstream infile;
    float rndoff;
    int currSpecies, speciesCount;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> speciesCount;

        for(int i = 0; i < speciesCount; i++)
        {
            int numPlants;
            float canopyRatio, hghtMin, hghtMax;

            infile >> currSpecies;

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
                    placePlant(ter, currSpecies, p);
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

void PlantGrid::inscribeAlpha(Terrain * ter, MapFloat * alpha, float aval, vpPoint p, float rcanopy)
{
    int gx, gy, tx, ty, gr;

    // convert to grid coordinates
    ter->toGrid(p, gx, gy);
    ter->getGridDim(tx, ty);
    gr = (int) (ter->toGrid(rcanopy/2.0f) + 0.5f);

    // bounding box around circle
    int sx = max(gx-gr, 0);
    int ex = min(gx+gr, tx-1);
    int sy = max(gy-gr, 0);
    int ey = min(gy+gr, ty-1);

    // iterate over square on terrain containing the circle
    for(int x = sx; x <= ex; x++)
        for(int y = sy; y <= ey; y++)
        {
            float delx = (float) x-gx;
            float dely = (float) y-gy;
            float distsq = delx*delx + dely*dely;
            if (distsq <= (float) (gr*gr)) // inside the circle
             {
                if(aval > alpha->get(x, y)) // max of occlusion written
                    alpha->set(x, y, aval);
             }
        }
}

void PlantGrid::sunSeeding(Terrain * ter, Biome * biome, MapFloat * alpha)
{
    int x, y, s, p, f;
    Plant plnt;

    for(x = 0; x < gx; x++)
        for(y = 0; y < gy; y++)
        {
            f = flatten(x, y);
            for(s = 0; s < (int) pgrid[f].pop.size(); s++)
                for(p = 0; p < (int) pgrid[f].pop[s].size(); p++)
                {
                    plnt = pgrid[f].pop[s][p];
                    inscribeAlpha(ter, alpha, biome->getAlpha(s), plnt.pos, plnt.canopy);
                }
        }
}

bool PlantGrid::writePDB(string filename, Biome * biome)
{
    ofstream outfile;
    std::vector<int> activeSpecies;
    int writtenPlants = 0;

    cerr << "write PDB " << filename << endl;
    outfile.open((char *) filename.c_str(), ios_base::out | ios_base::trunc);
    if(outfile.is_open())
    {
        // count number of non-zero element species
        for(int s = 0; s < biome->numPFTypes(); s++)
        {
            int specNum;
            bool found = false;

            // gather plants belonging to species and derive statistics
            for(int i = 0; i < gx; i++)
                for(int j = 0; j < gy; j++)
                    if((int) pgrid[flatten(i,j)].pop[s].size() > 0)
                        found = true;
            if(found)
                activeSpecies.push_back(s);
        }

        outfile << (int) activeSpecies.size() << endl;

        for(int k = 0; k < (int) activeSpecies.size(); k++)
        {
            int numPlants, s;
            float canopyRatio, hghtMin, hghtMax;
            std::vector<Plant> tpop; tpop.clear();

            s = activeSpecies[k];
            outfile << s << " ";

            canopyRatio = 0.0f; hghtMax = 0.0f; hghtMin = 200.0f;
            // gather plants belonging to species and derive statistics
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
                        tpop.push_back(plnt);
                    }
                }
            numPlants = (int) tpop.size();
            canopyRatio /= (float) numPlants;

            outfile << hghtMin << " " << hghtMax << " " << canopyRatio; // << " ";
            outfile << endl;	// XXX: if errors occur in PDB file, remove this line, and uncomment space above

            outfile << numPlants << endl;
            for(int j = 0; j < numPlants; j++)
            {
                Plant plnt = tpop[j];

                // terrain position and plant height
                float x = plnt.pos.x;
                float y = plnt.pos.z;
                float z = plnt.pos.y;
                float h = plnt.height;
                float r = plnt.canopy/2.0f;
                outfile << x  << " " << y << " " << z << " " << h << " " << r << endl;
                writtenPlants++;
            }
        }

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
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1f, 3, 1, tfm, false);

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
    shape.genCappedCylinder(trunkradius, trunkradius, trunkheight+0.1f, 3, 1, tfm, false);

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
            for(s = 0; s < maxSpecies; s++)
            {
                Shape shape;
                shapevec.push_back(shape);
            }
            shapes.push_back(shapevec);
        }
    genPlants();
}

void ShapeGrid::genPlants()
{
    float trunkheight, trunkradius;
    int s;

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



void ShapeGrid::bindPlantsSimplified(Terrain *ter, PlantGrid *esys, std::vector<bool> * plantvis)
{
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

            for(s = 0; s < (int) plnts->pop.size(); s++) // iterate over plant types
            {
                std::vector<glm::mat4> xform; // transformation to be applied to each instance
                std::vector<glm::vec4> colvar; // colour variation to be applied to each instance
                if((* plantvis)[s])
                    {
                    for(p = 0; p < (int) plnts->pop[s].size(); p++) // iterate over plant specimens
                    {
                        if(plnts->pop[s][p].height > 0.01f) // only display reasonably sized plants
                        {
                            // setup transformation for individual plant, including scaling and translation
                            glm::mat4 idt, tfm;
                            glm::vec3 trs, sc, rotate_axis = glm::vec3(0.0f, 1.0f, 0.0f);
                            vpPoint loc = plnts->pop[s][p].pos;
                            // GLfloat rotate_rad;

                            /*
                            if (plnts->pop[s+a][p].iscanopy)	// we use a different generator depending on whether we have a canopy or undergrowth plant - keeps rotations the same whether we render undergrowth plants or not
                            {
                                rotate_rad = rand_unif(generator_canopy) * glm::pi<GLfloat>() * 2.0f;
                            }
                            else
                            {
                                rotate_rad = rand_unif(generator_under) * glm::pi<GLfloat>() * 2.0f;
                            }*/

                            idt = glm::mat4(1.0f);
                            trs = glm::vec3(loc.x, loc.y, loc.z);
                            tfm = glm::translate(idt, trs);
                            sc = glm::vec3(plnts->pop[s][p].canopy, plnts->pop[s][p].height, plnts->pop[s][p].canopy);		// XXX: use this for actual tree models
                            tfm = glm::scale(tfm, sc);
                            // tfm = glm::rotate(tfm, rotate_rad, rotate_axis);
                            xform.push_back(tfm);

                            colvar.push_back(plnts->pop[s][p].col); // colour variation
                            bndplants++;
                        }
                        else
                        {
                            culledplants++;
                        }
                    }

                }

                if (xforms.size() < s + 1)
                {
                    xforms.resize(s  + 1);
                }
                if (colvars.size() < s + 1)
                {
                    colvars.resize(s + 1);
                }
                xforms[s].insert(xforms[s].end(), xform.begin(), xform.end());
                colvars[s].insert(colvars[s].end(), colvar.begin(), colvar.end());
                f = flatten(x, y);
                shapes[f][s].removeAllInstances();
            }
        }

    for (int i = 0; i < xforms.size(); i++)
    {
        shapes[0][i].removeAllInstances();
        shapes[0][i].bindInstances(nullptr, &xforms[i], &colvars[i]);
    }
}

void ShapeGrid::bindPlants(View * view, Terrain * ter, std::vector<bool> * plantvis, PlantGrid * esys, Region region)
{
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

            for(s = 0; s < (int) plnts->pop.size(); s+=3) // iterate over plant types
            {
                std::vector<glm::mat4> xform; // transformation to be applied to each instance
                std::vector<glm::vec4> colvar; // colour variation to be applied to each instance

                for(int a = 0; a < 3; a++)
                {
                    for(p = 0; p < (int) plnts->pop[s+a].size(); p++) // iterate over plant specimens
                    {
                        if(plnts->pop[s+a][p].height > 0.01f) // only display reasonably sized plants
                        {
                            // setup transformation for individual plant, including scaling and translation
                            glm::mat4 idt, tfm;
                            glm::vec3 trs, sc, rotate_axis = glm::vec3(0.0f, 1.0f, 0.0f);
                            vpPoint loc = plnts->pop[s+a][p].pos;
                            // GLfloat rotate_rad;

                            /*
                            if (plnts->pop[s+a][p].iscanopy)	// we use a different generator depending on whether we have a canopy or undergrowth plant - keeps rotations the same whether we render undergrowth plants or not
                            {
                                rotate_rad = rand_unif(generator_canopy) * glm::pi<GLfloat>() * 2.0f;
                            }
                            else
                            {
                                rotate_rad = rand_unif(generator_under) * glm::pi<GLfloat>() * 2.0f;
                            }*/

                            idt = glm::mat4(1.0f);
                            trs = glm::vec3(loc.x, loc.y, loc.z);
                            tfm = glm::translate(idt, trs);
                            sc = glm::vec3(plnts->pop[s+a][p].height, plnts->pop[s+a][p].height, plnts->pop[s+a][p].height);		// XXX: use this for actual tree models
                            tfm = glm::scale(tfm, sc);
                            xform.push_back(tfm);

                            colvar.push_back(plnts->pop[s+a][p].col); // colour variation
                            bndplants++;
                        }
                        else
                        {
                            culledplants++;
                        }
                    }

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
            }
        }


    for (int i = 0; i < xforms.size(); i++)
    {
        cerr << i << endl;
        shapes[0][i].removeAllInstances();
        shapes[0][i].bindInstances(nullptr, &xforms[i], &colvars[i]);
    }
    cerr << "num bound plants = " << bndplants << endl;
    cerr << "num culled plants = " << culledplants << endl;


}

void ShapeGrid::drawPlants(std::vector<ShapeDrawData> &drawParams)
{
    int x, y, s, f;
    ShapeDrawData sdd;

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
}

void EcoSystem::clear()
{
    esys = PlantGrid(pgdim, pgdim);

    for(int i = 0; i < (int) niches.size(); i++)
    {
        niches[i] = PlantGrid(pgdim, pgdim);
    }
}

bool EcoSystem::loadNichePDB(string filename, Terrain * ter, int niche)
{  
    bool success;

    // std::cout << "Number of niches: " << niches.size() << std::endl;
    success = niches[niche].readPDB(filename, biome, ter, maxtreehght);
    if(success)
    {
        dirtyPlants = true; drawPlants = true;
        cerr << "plants loaded for Niche " << niche << endl;
    }
    return success;
}

bool EcoSystem::saveNichePDB(string filename, int niche)
{
    return niches[niche].writePDB(filename, biome);
}

void EcoSystem::pickPlants(Terrain * ter, TypeMap * clusters)
{
    Region reg = clusters->getRegion();
    esys.clearRegion(ter, reg);
    for(int n = 0; n < (int) niches.size(); n++)
    {
        niches[n].pickPlants(ter, clusters, n, esys);
    }
    dirtyPlants = true;
}

void EcoSystem::pickAllPlants(Terrain * ter, bool canopyOn, bool underStoreyOn)
{
    esys.clear();
    for(int n = 0; n < (int) niches.size(); n++)
    {
        if(n == 0 && canopyOn)
            niches[n].pickAllPlants(ter, 0.0f, 0.0f, 1.0f, esys);
        if(n > 0 && underStoreyOn)
            niches[n].pickAllPlants(ter, 0.0f, 0.0f, 1.0f, esys);
    }
    dirtyPlants = true;
}

void EcoSystem::sunSeeding(Terrain * ter, Biome * biome, MapFloat * alpha)
{
    for(int n = 0; n < (int) niches.size(); n++)
    {
        getNiche(n)->sunSeeding(ter, biome, alpha);
    }
}

void EcoSystem::bindPlantsSimplified(Terrain *ter, std::vector<ShapeDrawData> &drawParams, std::vector<bool> * plantvis)
{
    if(dirtyPlants) // plant positions have been updated since the last bindPlants
    {
        drawPlants = true;
        dirtyPlants = false;
        eshapes.bindPlantsSimplified(ter, &esys, plantvis);
    }

    if(drawPlants)
        eshapes.drawPlants(drawParams);
}

void EcoSystem::bindPlants(View * view, Terrain * ter, TypeMap * clusters, std::vector<bool> * plantvis, std::vector<ShapeDrawData> &drawParams)
{
    cerr << "ecosys bind" << endl;
    if(dirtyPlants) // plant positions have been updated since the last bindPlants
    {
        // t.start();
        drawPlants = true;
        dirtyPlants = false;
        eshapes.bindPlants(view, ter, plantvis, &esys, clusters->getRegion());
    }

    if(drawPlants)
        eshapes.drawPlants(drawParams);

    cerr << "end ecosys bind" << endl;
}

