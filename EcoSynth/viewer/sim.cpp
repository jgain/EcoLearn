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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "sim.h"
#include "data_importer/data_importer.h"
#include "ClusterMatrices.h"

#include <QLabel>

////
// MapSimCell
///

void MapSimCell::initMap()
{
    smap.clear();
    smap.resize(gx*gy);
    for(int c = 0; c < gx*gy; c++) // empty cells are always sorted
    {
        smap[c].canopysorted = true;
        smap[c].rootsorted = true;
    }
}

void MapSimCell::inscribe(int plntidx, float px, float py, float rcanopy, float rroot)
{
    float grx, gry, grcanopy, grroot, rmax, distsq, grcanopysq, grrootsq;

    // assumes that plant index is not already inscribed in the simulation grid

    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);
    grcanopy = convert(rcanopy);
    grroot = convert(rroot);

    grcanopysq = grcanopy * grcanopy;
    grrootsq = grroot * grroot;
    rmax = fmax(grroot, grcanopy);

    for(int x = (int) grx-rmax; x <= (int) grx+rmax; x++)
        for(int y = (int) gry-rmax; y <= (int) gry+rmax; y++)
        {
            PlntInCell pic;

            if(ingrid(x,y))
            {
                pic.hght = 0.0f; // height will be instantiated later on demand
                pic.idx = plntidx;

                float delx = (float) x-grx;
                float dely = (float) y-gry;
                distsq = delx*delx + dely*dely;
                if(distsq <= grcanopysq)
                {
                    get(x,y)->canopies.push_back(pic);
                    if((int) get(x,y)->canopies.size() > 1)
                        get(x,y)->canopysorted = false;
                }
                if(distsq <= grrootsq)
                {
                    get(x,y)->roots.push_back(pic);
                    if((int) get(x,y)->roots.size() > 1)
                        get(x,y)->rootsorted = false;
                }
            }
        }
}

void MapSimCell::expand(int plntidx, float px, float py, float prevrcanopy, float prevrroot, float newrcanopy, float newrroot)
{
    float grx, gry, gprevrcanopy, gprevrroot, gnewrcanopy, gnewrroot, rmax, distsq, gprevrcanopysq, gprevrrootsq, gnewrcanopysq, gnewrrootsq;

    // assumes that plant index is not already inscribed in the simulation grid

    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);
    gprevrcanopy = convert(prevrcanopy);
    gprevrroot = convert(prevrroot);
    gnewrcanopy = convert(newrcanopy);
    gnewrroot = convert(newrroot);

    gprevrcanopysq = gprevrcanopy * gprevrcanopy;
    gprevrrootsq = gprevrroot * gprevrroot;
    gnewrcanopysq = gnewrcanopy * gnewrcanopy;
    gnewrrootsq = gnewrroot * gnewrroot;
    rmax = fmax(gnewrroot, gnewrcanopy);

    for(int x = (int) grx-rmax; x <= (int) grx+rmax; x++)
        for(int y = (int) gry-rmax; y <= (int) gry+rmax; y++)
        {
            PlntInCell pic;

            if(ingrid(x,y))
            {
                pic.hght = 0.0f; // height will be instantiated later on demand
                pic.idx = plntidx;

                float delx = (float) x-grx;
                float dely = (float) y-gry;
                distsq = delx*delx + dely*dely;
                if(distsq <= gnewrcanopysq && distsq > gprevrcanopysq)
                {
                    get(x,y)->canopies.push_back(pic);
                    if((int) get(x,y)->canopies.size() > 1)
                        get(x,y)->canopysorted = false;
                }
                if(distsq <= gnewrrootsq && distsq > gprevrrootsq)
                {
                    get(x,y)->roots.push_back(pic);
                    if((int) get(x,y)->roots.size() > 1)
                        get(x,y)->rootsorted = false;
                }

            }
        }
}

float cmpHeight(PlntInCell i, PlntInCell j)
{
    return i.hght > j.hght;
}

void MapSimCell::traverse(std::vector<SimPlant> *plnts, Biome * biome, MapFloat * sun, MapFloat * wet)
{
    plntpop = plnts;

    // clear sunlight and moisture plant pools
    for(int p = 0; p < (int) plntpop->size(); p++)
    {
        SimPlant * plnt = &(* plntpop)[p];
        if(plnt->state == PlantSimState::ALIVE)
        {
            plnt->sunlight = 0.0f;
            plnt->water = 0.0f;
            plnt->sunlightcnt = 0;
            plnt->watercnt = 0;
        }
    }

    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // sort canopy trees by height if necessary
            if(!get(x,y)->canopysorted)
            {
                // update heights
                for(int p = 0; p < (int) get(x,y)->canopies.size(); p++)
                    get(x,y)->canopies[p].hght = (* plntpop)[get(x,y)->canopies[p].idx].height;
                std::sort(get(x,y)->canopies.begin(), get(x,y)->canopies.end(), cmpHeight);
                get(x,y)->canopysorted = true;
            }

            // JGFIX - flag empty cells as possible seeding candidates

            float sunlight = sun->get(x/step, y/step);
            // traverse trees by height supplying and reducing sunlight
            for(int p = 0; p < (int) smap[flatten(x,y)].canopies.size(); p++)
            {
                if((* plntpop)[get(x,y)->canopies[p].idx].state != PlantSimState::DEAD)
                {
                    (* plntpop)[get(x,y)->canopies[p].idx].sunlight += sunlight;
                    (* plntpop)[get(x,y)->canopies[p].idx].sunlightcnt++;
                    // reduce sunlight by alpha of current plant
                    sunlight *= biome->getAlpha((* plntpop)[get(x,y)->canopies[p].idx].pft);
                }
            }

            // sort root trees by height if necessary
            if(!get(x,y)->rootsorted)
            {
                // update heights
                for(int p = 0; p < (int) get(x,y)->roots.size(); p++)
                    get(x,y)->roots[p].hght = (* plntpop)[get(x,y)->roots[p].idx].height;
                std::sort(get(x,y)->roots.begin(), get(x,y)->roots.end(), cmpHeight);
                get(x,y)->rootsorted = true;
            }

            float moisture = wet->get(x/step, y/step);
            float watershare;
            int livecnt = 0;
            // traverse trees by height (proxy for root depth) supplying and reducing moisture
            for(int p = 0; p < (int) smap[flatten(x,y)].roots.size(); p++)
            {
                if((* plntpop)[get(x,y)->roots[p].idx].state != PlantSimState::DEAD)
                {   // plants grab min ideal
                    watershare = fmin(moisture, biome->getMinIdealMoisture((* plntpop)[get(x,y)->roots[p].idx].pft));
                    (* plntpop)[get(x,y)->roots[p].idx].water += watershare;
                    (* plntpop)[get(x,y)->roots[p].idx].watercnt++;
                    moisture -= watershare;
                    livecnt++;
                }
            }
            // remainder spread equally
            if(moisture > 0.0f)
            {
                watershare = moisture / (float) livecnt;
                for(int p = 0; p < (int) smap[flatten(x,y)].roots.size(); p++)
                    if((* plntpop)[get(x,y)->roots[p].idx].state != PlantSimState::DEAD)
                        (* plntpop)[get(x,y)->roots[p].idx].water += watershare;
            }
    }

}

void MapSimCell::visualize(QImage * visimg, std::vector<SimPlant> *plnts)
{
    int numplnts = (int) plnts->size();
    std::vector<QColor> plntcols;

    // create table of random colours
    for(int p = 0; p < numplnts; p++)
    {
        int r = 256;
        while(r > 255)
            r = std::rand()/(RAND_MAX/255);
        int g = 256;
        while(g > 255)
            g = std::rand()/(RAND_MAX/255);
        int b = 256;
        while(b > 255)
            b = std::rand()/(RAND_MAX/255);

        QColor col = QColor(r, g, b);
        plntcols.push_back(col);
    }

    // create image corresponding to MapSim
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // canopies
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
            {
                for(int p = 1; p < 10; p++)
                    for(int q = 1; q < 10; q++)
                        visimg->setPixelColor(x*10+p, y*10+q, plntcols[get(x, y)->canopies[i].idx]);
            }

            // roots
            /*
            for(int i = 0; i < (int) get(x, y)->roots.size(); i++)
            {

            }*/
        }
}

bool MapSimCell::unitTests(QImage * visimg)
{
    bool valid = true;
    std::vector<SimPlant> testplnts;
    SimPlant plnt;

    setDim(50, 50, 5);

    // test simple placement of single plant, at domain center reaching to boundary
    inscribe(0, 5.0f, 5.0f, 5.0f, 2.0f);
    plnt.height = 10.0f;
    testplnts.push_back(plnt);

    // add shorter plant in center
    inscribe(1, 6.15f, 4.15f, 2.0f, 1.0f);
    plnt.height = 5.0f;
    testplnts.push_back(plnt);

    // grow shorter plant
    expand(1, 6.15f, 4.15f, 2.0f, 1.0f, 3.0f, 1.0f);
    visualize(visimg, &testplnts);

    PlntInCell p;
    p.idx = 2;
    get(0,0)->canopies.push_back(p);
    get(0,0)->canopies.push_back(p);

    if(valid)
        valid = validate(&testplnts);
    cerr << "validity status = " << valid << endl;

    return valid;
}

bool MapSimCell::validate(std::vector<SimPlant> *plnts)
{
    bool valid = true;

    // a given plant index must only appear once per cell
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // canopies
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
                for(int j = i+1; j < (int) get(x, y)->canopies.size(); j++)
                    if(get(x, y)->canopies[i].idx == get(x,y)->canopies[j].idx)
                    {
                        valid = false;
                        cerr << "MapSimCell validity: duplicate canopy index " << get(x,y)->canopies[i].idx << " at position " << i << " and " << j << endl;
                    }

            // roots
            for(int i = 0; i < (int) get(x, y)->roots.size(); i++)
                for(int j = i+1; j < (int) get(x, y)->roots.size(); j++)
                    if(get(x, y)->roots[i].idx == get(x,y)->roots[j].idx)
                    {
                        valid = false;
                        cerr << "MapSimCell validity: duplicate root index " << get(x,y)->roots[i].idx << " at position " << i << " and " << j << endl;
                    }

        }

    // static plants do not appear in the canopy list
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // canopies
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
                    if((* plnts)[get(x, y)->canopies[i].idx].state == PlantSimState::STATIC)
                    {
                        valid = false;
                        cerr << "MapSimCell validity: static plant " << get(x,y)->canopies[i].idx << " in canopy list at position " << i << endl;
                    }
        }

    // sorted is true for all canopy and root lists with less than 2 elements
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // canopies
            if((int) get(x,y)->canopies.size() < 2 && !get(x,y)->canopysorted)
            {
                valid = false;
                cerr << "MapSimCell validity: canopy list of less than 2 marked as unsorted in canopy list at position " << x << " " << y << endl;
            }

            // roots
            if((int) get(x,y)->roots.size() < 2 && !get(x,y)->rootsorted)
            {
                valid = false;
                cerr << "MapSimCell validity: canopy list of less than 2 marked as unsorted in root list at position " << x << " " << y << endl;
            }
        }

    return valid;
}


////
// Simulation
///
void Simulation::initSim(int dx, int dy, int subcellfactor)
{
    simcells.setDim(dx*subcellfactor, dy*subcellfactor, subcellfactor);
    for(int m = 0; m < 12; m++)
    {
        MapFloat sun, mst, sunland;
        sun.setDim(dx, dy);
        mst.setDim(dx, dy);
        sunland.setDim(dx, dy);
        sunlight.push_back(sun);
        landsun.push_back(sunland);
        moisture.push_back(mst);
        temperature.push_back(0.0f);
        cloudiness.push_back(0.0f);
        rainfall.push_back(0.0f);
    }
    slope.setDim(dx, dy);
    sunsim = new SunLight();
    dice = new DiceRoller(0,1000);
    time = 0.0f;
}

void Simulation::delSim()
{
    simcells.delMap();
    for(int m = 0; m < 12; m++)
    {
        sunlight[m].delMap();
        moisture[m].delMap();
    }
    plntpop.clear();
    temperature.clear();
    cloudiness.clear();
    rainfall.clear();
    slope.delMap();
    if(sunsim) delete sunsim;
    time = 0.0f;
}

bool Simulation::death(int pind, float stress)
{
    bool dead;

    // age and stress factors are combined using a probabilistic apprach
    // use constant background mortaility to model age effects
    // use bioclamatic envelope and a carbohydrate pool to model stress effects

    // background mortality
    float ageFactor = 1.0f / biome->getPFType(plntpop[pind].pft)->maxage;
    float age = 1.0f - pow(0.05f, ageFactor);

    // stress mortality. stress score is remainder after effects of carbohydrate pool accounted for
    float p = 1000.0f * (age + stress);

    // test against a uniform random variable in [0,1000]
    dead = dice->generate() < (int) p;
    if(dead) // if plant has died change its state
        plntpop[pind].state = PlantSimState::DEAD;
    return dead;
}

void Simulation::growth(int pind, float vitality)
{
    PFType * pft;

    if(vitality > 0.0f) // also check month range to see if it falls in growing season
    {
        // apply growth equation for particular pft moderated by vitality
        pft = biome->getPFType(plntpop[pind].pft);
        float maxgrowth = pft->grow_m * (pft->grow_c1 + exp(pft->grow_c2)) / (float) (5 * pft->grow_months); // scale from 5 years down to 1 month
        plntpop[pind].height += vitality * maxgrowth;

        // use allometries for canopy and root
        plntpop[pind].canopy = exp(pft->alm_c1 + pft->alm_m * log(plntpop[pind].height)); // r = e ** (c1 + m ln(h))
        plntpop[pind].root = plntpop[pind].canopy * pft->alm_rootmult;
    }
}

void Simulation::simStep(int month)
{
    // traverse all cells contributing moisture and sunlight to plant pool
    // simcells.traverse(plntpop);

    // calculate viability
    for(int p = 0; p < (int) plntpop.size(); p++)
    {
        float sun, wet, temp, slope, str;
        float pool, stress = 0.0f, vitality = 0.0f;

        sun = plntpop[p].sunlight / (float) plntpop[p].sunlightcnt; // average sunlight
        wet = plntpop[p].water / (float) plntpop[p].watercnt; // average moisture
        temp = getTemperature(plntpop[p].gx, plntpop[p].gy, month);
        slope = getSlopeMap()->get(plntpop[p].gx, plntpop[p].gy);
        str = biome->viability(plntpop[p].pft, sun, wet, temp, slope);

        // account for plant reserve pool
        pool = plntpop[p].reserves+str;
        if(pool < 0.0f) // potential death due to stress
        {
            stress = -1.0f * pool;
            pool = 0.0f;
        }
        else if(pool > reservecapacity) // reserves are full so growth is possible
        {
            vitality = pool - reservecapacity;
            pool = reservecapacity;
        }
        plntpop[p].reserves = pool;

        // run growth, death, seeding processes
        if(!death(p, stress)) // check for death from old age or stress
        {
            // use vitality to determine growth based on allometry
            // but check if this falls in the growing season
            growth(p, vitality);

            plntpop[p].age += 1;
        }
    }
}

Simulation::Simulation(Terrain * terrain, Biome * simbiome, int subcellfactor)
{
    int dx, dy;

    set_terrain(terrain);
    biome = simbiome;
    ter->getGridDim(dx, dy);
    initSim(dx, dy, subcellfactor);
    calcSlope();
    set_rocks();
}

void Simulation::set_terrain(Terrain *terrain)
{
    ter = terrain;
    calcSlope();	// all we need for slopemap is terrain, so we can compute it so long...?
}

/*
 *
 * calculates where there should be rocks, based on slope.
 * REQUIRED: slope map
 */

void Simulation::set_rocks()
{
    int tx, ty;
    int dx, dy;
    slope.getDim(dx, dy);
    ter->getGridDim(tx, ty);
    if (dx * dy <= 0 || dx != tx || dy != ty)
    {
        throw std::runtime_error("Slope map needs to be computed first before the rock map can be computed");
    }

    rocks.setDim(dx, dy);

    for (int y = 0; y < dy; y++)
    {
        for (int x = 0; x < dx; x++)
        {
            if (slope.get(x, y) > 40.0f)
                rocks.set(x, y, 1.0f);
            else
                rocks.set(x, y, 0.0f);
        }
    }
}

MapFloat * Simulation::get_rocks()
{
    return &rocks;
}


void Simulation::calcSunlight(GLSun * glsun, int minutestep, int nsamples)
{
    Timer t;

    t.start();
    sunsim->setLatitude(ter->getLatitude());
    sunsim->setNorthOrientation(Vector(0.0f, 0.0f, -1.0f));
    sunsim->setTerrainDimensions(ter);

    MapFloat diffusemap;
    std::vector<float> sunhours;
    glsun->bind();
    sunsim->projectSun(ter, sunlight, glsun, sunhours, minutestep);
    sunsim->diffuseSun(ter, &diffusemap, glsun, nsamples);
    sunsim->mergeSun(sunlight, &diffusemap, cloudiness, sunhours);
    t.stop();
    cerr << "SUNLIGHT SIMULATION TOOK " << t.peek() << "s IN TOTAL" << endl;
}

/*
void Simulation::calc_adaptsun(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata)
{
    int dx, dy;
    average_adaptsun.getDim(dx, dy);
    {
        int lx, ly;
        average_landsun.getDim(lx, ly);
        if (dx != lx || dy != ly)
        {
            average_adaptsun.setDim(lx, ly);
            dx = lx, dy = ly;
        }
    }

    float *datbegin = average_adaptsun.data();
    memcpy(datbegin, average_landsun.data(), sizeof(float) * dx * dy);

    auto trim = [](int &v, int dim) { if (v < 0) v = 0; if (v >= dim) v = dim - 1; };

    for (auto &t : trees)
    {
        int sx = (t.x - t.radius);
        int ex = (t.x + t.radius);
        int sy = (t.y - t.radius);
        int ey = (t.y + t.radius);

        trim(sx, dx);
        trim(ex, dx);
        trim(sy, dy);
        trim(ey, dy);

        int maxradsq = t.radius * t.radius;
        for (int y = sy; y <= ey; y++)
        {
            for (int x = sx; x < ex; x++)
            {
                int d = (t.x - x) * (t.x - x) + (t.y - y) * (t.y - y);
                if (d <= maxradsq)
                {
                    float alpha = cdata.canopy_and_under_species.at(t.species).alpha;
                    average_adaptsun.set(x, y, average_adaptsun.get(x, y) * (1.0f - alpha));
                }
            }
        }
    }
}
*/

void Simulation::calc_adaptsun(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float rw, float rh)
{
    int dx, dy;
    average_adaptsun.getDim(dx, dy);
    {
        int lx, ly;
        average_landsun.getDim(lx, ly);
        if (dx != lx || dy != ly)
        {
            average_adaptsun.setDim(lx, ly);
            dx = lx, dy = ly;
        }
    }
    float *datbegin = average_adaptsun.data();
    memcpy(datbegin, average_landsun.data(), sizeof(float) * dx * dy);

    auto alphamap = create_alphamap(trees, cdata, rw, rh);

    for (int y = 0; y < dy; y++)
    {
        for (int x = 0; x < dx; x++)
        {
            average_adaptsun.set(x, y, average_adaptsun.get(x, y) * (1.0f - alphamap.get(x, y)));
        }
    }
}

int Simulation::inscribe_alpha(const basic_tree &plnt, float plntalpha, ValueGridMap<float> &alphamap)
{
    const xy<int> gridcoords = alphamap.togrid_safe(plnt.x, plnt.y);
    const xy<int> gridstart = alphamap.togrid_safe(plnt.x - plnt.radius, plnt.y - plnt.radius);
    const xy<int> gridend = alphamap.togrid_safe(plnt.x + plnt.radius, plnt.y + plnt.radius);
    const int gridrad = alphamap.togrid_safe(plnt.radius, plnt.radius).x;

    float avggdist = 0.0f;
    int nwrite = 0;
    for (int x = gridstart.x; x <= gridend.x; x++)
    {
        for (int y = gridstart.y; y <= gridend.y; y++)
        {
            float griddist = (x - gridcoords.x) * (x - gridcoords.x) + (y - gridcoords.y) * (y - gridcoords.y);
            if (griddist <= ((float) gridrad * gridrad) && alphamap.get(x, y) < plntalpha)
            {
                alphamap.set(x, y, plntalpha);
                nwrite++;
                avggdist += sqrt(griddist);
            }
        }
    }
    /*
    if (nwrite == 0)
    {
        std::cout << "no writes made. gridrad: " << gridrad << ", ";
        std::cout << "avg griddist: " << avggdist << ", ";
        std::cout << "gridstart: " << gridstart.x << ", " << gridstart.y << std::endl;
    }
    */
    return nwrite;
}

ValueGridMap<float> Simulation::create_alphamap(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float rw, float rh)
{
    ValueGridMap<float> alphamap;
    alphamap.setDim(average_landsun);
    alphamap.setDimReal(rw, rh);
    alphamap.fill(0.0f);
    for (const auto &plnt : trees)
    {
        float alpha = cdata.canopy_and_under_species.at(plnt.species).alpha;
        int nwrite = inscribe_alpha(plnt, alpha, alphamap);
        //if (nwrite == 0)
        //    std::cout << "Number of cells written with alpha " << alpha << ": " << nwrite << std::endl;
    }

    int gw, gh;
    alphamap.getDim(gw, gh);
    int smrad = 15;

    auto alphamap_copy = alphamap;

    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            int count = 0;
            float sum = 0.0f;
            int sx = common_funcs::trimret(0, gw - 1, x - smrad);
            int ex = common_funcs::trimret(0, gw - 1, x + smrad);
            int sy = common_funcs::trimret(0, gh - 1, y - smrad);
            int ey = common_funcs::trimret(0, gh - 1, y + smrad);

            for (int cx = sx; cx <= ex; cx++)
            {
                for (int cy = sy; cy <= ey; cy++)
                {
                    int gdist = (cx - x) * (cx - x) + (cy - y) * (cy - y);
                    if (gdist <= smrad * smrad)
                    {
                        count++;
                        sum += alphamap_copy.get(cx, cy);
                    }
                }
            }
            if (count > 0)
            {
                sum /= count;
                alphamap.set(x, y, sum);
            }
            else	// impossible for count == 0, so throw error
            {
                throw std::runtime_error("In Simulation::create_alphamap, count is zero in smoothing. It cannot be. This indicates a bug");
            }
        }
    }

    return alphamap;
}

void Simulation::calcSunlightSelfShadowOnly(GLSun *glsun)
{
    Timer t;

    t.start();
    sunsim->setLatitude(ter->getLatitude());
    sunsim->setNorthOrientation(Vector(0.0f, 0.0f, -1.0f));
    sunsim->setTerrainDimensions(ter);

    MapFloat diffusemap;
    std::vector<float> sunhours;
    //glsun->bind();
    sunsim->projectSunSelfShadowOnly(ter, sunlight, glsun, sunhours, 30);
    t.stop();
    cerr << "SUNLIGHT SIMULATION TOOK " << t.peek() << "s IN TOTAL" << endl;
}

void Simulation::calcMoisture()
{
    Timer t;
    MoistureSim wet;

    t.start();

     // alpine params
     // wet.simSoilCycle(ter, &slope, rainfall, 10.0f, (float) slope, 0.25f, 120.0f, 200.0f, 15000.0f, moisture);

     // savannah params
     // wet.simSoilCycle(ter, &slope, rainfall, 20.0f, (float) slope, 0.3f, 180.0f, 300.0f, 120000.0f, moisture);

     // canyon params
     // wet.simSoilCycle(ter, &slope, rainfall, 5.0f, (float) slope, 0.5f, 50.0f, 100.0f, 10000.0f, moisture);

     // wet canyon params
     // wet.simSoilCycle(ter, &slope, rainfall, 5.0f, (float) slope, 0.5f, 100.0f, 180.0f, 5000.0f, moisture);

     // med params
     // wet.simSoilCycle(ter, &slope, rainfall, 5.0f, (float) slope, 0.25f, 100.0f, 180.0f, 5000.0f, moisture);

    wet.simSoilCycle(ter, &slope, rainfall, biome->slopethresh, biome->slopemax, biome->evaporation,
                     biome->runofflevel, biome->soilsaturation, biome->waterlevel, moisture);
    t.stop();
    cerr << "MOISTURE SIMULATION TOOK " << t.peek() << "s IN TOTAL" << endl;
}

 void Simulation::calc_average_monthly_map(std::vector<MapFloat> &mmap, MapFloat &avgmap)
{
    int sizex, sizey;
    mmap[0].getDim(sizex, sizey);
    avgmap.setDim(sizex, sizey);
    avgmap.fill(0.0f);
    for (int m = 0; m < 12; m++)
    {
        for (int x = 0; x < sizex; x++)
        {
            for (int y = 0; y < sizey; y++)
            {
                avgmap.set(x, y, avgmap.get(x, y) + mmap[m].get(x, y));
            }
        }
    }
    for (int x = 0; x < sizex; x++)
    {
        for (int y = 0; y < sizey; y++)
        {
            avgmap.set(x, y, avgmap.get(x, y) / 12.0f);
        }
    }
}

void Simulation::calc_average_sunlight_map()
{
    calc_average_monthly_map(sunlight, average_sunlight);
}

void Simulation::calc_average_landsun_map()
{
    calc_average_monthly_map(landsun, average_landsun);
}

void Simulation::calc_average_moisture_map()
{
    calc_average_monthly_map(moisture, average_moisture);
}

void Simulation::calc_temperature_map()
{
    int xsize, ysize;
    ter->getGridDim(xsize, ysize);
    temperate_mapfloat.setDim(xsize, ysize);
    float avg_temp = 0.0f;
    for (auto &t : temperature)
    {
        avg_temp += t;
    }
    avg_temp /= 12.0f;
    for (int x = 0; x < xsize; x++)
    {
        for (int y = 0; y < ysize; y++)
        {
            float val = avg_temp - ter->getHeight(x, y) / 1000.0f * lapserate;
            temperate_mapfloat.set(x, y, val);
        }
    }
}

void Simulation::copy_temperature_map(const ValueGridMap<float> &tempmap)
{
    int xsize, ysize;
    tempmap.getDim(xsize, ysize);
    temperate_mapfloat.setDim(xsize, ysize);
    memcpy(temperate_mapfloat.data(), tempmap.data(), sizeof(float) * xsize * ysize);
}

bool Simulation::readSun(std::string filename)
{
    bool result = readMonthlyMap(filename, sunlight);
    if (result)
        calc_average_sunlight_map();
    return result;
}

bool Simulation::readLandscapeSun(std::string filename)
{
    bool result = readMonthlyMap(filename, landsun);
    if (result)
        calc_average_landsun_map();		// this is probably obsolete and slows down the interface - ditto for readSun
    return result;
}

void Simulation::copy_map(const ValueGridMap<float> &srcmap, abiotic_factor f)
{
    MapFloat *dest;
    if (f == abiotic_factor::SUN)
    {
        dest = &average_landsun;
        average_adaptsun.setDim(srcmap);
    }
    else if (f == abiotic_factor::MOISTURE)
    {
        dest = &average_moisture;
    }
    else return;
    int xsize, ysize;
    srcmap.getDim(xsize, ysize);
    dest->setDim(xsize, ysize);
    memcpy(dest->data(), srcmap.data(), sizeof(float) * xsize * ysize);

    if (f == abiotic_factor::SUN)
    {
        memcpy(average_adaptsun.data(), srcmap.data(), sizeof(float) * xsize * ysize);
    }
}

bool Simulation::readMoisture(std::string filename)
{
    bool result = readMonthlyMap(filename, moisture);
    calc_average_moisture_map();
    return result;
}

bool Simulation::readMonthlyMap(std::string filename, std::vector<MapFloat> &monthly)
{
    float val;
    ifstream infile;
    int gx, gy, dx, dy;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> gx >> gy;
        ter->getGridDim(dx, dy);
        if((gx != dx) || (gy != dy))
            cerr << "Error Simulation::readMonthlyMap: map dimensions do not match terrain" << endl;

        for(int m = 0; m < 12; m++)
            monthly[m].setDim(gx, gy);

        for (int y = 0; y < gy; y++)
            for (int x = 0; x < gx; x++)
                for(int m = 0; m < 12; m++)
                {
                    infile >> val;
                    monthly[m].set(x, y, val);
                }
        infile.close();
        return true;
    }
    else
    {
        cerr << "Error Simulation::readMonthlyMap: unable to open file" << filename << endl;
        return false;
    }
}

bool Simulation::writeMonthlyMap(std::string filename, std::vector<MapFloat> &monthly)
{
    int gx, gy;
    ofstream outfile;
    monthly[0].getDim(gx, gy);

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << gx << " " << gy << " " << 0.9144 << endl;
        for (int y = 0; y < gy; y++)
            for (int x = 0; x < gx; x++)
                for(int m = 0; m < 12; m++)
                    outfile << monthly[m].get(x, y) << " ";

        outfile << endl;
        outfile.close();
        return true;
    }
    else
    {
        cerr << "Error Simulation::writeMonthlyMap:unable to open file " << filename << endl;
        return true;
    }

}

bool Simulation::readClimate(std::string filename)
{
    float elv, val;
    ifstream infile;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> elv;

        // temperature values
        for(int m = 0; m < 12; m++)
        {
            infile >> val;
            temperature[m] = val - elv / 1000.0f * lapserate;
        }

        // sky clarity
        for(int m = 0; m < 12; m++)
        {
            infile >> val;
            cloudiness[m] = 1.0f - val;
        }

        // rainfall
        for(int m = 0; m < 12; m++)
        {
            infile >> val;
            rainfall[m] = val;
        }

        infile.close();

        return true;
    }
    else
    {
        cerr << "Error Simulation::readClimate: unable to open file" << filename << endl;
        return false;
    }
}

float Simulation::getTemperature(int x, int y, int mth)
{
    return temperature[mth] - ter->getHeight(x, y) / 1000.0f * lapserate;
}

void Simulation::calcSlope()
{
    int dx, dy;
    Vector up, n;

    // slope is dot product of terrain normal and up vector
    up = Vector(0.0f, 1.0f, 0.0f);
    ter->getGridDim(dx, dy);
    slope.setDim(dx, dy);
    slope.fill(0.0f);
    for(int x = 0; x < dx; x++)
        for(int y = 0; y < dy; y++)
        {
            ter->getNormal(x, y, n);
            float rad = acos(up.dot(n));
            float deg = RAD2DEG * rad;

            slope.set(y, x, deg);
            //slope.set(x, y, deg);
        }
}

void Simulation::importCanopy(EcoSystem * eco)
{
    std::vector<Plant> plnts;

    eco->pickAllPlants(ter); // must gather plants before vectorizing
    plntpop.clear();

    // iterate over plant functional types
    for(int pft = 0; pft < biome->numPFTypes(); pft++)
    {
        eco->getPlants()->vectoriseByPFT(pft, plnts);
        for(int p = 0; p < (int) plnts.size(); p++)
        {
            SimPlant sp;
            sp.state = PlantSimState::STATIC;
            sp.age = 0.0f; // PFTSTATS: derive from height via pft allometry
            sp.pos = plnts[p].pos;
            sp.height = plnts[p].height;
            sp.canopy = plnts[p].canopy;
            sp.root = 1.0f; // PFTSTATS: derive from height via pft allometry
            sp.reserves = reservecapacity;
            sp.col = plnts[p].col;
            sp.pft = pft;
            sp.water = 0.0f;
            sp.sunlight = 0.0f;
            plntpop.push_back(sp);
        }
    }
}

void Simulation::exportUnderstory(EcoSystem * eco)
{
    for(int n = 0; n < 2; n++)
    {
        PlantGrid * outplnts = eco->getNiche(n);
        outplnts->clear();

        // canopy into niche 0, understory into niche 1
        for(int p = 0; p < (int) plntpop.size(); p++)
        {
            Plant op;
            bool place;
            if(n == 0) // only include static plants in niche 0
                place = (plntpop[p].state == PlantSimState::STATIC);
            else // only include live plants in niche 1
                place = (plntpop[p].state == PlantSimState::ALIVE);

            if(place)
            {
                op.canopy = plntpop[p].canopy;
                op.col = plntpop[p].col;
                op.height = plntpop[p].height;
                op.pos = plntpop[p].pos;
                outplnts->placePlant(ter, plntpop[p].pft, op);
            }
        }

    }

    eco->pickAllPlants(ter); // gather all plants for display
}

bool Simulation::writeAssignSun(std::string filename)
{
    int gx, gy;
    ofstream outfile;
    sunlight[0].getDim(gx, gy);

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        // sunlight
        outfile << gx << " " << gy << " " << endl;
        for (int x = 0; x < gx; x++)
            for (int y = 0; y < gy; y++)
                for(int m = 1; m < 2; m++)
                    outfile << sunlight[m].get(x, y) << " ";
        outfile << endl;

        // block plant placement
        outfile << (gx / 3 * gy / 3) << endl;
        cerr << "num trees = " << (gx / 3 * gy / 3) << endl;
        int countt = 0;
        for (int x = 0; x < gx; x+= 3)
            for (int y = 0; y < gy; y+= 3)
            {
                // random position and size
                int rx = x+dice->generate() % 3;
                int ry = y+dice->generate() % 3;
                if(rx >= gx)
                    rx = gx-1;
                if(ry >= gy)
                    ry = gy-1;
                int re = dice->generate() % 8 + 1;
                outfile << rx << " " << ry << " " << re << endl;
                countt++;
            }
        cerr << "actual num trees = " << countt << endl;


        outfile << endl;
        outfile.close();
        return true;
    }
    else
    {
        cerr << "Error Simulation::writeMonthlyMap:unable to open file " << filename << endl;
        return true;
    }


}


void Simulation::pickInfo(int x, int y)
{
    cerr << "Sunlight (Hrs): ";
    //for(int m = 0; m < 12; m++)
    //    cerr << sunlight[m].get(x, y) << " ";
    cerr << endl;
    cerr << "Slope (rad): " << slope.get(x, y) << endl;
    cerr << "Soil Moisture (mm): ";
    for(int m = 0; m < 12; m++)
        cerr << moisture[m].get(x, y) << " ";
    cerr << endl;
    cerr << "Temperature (C): ";
    for(int m = 0; m < 12; m++)
        cerr << getTemperature(x, y, m) << " ";
    cerr << endl;
}


void Simulation::simulate(int delYears)
{
    for(int y = 0; y < delYears; y++)
        for(int m = 0; m < 12; m++)
            simStep(m);
}




