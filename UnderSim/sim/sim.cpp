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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <utility>
#include "sim.h"
#include <sys/stat.h>

#include <QLabel>

// #define TESTING

////
// MapSimCell
///


void MapSimCell::delMap()
{
    for(int c = 0; c < gx*gy; c++) // empty cells are always sorted
    {
        smap[c].canopies.clear();
        smap[c].roots.clear();
        smap[c].seedbank.clear();
    }
    smap.clear();
}

void MapSimCell::initMap()
{
    smap.clear();
    smap.resize(gx*gy);
    for(int c = 0; c < gx*gy; c++) // empty cells are always sorted
    {
        smap[c].canopysorted = true;
        smap[c].rootsorted = true;
        smap[c].growing = false;
        smap[c].available = true;
        smap[c].seed_chance = 0.0f;
    }
    resetSeeding();

}

void MapSimCell::resetSeeding()
{
    for(int c = 0; c < gx*gy; c++) // empty cells are always sorted
    {
        smap[c].leftoversun = 0.0f;
        smap[c].leftoverwet = 0.0f;
    }
}

void MapSimCell::init_countmaps(const std::set<int> &species, int gw, int gh, int tw, int th)
{
    closest_distances.clear();
    species_counts.clear();
    for (auto specidx : species)
    {
        closest_distances.emplace(std::make_pair<int, ValueGridMap<float> >(std::move(specidx), ValueGridMap<float>(gw, gh, tw, th)));
        closest_distances.at(specidx).fill(0.0f);
        species_counts.emplace(std::make_pair<int, ValueGridMap<int> >(std::move(specidx), ValueGridMap<int>(gw, gh, tw, th)));
        species_counts.at(specidx).fill(0);
    }
}

void MapSimCell::toTerGrid(int mx, int my, int &tx, int &ty, Terrain * ter)
{
    int dx, dy;

    tx = mx / step;
    ty = my / step;

    // upper bounds check
    ter->getGridDim(dx, dy);
    tx = std::min(tx, dx-1);
    ty = std::min(ty, dy-1);
}

bool MapSimCell::notInSeedbank(int sbidx, int x, int y)
{
    bool found = false;
    int i = 0;

    while(!found && i < (int) get(x,y)->seedbank.size())
    {
        found = (get(x,y)->seedbank[i] == sbidx);
        i++;
    }

    return !found;
}

// side effects:
// - add seed index sbidx to each cell in the seedbank, in rcanopy radius around px, py
void MapSimCell::inscribeSeeding(int sbidx, int spec_idx, float px, float py, float rcanopy, Terrain * ter)
{
    float grx, gry, grcanopy, rmax, distsq, grcanopysq;

    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);
    grcanopy = convert(rcanopy);

    grcanopysq = grcanopy * grcanopy;
    rmax = grcanopy * radius_mult;

    float rter = ter->toWorld(rcanopy);
    float real_rcanopysq = rter * rter;

    // bounding box around circle
    for(int x = (int) (grx-rmax); x <= (int) (grx+rmax); x++)
        for(int y = (int) (gry-rmax); y <= (int) (gry+rmax); y++)
        {
            if(ingrid(x,y))
            {
                float delx = convert_to_tergrid((float) x-grx);
                float dely = convert_to_tergrid((float) y-gry);
                vpPoint temp = ter->toWorld(delx, dely, 0.0f);
                delx = temp.x;
                dely = temp.z;
                distsq = delx*delx + dely*dely;

                if (distsq <= real_rcanopysq)
                {
                    // not already in seedbank and under chm
                    if(notInSeedbank(sbidx, x, y) && get(x, y)->growing)
                    {
                        // add with a probabliity that depends on distance
                        float d = distsq / real_rcanopysq;
                        float p = 1.0f;
                        if(d > 0.6f)
                            p -= (d - 0.6f) / 0.4f;

                        if((float) dice->generate() / 1000.0f < p)
                            get(x,y)->seedbank.push_back(sbidx);
                    }
                    float terdist = convert_to_tergrid(sqrt(distsq));
                    int tx = convert_to_tergrid(x) + 1e-5;
                    int ty = convert_to_tergrid(y) + 1e-5;
                    if (terdist < closest_distances.at(spec_idx).get(tx, ty))
                    {
                        closest_distances.at(spec_idx).set(tx, ty, terdist);
                    }
                    int curr_count = species_counts.at(spec_idx).get(tx, ty);
                    species_counts.at(spec_idx).set(tx, ty, curr_count + 1);
                }
            }
        }
}


// side effects:
// - set 'growing' and 'available' attributes of SimCell objects
// - establish what plants' canopies and roots intersect each SimCell
void MapSimCell::inscribe(std::list<SimPlant *>::iterator plntidx, float px, float py, float rcanopy, float rroot, bool isStatic, Terrain *ter, Simulation * sim)
{
    float grx, gry, grcanopy, grroot, rmax, distsq, grcanopysq, grrootsq;

    // assumes that plant index is not already inscribed in the simulation grid

    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);

    // also convert from diameter to radius
    grcanopy = convert(rcanopy/2.0f);
    grroot = convert(rroot/2.0f);

    grcanopysq = grcanopy * grcanopy;
    grrootsq = grroot * grroot;
    rmax = fmax(grroot * radius_mult, grcanopy * radius_mult);

    vpPoint pt = ter->toWorld(rcanopy/2.0f, rcanopy/2.0f, rcanopy/2.0f);
    float real_rcanopysq = pt.x * pt.x;
    pt = ter->toWorld(rroot/2.0f, rroot/2.0f, rroot/2.0f);
    float real_rrootsq = pt.x * pt.x;

    float tercellsize = ter->getCellExtent();

    float distsqmax = fmax(real_rcanopysq, real_rrootsq) * radius_mult * radius_mult;
    float distmax = sqrt(distsqmax);
    float distsqmax_error = 2.0f * (distmax + tercellsize) * (distmax + tercellsize);

    get(grx, gry)->available = false;
    get(grx, gry)->growing = false;

    for(int x = (int) (grx-rmax); x <= (int) (grx+rmax); x++)
        for(int y = (int) (gry-rmax); y <= (int) (gry+rmax); y++)
        {
            PlntInCell pic;

            if(ingrid(x,y))		// cell is in grid
            {
                pic.hght = 0.0f; // height will be instantiated later on demand
                pic.plnt = (* plntidx);

                float delx = convert_to_tergrid((float) x-grx);
                float dely = convert_to_tergrid((float) y-gry);
                vpPoint temp = ter->toWorld(delx, dely, 0.0f);
                delx = temp.x;
                dely = temp.z;
                distsq = delx*delx + dely*dely;

                assert(distsq < distsqmax_error);
                if (distsq > distsqmax) distsq = distsqmax;

                if (!get(x, y)->available)
                    assert(!get(x, y)->growing);

                if (!get(x, y)->available)   //if cell is covered by any canopy tree trunks, ignore
                {
                    get(x, y)->growing = false;
                    continue;
                }
                if(isStatic) // under a canopy
                {
                    if (distsq > 2.25f + 1e-5)
                    {
                        //float expval = std::exp(-distsq / (2 * grcanopysq));
                        //get(x, y)->seed_chance += (sim->sparams.seedprob * 10) * expval;
                        get(x, y)->seed_chance += sim->sparams.seedprob * (1 - (distsq / (distsqmax)));
                        get(x,y)->growing = true;
                    }
                    else
                    {
                        get(x, y)->available = false;
                        get(x, y)->growing = false;
                    }
                }

                //if(distsq <= grcanopysq)
                if (distsq <= real_rcanopysq)
                {
                    get(x,y)->canopies.push_back(pic);
                    if((int) get(x,y)->canopies.size() > 1)
                        get(x,y)->canopysorted = false;
                }
                // if(distsq <= grrootsq)
                if(distsq <= real_rrootsq)
                {
                    get(x,y)->roots.push_back(pic);
                    if((int) get(x,y)->roots.size() > 1)
                        get(x,y)->rootsorted = false;
                }
            }
        }
}

void MapSimCell::expand(std::list<SimPlant *>::iterator plntidx, float px, float py, float prevrcanopy, float prevrroot, float newrcanopy, float newrroot)
{
    float grx, gry, gprevrcanopy, gprevrroot, gnewrcanopy, gnewrroot, rmax, distsq, gprevrcanopysq, gprevrrootsq, gnewrcanopysq, gnewrrootsq;

    if(prevrcanopy > newrcanopy)
        cerr << "EXPAND: CANOPY INCORRECTLY SHRUNK" << endl;
    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);
    gprevrcanopy = convert(prevrcanopy/2.0f);
    gprevrroot = convert(prevrroot/2.0f);
    gnewrcanopy = convert(newrcanopy/2.0f);
    gnewrroot = convert(newrroot/2.0f);

    gprevrcanopysq = gprevrcanopy * gprevrcanopy;
    gprevrrootsq = gprevrroot * gprevrroot;
    gnewrcanopysq = gnewrcanopy * gnewrcanopy;
    gnewrrootsq = gnewrroot * gnewrroot;
    rmax = fmax(gnewrroot, gnewrcanopy);

    for(int x = (int) (grx-rmax); x <= (int) (grx+rmax); x++)
        for(int y = (int) (gry-rmax); y <= (int) (gry+rmax); y++)
        {
            PlntInCell pic;

            if(ingrid(x,y))
            {
                pic.hght = 0.0f; // height will be instantiated later on demand
                pic.plnt = (* plntidx);

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

void MapSimCell::uproot(std::list<SimPlant *>::iterator plntidx, float px, float py, float rcanopy, float rroot, Terrain * ter)
{
    float grx, gry, grcanopy, grroot, rmax, distsq, grcanopysq, grrootsq;
    std::vector<PlntInCell>::iterator cidx, delidx;

    // convert to sim grid coordinates
    grx = convert(px);
    gry = convert(py);
    grcanopy = convert(rcanopy/2.0f);
    grroot = convert(rroot/2.0f);

    grcanopysq = grcanopy * grcanopy;
    grrootsq = grroot * grroot;
    rmax = fmax(grroot * radius_mult, grcanopy * radius_mult)+2.0f;

    for(int x = (int) (grx-rmax); x <= (int) (grx+rmax); x++)
        for(int y = (int) (gry-rmax); y <= (int) (gry+rmax); y++)
        {
            if(ingrid(x,y))		// cell is in grid
            {

                // find and remove plntidx from canopies
                // assumes it appears only once
                bool found = false;
                for(cidx = get(x,y)->canopies.begin(); cidx != get(x,y)->canopies.end(); cidx++)
                    if(cidx->plnt == (* plntidx))
                    {
                        if(found)
                            cerr << "REMOVE CANOPY: MULTIPLE FINDS ON PLANT CANOPY" << endl;
                        found = true;
                        delidx = cidx;
                    }
                if(found)
                    get(x,y)->canopies.erase(delidx);


                // find and remove plntidx from roots
                // assumes it appears only once
                found = false;
                for(cidx = get(x,y)->roots.begin(); cidx != get(x,y)->roots.end(); cidx++)
                   if(cidx->plnt == (* plntidx))
                   {
                       if(found)
                          cerr << "REMOVE ROOT: MULTIPLE FINDS ON PLANT CANOPY" << endl;
                       found = true;
                       delidx = cidx;
                   }
                if(found)
                   get(x,y)->roots.erase(delidx);
            }
        }
}


float cmpHeight(PlntInCell i, PlntInCell j)
{
    return i.hght > j.hght;
}

void MapSimCell::traverse(std::list<SimPlant *> *plntpop, Simulation * sim, Biome * biome, MapFloat * sun, MapFloat * wet, bool seedable)
{
    int tx, ty;

    // clear sunlight and moisture plant pools
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop->begin(); plnt != plntpop->end(); plnt++)
    {
        if((* plnt)->state == PlantSimState::ALIVE)
        {
            (* plnt)->sunlight = 0.0f;
            (* plnt)->water = 0.0f;
            (* plnt)->sunlightcnt = 0;
            (* plnt)->watercnt = 0;
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
                    get(x,y)->canopies[p].hght = get(x,y)->canopies[p].plnt->height;
                std::sort(get(x,y)->canopies.begin(), get(x,y)->canopies.end(), cmpHeight);
                get(x,y)->canopysorted = true;
            }

            toTerGrid(x, y, tx, ty, sim->getTerrain());
            float sunlight = sun->get(tx, ty);
            // traverse trees by height supplying and reducing sunlight
            for(int p = 0; p < (int) smap[flatten(x,y)].canopies.size(); p++)
            {
                PlantSimState pstate = get(x,y)->canopies[p].plnt->state;

                assert(pstate != PlantSimState::DEAD); // dead plants should already be removed from grid
                if(pstate == PlantSimState::DEAD)
                    cerr << "WARNING: DEAD PLANTS IN GRID" << endl;

                if(pstate != PlantSimState::STATIC)
                {
                    // cerr << "sunlight = " << sunlight << " at " << (int) x/step << ", " << (int) y/step << endl;
                    get(x,y)->canopies[p].plnt->sunlight += sunlight;
                    get(x,y)->canopies[p].plnt->sunlightcnt++;
                    // reduce sunlight by alpha of current plant. note reciprocal
                    sunlight *= (1.0f - biome->getAlpha(get(x,y)->canopies[p].plnt->pft));
                }
            }

            // add value to sunlight accumulation for later seeding check
            if(seedable)
                get(x,y)->leftoversun += sunlight;

            // sort root trees by height if necessary
            if(!get(x,y)->rootsorted)
            {
                // update heights
                for(int p = 0; p < (int) get(x,y)->roots.size(); p++)
                    get(x,y)->roots[p].hght = get(x,y)->roots[p].plnt->height;
                std::sort(get(x,y)->roots.begin(), get(x,y)->roots.end(), cmpHeight);
                get(x,y)->rootsorted = true;
            }

            float moisture = wet->get(tx, ty);
            float watershare;
            int livecnt = 0;
            // traverse trees by height (proxy for root depth) supplying and reducing moisture
            for(int p = 0; p < (int) smap[flatten(x,y)].roots.size(); p++)
            {
                PlantSimState pstate = get(x,y)->roots[p].plnt->state;

                if(pstate == PlantSimState::DEAD)
                    cerr << "WARNING: DEAD PLANTS IN GRID" << endl;

                // plants grab flat min
                watershare = fmin(moisture, sim->sparams.moisturedemand);
                //if(pstate == PlantSimState::STATIC)
                //     watershare *= 0.0f;
                // watershare = fmin(moisture, biome->getMinIdeal   dice = new DiceRoller(0,1000);Moisture(get(x,y)->roots[p].plnt->pft));
                get(x,y)->roots[p].plnt->water += watershare;
                get(x,y)->roots[p].plnt->watercnt++;
                moisture -= watershare;
                if(pstate != PlantSimState::STATIC)
                    livecnt++;
            }
            // remainder spread equally
            if(moisture > 0.0f)
            {
                if(livecnt > 0)
                    watershare = moisture / (float) livecnt;
                else
                    watershare = moisture;

                // add share of leftover water to water accumulator for later seeding check
                if(seedable)
                    // get(x,y)->leftoverwet += watershare;
                    get(x,y)->leftoverwet += moisture; // potential water for seed access

                for(int p = 0; p < (int) smap[flatten(x,y)].roots.size(); p++)
                    if(get(x,y)->roots[p].plnt->state != PlantSimState::DEAD)
                        get(x,y)->roots[p].plnt->water += watershare;
            }
    }
}

void MapSimCell::establishSeedBank(std::list<SimPlant *> * plntpop, int plntpopsize, Biome * biome, Terrain * ter)
{
    int p = 0;
    // iterate over every plant, check whether it is a canopy plant and write its species index into the simulation cells
    // in a circle that extends out to a multiple of the canopy radius

    cerr << "Establishing Seedbank" << endl;
    int div = plntpopsize / 5;

    // clear sunlight and moisture plant pools
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop->begin(); plnt != plntpop->end(); plnt++)
    {
        if((* plnt)->state == PlantSimState::STATIC) // denotes a canopy plant
        {
            float x, y, h;
            //std::cout << "Doing subbiome lookup..." << std::endl;
            int sb = biome->getSubBiomeFromSpecies((* plnt)->pft);
            int spec_idx = (* plnt)->pft;
            //std::cout << "getting x, y location..." << std::endl;
            ter->toGrid((* plnt)->pos, x, y, h);

            //std::cout << "inscribing seeding..." << std::endl;
            inscribeSeeding(sb, spec_idx, x, y, ter->toGrid((* plnt)->canopy * seed_radius_mult), ter);
            if (p % div == 0 && p != 0)
            {
                std::cout << int((float) p / (float) plntpopsize * 100) << "% done" << std::endl;
            }
        }
        p++;
    }
}

// testing seeding:
// artificially set values in particular cell
// conversion to and from terrain and mapsimcell grid
// no seeds in non growth areas
// check sub-biome lists correctly compiled over multiple biomes
/*
void MapSimCell::testSeeding(vpPoint pos, Simulation * sim, Terrain * ter, Biome * biome)
{
    std::vector<int> noseed_count(4, 0);
    float x, y, h;
    int grx, gry;
    std::vector<SimPlant  *> plnts;

    // specific point on map
    pos.x = 500.0f; pos.z = 200.0f; pos.y = 0.0f;
    cerr << "picked terrain location = " << pos.x << ", " << pos.z << endl;

    // map to simcell
    ter->toGrid(pos, x, y, h);
    grx = (int) convert(x);
    gry = (int) convert(y);

    // set simcell parameters
    // assumes testing with Sonoma biome
    get(grx, gry)->growing = true;
    get(grx, gry)->leftoversun = 12.0f;
    get(grx, gry)->leftoverwet = 360.0f;
    get(grx, gry)->seedbank.clear();
    // two sub-biomes
    get(grx, gry)->seedbank.push_back(0);
    get(grx, gry)->seedbank.push_back(1);

    // call singleSeed
    singleSeed(grx, gry, &plnts, sim, ter, biome, noseed_count);

    // report final position
    if((int) plnts.size() == 1)
    {
        cerr << "final location = " << plnts[0]->pos.x << ", " << plnts[0]->pos.z << ", terrain y = " << plnts[0]->pos.y << endl;
        cerr << "tree height = " << plnts[0]->height << endl;
        delete plnts[0];
    }
    else
    {
        cerr << "Error MapSimCell::testSeeding: plant not created properly" << endl;
    }
}*/

bool MapSimCell::singleSeed(int x, int y, std::list<SimPlant *> * plntpop, Simulation * sim, Terrain * ter, Biome * biome, std::vector<int> &noseed_count)
{
    // roll dice to see if a seedling could happen, if so check seedbank and assign viability to possible seeds.
    // choose seed randomly weighted by viability

    std::vector<float> adaptvals(4);
    std::vector<float> seedviability;
    std::vector<int> seedspecies;
    float totalstr, cumstr, sun, wet, temp, slope;
    int sidx, dx, dy;
    bool found;
    bool seeded = false;

    ter->getGridDim(dx, dy);

    // cerr << "(" << x << ", " << y << ") " << endl;
    // gather understorey plants
    for(int b = 0; b < (int) get(x,y)->seedbank.size(); b++)
    {
        // UNDO

        SubBiome * sb = biome->getSubBiome(get(x,y)->seedbank[b]);
        for(int u = 0; u < (int) sb->understorey.size(); u++)
            seedspecies.push_back(sb->understorey[u]);
        for(int o = 0; o < (int) sb->canopies.size(); o++)
            seedspecies.push_back(sb->canopies[o]);
       // seedspecies.push_back(11);

    }

    // determine viability of plants
    totalstr = 0.0f;
    for(int s = 0; s < (int) seedspecies.size(); s++)
    {
        float str;
        int tgx, tgy;
        // convert from x, y simcell to tx, ty terrain grid
        toTerGrid(x, y, tgx, tgy, sim->getTerrain());
        // average sunlight and moisture over growing season

        sun = get(x,y)->leftoversun / (float) shortgrowmonths;
        wet = get(x,y)->leftoverwet / (float) shortgrowmonths;

        //sun = sim->getSunlightMap(shortgrowend)->get(tgx, tgy);
        //wet = sim->getMoistureMap(shortgrowend)->get(tgx, tgy);

        temp = sim->getTemperature(tgx, tgy, shortgrowend); // use end of growing season
        slope = sim->getSlopeMap()->get(tgx, tgy);
        str = max(0.0f, biome->viability(seedspecies[s], sun, wet, temp, slope, adaptvals, sim->sparams.viabilityneg));
        //str = 1.0f;

        str /= pow((float) biome->getPFType(seedspecies[s])->maxage, 1.2f);
        totalstr += str;
        seedviability.push_back(totalstr);

        for (int i = 0; i < adaptvals.size(); i++)
        {
            if (adaptvals.at(i) <= 0.0f)
                noseed_count.at(i) += 1;
        }
        // cerr << "  " << str << "(" << totalstr << ")";
    }
    // cerr << endl << "   str = " << totalstr << endl;

#ifdef TESTING
    cerr << "SEEDBANK" << endl;
    for(int s = 0; s < (int) seedspecies.size(); s++)
    {
        cerr << "s: " << seedspecies[s] << " v: " << seedviability[s] << endl;
    }
    cerr << "total viability = " << totalstr << endl;
#endif

    // choose particular seed randomly according to viability
    if(totalstr > 0.0f)
    {
        // first check random suitability of location using totalstr
        // plants will thus be more likely to seed in generally more favourable areas
        // creating

        // / numspecies
        float select = (float) dice->generate() / 10000.0f * totalstr;
        cumstr = 0.0f; found = false; sidx = 0;

        // find particular seed according to random selection
        while(!found)
        {
            cumstr = seedviability[sidx];
            if(select <= cumstr)
            {
                found = true;
            }
            else
            {
                sidx++;
                if(sidx >= (int) seedviability.size())
                    found = true;
            }
        }

#ifdef TESTING
        cerr << "RND GEN" << endl;
        cerr << "rnd number = " << select << " for entry = " << sidx << endl;
#endif

        // cerr << "   sidx = " << sidx << endl;
        // assign height, random position within cell, and other attribues
        if(sidx < (int) seedviability.size())
        {
            SimPlant * sp = new SimPlant;
            float sgx, sgy, rndoff;
            float wx, wy, wz;
            int pft = seedspecies[sidx];
            sp->state = PlantSimState::ALIVE;
            sp->age = 1.0f;

            // random subgrid position within simcell
            rndoff = (float) dice->generate() / 10000.0f;
            sgx = (float) x + rndoff;
            rndoff = (float) dice->generate() / 10000.0f;
            sgy = (float) y + rndoff;
            // convert to terrain position
            sgx /= (float) step; sgy /= (float) step; // terrain grid

            // clamp to terrain grid if necessary
            if(sgx > (float) dx)
                sgx = (float) dx;
            if(sgy > (float) dy)
                sgy = (float) dy;

            sp->pos = ter->toWorld(sgx, sgy, ter->getHeight(sgy, sgx)); // BUG? note inversion in getHeight
            sp->height = biome->growth(pft, 0.0f, seedviability[sidx]); // one years growth in proportion to viability

            sp->canopy = 2.0f * biome->allometryCanopy(pft, sp->height); // diameter not radius
            sp->root = sp->canopy;
            sp->reserves = sim->sparams.reservecapacity;
            sp->stress = 0.0f;
            rndoff = (float) dice->generate() / 10000.0f * 0.4f;
            sp->col = glm::vec4(-0.2f+rndoff, -0.2f+rndoff, -0.2f+rndoff, 1.0f);
            sp->pft = pft;
            sp->water = 0.0f;
            sp->sunlight = 0.0f;
            ter->toGrid(sp->pos, wx, wy, wz);
            if(wx > dx-1)
                wx = dx-1;
            if(wy > dy-1)
                wy = dy-1;
            sp->gx = wx;
            sp->gy = wy;
            plntpop->push_front(sp);
            sim->incrPlantPop();

            get(x, y)->growing = false; // no other plants can intersect the trunk
            get(x, y)->available = false;

            seeded = true;

#ifdef TESTING
            cerr << "SEED" << endl;
            cerr << "pft = " << sp->pft << " subgrid pos = " << sgx << ", " << sgy << endl;
            cerr << "height = " << sp->height << ", canopy = " << sp->canopy << endl;
#endif
        }
        else
        {
            cerr << "Error MapSimCell::singleSeed: no sapling generated." << endl;
        }
    }
    seedviability.clear();
    seedspecies.clear();

    return seeded;
}

void MapSimCell::writeSeedBank(string outfile)
{
    std::ofstream ofs(outfile);

    ofs << smap.size() << " ";

    for (auto &simc : smap)
    {
        ofs << simc.seedbank.size() << " ";
        for (auto &sb : simc.seedbank)
        {
            ofs << sb << " ";
        }
    }
}

void MapSimCell::readSeedBank(string infile)
{
    std::ifstream ifs(infile);

    std::string alldata;

    for (auto &sm : smap)
    {
        sm.seedbank.clear();
    }

    if (ifs.is_open() && ifs.good())
    {
        std::getline(ifs, alldata);
    }

    std::stringstream ss(alldata);

    int banksize, nseeds, currseed;

    ss >> banksize;

    if (smap.size() != banksize)
    {
        throw std::invalid_argument("Seedbank file " + infile + " is corrupted. Generating new seedbank...");
    }
    //smap.resize(banksize);		// we rather do the assert, since smap is supposed to be the right size already...
    int currloc = 0;
    while (currloc < banksize && ss.good())
    {
        ss >> nseeds;
        for (int i = 0; i < nseeds; i++)
        {
            ss >> currseed;
            smap.at(currloc).seedbank.push_back(currseed);
        }
        currloc++;
    }
}

void MapSimCell::clamp(int &x, int &y)
{
    if(x < 0)
        x = 0;
    if(x > gx-1)
        x = gx-1;
    if(y < 0)
        y = 0;
    if(y > gy-1)
        y = gy-1;
}

void MapSimCell::seeding(std::list<SimPlant *> * plntpop, int plntpopsize, Simulation * sim, Terrain * ter, Biome * biome)
{
    /*
    int count = 0;
    for (int y = 0; y < gy; y++)
    {
        for (int x = 0; x < gx; x++)
        {
            if (get(x, y)->seed_chance > 1e-4)
            {
                count++;
            }
        }
    }
    std::cout << "seed chance nonzero in " << count << " cells" << std::endl;
    */

    int nseed_attempts = 0;
    int nseeded = 0;
    int nseeded_total = 0;
    int size_before = plntpopsize;
    std::vector<int> noseed_count(4, 0);
    std::vector<int> noseed_count_outside(4, 0);

    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
            if(get(x, y)->growing) // is cell in a growth zone
            {
                float seedchance = get(x, y)->seed_chance;
                // check chance to seed
                //bool seed = dice->generate() < (int) (seedprob * 10000.0f);
                bool seed = dice->generate() < (int) (seedchance * 10000.0f);

                if(seed)
                {
                    bool seeded = singleSeed(x, y, plntpop, sim, ter, biome, noseed_count_outside);
                    if (seeded)
                        nseeded_total++;
                }
            }
    nseeded_total += nseeded;

    /*
    cerr << "num plants before seeding = " << size_before << std::endl;
    cerr << "num plants after seeding = " << plntpopsize << endl;
    cerr << "Number seeded in target location, attempted: " << nseeded << ", " << nseed_attempts << std::endl;
    cerr << "Total number seeded: " << nseeded_total << std::endl;

    cerr << "No seed count due to sun, moisture, temp slope in target area:  ";
    for (auto &v : noseed_count)
    {
        cerr << v << " ";
    }
    cerr << std::endl;
    */
}

/*
void MapSimCell::visualize(QImage * visimg, std::vector<SimPlant *> *plnts)
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
                // cerr << "total sun = " << plntpop[p].sunlight << " lightcnt = " << plntpop[p].sunlightcnt  << endl;
                // cerr << "total water = " << plntpop[p].water << " watercnt = " << plntpop[p].watercnt << endl;
                // cerr << "simcell occupancy = " << plntpop[p].sunlightcnt << endl;= (int) get(x, y)->canopies.size()-1; i >= 0; i--)
            {
                for(int p = 1; p < 10; p++)
                    for(int q = 1; q < 10; q++)
                        visimg->setPixelColor(x*10+p, y*10+q, plntcols[get(x, y)->canopies[i].idx]);
            }


        }
}*/

/*
bool MapSimCell::unitTests(QImage * visimg, Terrain *ter, Simulation * sim)
{
    bool valid = true;
    std::vector<SimPlant *> testplnts;
    SimPlant  * plnt1, * plnt2;
    plnt1 = new SimPlant;
    plnt2 = new SimPlant;

    setDim(50, 50, 5);

    // test simple placement of single plant, at domain center reaching to boundary
    inscribe(testplnts.begin(), 5.0f, 5.0f, 5.0f, 2.0f, false, ter, sim);
    plnt1->height = 10.0f;
    testplnts.push_back(plnt1);

    // add shorter plant in offset from center
    inscribe(1, 6.15f, 4.15f, 2.0f, 1.0f, false, ter, sim);
    plnt2->height = 5.0f;
    testplnts.push_back(plnt2);

    // grow shorter plant
    expand(1, 6.15f, 4.15f, 2.0f, 1.0f, 3.0f, 1.0f);
    visualize(visimg, &testplnts);

    if(valid)
        valid = validate(&testplnts);
    cerr << "validity status = " << valid << endl;

    for(int p = 0; p < (int) testplnts.size(); p++)
        delete testplnts[p];
    return valid;
}*/

/*
bool MapSimCell::validate(std::vector<SimPlant *> *plnts)
{
    bool valid = true;

    // a given plant index must only appear once per cell
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // print canopies
            cerr << "(" << x << ", " << y << ") = ";
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
            {
                cerr << (int) get(x, y)->canopies[i].idx << " ";
            }
            cerr << endl;
            // canopies
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
                for(int j = i+1; j < (int) get(x, y)->canopies.size(); j++)
                    if(get(x, y)->canopies[i].plnt == get(x,y)->canopies[j].plnt)
                    {
                        valid = false;
                        cerr << "MapSimCell validity: duplicate canopy index " << get(x,y)->canopies[i].idx << " at position " << i << " and " << j;
                        cerr << "in cell " << x << ", " << y << endl;
                    }

            // roots
            for(int i = 0; i < (int) get(x, y)->roots.size(); i++)
                for(int j = i+1; j < (int) get(x, y)->roots.size(); j++)
                    if(get(x, y)->roots[i].plnt == get(x,y)->roots[j].plnt)
                    {
                        valid = false;
                        cerr << "MapSimCell validity: duplicate root index at position " << i << " and " << j << endl;
                    }

        }

    // static plants do not appear in the canopy list
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            // canopies
            for(int i = 0; i < (int) get(x, y)->canopies.size(); i++)
                    if((* plnts)[get(x, y)->canopies[i].idx]->state == PlantSimState::STATIC)
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
}*/


////
// Simulation
///
void Simulation::initSim(int dx, int dy, int subcellfactor)
{
    simcells.setDim(dx*subcellfactor, dy*subcellfactor, subcellfactor);
    for(int m = 0; m < 12; m++)
    {
        MapFloat sun, mst;
        sun.setDim(dx, dy);
        mst.setDim(dx, dy);
        sunlight.push_back(sun);
        sunlight.back().fill(0.0f);
        moisture.push_back(mst);
        moisture.back().fill(0.0f);
        temperature.push_back(0.0f);
        cloudiness.push_back(0.0f);
        rainfall.push_back(0.0f);
    }
    slope.setDim(dx, dy);
    sunsim = new SunLight();
    dice = new DiceRoller(0,1000);
    time = 0.0f;
    plntpopsize = 0;

    // set simulation parameters to default values
    sparams.reservecapacity = def_reservecapacity;
    sparams.moisturedemand = def_moisturedemand;
    sparams.seeddistmult = def_seeddistmult;
    sparams.seedprob = def_seedprob;
    sparams.stresswght = def_stresswght;
    sparams.mortalitybase = def_mortalitybase;
    sparams.viabilityneg = def_viabilityneg;
}

void Simulation::delSim()
{
    simcells.delMap();
    for(int m = 0; m < 12; m++)
    {
        sunlight[m].delMap();
        moisture[m].delMap();
    }
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
        delete (* plnt);
    plntpop.clear();
    temperature.clear();
    cloudiness.clear();
    rainfall.clear();
    slope.delMap();
    if(sunsim) delete sunsim;
    time = 0.0f;
    plntpopsize = 0;
}

void Simulation::clearPass()
{
    int dx, dy, subcell;

    subcell = simcells.getStep();
    simcells.delMap();
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
        delete (* plnt);
    plntpop.clear();
    ter->getGridDim(dx, dy);
    simcells.setDim(dx*subcell, dy*subcell, subcell);
    time = 0.0f;
    plntpopsize = 0;
}

void Simulation::reportSunAverages()
{
    int dx, dy;
    float sum;
    ter->getGridDim(dx, dy);

    for(int m = 0; m < 12; m++)
    {
        sum = 0.0f;
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
            {
                sum += sunlight[m].get(x, y);
            }
        cerr << "sun month " << m << " avg = " << sum / (float) (dx*dy) << endl;
    }
}

bool Simulation::death(std::list<SimPlant *>::iterator pind, float stress)
{
    int dx, dy;
    bool dead = false;

    // age and stress factors are combined using a probabilistic apprach
    // use constant background mortaility to model age effects
    // use bioclamatic envelope and a carbohydrate pool to model stress effects

    // immediately cull tree if it is taller than threshold because this means that it
    // is no longer strictly undergrowth but now part of the canopy
    if((* pind)->height > def_hghtthreshold)
    {
        dead = true;
    }
    else
    {
        // background mortality
        float ageFactor = 1.0f / (biome->getPFType((* pind)->pft)->maxage);
        float age = 1.0f - pow(sparams.mortalitybase, ageFactor);
        // cerr << "maxage = " << biome->getPFType((* currind)->pft)->maxage << " ";

        // stress mortality. stress score is remainder after effects of carbohydrate pool accounted for
        float p = 1000.0f * (age + sparams.stresswght * stress);
        // cerr << "p = " << p << " ";

        // test against a uniform random variable in [0,1000]
        dead = dice->generate() < (int) p;
    }

    if(dead) // if plant has died change its state
    {
        float x, y, h;
        int sx, sy;
        (* pind)->state = PlantSimState::DEAD;
        ter->toGrid((* pind)->pos, x, y, h);
        sx = (int) simcells.convert((* pind)->gx);
        sy = (int) simcells.convert((* pind)->gy);
        simcells.clamp(sx, sy);
        simcells.get(sx, sy)->growing = true; // flag mapsimcell location as growable so that a new plant can be placed there
        simcells.get(sx, sy)->available = true;
    }
    return dead;
}

void Simulation::growth(std::list<SimPlant *>::iterator pind, float vitality)
{
    PFType * pft;
    float x, y, h, prevcanopy, prevroot, newcanopy;

    assert((* pind)->state != PlantSimState::DEAD);

    if(vitality > 0.0f)
    {
        // apply growth equation for particular pft moderated by vitality
        pft = biome->getPFType((* pind)->pft);
        (* pind)->height += biome->growth((* pind)->pft, (* pind)->height, min(vitality, 1.0f));

        // use allometries for canopy and root
        prevcanopy = (* pind)->canopy;
        // due to randomness canopy can actually shrink so check for this
        newcanopy = 2.0f * biome->allometryCanopy((* pind)->pft, (* pind)->height);
        if(newcanopy > prevcanopy)
            (* pind)->canopy = newcanopy;

        // (* pind)->canopy = 1.0f * biome->allometryCanopy((* pind)->pft, (* pind)->height);
        prevroot = (* pind)->root;
        (* pind)->root = (* pind)->canopy * pft->alm_rootmult;

        // adjust coverage in simcells accordingly
        if((* pind)->canopy > prevcanopy)
        {
            ter->toGrid((* pind)->pos, x, y, h);
            simcells.expand(pind, x, y, ter->toGrid(prevcanopy), ter->toGrid(prevroot), ter->toGrid((* pind)->canopy), ter->toGrid((* pind)->root));
        }

        // cerr << " * GROWTH by " << vitality << " to new hght " << plntpop[pind].height << " * ";
    }
}

void Simulation::simStep(int month)
{
    std::vector<float> adaptvals(4);
    bool shortgrow;
    float x, y, h;

    // traverse all cells contributing moisture and sunlight to plant pool
    shortgrow = (month >= shortgrowstart  || month <= shortgrowend);
    simcells.traverse(&plntpop, this, biome, &sunlight[month], &moisture[month], shortgrow);

    int ndied = 0;
    int ntotal_died = 0;
    int p = 0;

    // cerr << month << endl;
    std::list<SimPlant *>::iterator plnt;

    for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
    {
        float sun, wet, temp, slope, str;
        float pool, stress = 0.0f, vitality = 0.0f;
        bool died = false;

        if((* plnt)->state == PlantSimState::DEAD)
            cerr << "WARNING DEAD PLANT IN PLANT POPULATION" << endl;

        if((* plnt)->state == PlantSimState::ALIVE)
        {
            if(month == 11) // check for death and seeding once a year
            {
                // check for death from old age or stress or simplyt because the plant is too tall
                died = death(plnt, (* plnt)->stress);
                (* plnt)->stress = 0.0f;
                if (died)
                {
                    // remove from simulation grid
                    ter->toGrid((* plnt)->pos, x, y, h);
                    simcells.uproot(plnt, x, y, ter->toGrid((* plnt)->canopy), ter->toGrid((* plnt)->root), ter);

                    // remove from plant population, but make sure iterator remains valid
                    delete (* plnt);
                    plnt = plntpop.erase(plnt); // require c++11
                    plntpopsize--;

                    ntotal_died++;
                }
            }

            if(!died) // still alive?
            {
                // cerr << endl << "PLANT " << p << " of " << plntpopsize << endl;
                sun = (* plnt)->sunlight / (float) (* plnt)->sunlightcnt; // average sunlight
                wet = (* plnt)->water / (float) (* plnt)->watercnt; // average moisture
                // cerr << "total sun = " << plntpop[p].sunlight << " lightcnt = " << plntpop[p].sunlightcnt  << endl;
                // cerr << "total water = " << plntpop[p].water << " watercnt = " << plntpop[p].watercnt << endl;
                // cerr << "simcell occupancy = " << plntpop[p].sunlightcnt << endl;
                temp = getTemperature((* plnt)->gx, (* plnt)->gy, month);
                slope = getSlopeMap()->get((* plnt)->gx, (* plnt)->gy);
                str = biome->viability((* plnt)->pft, sun, wet, temp, slope, adaptvals, sparams.viabilityneg);

                // account for plant reserve pool
                pool = (* plnt)->reserves+str;
                // cerr << "pool = " << pool << " ";
                if(pool < 0.0f) // potential death due to stress
                {
                    (* plnt)->stress += -1.0f * pool;
                    pool = 0.0f;
                }
                else if(pool > sparams.reservecapacity) // reserves are full so growth is possible
                {
                    vitality = pool - sparams.reservecapacity;
                    pool = sparams.reservecapacity;
                }
                (* plnt)->reserves = pool;
                // cerr << "vitality = " << vitality << " reserves = " << plntpop[p].reserves << " ";
                // cerr << "stress = " << plntpop[p].stress << " ";

                // use vitality to determine growth based on allometry
                // but check if this falls in the growing season
                // cerr << "pre-growth" << endl;
                if(month >= biome->getPFType((* plnt)->pft)->grow_start || month <= biome->getPFType((* plnt)->pft)->grow_end)
                {
                    growth(plnt, vitality);
                }
                // cerr << "post-growth" << endl;
                (* plnt)->age += 1;
            }
        }
        p++;
    }
    /*
    if (month == 11)
    {
        cerr << "Number of plants that died at target location: " << ndied << std::endl;
        cerr << "Total number of plants that died: " << ntotal_died << std::endl;
    }*/
    // cerr << endl;

}

Simulation::Simulation(Terrain * terrain, Biome * simbiome, int subcellfactor)
{
    int dx, dy;

    ter = terrain;
    biome = simbiome;
    ter->getGridDim(dx, dy);
    initSim(dx, dy, subcellfactor);
    calcSlope();
}

void Simulation::writeSeedBank(string outfile)
{
    simcells.writeSeedBank(outfile);
}

void Simulation::writeMap(std::string filename, const MapFloat &map)
{
    int dx, dy;
    map.getDim(dx, dy);

    std::ofstream ofs(filename);

    ofs << dx << " " << dy << std::endl;
    for (int y = 0; y < dy; y++)
    {
        for (int x = 0; x < dx; x++)
        {
            ofs << map.get(x, y) << " ";
        }
    }
}

void Simulation::calcCanopyDensity(EcoSystem *eco, MapFloat *density, std::string outfilename)
{

    int gw, gh;
    ter->getGridDim(gw, gh);
    density->setDim(gw, gh);
    density->fill((float)1.0f);

    auto trim = [gw, gh](int &x, int &y)
    {
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= gw) x = gw - 1;
        if (y >= gh) y = gh - 1;
    };

    std::vector<Plant> plnts;
    for (int pft = 0; pft < biome->numPFTypes(); pft++)
    {
        float alpha = biome->getAlpha(pft);
        eco->getPlants()->vectoriseByPFT(pft, plnts);
        for (int p = 0; p < plnts.size(); p++)
        {
            const vpPoint &pos = plnts[p].pos;
            float rad = plnts[p].canopy;
            float srx = pos.x - rad;
            float sry = pos.z - rad;
            vpPoint startpt_r(srx, 0.0f, sry);
            float erx = pos.x + rad;
            float ery = pos.z + rad;
            vpPoint endpt_r(erx, 0.0f, ery);

            int sx, ex, sy, ey;		// start, end grid locations of possible circular area that intersects plant canopy
            int gx, gy;		// grid location of current plant

            ter->toGrid(startpt_r, sx, sy);
            ter->toGrid(endpt_r, ex, ey);
            ter->toGrid(pos, gx, gy);
            trim(sx, sy), trim(ex, ey);
            float canopygridsq = ter->toGrid(plnts[p].canopy);
            canopygridsq *= canopygridsq;

            for (int y = sy; y <= ey; y++)
            {
                for (int x = sx; x <= ex; x++)
                {
                    int gridsq = (y - gy) * (y - gy) + (x - gx) * (x - gx);
                    if (gridsq <= canopygridsq)
                    {
                        float prevval = density->get(x, y);
                        float newval = prevval * (1.0f - alpha);
                        density->set(x, y, newval);
                    }
                }
            }
        }
    }
    // we actually calculated above the amount of light that reaches the ground. Canopy density is the additive inverse of this
    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            density->set(x, y, 1.0f - density->get(x, y));
        }
    }

    if (outfilename.size() > 0)
    {
        writeMap(outfilename, *density);
    }
}

void Simulation::calcSunlight(GLSun * glsun, int minstep, int nsamples, bool inclCanopy)
{
    Timer t;

    t.start();
    sunsim->setLatitude(ter->getLatitude());
    sunsim->setNorthOrientation(Vector(0.0f, 0.0f, -1.0f));
    sunsim->setTerrainDimensions(ter);

    MapFloat diffusemap;
    std::vector<float> sunhours;
    std::cout << "Binding glsun..." << std::endl;
    glsun->bind();
    std::cout << "projecting sun..." << std::endl;
    bool projectenable = true;
    int startm = 1, endm = 12, mincr = 1;
    if (minstep == 0)
    {
        projectenable = false;
        minstep = 60;		// otherwise we have an infinite loop
        startm = 1;
        endm = 12;
        mincr = 1;
    }
    bool diffuseenable = true;
    if (nsamples == 0)
        diffuseenable = false;
    sunsim->projectSun(ter, sunlight, glsun, sunhours, minstep, startm, endm , mincr, projectenable);
    std::cout << "diffusing sun..." << std::endl;
    sunsim->diffuseSun(ter, &diffusemap, glsun, nsamples, diffuseenable);
    std::cout << "merging sun..." << std::endl;
    sunsim->mergeSun(sunlight, &diffusemap, cloudiness, sunhours);

    if(inclCanopy)
    {
        std::cout << "inscribing plants..." << std::endl;
        sunsim->applyAlpha(glsun, sunlight);
    }
    t.stop();
    cerr << "SUNLIGHT SIMULATION TOOK " << t.peek() << "s IN TOTAL" << endl;
}

void Simulation::calcMoisture()
{
    Timer t;
    MoistureSim wet;

    t.start();

    cerr << "MOISTURE SIM PARAMETERS:" << endl;
    cerr << " slope threshold = " << biome->slopethresh << " slope max = " << biome->slopemax << " evaporation = " << biome->evaporation;
    cerr << " runofflevel = " << biome->runofflevel << " soilsaturation = " << biome->soilsaturation << " water level = " << biome->waterlevel << endl;
    wet.simSoilCycle(ter, &slope, rainfall, biome->slopethresh, biome->slopemax, biome->evaporation,
                     biome->runofflevel, biome->soilsaturation, biome->waterlevel, moisture);
    t.stop();
    cerr << "MOISTURE SIMULATION TOOK " << t.peek() << "s IN TOTAL" << endl;
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
#ifdef STEPFILE
        float step;
        infile >> step; // new format
#endif
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

void Simulation::write_monthly_map_copy(std::string filename, std::vector<MapFloat> &monthly)
{
    std::vector<ValueGridMap<float> > mapcopies;
    for (auto &mmap : monthly)
    {
        float rw, rh;
        int gw, gh;
        mmap.getDim(gw, gh);
        rw = gw * 0.9144f;
        rh = gh * 0.9144f;
        mapcopies.push_back(ValueGridMap<float>());
        mapcopies.back().setDim(gw, gh);
        mapcopies.back().setDimReal(rw, rh);

        for (int y = 0; y < gh; y++)
        {
            for (int x = 0; x < gw; x++)
            {
                mapcopies.back().set(x, y, mmap.get(x, y));
            }
        }
    }

    data_importer::write_monthly_map<ValueGridMap<float> >(filename, mapcopies);
}

bool Simulation::writeMonthlyMap(std::string filename, std::vector<MapFloat> &monthly)
{
    int gx, gy;
    ofstream outfile;
    monthly[0].getDim(gx, gy);

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << gx << " " << gy;
#ifdef STEPFILE
        outfile << " 0.9144"; // hardcoded step
#endif
        outfile << endl;
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
        infile >> elv; // in most cases this should be zero. Only use to offset DEM elevations for temperature

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

void Simulation::setTemperature(std::array<float, 12> temp)
{
    temperature.clear();
    for (auto &val : temp)
    {
        temperature.push_back(val);
    }
}

void Simulation::setRainfall(std::array<float, 12> rain)
{
    rainfall.clear();
    for (auto &val : rain)
    {
        rainfall.push_back(val);
    }
}

void Simulation::setCloudiness(std::array<float, 12> cloud)
{
    cloudiness.clear();
    for (auto &val : cloud)
    {
        cloudiness.push_back(val);
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
        }
}


void Simulation::importCanopy(EcoSystem * eco, std::string seedbank_file, std::string seedchance_filename)
{
    auto start_tp = std::chrono::steady_clock::now().time_since_epoch();

    std::vector<Plant> plnts;
    float x, y, h;
    int dx, dy;

    hghts.clear();
    eco->pickAllPlants(ter); // must gather plants before vectorizing
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
        delete (* plnt);
    plntpop.clear();
    simcells.initMap();
    ter->getGridDim(dx, dy);
    // cerr << "ter dimensions = " << dx << " X " << dy << endl;
    simcells.getDim(dx, dy);
    // cerr << "sim map dimensions = " << dx << " X " << dy << endl;

    std::set<int> all_specidxes;

    // iterate over plant functional types
    for(int pft = 0; pft < biome->numPFTypes(); pft++)
    {
        eco->getPlants()->vectoriseByPFT(pft, plnts);
        // cerr << "PFT = " << pft << "numplants = " << (int) plnts.size() << endl;
        for(int p = 0; p < (int) plnts.size(); p++)
        {
            SimPlant * sp = new SimPlant;
            sp->state = PlantSimState::STATIC;
            sp->age = plnts[p].height / biome->getPFType(pft)->maxhght * (float) biome->getPFType(pft)->maxage;
            sp->pos = plnts[p].pos;
            // cerr << "POSITION = " << sp.pos.x << ", " << sp.pos.y << endl;
            sp->height = plnts[p].height;
            hghts.push_back(sp->height);
            // cerr << "HEIGHT = " << sp.height << endl;
            sp->canopy = plnts[p].canopy;
            // cerr << "CANOPY = " << sp.canopy << endl;
            sp->root = sp->canopy;
            sp->reserves = sparams.reservecapacity;
            sp->stress = 0.0f;
            sp->col = plnts[p].col;
            sp->pft = pft;
            sp->water = 0.0f;
            sp->sunlight = 0.0f;
            plntpop.push_front(sp);
            plntpopsize++;

            all_specidxes.insert(sp->pft);

            // inscribe plant into simcells
            // cerr << "TERRAIN CELL SIZE = " << ter->getCellExtent() << endl;

            ter->toGrid(sp->pos, x, y, h);
            if(x > dx-1)
                x = dx-1;
            if(y > dy-1)
                y = dy-1;
            sp->gx = x;
            sp->gy = y;
            simcells.inscribe(plntpop.begin(), x, y, ter->toGrid(sp->canopy), ter->toGrid(sp->root), true, ter, this);
        }
    }


    float tw, th;
    uint gw, gh;
    ter->getGridDim(gw, gh);
    ter->getTerrainDim(tw, th);

    if (seedchance_filename.size() > 0)
    {
        int simc_gw, simc_gh;
        simcells.getDim(simc_gw, simc_gh);

        ValueGridMap<float> seedchance_map;
        seedchance_map.setDim(int(std::round(tw) + 1e-5f), int(std::round(th) + 1e-5f));
        seedchance_map.setDimReal(tw, th);

        for (int y = 0; y < simc_gh; y++)
        {
            for (int x = 0; x < simc_gw; x++)
            {
                float chance = simcells.get(x, y)->seed_chance;
                float tgx = simcells.convert_to_tergrid(x);
                float tgy = simcells.convert_to_tergrid(y);
                vpPoint pt = ter->toWorld(tgx, tgy, 0);
                float ry = pt.z;		// vpPoint z is actually y, according to the toWorld conversion above
                float rx = pt.x;
                if (rx > tw || ry > th || rx < 0 || ry < 0)
                {
                    // get size of terrain grid cells width and height, to see if any out of range coordinates are "too much" out of range
                    float tercw = tw / gw;
                    float terch = th / gh;
                    if (rx >= tw + tercw || ry >= th + terch || rx <= -tercw || ry <= -terch)	// converted coords are too much higher than terrain size, so we don't correct them and allow exception to be thrown in set_fromreal function below
                        std::cerr << "Warning: converted coordinates (" << rx << ", " << ry << ") are larger than terrain real dim (" << tw << ", " << th << ")" << std::endl;
                    else
                    {
                        // probably just a cell that falls partly outside the actual terrain, so we clamp to the terrain width/height
                        if (rx > tw)
                            rx = tw - 1e-5f;
                        else
                            rx = 1e-5f;
                        if (ry > th)
                            ry = th - 1e-5f;
                        else
                            ry = 1e-5f;
                    }
                }
                seedchance_map.set_fromreal(rx, ry, chance);
            }
        }

        data_importer::write_txt<ValueGridMap<float> >(seedchance_filename, &seedchance_map);
        // std::cout << "Done writing seedchance file at " << seedchance_filename << std::endl;
    }

    simcells.init_countmaps(all_specidxes, gw, gh, tw, th);

    // std::cout << "Establishing seed bank..." << std::endl;

    bool readsuccess = false;
    if (seedbank_file.size() > 0 && static_cast<bool>(std::ifstream(seedbank_file)))		// if a seedbank file already exists, we just read it in
    {
        try
        {
            simcells.readSeedBank(seedbank_file);
            readsuccess = true;
        }
        catch (std::invalid_argument &e)
        {
            std::cout << "Invalid seedbank file at " + seedbank_file + ". Simulating new seedbank..." << std::endl;
            readsuccess = false;
        }
    }

    if (!readsuccess)
    {
        simcells.establishSeedBank(&plntpop, plntpopsize, biome, ter);
        if (seedbank_file.size() > 0)
        {
            simcells.writeSeedBank(seedbank_file);					// write newly simulated seedbank to a file that can be imported later for faster simulation
#ifndef NDEBUG
            std::vector<SimCell> cells = simcells.get_smap();
            simcells.readSeedBank(seedbank_file);
            std::vector<SimCell> cells2 = simcells.get_smap();
            assert(cells.size() == cells2.size());
            for (int i = 0; i < cells.size(); i++)
            {
                for (int j = 0; j < cells.at(i).seedbank.size(); j++)
                {
                    assert(cells.at(i).seedbank.at(j) == cells2.at(i).seedbank.at(j));
                }
            }
#endif // NDEBUG
        }
    }


    auto end_tp = std::chrono::steady_clock::now().time_since_epoch();
}

void Simulation::exportUnderstory(EcoSystem * eco)
{

    for(int n = 0; n < maxNiches; n++)
    {
        int plntcnt = 0;
        int deadcount = 0;
        PlantGrid * outplnts = eco->getNiche(n);
        outplnts->clear();
        plntcnt = 0;

        // canopy into niche 0, understory into niche 1
        std::list<SimPlant *>::iterator plnt;
        for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
        {
            Plant op;
            bool place;

            if(n == 0) // only include static plants in niche 0
            {
                place = ((* plnt)->state == PlantSimState::STATIC);
            }
            else // only include live plants in niche 1
            {
                place = ((* plnt)->state == PlantSimState::ALIVE);
            }


            if(place)
            {
                float cullhght = biome->getPFType((* plnt)->pft)->minhght;
                if ((* plnt)->state != PlantSimState::DEAD && (* plnt)->height >= cullhght)
                {
                    op.canopy = (* plnt)->canopy;
                    op.col = (* plnt)->col;
                    op.height = (* plnt)->height;
                    if(n == 0)
                        op.pos = (* plnt)->pos;
                    else
                        op.pos = vpPoint((* plnt)->pos.x, (* plnt)->pos.y, (* plnt)->pos.z);
                    outplnts->placePlant(ter, (* plnt)->pft, op);
                    plntcnt++;
                    if((* plnt)->state == PlantSimState::DEAD)
                        cerr << " DEAD";
                }
                else
                {
                    deadcount++;
                }
            }
        }
        // cerr << "niche " << n << " numplants = " << plntcnt << endl;
        // cerr << "number of dead plants in niche: " << deadcount << endl;
    }

    // eco->pickAllPlants(ter); // gather all plants for display
    // eco->redrawPlants();
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
    for(int m = 0; m < 12; m++)
        cerr << sunlight[m].get(x, y) << " ";
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

void Simulation::printParams()
{
    cerr << "Simulation Parameters" << endl;
    cerr << "Reserve Capacity = " << sparams.reservecapacity << " Moisture Demand = " << sparams.moisturedemand << " Seed Probability = " << sparams.seedprob << endl;
    cerr << "Stress Weight = " << sparams.stresswght << " Mortality Base = " << sparams.mortalitybase << " Viability Neg Range = " << sparams.viabilityneg << endl;
    cerr << endl;
}

void Simulation::averageViability(std::vector<float> &targetnums)
{
    std::vector<float> adaptvals(4);
    std::vector<float> totadaptvals(4);
    float totstr, str, sun, wet, temp, slope;
    int gx, gy, step, avgcount, wetcount, numover, nonviable;

    simcells.getDim(gx, gy);
    step = simcells.getStep();

    cerr << "Average Viability by Species: " << endl;

    // sum viability per species over growing areas of map
    for(int s = 0; s < biome->numPFTypes(); s++)
    {
        totstr = 0.0f;
        avgcount = 0;
        wetcount = 0; numover = 0; nonviable = 0;
        for(int i = 0; i < 4; i++)
            totadaptvals[i] = 0.0f;

        if(targetnums[s] >= 0.0f)
        {
            for(int x = 0; x < gx; x++)
                for(int y = 0; y < gy; y++)
                    if(simcells.get(x, y)->growing)
                    {
                        float str;
                        int tgx, tgy;

                        // convert from x, y simcell to tx, ty terrain grid
                        simcells.toTerGrid(x, y, tgx, tgy, ter);

                        // average sunlight and moisture over growing season
                        sun = simcells.get(x,y)->leftoversun / (float) shortgrowmonths;
                        wet = simcells.get(x,y)->leftoverwet / (float) shortgrowmonths;
                        temp = getTemperature(tgx, tgy, shortgrowend); // use end of growing season
                        slope = getSlopeMap()->get(tgx, tgy);
                        str = max(0.0f, biome->viability(s, sun, wet, temp, slope, adaptvals, sparams.viabilityneg));
                        totstr += str;
                        for(int i = 0; i < 4; i++)
                            totadaptvals[i] += adaptvals[i];
                        avgcount++;
                        if(adaptvals[1] < 0.0f) // negative wetness
                        {
                            if(biome->overWet(s, wet))
                                numover++;
                            wetcount++;
                        }
                        if(str < 0.5f) // low viability here
                        {
                            nonviable++;
                        }


                    }

                cerr << s << ", " << totstr / (float) avgcount << ", " << totadaptvals[0] / (float) avgcount << ", " << totadaptvals[1] / (float) avgcount << ", " << totadaptvals[2] / (float) avgcount << ", " << totadaptvals[3] / (float) avgcount;
                cerr << endl;
                cerr << "overwet proportion = " << (float) numover / (float) wetcount << endl;
                cerr << "nonviable proportion = " << (float) nonviable / (float) avgcount;
        }
        cerr << endl;
    }
}

float Simulation::plantTarget(std::vector<float> &targetnums)
{
    std::vector<int> actualnums;
    std::vector<float> avghght;
    int canopynums = 0;
    float diff = 0.0f, cullhght;

    // init count per species
    for(int s = 0; s < biome->numPFTypes(); s++)
    {
        actualnums.push_back(0);
        avghght.push_back(0.0f);
    }


    // canopy into niche 0, understory into niche 1
    std::list<SimPlant *>::iterator plnt;
    for(plnt = plntpop.begin(); plnt != plntpop.end(); plnt++)
    {
        cullhght = biome->getPFType((* plnt)->pft)->minhght;
        if((* plnt)->state == PlantSimState::ALIVE && (* plnt)->height > cullhght)
        {
            actualnums[(* plnt)->pft]++;
            avghght[(* plnt)->pft] += (* plnt)->height;
        }
        else
        {
            if((* plnt)->state == PlantSimState::STATIC)
                canopynums++;
        }
    }

    // print different species
//    cerr << "Number of Canopy Plant = " << canopynums << endl;
//    cerr << "Number of Undergrowth Plant by Species: " << endl;
     for(int s = 0; s < biome->numPFTypes(); s++)
         if(targetnums[s] >= 0.0f) // in subbiome
         {
              cerr << biome->getPFType(s)->code << " [" << s << "] = " << actualnums[s];
              cerr << " avg. height = " << avghght[s] / (float) actualnums[s] << endl;
              cerr << " as proportion = " << (float) actualnums[s] / (float) canopynums << endl;
                // cerr << "diff from target " << (float) actualnums[s] / (float) canopynums - targetnums[s] << endl;
          }
     cerr << endl;

    // calculate diff from target
    // actual nums need to be converted to a total water available in mmper canopy tree basis for comparison against targetnums
    /*
    for(int s = 0; s < biome->numPFTypes(); s++)
    {
    if(targetnums[s] >= 0.0f) // negative results is signal to ignore this species in calculations
        diff += fabs((float) actualnums[s] / (float) canopynums - targetnums[s]);
    }*/
    return diff;
}

void Simulation::simulate(EcoSystem * eco, std::string seedbank_file, std::string seedchance_filename, int delYears)
{
    Timer simt;
    std::string tmp_filename;

    simt.start();

    std::vector<float> targetnums;
    for(int s = 0; s < biome->numPFTypes(); s++)
        targetnums.push_back(1.0f);

    // tuned simulation parameters
    biome->getPFType(11)->sun.r = 8.0f;
    sparams.reservecapacity = 3.0f;
    sparams.seedprob = 0.003f;
    sparams.stresswght = 1.25f;
    sparams.viabilityneg = -0.4f;

    // clear undergrowth
    clearPass();
    importCanopy(eco, seedbank_file, seedchance_filename);

    /*
    // checkpointing functionality
    // commented out because it creates large files and slows simulations
    if(chkpnt > 0)
    {
        tmp_filename = out_filename + "_chk" + std::to_string(chkpnt) + ".pdb";
        readCheckPoint(tmp_filename);
    }*/

    for(int y = 0; y < delYears; y++)
    {
        Timer simy;

        simy.start();

        simcells.resetSeeding(); // clear seeding

        for(int m = 0; m < 12; m++)
            simStep(m);

        if(y == 0)
            averageViability(targetnums);

        simcells.seeding(&plntpop, plntpopsize, this, ter, biome);

        simy.stop();
        cerr << "YEAR " << y << " took " << simy.peek() << "s" << endl;

        // report species numbers and save to checkpoint file
        /*
        if(y > 0 && y%25 == 0)
        {
            float res = plantTarget(targetnums);
            tmp_filename = out_filename + "_" + std::to_string(y+chkpnt) + ".pdb";
            exportUnderstory(eco); // transfer plants from simulation
            eco->saveNichePDB(tmp_filename, 1);
        }*/
    }

    // print viability params
    // biome->printParams();

    simt.stop();
    cerr << "simulation took " << simt.peek() << "s" << endl;   
}
