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
#include <iostream>
// #include <fstream>
#include "pft.h"

////
// Viability
///

float Viability::eval(float val, float neg)
{
    // evaluate exponential. Power term introduces flat top
    float s = log(0.2f) / pow(r/2.0, 4.5f);
    float v = pow(M_E, s * pow(fabs(val-c), 4.5f));

    // adjust to include negative viability for stress purposes
    v *= 1.0-neg;
    v += neg;
    return v;
}

////
// Biome
///

void Biome::categoryNameLookup(int idx, std::string &catName)
{
    if(idx >= 1 && idx <= (int) catTable.size())
        catName = catTable[idx-1]; // categories start at 1, whereas table is indexed from 0
    else
        catName = "OutOfCategoryTableRange";
}

float Biome::getMinIdealMoisture(int i)
{
    float c, r;
    char cmin, cmax;

    pftypes[i].wet.getValues(cmin, cmax, c, r);
    return c - 0.5f * r; // minimum water needed for survival
}

int Biome::getSubBiomeFromSpecies(int species)
{
    if(subLookup[species] == -1)
    {
        cerr << "Error Biome::getSubBiomeFromSpecies: species " << species << " is not in the canopy list of any sub-biome" << endl;
        return 0;
    }
    return subLookup[species];
}

float Biome::allometryCanopy(int pft, float height)
{
    /*
    // allometry-based approach
    float radius;
    if(pft == 0)
    {
        radius = 4.0f * height;
    }
    else
    {
         radius = exp(pftypes[pft].alm_a + pftypes[pft].alm_b * log(height)); // r = e ** (a + b ln(h))
    }*/

    // model-based approach
    int vueid;
    float radius = 0.0f;

    if(pft == 0)
    {
        radius = 4.0f * height;
    }
    else
    {
        if(cdata != nullptr)
            radius = cdata->modelsamplers.at(pft).sample_rh_ratio(height, &vueid) * height;
    }
    return radius;
}

float Biome::maxGrowthTest(int pft)
{
    float hght = 0.0f;

    // run mini growth simulation
    for(int a = 0; a < pftypes[pft].maxage; a++)
        for(int g = 0; g < pftypes[pft].grow_months; g++)
            hght += growth(pft, hght, 1.0f);

    return hght;
}

float Biome::growthfn(float t)
{
    // sigmoidal-like (smoothstep) function for growth, accounting for maximum tree age and height

    // bounds checks
    if(t < 0.0f)
    {
        cerr << "Error Biome::growthfn: age is below zero" << endl;
        return 0.0f;
    }

    if(t > 1.0f)
    {
        return  1.0f;
        cerr << "Error Biome::growthfn: age is above maximum" << endl;
    }
    return t*t*(3.0f - 2.0f*t);
}

float Biome::invgrowthfn(float h)
{
    // inverse of sigmoidal-like (smoothstep) function

    // bounds checks
    if(h < 0.0f)
    {
        cerr << "Error Biome::invgrowthfn: height is below zero" << endl;
        return 0.0f;
    }

    if(h > 1.0f)
    {
        return  1.0f;
        cerr << "Error Biome::invgrowthfn: height is above maximum" << endl;
    }
    return 0.5f - sinf(asinf(1.0f - 2.0f*h)/3.0f);
}

float Biome::growth(int pft, float currhght, float viability)
{
    float hghtnorm, hghtincr, agenominal, ageincr;

    ageincr = 1.0f / (float) (pftypes[pft].maxage * pftypes[pft].grow_months); // month in growing season as a proportion of overall growth lifespan

    // find normalized nominal age on sigmoidal growth curve
    // this does not correspond to actual age because of potentially stunted growth
    hghtnorm = currhght / pftypes[pft].maxhght;
    agenominal = invgrowthfn(hghtnorm);

    hghtincr = pftypes[pft].maxhght * (growthfn(agenominal+ageincr) - growthfn(agenominal));
    return hghtincr * viability;
}

float Biome::viability(int pft, float sunlight, float moisture, float temperature, float slope, std::vector<float> &adaptvals, float negval)
{
    float val[4], vmin;

    val[0] = pftypes[pft].sun.eval(sunlight, negval);
    val[1] = pftypes[pft].wet.eval(moisture, negval);
    val[2] = pftypes[pft].temp.eval(temperature, negval);
    val[3] = pftypes[pft].slope.eval(slope, negval);

    if (adaptvals.size() != 4)
        adaptvals.resize(4);
    memcpy(adaptvals.data(), val, sizeof(float) * 4);

    // the least satisfied resource dominates, so find min value
    vmin = val[0];
    for(int i = 1; i < 4; i++)
        vmin = fmin(vmin, val[i]);

    return vmin;
}

bool Biome::overWet(int pft, float moisture)
{
    return moisture > pftypes[pft].wet.c;
}

bool Biome::read_dataimporter(std::string cdata_fpath)
{
    // data_importer::common_data cdata = data_importer::common_data(cdata_fpath);
    cdata = new data_importer::common_data(cdata_fpath);

    return read_dataimporter((* cdata));
}

float Biome::cullHght(int id)
{
    float hght = 0.0f;

    switch(id)
    {
    case 0:
        hght = 0.05f;
        break;
    case 1:
        hght = 0.1f;
        break;
    case 2:
        hght = 0.1f;
        break;
    case 3:
        hght = 0.1f;
        break;
    case 4:
        hght = 0.1f;
        break;
    case 5:
        hght = 0.2f;
        break;
    case 6:
        hght = 0.1f;
        break;
    case 7:
        hght = 0.25f;
        break;
    case 8:
        hght = 0.25f;
        break;
    case 9:
        hght = 0.25f;
        break;
    case 10:
        hght = 0.25f;
        break;
    case 11:
        hght = 0.25f;
        break;
    case 12:
        hght = 0.25f;
        break;
    case 13:
        hght = 0.25f;
        break;
    case 14:
        hght = 0.3f;
        break;
    case 15:
        hght = 0.2f;
        break;
    default:
        hght = 0.1f;
        break;
    }

    return hght;
}

bool Biome::read_dataimporter(data_importer::common_data &cdata)
{
    pftypes.clear();

    PFType pft;

    int maxspec_id = -1;

    // cerr << "--- BIOME READ FROM DATABASE ---" << endl;

    for (auto &sppair : cdata.canopy_and_under_species)
    {
        data_importer::species &spec = sppair.second;
        pft.code = spec.name;
        for (int i = 0; i < 4; i++)
            pft.basecol[i] = spec.basecol[i];
        pft.draw_hght = spec.draw_hght;
        pft.draw_radius = spec.draw_radius;
        pft.draw_box1 = spec.draw_box1;
        pft.draw_box2 = spec.draw_box2;
        switch (spec.shapetype)
        {
            case (data_importer::treeshape::BOX):
                pft.shapetype = TreeShapeType::BOX;
                break;
            case (data_importer::treeshape::CONE):
                pft.shapetype = TreeShapeType::CONE;
                break;
            case (data_importer::treeshape::SPHR):
                pft.shapetype = TreeShapeType::SPHR;
                break;
            case (data_importer::treeshape::INVCONE):
                pft.shapetype = TreeShapeType::INVCONE;
                break;
            default:
                assert(false);
                break;
        }
        // cerr << "PFT " << pft.code << " shape type = " << pft.shapetype << endl;
        pft.maxage = spec.maxage;
        pft.maxhght = spec.maxhght;
        pft.alpha = spec.alpha;


        pft.sun.setValues(' ', ' ', spec.sun.c, spec.sun.r);
        pft.wet.setValues(' ', ' ', spec.wet.c, spec.wet.r);
        pft.temp.setValues(' ', ' ', spec.temp.c, spec.temp.r);
        pft.slope.setValues(' ', ' ', spec.slope.c, spec.slope.r);

        /*
        cerr << pft.code << " " << "sun (" << spec.sun.c << ", " << spec.sun.r << ")" << " wet (" << spec.wet.c << ", " << spec.wet.r << ") ";
        cerr << "slope (" << spec.slope.c << ", " << spec.slope.r << endl;
        cerr << "alpha = " << pft.alpha << endl;
        cerr << "age = " << pft.maxage << endl;
        */

        pft.grow_start = spec.grow_start;
        pft.grow_end = spec.grow_end;
        pft.grow_months = spec.grow_months;

        pft.alm_a = spec.a;
        pft.alm_b = spec.b;
        pft.alm_rootmult = 1.0f;
        pft.grow_m = spec.grow_m;
        pft.grow_c1 = spec.grow_c1;
        pft.grow_c2 = spec.grow_c2;
        pft.minhght = cullHght((int) pftypes.size());

        pftypes.push_back(pft);

        if (sppair.first > maxspec_id)
            maxspec_id = sppair.first;

    }

    slopethresh = cdata.soil_info.slopethresh;
    slopemax = cdata.soil_info.slopemax;
    evaporation = cdata.soil_info.evap;
    runofflevel = cdata.soil_info.runofflim;
    soilsaturation = cdata.soil_info.soilsat;
    waterlevel = cdata.soil_info.riverlevel;

    subLookup.resize(maxspec_id + 1, -1);

    for (const std::pair<int, data_importer::sub_biome> &sb : cdata.subbiomes_all_species)
    {
        SubBiome sbadd;
        for (const data_importer::species_encoded &spec : sb.second.species)
        {  
            if (spec.canopy)
                subLookup[spec.id] = sb.first;
            if (spec.canopy)
                sbadd.canopies.push_back(spec.id);
            else
                sbadd.understorey.push_back(spec.id);
        }

#ifndef NDEBUG

        auto has_dupl = [](const std::vector<int> &vec, bool ignore = false, int ignoreval = -1)
        {
            for (int i = 0; i < vec.size(); i++)
            {
                for (int j = i + 1; j < vec.size(); j++)
                {
                    if ((!ignore || (vec[i] != ignoreval)) && (vec[i] == vec[j]))
                    {
                        std::cerr << vec[i] << ", " << vec[j] << std::endl;
                        return true;
                    }
                }
            }
            return false;
        };

        assert(!has_dupl(sbadd.canopies));
        assert(!has_dupl(sbadd.understorey));

#endif

        subTable.push_back(sbadd);
    }


    return true;
}

void Biome::sunmapping(PFType &pft, char cmin, char cmax)
{
    float c, r, fmin, fmax;

    // shade tolerance mapping
    switch(cmin)
    {
    case 'L':
        fmin = shadeLow;
        break;
    case 'M':
        fmin = shadeMedium;
        break;
    case 'H':
        fmin = shadeHigh;
        break;
    default:
        cerr << "Biome::sunmapping: unrecognized shade code." << cmin << endl;
        break;
    }

    // sun tolerance mapping
    switch(cmax)
    {
    case 'L':
        fmax = scorchLow;
        break;
    case 'M':
        fmax = scorchMedium;
        break;
    case 'H':
        fmax = scorchHigh;
        break;
    default:
        cerr << "Biome::sunmapping: unrecognized scorch code." << cmax << endl;
        break;
    }

    c = (fmax - fmin) / 2.0f + fmin;
    r = (fmax - fmin);
    pft.sun.setValues(cmin, cmax, c, r);
}

void Biome::wetmapping(PFType &pft, char cmin, char cmax)
{
    float c, r, fmin, fmax;

    // drought tolerance mapping
    switch(cmin)
    {
    case 'L':
        fmin = droughtLow;
        break;
    case 'M':
        fmin = droughtMedium;
        break;
    case 'H':
        fmin = droughtHigh;
        break;
    default:
        cerr << "Biome::wetmapping: unrecognized moisture code." << endl;
        break;
    }

    // flood tolerance mapping
    switch(cmax)
    {
    case 'L':
        fmax = floodLow;
        break;
    case 'M':
        fmax = floodMedium;
        break;
    case 'H':
        fmax = floodHigh;
        break;
    default:
        cerr << "Biome::wetmapping: unrecognized moisture code." << endl;
        break;
    }

    c = (fmax - fmin) / 2.0f + fmin;
    r = (fmax - fmin);
    pft.wet.setValues(cmin, cmax, c, r);
}

void Biome::tempmapping(PFType &pft, char cmin)
{
    float c, r, fmin, fmax;
    char cmax = 'E';

    // lower temperature tolerance mapping
    switch(cmin)
    {
    case 'L':
        fmin = coldLow;
        break;
    case 'M':
        fmin = coldMedium;
        break;
    case 'H':
        fmin = coldHigh;
        break;
    case 'V': // very low cold tolerance
        fmin = coldVeryLow;
        break;
    default:
        cerr << "Biome::wetmapping: unrecognized temperature code." << endl;
        break;
    }

    fmax = 35.0f; // fixed upper bound

    c = (fmax - fmin) / 2.0f + fmin;
    r = (fmax - fmin);
    pft.temp.setValues(cmin, cmax, c, r);
}

void Biome::slopemapping(PFType &pft, char cmax)
{
    float c, r, fmin, fmax;
    char cmin = 'E';

    // upper slope tolerance mapping
    switch(cmax)
    {
    case 'M':
        fmax = steepMedium;
        break;
    case 'H':
        fmax = steepHigh;
        break;
    default:
        cerr << "Biome::slopemapping: unrecognized slope code." << endl;
        break;
    }

    fmin = -5.0f; // fixed lower bound

    c = (fmax - fmin) / 2.0f + fmin;
    r = (fmax - fmin);
    pft.slope.setValues(cmin, cmax, c, r);
}

bool Biome::read(const std::string &filename)
{
    int nb, nc, ns;
    ifstream infile;
    char cmin, cmax;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> name;

        // cerr << "--- BIOME READ FROM FILE ---" << endl;
        // plant functional types
        infile >> nb; // number of pft
        subLookup.resize(0);
        subLookup.resize(nb, -1);
        for(int t = 0; t < nb; t++)
        {
            PFType pft;
            string shapestr;

            infile >> pft.code >> pft.basecol[0] >> pft.basecol[1] >> pft.basecol[2] >> pft.draw_hght >> pft.draw_radius >> pft.draw_box1 >> pft.draw_box2;
            infile >> shapestr;

            if(shapestr == "SPHR")
                pft.shapetype = TreeShapeType::SPHR;
            else if(shapestr == "BOX")
                pft.shapetype = TreeShapeType::BOX;
            else if(shapestr == "CONE")
                pft.shapetype = TreeShapeType::CONE;
            else
                cerr << "Error Biome::read: malformed shape type" << endl;
            pft.basecol[3] = 1.0f;

            infile >> pft.maxage;
            infile >> pft.maxhght;
            infile >> pft.alpha;

            //cerr << pft.maxage << " " << pft.alpha << " ";
            // viability response values
            infile >> cmin >> cmax;
            //cerr << cmin << " " << cmax << " ";
            sunmapping(pft, cmin, cmax);
            infile >> cmin >> cmax;
            //cerr << cmin << " " << cmax << " ";
            wetmapping(pft, cmin, cmax);
            infile >> cmin;
            //cerr << cmin << " ";
            tempmapping(pft, cmin);
            infile >> cmax;
            //cerr << cmax << " ";
            slopemapping(pft, cmax);

            //< growth parameters
            infile >> pft.growth_period;
            //cerr << pft.growth_period << " ";
            switch(pft.growth_period)
            {
            case 'S': // short growing season December to May
                pft.grow_start = shortgrowstart;
                pft.grow_end = shortgrowend;
                pft.grow_months = shortgrowmonths;
                break;
            case 'L': // long growing season September to May
                pft.grow_start = longgrowstart;
                pft.grow_end = longgrowend;
                pft.grow_months = longgrowmonths;
                break;
            default:
                cerr << "Biome::read: incorrect growth period code in file" << endl;
                // error message
                break;
            }

            //< allometry parameters
            infile >> pft.allometry_code;
            //cerr << pft.allometry_code << " ";
            switch(pft.allometry_code)
            {
            case 'A':
                pft.alm_a = -0.584012589f;
                pft.alm_b = 0.923859482f;
                break;
            case 'B':
                pft.alm_a = -1.524125074f;
                pft.alm_b = 0.976098575f;
                break;
            case 'C':
                pft.alm_a = -1.8185f;
                pft.alm_b = 1.1471f;
                break;
            case 'D':
                pft.alm_a = -0.728451431f;
                pft.alm_b = 0.656991006f;
                break;
            /*
            case 'E':
                pft.alm_a = 0.5f;
                pft.alm_b = 0.0f;
                break;*/
            default:
                cerr << "Biome::read: incorrect allometry code in file" << endl;
                break;
            }
            pft.alm_rootmult = 1.0f;

            infile >> pft.grow_m >> pft.grow_c1 >> pft.grow_c2;
            //cerr << pft.grow_m << " " << pft.grow_c1 << " " << pft.grow_c2 << endl;

            pft.minhght = cullHght((int) pftypes.size());

            pftypes.push_back(pft);
        }

        // sub-biome names and species
        infile >> ns; // number of categories
        for(int s = 0; s < ns; s++)
        {
            SubBiome sb;
            int over, under, spec;

            infile >> sb.code;
            infile >> over;
            for(int o = 0; o < over; o++)
            {
                infile >> spec;
                sb.canopies.push_back(spec);
                subLookup[spec] = s;
                std::cout << "For " << sb.code << ", adding subbiome " << s << " to lookup for canopy species " << spec << std::endl;
            }
            infile >> under;
            for(int u = 0; u < under; u++)
            {
                infile >> spec;
                sb.understorey.push_back(spec);
            }
            subTable.push_back(sb);
        }

        // category names
        infile >> nc; // number of categories
        for(int c = 0; c < nc; c++)
        {
            std::string str;
            infile >> str;
            catTable.push_back(str);
        }

        // soil moisture parameters
        infile >> slopethresh;
        infile >> slopemax;
        infile >> evaporation;
        infile >> runofflevel;
        infile >> soilsaturation;
        infile >> waterlevel;

        infile.close();
        return true;
    }
    else
    {
        cerr << "Error Biome::read: unable to open file" << filename << endl;
        return false;
    }
}

bool Biome::write(const std::string &filename)
{
    ofstream outfile;
    float c, r;
    char cmin, cmax;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << name << endl;

        // plant functional types
        outfile << numPFTypes() << endl;
        for (int t = 0; t < numPFTypes(); t++)
        {
            outfile << pftypes[t].code << " " << pftypes[t].basecol[0] << " " << pftypes[t].basecol[1] << " " << pftypes[t].basecol[2] << " ";
            outfile << pftypes[t].draw_hght << " " << pftypes[t].draw_radius << " " << pftypes[t].draw_box1 << " " << pftypes[t].draw_box2 << " ";
            switch(pftypes[t].shapetype)
            {
            case TreeShapeType::SPHR:
                outfile << "SPHR";
                break;
            case TreeShapeType::BOX:
                outfile << "BOX";
                break;
            case TreeShapeType::CONE:
                outfile << "CONE";
                break;
            }
            outfile << endl;
            // viability response values
            outfile << pftypes[t].maxage << " " << pftypes[t].maxhght << " " << pftypes[t].alpha << " ";
            pftypes[t].sun.getValues(cmin, cmax, c, r);
            outfile << cmin << " " << cmax << " ";
            pftypes[t].wet.getValues(cmin, cmax, c, r);
            outfile << cmin << " " << cmax << " ";
            pftypes[t].temp.getValues(cmin, cmax, c, r);
            outfile << cmin << " ";
            pftypes[t].slope.getValues(cmin, cmax, c, r);
            outfile << cmax << " ";

            //< growth and allometry parameters
            outfile << pftypes[t].growth_period << " " << pftypes[t].allometry_code << " ";
            outfile << pftypes[t].grow_m << " " << pftypes[t].grow_c1 << " " << pftypes[t].grow_c2 << endl;
        }

        // sub-biomes
        outfile << (int) subTable.size() << endl;
        for(int s = 0; s < (int) subTable.size(); s++)
        {
            outfile << subTable[s].code << " ";
            int over = (int) subTable[s].canopies.size();
            outfile << over << " ";
            for(int o = 0; o < over; o++)
            {
                outfile << subTable[s].canopies[o] << " ";
            }
            int under = (int) subTable[s].understorey.size();
            outfile << under << " ";
            for(int u = 0; u < under; u++)
            {
                outfile << subTable[s].understorey[u] << " ";
            }
            outfile << endl;
        }

        // category names
        outfile << (int) catTable.size() << endl;
        for(int c = 0; c < (int) catTable.size(); c++)
            outfile << catTable[c] << endl;

        // soil moisture parameters
        outfile << slopethresh << " ";
        outfile << slopemax << " ";
        outfile << evaporation << " ";
        outfile << runofflevel << " ";
        outfile << soilsaturation << " ";
        outfile << waterlevel << endl;
        outfile.close();
        return true;
    }
    else
    {
        cerr << "Error Biome::write: unable to open file " << filename << endl;
        return false;
    }
}

void Biome::printParams()
{
    char cmin, cmax;
    float c, r;

    for (int t = 0; t < numPFTypes(); t++)
    {
        cerr << pftypes[t].code << " [" << t << "]: " << endl;

        // viability response values
        pftypes[t].sun.getValues(cmin, cmax, c, r);
        cerr << "   sun: " << c << " " << r << endl;
        pftypes[t].wet.getValues(cmin, cmax, c, r);
        cerr << "   wet: " << c << " " << r << endl;
        pftypes[t].temp.getValues(cmin, cmax, c, r);
        cerr << "   temp: " << c << " " << r << endl;
        pftypes[t].slope.getValues(cmin, cmax, c, r);
        cerr << "   slope: " << c << " " << r << endl << endl;
    }
}
