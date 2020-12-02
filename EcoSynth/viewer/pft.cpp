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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
// #include <fstream>
#include "pft.h"
#include "data_importer/data_importer.h"

////
// Viability
///

float Viability::eval(float val)
{
    if(val < o1)
        return -1.0f;
    else if(val < z1)
        return -1.0f + (val-o1) / (z1-o1);
    else if(val < i1)
        return (val-z1) / (i1-z1);
    else if(val < i2)
        return 1.0f;
    else if(val < z2)
        return (z2-val) / (z2-i2);
    else if(val < o2)
        return -1.0f + (o2-val) / (o2-z2);
    else
        return -1.0f;
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

float Biome::viability(int pft, float sunlight, float moisture, float temperature, float slope)
{
    float val[4], vmin;

    val[0] = pftypes[pft].sun.eval(sunlight);
    val[1] = pftypes[pft].wet.eval(moisture);
    val[2] = pftypes[pft].temp.eval(temperature);
    val[3] = pftypes[pft].slope.eval(slope);

    // the least satisfied resource dominates, so find min value
    vmin = val[0];
    for(int i = 1; i < 4; i++)
        vmin = fmin(vmin, val[i]);
    return vmin;
}

bool Biome::read_dataimporter(std::string cdata_fpath)
{
    data_importer::common_data cdata = data_importer::common_data(cdata_fpath);

    return read_dataimporter(cdata);
}

GLfloat *Biome::get_species_colour(int specid)
{
    return pftypes.at(specid).basecol;
}

bool Biome::read_dataimporter(data_importer::common_data &cdata)
{
    pftypes.clear();

    PFType pft;

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
        pftypes.push_back(pft);
    }

    slopethresh = cdata.soil_info.slopethresh;
    slopemax = cdata.soil_info.slopemax;
    evaporation = cdata.soil_info.evap;
    runofflevel = cdata.soil_info.runofflim;
    soilsaturation = cdata.soil_info.soilsat;
    waterlevel = cdata.soil_info.riverlevel;


    return true;
}

bool Biome::read(const std::string &filename)
{
    int nb, nc;
    ifstream infile;
    float o1, o2, z1, z2, i1, i2;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> name;

        // plant functional types
        infile >> nb; // number of pft
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
            else if (shapestr == "INVCONE")
                pft.shapetype = TreeShapeType::INVCONE;
            else
                cerr << "Error Biome::read: malformed shape type" << endl;
            pft.basecol[3] = 1.0f;
            pftypes.push_back(pft);

            // viability response values
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.sun.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.wet.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.temp.setValues(o1, o2, z1, z2, i1, i2);
            infile >> o1 >> z1 >> i1 >> i2 >> z2 >> o2;
            pft.slope.setValues(o1, o2, z1, z2, i1, i2);

            //infile >> pft.alpha;
            //infile >> pft.maxage;

            //< growth parameters
            /*
            infile >> pft.grow_months;
            infile >> pft.grow_m >> pft.grow_c1 >> pft.grow_c2;

            //< allometry parameters
            infile >> pft.alm_m >> pft.alm_c1;
            infile >> pft.alm_rootmult;*/
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
    float o1, o2, z1, z2, i1, i2;

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
            case TreeShapeType::INVCONE:
                outfile << "INVCONE";
                break;
            }
            outfile << endl;
            // viability response values
            pftypes[t].sun.getValues(o1, o2, z1, z2, i1, i2);
            outfile << o1 << " " << z1 << " " << i1 << " " << i2 << " " << z2 << " " << o2 << " ";
            pftypes[t].wet.getValues(o1, o2, z1, z2, i1, i2);
            outfile << o1 << " " << z1 << " " << i1 << " " << i2 << " " << z2 << " " << o2 << " ";
            pftypes[t].temp.getValues(o1, o2, z1, z2, i1, i2);
            outfile << o1 << " " << z1 << " " << i1 << " " << i2 << " " << z2 << " " << o2 << " ";
            pftypes[t].slope.getValues(o1, o2, z1, z2, i1, i2);
            outfile << o1 << " " << z1 << " " << i1 << " " << i2 << " " << z2 << " " << o2 << " ";
            outfile << pftypes[t].alpha << " " << pftypes[t].maxage << endl;

            //< growth parameters
            /*
            outfile << pftypes[t].grow_months << " ";
            outfile << pftypes[t].grow_m << " " << pftypes[t].grow_c1 << " " << pftypes[t].grow_c2 << " ";

            //< allometry parameters
            outfile << pftypes[t].alm_m << " " << pftypes[t].alm_c1 << " ";
            outfile << pftypes[t].alm_rootmult << endl;*/
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
