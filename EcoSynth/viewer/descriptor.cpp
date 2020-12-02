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

#include "descriptor.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
// #include <sstream>

using namespace std;

bool MapDescriptor::read(std::string filename)
{
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
                SampleDescriptor sd;
                infile >> sd.slope;
                infile >> sd.moisture[0] >> sd.moisture[1];
                infile >> sd.sunlight[0] >> sd.sunlight[1];
                infile >> sd.temperature[0] >> sd.temperature[1];
                infile >> sd.age;
                set(x, y, sd);
            }
        }
        infile.close();
        return true;
    }
    else
    {
        cerr << "Error MapDescriptor::read: unable to open file" << filename << endl;
        return false;
    }
}

bool MapDescriptor::write(std::string filename)
{
    ofstream outfile;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << gx << " " << gy << endl;

        for (int x = 0; x < gx; x++)
            for (int y = 0; y < gy; y++)
            {
                SampleDescriptor sd = get(x, y);
                outfile << sd.slope << " ";
                outfile << sd.moisture[0] << " " << sd.moisture[1] << " ";
                outfile << sd.sunlight[0] << " " << sd.sunlight[1] << " ";
                outfile << sd.temperature[0] << " " << sd.temperature[1] << " ";
                outfile << sd.age << " ";
            }

        outfile.close();
        return true;
    }
    else
    {
        cerr << "Error MapDescriptor::write: unable to write file" << filename << endl;
        return false;
    }

}
