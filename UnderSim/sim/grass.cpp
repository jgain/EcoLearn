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


#include "grass.h"
#include "dice_roller.h"

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
