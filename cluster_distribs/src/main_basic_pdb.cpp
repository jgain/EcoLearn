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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include <iostream>
//#include "distribution.h"
#include "GridDistribs.h"
#include "dice.h"

int main(int argc, char * argv [])
{
    GridDistribs distribs("/home/konrad/PhDStuff/prototypes/repo/data/basic_posses1.pdb", 32, 32, 32, 32, 8, {0, 10, 20, 30, 40, 50, 60}, "/home/konrad/PhDStuff/prototypes/repo/data/vh0090.png");
    distribs.analyse_all_sandboxes_src();
    distribs.write_sandboxes("/home/konrad/PhDStuff/prototypes/repo/data/basic_posses1.json");
    return 0;
}
