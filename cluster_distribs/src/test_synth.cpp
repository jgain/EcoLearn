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


#include "GridDistribs.h"

int main(int argc, char * argv [])
{
    //GridDistribs d("/home/konrad/PhDStuff/prototypes/repo/data/test_all_hists_out.json");
    //GridDistribs d("/home/konrad/PhDStuff/prototypes/repo/data/basic_test.json");
    GridDistribs d("/home/konrad/PhDStuff/prototypes/repo/data/optim90.pdb", 234, 234, 4, 4, 4, {0, 10, 20, 30, 40, 50, 60}, "/home/konrad/PhDStuff/prototypes/repo/data/vh0090.png");
    d.analyse_all_sandboxes_src();
    std::cout << "Synthesizing..." << std::endl;
    int npasses = 1;
    int movetol = 0;
    int rolltol = 20;
    d.synthesize(npasses, movetol, rolltol, 0, false);
    std::cout << "done" << std::endl;
    d.write_synth_to_pdb("/home/konrad/PhDStuff/prototypes/data/test_gridsynth_done.pdb");
    //d.write_synth_to_pdb("/home/konrad/PhDStuff/prototypes/repo/data/basic_test_synth.pdb");

	return 0;
}
