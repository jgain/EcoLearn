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
    //GridDistribs::test_histogram_modify_oneclass_oneblock();
    //GridDistribs::test_histogram_modify_multiclass_oneblock();
    //GridDistribs::test_histogram_modify_oneclass_3x3blocks();
    //GridDistribs::test_histogram_modify_multiclass_3x3blocks();
    //GridDistribs::test_histogram_modify_multiclass_5x5blocks();
    //GridDistribs::test_write_read_synth_stat_data(true);
    GridDistribs::test_synth_from_distrib_and_plants();
    //GridDistribs::time_synthesis();

    return 0;
}
