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


#ifndef MAP_PROCS_H
#define MAP_PROCS_H

#include <vector>

template<typename T> class ValueGridMap;

ValueGridMap<float> average_monthly_data_hostcall(const std::vector<float> &data, int w, int h, float rw, float rh);
ValueGridMap<float> average_monthly_data_hostcall(const std::vector<ValueGridMap<float> > &data);

#endif 		// MAP_PROCS_H
