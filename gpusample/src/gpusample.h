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


#include <vector>
#include <functional>

struct basic_tree;
class ClusterMaps;
class ClusterMatrices;
template<typename T> class ValueGridMap;


// See gpusample.cu for comments on all functions. Many functions are not declared here to prevent pollution of the
// namespace where gpusample.h is included

/*
 * Main function to be called from the host for sampling undergrowth plants.
 * This function computes adapted sunlight, after which it computes clusters/regions over the whole landscape, along
 * with required plant counts for each region, etc.
 * Then it samples undergrowth plants given the criteria of the regions computed
 */
std::vector<basic_tree> compute_and_sample_plants(ClusterMaps &clmaps, ClusterMatrices &model, const std::vector<basic_tree> &canopytrees, std::function<void(int)> update_func);

/*
 * Calculate sunshade from trees in 'trees' vector, based on base sunmap 'average_landsun'.
 */
ValueGridMap<float> calc_adaptsun(const std::vector<basic_tree> &trees, ValueGridMap<float> average_landsun, const data_importer::common_data &cdata, float rw, float rh);
