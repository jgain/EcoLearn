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


#ifndef CLUSTERASSIGN_H
#define CLUSTERASSIGN_H

#include "EcoSynth/kmeans/src/kmeans.h"
#include "AllClusterInfo.h"
#include <vector>

class abiotic_maps_package;

/*
 * Class that assigns clusters based on a trained kmeans model ('means' member), and can also train a kmeans model
 * based on data that we pass to it. 
 * 
 * This could also have been subclassed from the kMeans class.
 */

class ClusterAssign
{
public:
    ClusterAssign();
    ClusterAssign(const std::vector<std::array<float, 4> > &means, const std::array< std::pair<float, float>, 4> &minmax_ranges);
    ClusterAssign(const AllClusterInfo clusterinfo);

	/*
	 * Assign cluster based on 4 abiotic conditions
	 */
    int assign(float moisture, float sun, float slope, float temp) const;

	/*
	 * Getter for number of means
	 */
    int get_nmeans() const;

	/*
	 * Check if this object has been assigned a trained kmeans model
	 */
    bool has_model() const;

	/*
	 * Get the minmax ranges which we use to scale values. These have to be the same minmax ranges that the model
	 * was trained with
	 */
    const std::array<std::pair<float, float>, 4> &get_minmax_ranges() const;

	/*
	 * Get the means that resulted from model training
	 */
    const std::vector<std::array<float, 4> > &get_means() const;
	
	/* 
	 * Train a kmeans model based on the 'all_maps' parameter, and assign it to this object
	 */
    void do_kmeans(const std::vector<abiotic_maps_package> &all_amaps, int nmeans, int niters, const data_importer::common_data &cdata);
	/* 
	 * Train a kmeans model based on the data imported based on the 'datadirs' string parameter, and assign it to this object
	 */
    void do_kmeans(const std::vector<std::string> &datadirs, int nmeans, int niters, const data_importer::common_data &cdata);
private:
    std::array<std::pair<float, float>, 4> minmax_ranges;
    std::vector<std::array<float, 4> > means;
};

#endif		// CLUSTERASSIGN_H
