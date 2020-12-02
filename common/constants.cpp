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


#include "constants.h"
#include "cluster_distribs/src/HistogramDistrib.h"
#include "cluster_distribs/src/HistogramMatrix.h"

HistogramDistrib::Metadata HistogramMatrix::global_canopy_metadata = {
    5.0f,		// maxdist
    3,			// nreserved_bins
    5,			// nreal_bins
    8,			// ntotal_bins
    5.0f / 5	// binwidth
};

HistogramDistrib::Metadata HistogramMatrix::global_under_metadata = {
    5.0f,		// maxdist
    3,			// nreserved_bins
    5,			// nreal_bins
    8,			// ntotal_bins
    5.0f / 5	// binwidth
};
