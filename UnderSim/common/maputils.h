/**
 * @file
 *
 * Some utility functions for dealing with maps.
 */

#ifndef UTS_COMMON_MAPUTILS_H
#define UTS_COMMON_MAPUTILS_H

#include <utility>
#include "map.h"

/**
 * Split a combined height-and-mask image into separate height and mask
 * images. The outputs are reallocated at the appropriate size.
 */
std::pair<MemMap<height_tag>, MemMap<mask_tag> >
demultiplex(const MemMap<height_and_mask_tag> &in);

#endif /* !UTS_COMMON_MAPUTILS_H */
