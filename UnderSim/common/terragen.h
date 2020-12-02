/**
 * @file
 *
 * Terragen format import and export.
 */

#ifndef UTS_COMMON_TERRAGEN_H
#define UTS_COMMON_TERRAGEN_H

#include <common/debug_string.h>
#include "map.h"

/**
 * Load a Terragen file.
 */
MemMap<height_tag> readTerragen(const uts::string &filename);

/**
 * Save a Terragen file.
 */
void writeTerragen(const uts::string &filename, const MemMap<height_tag> &map, const Region &region);

#endif /* !UTS_COMMON_TERRAGEN_H */
