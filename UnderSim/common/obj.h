/**
 * @file
 *
 * OBJ format export.
 */

#ifndef UTS_COMMON_OBJ_H
#define UTS_COMMON_OBJ_H

#include <common/debug_string.h>
#include "map.h"

/**
 * Save an OBJ file.
 */
void writeOBJ(const uts::string &filename, const MemMap<height_tag> &map, const Region &region);

#endif /* !UTS_COMMON_TERRAGEN_H */
