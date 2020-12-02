#ifndef _INC_FILL
#define _INC_FILL

#include <vector>
#include <list>
#include "view.h"
#include "typemap.h"
#include "terrain.h"

/**
 * Scan line polygon fill inside a loop stroke
 * @param loop          vertices of loop
 * @param ter           local terrain map
 * @param tmap          typemap to be polygon filled
 * @param brushtype     type value to write to map
 */
void scanLoopFill(std::vector<vpPoint> * loop, Terrain * ter, TypeMap * tmap, int brushtype);

/**
 * Scan line polygon fill mask inside a loop stroke
 * @param loop          vertices of loop
 * @param ter           local terrain map
 * @param mask          mask to be polygon filled
 */
void scanLoopMaskFill(std::vector<vpPoint> * loop, Terrain * ter, MemMap<bool> * mask);

# endif // _INC_FILL