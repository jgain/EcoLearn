/**
 * @file
 *
 * Some utility functions for dealing with maps.
 */

#include <utility>
#include "map.h"
#include "maputils.h"

std::pair<MemMap<height_tag>, MemMap<mask_tag> >
demultiplex(const MemMap<height_and_mask_tag> &in)
{
    std::pair<MemMap<height_tag>, MemMap<mask_tag> > out;
    const Region region = in.region();

    auto &height = out.first;
    height.allocate(region);
    height.setStep(in.step());
    auto &mask = out.second;
    mask.allocate(region);
    mask.setStep(in.step());

#pragma omp parallel for schedule(static) default(none) shared(in, height, mask)
    for (int y = region.y0; y < region.y1; y++)
        for (int x = region.x0; x < region.x1; x++)
        {
            height[y][x] = in[y][x].height;
            mask[y][x] = in[y][x].mask;
        }

    return out;
}
