/**
 * @file
 *
 * Specialization of @ref MapTraits for color images.
 */

#ifndef UTS_COMMON_MAP_COLOR_H
#define UTS_COMMON_MAP_COLOR_H

#include <array>
#include <common/debug_string.h>
#include <common/debug_vector.h>
#include "map.h"
#include "region.h"

struct gray_tag {};
struct rgba_tag {};

template<>
class MapTraits<gray_tag>
{
public:
    typedef float type;
    typedef type io_type;
    typedef std::false_type needs_conversion;

    /**
     * Reads from an arbitrary file format using ImageMagick. If the file contains
     * color, it is converted to grayscale.
     *
     * @return whether the file format was recognised.
     */
    static bool customRead(MemMap<gray_tag> &out, const uts::string &filename);

    /**
     * Writes to an arbitrary file format using ImageMagick.
     *
     * @return whether the file format was recognised.
     */
    static bool customWrite(const MemMap<gray_tag> &out, const uts::string &filename,
                            const Region &region);

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width);
};

template<>
class MapTraits<rgba_tag>
{
public:
    typedef std::array<float, 4> type;
    typedef type io_type;
    typedef std::false_type needs_conversion;

    /**
     * Reads from an arbitrary file format using ImageMagick.
     *
     * @return whether the file format was recognised.
     */
    static bool customRead(MemMap<rgba_tag> &out, const uts::string &filename);
    /**
     * Writes to an arbitrary file format using ImageMagick.
     *
     * @return whether the file format was recognised.
     */
    static bool customWrite(const MemMap<rgba_tag> &in, const uts::string &filename, const Region &region);

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width);

    /**
     * Color palette used for type painting
     */
    static const uts::vector<type> &colorPalette();
};

#include <set>
#include <iostream>

/**
 * Look up colors in the color palette and generate corresponding type masks.
 *
 * @param region     Region to allocate for the output.
 * @param paint      Color image
 * @param callback   Callback provided with x, y and mask for each element of region
 *
 * @pre @a paint.region() is a superset of @a region
 */
template<typename Callback>
void colorsToMasks(const Region &region, const MemMap<rgba_tag> &paint, const Callback &callback)
{
    typedef MapTraits<mask_tag>::type mask_t;
    typedef MapTraits<rgba_tag>::type color;
    const uts::vector<color> &palette = MapTraits<rgba_tag>::colorPalette();

    assert(paint.region().contains(region));

    // Apply labels to the map
    std::set<color> seenBad; // unrecognised colors - saved to avoid duplicate errors

#pragma omp parallel for schedule(static)
    for (int y = region.y0; y < region.y1; y++)
        for (int x = region.x0; x < region.x1; x++)
        {
            mask_t value = MapTraits<mask_tag>::all;
            if (paint[y][x][3] != 0) // transparent
            {
                int label = -1;
                for (std::size_t i = 0; i < palette.size(); i++)
                    if (paint[y][x] == palette[i])
                    {
                        label = i;
                        break;
                    }
                if (label == -1)
                {
#pragma omp critical
                    {
                        if (seenBad.insert(paint[y][x]).second)
                        {
                            std::cerr << "Warning: unrecognized color:";
                            for (float c : paint[y][x])
                                std::cerr << ' ' << c;
                            std::cerr << '\n';
                        }
                    }
                }
                else
                    value = ~(mask_t(1) << label);
            }
            callback(x, y, value);
        }
}

#endif /* !UTS_COMMON_MAP_COLOR_H */
