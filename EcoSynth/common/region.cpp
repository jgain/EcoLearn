/**
 * @file
 *
 * Data structure representing a rectangular region of 2D space.
 */

#include <cassert>
#include <algorithm>
#include "region.h"

Region::Region()
    : x0(0), y0(0), x1(0), y1(0)
{
}

Region::Region(int x0, int y0, int x1, int y1)
    : x0(x0), y0(y0), x1(x1), y1(y1)
{
    assert(x0 <= x1);
    assert(y0 <= y1);
}

Region Region::operator|(const Region &other) const
{
    // Empty regions need special treatment, because otherwise they are
    // treated as containing the point (0, 0).
    if (empty())
        return other;
    else if (other.empty())
        return *this;
    else
        return Region(
            std::min(x0, other.x0), std::min(y0, other.y0),
            std::max(x1, other.x1), std::max(y1, other.y1));
}

Region &Region::operator|=(const Region &other)
{
    return *this = *this | other;
}

Region Region::operator&(const Region &other) const
{
    Region ans = *this;
    return ans &= other;
}

Region &Region::operator&=(const Region &other)
{
    x0 = std::max(x0, other.x0);
    y0 = std::max(y0, other.y0);
    x1 = std::min(x1, other.x1);
    y1 = std::min(y1, other.y1);
    if (empty())
    {
        x0 = 0;
        y0 = 0;
        x1 = 0;
        y1 = 0;
    }
    return *this;
}

bool Region::operator==(const Region &other) const
{
    return x0 == other.x0 && y0 == other.y0 && x1 == other.x1 && y1 == other.y1;
}

bool Region::operator!=(const Region &other) const
{
    return !(*this == other);
}

int Region::width() const
{
    return x1 - x0;
}

int Region::height() const
{
    return y1 - y0;
}

bool Region::empty() const
{
    return x0 >= x1 || y0 >= y1;
}

std::size_t Region::pixels() const
{
    return empty() ? 0 : std::size_t(width()) * height();
}

Region Region::dilate(int border) const
{
    Region out;
    if (!empty())
    {
        out.x0 = x0 - border;
        out.x1 = x1 + border;
        out.y0 = y0 - border;
        out.y1 = y1 + border;
        if (out.empty())
            out = Region(); // make it the canonical empty region
    }
    return out;
}

Region Region::erode(int border) const
{
    return dilate(-border);
}

bool Region::contains(const Region &other) const
{
    if (other.empty())
        return true;
    else
        return x0 <= other.x0 && y0 <= other.y0 && x1 >= other.x1 && y1 >= other.y1;
}
