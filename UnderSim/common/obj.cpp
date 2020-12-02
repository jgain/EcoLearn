/**
 * @file
 *
 * OBJ format export.
 */

#include <fstream>
#include <stdexcept>
#include <locale>
#include "map.h"
#include "obj.h"

void writeOBJ(const uts::string &filename, const MemMap<height_tag> &map, const Region &region)
{
    try
    {
        if (region.empty())
            throw std::runtime_error("Empty region cannot be written to OBJ format");
        if (map.step() <= 0.0f)
            throw std::runtime_error("Map with zero step cannot be written to OBJ format");

        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Could not open file");

        out.imbue(std::locale::classic());
        out.exceptions(std::ios::failbit | std::ios::badbit);
        float scale = 0.001; // convert to km to keep size manageable
        float step = map.step() * scale;
        for (int y = region.y0; y < region.y1; y++)
            for (int x = region.x0; x < region.x1; x++)
            {
                // y and z are swapped to be consistent with the GUI
                out << "v " << x * step << ' ' << map[y][x] * scale << ' ' << y * step << '\n';
            }
        for (int y = 0; y < region.height() - 1; y++)
            for (int x = 0; x < region.width() - 1; x++)
            {
                int ids[4];
                ids[0] = y * region.width() + x + 1; // OBJ counts from 1
                ids[1] = ids[0] + region.width();
                ids[2] = ids[1] + 1;
                ids[3] = ids[0] + 1;
                out << "f " << ids[0] << ' ' << ids[1] << ' ' << ids[2] << ' ' << ids[3] << '\n';
            }
        out.close();
    }
    catch (std::runtime_error &e)
    {
        throw std::runtime_error("Failed to write " + filename + ": " + e.what());
    }
}
