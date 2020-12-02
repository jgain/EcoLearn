/**
 * @file
 *
 * Terragen format import and export.
 * @see http://www.planetside.co.uk/terragen/dev/tgterrain.html
 */

#include <fstream>
#include <algorithm>
#include <limits>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <locale>
#include <cstdint>
#include <cmath>
#include <common/debug_string.h>
#include "map.h"

/// Reads a binary little-endian float
static float readFloat(std::istream &in)
{
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "float is not IEEE-754 compliant");
    unsigned char data[4];
    std::uint32_t host = 0;
    float out;

    in.read(reinterpret_cast<char *>(data), 4);
    for (int i = 0; i < 4; i++)
        host |= std::uint32_t(data[i]) << (i * 8);
    std::memcpy(&out, &host, sizeof(out));
    return out;
}

/// Writes a binary little-endian float
static void writeFloat(std::ostream &out, float f)
{
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "float is not IEEE-754 compliant");
    std::uint32_t host;
    unsigned char data[4];

    std::memcpy(&host, &f, sizeof(host));
    for (int i = 0; i < 4; i++)
        data[i] = (host >> (i * 8)) & 0xff;
    out.write(reinterpret_cast<char *>(data), 4);
}

/// Reads a binary little-endian 16-bit unsigned int
static std::uint16_t readUint16(std::istream &in)
{
    unsigned char data[2];
    in.read(reinterpret_cast<char *>(data), 2);
    return data[0] | (std::uint16_t(data[1]) << 8);
}

/// Writes a binary little-endian 16-bit unsigned int
static void writeUint16(std::ostream &out, std::uint16_t v)
{
    unsigned char data[2];
    data[0] = v & 0xff;
    data[1] = v >> 8;
    out.write(reinterpret_cast<char *>(data), 2);
}

/// Reads a binary little-endian 16-bit signed int
static std::int16_t readInt16(std::istream &in)
{
    return static_cast<std::int16_t>(readUint16(in));
}

/// Writes a binary little-endian 16-bit signed int
static void writeInt16(std::ostream &out, std::int16_t v)
{
    writeUint16(out, static_cast<std::uint16_t>(v));
}

MemMap<height_tag> readTerragen(const uts::string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open " + filename);

    MemMap<height_tag> ans;
    try
    {
        in.imbue(std::locale::classic());
        in.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
        char signature[16];
        in.read(signature, 16);
        if (uts::string(signature, 16) != "TERRAGENTERRAIN ")
        {
            throw std::runtime_error("signature did not match");
        }

        int width = -1;
        int height = -1;
        float step = 0.0f;
        while (true)
        {
            // Markers are aligned to 4-byte boundaries
            auto pos = in.tellg();
            if (pos & 3)
                in.seekg(4 - (pos & 3), std::ios::cur);

            char markerData[4];
            in.read(markerData, 4);
            uts::string marker(markerData, 4);
            if (marker == "XPTS")
                width = readUint16(in);
            else if (marker == "YPTS")
                height = readUint16(in);
            else if (marker == "SIZE")
                width = height = readUint16(in) + 1;
            else if (marker == "SCAL")
            {
                float stepX = readFloat(in);
                float stepY = readFloat(in);
                float stepZ = readFloat(in);
                if (stepY != stepX || stepZ != stepX)
                {
                    std::cerr << "stepX = " << stepX << " stepY = " << stepY << " stepZ = " << stepZ << std::endl;
                    // throw std::runtime_error("SCAL values are not all equal");
                }
                else if (stepX <= 0.0f)
                    throw std::runtime_error("SCAL value is negative");
                step = stepX;
            }
            else if (marker == "CRAD")
                readFloat(in); // radius of planet
            else if (marker == "CRVM")
                in.ignore(4);
            else if (marker == "ALTW")
            {
                if (step == 0.0f)
                {
                    std::cerr << "Warning: no scale found. Using spec default of 30\n";
                    step = 30.0f;
                }
                float heightScale = readInt16(in) / 65536.0f;
                float baseHeight = readInt16(in);
                if (width <= 0 || height <= 0)
                    throw std::runtime_error("ALTW found before dimensions");

                ans.setStep(step);
                ans.allocate({0, 0, width, height});
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        float h = readInt16(in);
                        h = baseHeight + heightScale * h;
                        h = h * step;
                        ans[y][x] = h;
                    }
            }
            else if (marker == "EOF ")
                break;
            else
                throw std::runtime_error("unexpected chunk `" + marker + "'");
        }
    }
    catch (std::runtime_error &e)
    {
        throw std::runtime_error("Failed to read " + filename + ": " + e.what());
    }
    return ans;
}

void writeTerragen(const uts::string &filename, const MemMap<height_tag> &map, const Region &region)
{
    try
    {
        if (region.width() >= 65536 || region.height() >= 65536)
            throw std::runtime_error("Region is too large for Terragen format");
        if (region.empty())
            throw std::runtime_error("Empty region cannot be written to Terragen format");
        if (map.step() <= 0.0f)
            throw std::runtime_error("Map with zero step cannot be written to Terragen format");

        float minHeight = std::numeric_limits<float>::infinity();
        float maxHeight = -std::numeric_limits<float>::infinity();
// #pragma omp parallel for schedule(static) reduction(min:minHeight) reduction(max:maxHeight)
        for (int y = region.y0; y < region.y1; y++)
            for (int x = region.x0; x < region.x1; x++)
            {
                float h = map[y][x];
                minHeight = std::min(minHeight, h);
                maxHeight = std::max(maxHeight, h);
            }
        if (!std::isfinite(minHeight) || !std::isfinite(maxHeight))
            throw std::runtime_error("Non-finite heights cannot be written to Terragen format");

        float baseHeight = std::round((minHeight + maxHeight) * 0.5 / map.step());
        if (baseHeight < INT16_MIN || baseHeight > INT16_MAX)
            throw std::runtime_error("Heights are too large for Terragen format");
        float heightScale = std::max(
            (minHeight / map.step() - baseHeight) * 65536 / INT16_MIN,
            (maxHeight / map.step() - baseHeight) * 65536 / INT16_MAX);
        heightScale = std::ceil(heightScale);
        if (heightScale > INT16_MAX)
            throw std::runtime_error("Height range is too large for Terragen format");

        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Could not open file");

        out.imbue(std::locale::classic());
        out.exceptions(std::ios::failbit | std::ios::badbit);

        out.write("TERRAGENTERRAIN ", 16);
        out.write("SIZE", 4);
        writeUint16(out, std::min(region.width(), region.height()) - 1);
        writeUint16(out, 0); // padding
        out.write("XPTS", 4);
        writeUint16(out, region.width());
        writeUint16(out, 0); // padding
        out.write("YPTS", 4);
        writeUint16(out, region.height());
        writeUint16(out, 0); // padding
        out.write("SCAL", 4);
        writeFloat(out, map.step());
        writeFloat(out, map.step());
        writeFloat(out, map.step());
        out.write("ALTW", 4);
        writeInt16(out, std::int16_t(heightScale));
        writeInt16(out, std::int16_t(baseHeight));

        float scale = 65536.0f / (map.step() * heightScale);
        float bias = -baseHeight * 65536.0f / heightScale;
        for (int y = region.y0; y < region.y1; y++)
            for (int x = region.x0; x < region.x1; x++)
            {
                float h = map[y][x] * scale + bias;
                writeInt16(out, std::int16_t(std::round(h)));
            }
        if (region.pixels() & 1)
            writeInt16(out, 0); // padding
        out.write("EOF ", 4);
        out.close();
    }
    catch (std::runtime_error &e)
    {
        throw std::runtime_error("Failed to write " + filename + ": " + e.what());
    }

}
