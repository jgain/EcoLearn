/**
 * @file
 *
 * Specialization of @ref MapTraits for color images.
 */

#include <cmath>
#include <Magick++.h>
#include <ImfFrameBuffer.h>
#include <thread>
#include "map.h"
#include "map_rgba.h"
#include "str.h"

static constexpr Magick::Quantum getQuantumRange()
{
    using namespace Magick; // otherwise the macro breaks
    return QuantumRange;
}

static Magick::Quantum floatToQuantum(float x)
{
    if (!(x >= 0.0f)) // also catches NaN
        x = 0.0f;
    if (x > 1.0f)
        x = 1.0f;
    return static_cast<Magick::Quantum>(std::round(x * getQuantumRange()));
}

bool MapTraits<gray_tag>::customRead(MemMap<gray_tag> &out, const uts::string &filename)
{
    if (endsWith(filename, ".exr"))
        return false; // use OpenEXR for EXR images (default path)

    Magick::Image image(filename);
    // This is needed to make the grayscale conversion happen in linear space
    image.image()->intensity = MagickCore::Rec709LuminancePixelIntensityMethod;
    image.colorSpace(Magick::GRAYColorspace);

    Region r(0, 0, image.columns(), image.rows());
    out.allocate(r);
    image.write(0, 0, r.x1, r.y1, "I", Magick::FloatPixel, out.get());
    return true;
}

bool MapTraits<gray_tag>::customWrite(const MemMap<gray_tag> &in, const uts::string &filename,
                                  const Region &region)
{
    if (endsWith(filename, ".exr"))
        return false;

    const Region &r = in.region();
    Magick::Image image(Magick::Geometry(r.width(), r.height()), Magick::Color(0, 0, 0, 0));
    image.colorSpace(Magick::RGBColorspace);

    Magick::Pixels view(image);
    Magick::PixelPacket *pixels = view.set(0, 0, r.width(), r.height());
    for (int y = r.y0; y < r.y1; y++)
        for (int x = r.x0; x < r.x1; x++, pixels++)
        {
            Magick::Quantum v = floatToQuantum(in[y][x]);
            pixels->red = v;
            pixels->green = v;
            pixels->blue = v;
            pixels->opacity = 0;
        }
    view.sync();

    /* Convert to sRGB. This shouldn't be necessary, but without it, the PNG
     * encoder writes linear values but stores a gAMA chunk of 1/2.2.
     */
    image.colorSpace(Magick::sRGBColorspace);
    image.write(filename);
    return true;
}

void MapTraits<gray_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    MapTraits<height_tag>::prepareFrameBuffer(fb, base, width);
}

bool MapTraits<rgba_tag>::customRead(MemMap<rgba_tag> &out, const uts::string &filename)
{
    if (endsWith(filename, ".exr"))
        return false; // use OpenEXR for EXR images (default path)

    Magick::Image image(filename);
    image.colorSpace(Magick::RGBColorspace);

    Region r(0, 0, image.columns(), image.rows());
    out.allocate(r);

    image.write(0, 0, r.x1, r.y1, "RGBA", Magick::FloatPixel, out.get());
    return true;
}

bool MapTraits<rgba_tag>::customWrite(const MemMap<rgba_tag> &in, const uts::string &filename,
                                      const Region &region)
{
    if (endsWith(filename, ".exr"))
        return false;

    const Region &r = in.region();
    Magick::Image image(Magick::Geometry(r.width(), r.height()), Magick::Color(0, 0, 0, getQuantumRange()));
    image.colorSpace(Magick::RGBColorspace);

    Magick::Pixels view(image);
    Magick::PixelPacket *pixels = view.set(0, 0, r.width(), r.height());
    for (int y = r.y0; y < r.y1; y++)
        for (int x = r.x0; x < r.x1; x++, pixels++)
        {
            pixels->red = floatToQuantum(in[y][x][0]);
            pixels->green = floatToQuantum(in[y][x][1]);
            pixels->blue = floatToQuantum(in[y][x][2]);
            // ImageMagick uses reverse alpha
            pixels->opacity = getQuantumRange() - floatToQuantum(in[y][x][3]);
        }
    view.sync();

    /* Convert to sRGB. The ImageMagick decoder doesn't understand that gamma=1
     * means a linear image, so we have to write an sRGB image. Many other decoders
     * are also unlikely to get this right.
     */
    image.colorSpace(Magick::sRGBColorspace);
    image.write(filename);
    return true;
}

void MapTraits<rgba_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][0]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][1]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][2]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("A", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][3]),
                              sizeof(io_type), sizeof(io_type) * width));
}

const uts::vector<MapTraits<rgba_tag>::type> &MapTraits<rgba_tag>::colorPalette()
{
    static const uts::vector<type> palette{
        {{ 0.0f, 0.0f, 0.0f, 1.0f }},
        {{ 0.0f, 0.0f, 1.0f, 1.0f }},
        {{ 0.0f, 1.0f, 0.0f, 1.0f }},
        {{ 1.0f, 0.0f, 0.0f, 1.0f }},
        {{ 1.0f, 1.0f, 0.0f, 1.0f }},
        {{ 1.0f, 1.0f, 1.0f, 1.0f }},
        {{ 1.0f, 0.0f, 1.0f, 1.0f }},
        {{ 0.0f, 1.0f, 1.0f, 1.0f }}
    };

    return palette;
}
