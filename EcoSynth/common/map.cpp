/**
 * @file
 *
 * Classes for 2D arrays of data.
 */

#include <ImfFrameBuffer.h>
#include <ImfInputFile.h>
#include <ImfFloatAttribute.h>
#include <algorithm>
#include <string>
#include <utility>
#include <cassert>
#include "map.h"
#include "str.h"
#include "terragen.h"
#include "obj.h"

namespace detail
{

float getOpenEXRStep(const Imf::InputFile &in)
{
    const Imf::FloatAttribute *s = in.header().findTypedAttribute<Imf::FloatAttribute>("heightMapResolution");
    if (s != NULL)
        return s->value();
    else
        return 0.0f;
}

uts::string mapArrayFilename(const uts::string &prefix, int slice)
{
    return prefix + "-" + std::to_string(slice) + ".exr";
}

void writeFrameBuffer(const uts::string &filename, float step, const Imf::FrameBuffer &fb,
                      const Region &region)
{
    Imath::Box2i window({region.x0, region.y0}, {region.x1 - 1, region.y1 - 1});
    Imf::Header header(window, window);
    if (step != 0.0f)
        header.insert("heightMapResolution", Imf::FloatAttribute(step));
    header.compression() = Imf::ZIP_COMPRESSION;
    for (auto i = fb.begin(); i != fb.end(); ++i)
        header.channels().insert(i.name(), Imf::Channel(i.slice().type));
    Imf::OutputFile out(filename.c_str(), header);
    out.setFrameBuffer(fb);
    out.writePixels(region.height());
}

} // namespace detail

/**********************************************************************/

void MapTraits<height_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("Y", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(const_cast<io_type *>(base)),
                              sizeof(io_type), sizeof(io_type) * width));
}

bool MapTraits<height_tag>::customRead(MemMap<height_tag> &out, const uts::string &filename)
{
    if (endsWith(filename, ".ter"))
    {
        out = readTerragen(filename);
        return true;
    }
    else
        return false;
}

bool MapTraits<height_tag>::customWrite(
    const MemMap<height_tag> &out, const uts::string &filename, const Region &region)
{
    if (endsWith(filename, ".ter"))
    {
        writeTerragen(filename, out, region);
        return true;
    }
    else if (endsWith(filename, ".obj"))
    {
        writeOBJ(filename, out, region);
        return true;
    }
    else
        return false;
}

void MapTraits<distance_field_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("dist", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].dist),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("param", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].param),
                                 sizeof(io_type), sizeof(io_type) * width));
}

constexpr MapTraits<mask_tag>::type MapTraits<mask_tag>::all;
constexpr MapTraits<mask_tag>::type MapTraits<mask_tag>::none;
constexpr int MapTraits<mask_tag>::numTypes;

void MapTraits<mask_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("mask", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(const_cast<io_type *>(base)),
                                 sizeof(io_type), sizeof(io_type) * width));
}

void MapTraits<height_and_mask_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("Y", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].height),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("mask", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].mask),
                                 sizeof(io_type), sizeof(io_type) * width));
}

bool MapTraits<height_and_mask_tag>::customRead(MemMap<height_and_mask_tag> &out, const uts::string &filename)
{
    if (endsWith(filename, ".ter"))
    {
        MemMap<height_tag> heights = readTerragen(filename);
        out.allocate(heights.region());
        out.setStep(heights.step());
        const Region &r = heights.region();
#pragma omp parallel for schedule(static)
        for (int y = r.y0; y < r.y1; y++)
            for (int x = r.x0; x < r.x1; x++)
            {
                out[y][x].height = heights[y][x];
                out[y][x].mask = MapTraits<mask_tag>::all;
            }
        return true;
    }
    else
        return false;
}

bool MapTraits<height_and_mask_tag>::customWrite(
    const MemMap<height_and_mask_tag> &out, const uts::string &filename, const Region &region)
{
    if (endsWith(filename, ".ter") || endsWith(filename, ".obj"))
    {
        MemMap<height_tag> tmp(region);
        tmp.setStep(out.step());
#pragma omp parallel for schedule(static)
        for (int y = region.y0; y < region.y1; y++)
            for (int x = region.x0; x < region.x1; x++)
                tmp[y][x] = out[y][x].height;
        tmp.write(filename);
        return true;
    }
    else
        return false;
}

void MapTraits<appearance_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    static_assert(APPEARANCE_MODES == 4, "Need to update for new APPEARANCE_MODES");
    fb.insert("R", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][0]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("G", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][1]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("B", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][2]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("appearance.3", Imf::Slice(Imf::UINT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0][3]),
                              sizeof(io_type), sizeof(io_type) * width));
}

void MapTraits<appearance_tag>::convert(type in, io_type &out)
{
    std::copy(begin(in), end(in), begin(out));
}

void MapTraits<appearance_tag>::convert(io_type in, type &out)
{
    std::copy(begin(in), end(in), begin(out));
}

void MapTraits<coords_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    fb.insert("R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].flat[0]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].flat[1]),
                              sizeof(io_type), sizeof(io_type) * width));
}

void MapTraits<coords_offset_tag>::prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t width)
{
    // This can't chain to MapTraits<coords_tag>, because sizeof(io_type) is different
    fb.insert("R", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].flat[0]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("G", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].flat[1]),
                              sizeof(io_type), sizeof(io_type) * width));
    fb.insert("B", Imf::Slice(Imf::FLOAT, reinterpret_cast<char *>(&const_cast<io_type *>(base)[0].offset),
                                   sizeof(io_type), sizeof(io_type) * width));
}

/**********************************************************************/

int MapRegion::width() const
{
    return region_.width();
}

int MapRegion::height() const
{
    return region_.height();
}

const Region &MapRegion::region() const
{
    return region_;
}

void MapRegion::setRegion(const Region &newRegion)
{
    region_ = newRegion;
}

void MapRegion::translateTo(int x, int y)
{
    setRegion(Region(x, y, x + region_.width(), y + region_.height()));
}

/**********************************************************************/

float MapBase::step() const
{
    return step_;
}

void MapBase::setStep(float step)
{
    assert(step >= 0.0f);
    step_ = step;
}

/**********************************************************************/

int MapArrayBase::arraySize() const
{
    return step_.size();
}

float MapArrayBase::step(int slice) const
{
    assert(0 <= slice && slice < arraySize());
    return step_[slice];
}

const uts::vector<float> &MapArrayBase::step() const
{
    return step_;
}

void MapArrayBase::reserveStep(int newSize)
{
    step_.reserve(newSize);
}

void MapArrayBase::resizeStep(int newSize)
{
    if (std::size_t(newSize) != step_.size())
    {
        step_.resize(newSize);
        step_.shrink_to_fit();
    }
}

void MapArrayBase::setStep(int slice, float step)
{
    assert(0 <= slice && slice < arraySize());
    step_[slice] = step;
}

void MapArrayBase::setStep(const MapArrayBase &other)
{
    assert(other.step_.size() == step_.size());
    step_ = other.step_;
}

void MapArrayBase::setStep(uts::vector<float> &&step)
{
    assert(step.size() == step_.size());
    step_ = std::move(step);
}

/**********************************************************************/

void Map::allocate(const Region &region)
{
    assert(region.x0 <= region.x1);
    assert(region.y0 <= region.y1);
    if (this->region().width() != region.width()
        || this->region().height() != region.height())
        allocateImpl(region.width(), region.height());
    setRegion(region);
}

void Map::clear()
{
    allocate(Region());
    setStep(0.0f);
}

std::tuple<Region, float> Map::fileDimensions(const uts::string &filename)
{
    Imf::InputFile in(filename.c_str());
    auto dw = in.header().dataWindow();
    Region r(dw.min.x, dw.min.y, dw.max.x + 1, dw.max.y + 1);
    float step = detail::getOpenEXRStep(in);
    return std::make_tuple(r, step);
}

/**********************************************************************/

void MapArray::allocate(const Region &region, int arraySize)
{
    assert(region.x0 <= region.x1);
    assert(region.y0 <= region.y1);

    reserveStep(arraySize);
    if (arraySize != this->arraySize()
        || region.width() != this->region().width()
        || region.height() != this->region().height())
    {
        allocateImpl(region.width(), region.height(), arraySize);
        resizeStep(arraySize);
    }
    setRegion(region);
}

void MapArray::clear()
{
    allocate(Region(), 0);
}

/**********************************************************************/

template class MemMap<height_tag>;
template class MemMap<distance_field_tag>;
template class MemMap<mask_tag>;
template class MemMap<height_and_mask_tag>;
template class MemMap<appearance_tag>;
template class MemMap<coords_tag>;
template class MemMap<coords_offset_tag>;
template class MemMap<height_constraint_tag>;

template class MemMapArray<height_tag>;
template class MemMapArray<appearance_tag>;
template class MemMapArray<coords_tag>;
template class MemMapArray<coords_offset_tag>;
