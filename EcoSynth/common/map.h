/**
 * @file
 *
 * Classes for 2D arrays of data.
 */

#ifndef UTS_COMMON_MAP_H
#define UTS_COMMON_MAP_H

#include <array>
#include <string>
#include <cstdint>
#include <memory>
#include <limits>
#include <cassert>
#include <utility>
#include <type_traits>
#include <tuple>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#include <ImfFrameBuffer.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
#include <ImfTestFile.h>
#include "region.h"
#include "debug_vector.h"
#include "debug_string.h"
#include "serialize.h"

/// Tag class for heights (currently 32-bit floating point)
struct height_tag {};

/// Tag class for distance field. Pair for distance to closest point and its index
struct distance_field_tag {};

/**
 * Tag class for terrain type masks. Masks specify a bitfield of terrain
 * types that are @em not permitted/provided at a particular point. This
 * choice make a default of zero represent any type.
 */
struct mask_tag {};

/// Tag class for pair of height and mask
struct height_and_mask_tag {};

/**
 * Tag class for appearance-space feature vectors. The values are quantised to
 * 16 bit unsigned normalized.
 */
struct appearance_tag {};

/**
 * Tag for coordinates indexing the exemplars. The internal representation is
 * a packed 32-bit integer (11:11:10) for x, y, exemplar.
 */
struct coords_tag {};

/// Tag for height constraint map (currently a height and an influence factor)
struct height_constraint_tag {};

/// Like @ref coords_tag, but also includes a height adjustment as a 32-bit float.
struct coords_offset_tag {};

constexpr int APPEARANCE_MODES = 4;

/**
 * Generic traits class templated on one of the tags. Only the specializations
 * are serializable, and use tags instead of the actual storage type. A
 * specialization is serializable if it has a @c prepareFrameBuffer method.
 * It may also provide @c customRead and/or @c customWrite methods to support
 * other file formats.
 */
template<typename T>
class MapTraits
{
public:
    typedef T type;
};

namespace detail
{

/**
 * Creates a traits class that determines whether a tag's @c MapTraits has
 * a particular member.
 *
 * @param className      Name for this helper class
 * @param member         Member to test for
 */
#define DEFINE_MAP_TRAITS_HELPER_CLASS(className, member) \
    template<typename T> \
    class className \
    { \
    private: \
        /* Eliminated by SFINAE if member not defined */ \
        template<typename U> \
        static constexpr bool helper(decltype(MapTraits<U>::member) *dummy) \
        { \
            return true; \
        } \
        /* Fallback if SFINAE kicks in */ \
        template<typename U> \
        static constexpr bool helper(...) \
        { \
            return false; \
        } \
    public: \
        static constexpr bool value = helper<T>(nullptr); \
        typedef bool value_type; \
        typedef std::integral_constant<bool, value> type; \
    }

DEFINE_MAP_TRAITS_HELPER_CLASS(Serializable, prepareFrameBuffer);
DEFINE_MAP_TRAITS_HELPER_CLASS(HasCustomRead, customRead);
DEFINE_MAP_TRAITS_HELPER_CLASS(HasCustomWrite, customWrite);

template<typename Owner>
class RowView
{
public:
    typedef typename std::conditional<
        std::is_const<Owner>::value,
        const typename Owner::value_type,
        typename Owner::value_type>::type value_type;
    typedef value_type &reference;

private:
    Owner &owner;
    value_type *data;   ///< First value, not origin

public:
    RowView(Owner &owner, value_type *data) : owner(owner), data(data) {}

    reference operator[](int x) const
    {
        assert(x >= owner.region().x0 && x < owner.region().x1);
        return data[x - owner.region().x0];
    }
};

template<typename Owner>
class SliceView
{
public:
    typedef typename std::conditional<
        std::is_const<Owner>::value,
        const typename Owner::value_type,
        typename Owner::value_type>::type value_type;
    typedef value_type &reference;

private:
    Owner &owner;
    int slice;
    value_type *data;   ///< First value, not origin

public:
    SliceView(Owner &owner, int slice, value_type *data) : owner(owner), slice(slice), data(data) {}

    RowView<Owner> operator[](int y) const
    {
        assert(y >= owner.region().y0 && y < owner.region().y1);
        return {owner, data + (y - owner.region().y0) * owner.rowStride()};
    }

    float step() const
    {
        return owner.step(slice);
    }

    const Region &region() const
    {
        return owner.region();
    }
};

/**
 * Reads the @c heightMapResolution header from the file and returns it. If
 * the header is not present, returns 0 (there is no way to distinguish this
 * from an encoded value of 0).
 */
float getOpenEXRStep(const Imf::InputFile &in);

/**
 * Helper function for @ref MemMap and @ref MemMapArray. It writes a prepared
 * framebuffer to an OpenEXR file.
 *
 * @param filename      Output filename
 * @param step          Horizontal resolution, to be placed in a header field if non-zero
 * @param fb            Prepared framebuffer
 * @param region        Data window for the output image
 */
void writeFrameBuffer(const uts::string &filename, float step, const Imf::FrameBuffer &fb,
                      const Region &region);

/**
 * Helper for @ref MemMap and @ref MemMapArray, for the case where the type
 * needs conversion prior to being written to file. This does the conversion
 * and writes the data to file.
 *
 * @param filename      Output filename
 * @param step          Horizontal resolution, to be placed in a header field if non-zero
 * @param region        Region of data to write
 * @param stride        Number of elements per row in @a data
 * @param data          Pointer to the first pixel to write.
 * @param tmp           Scratch space with at least @a region.width() * @a region.height() elements
 */
template<typename Tag>
void writeSlice(
    const typename std::enable_if<MapTraits<Tag>::needs_conversion::value, uts::string>::type &filename,
    float step,
    const Region &region,
    int stride,
    const typename MapTraits<Tag>::type *data,
    typename MapTraits<Tag>::io_type *tmp)
{
    typename MapTraits<Tag>::io_type *out = tmp;
    for (std::size_t y = 0; y < region.height(); y++)
        for (std::size_t x = 0; x < region.width(); x++)
            MapTraits<Tag>::convert(data[y * stride + x], *out++);

    Imf::FrameBuffer fb;
    std::ptrdiff_t tmpStride = region.width();
    MapTraits<Tag>::prepareFrameBuffer(
        fb, tmp - region.y0 * tmpStride - region.x0, tmpStride);
    writeFrameBuffer(filename, step, fb, region);
}

/**
 * Helper for @ref MemMap and @ref MemMapArray, for the case where the type
 * does not need conversion prior to being written to file.
 *
 * @param filename      Output filename
 * @param step          Horizontal resolution, to be placed in a header field if non-zero
 * @param region        Region of data to write
 * @param stride        Number of elements per row in @a data
 * @param data          Pointer to the first pixel to write.
 * @param dummy         Unused (necessary to make signature matching work)
 */
template<typename Tag>
void writeSlice(
    const typename std::enable_if<!MapTraits<Tag>::needs_conversion::value, uts::string>::type &filename,
    float step,
    const Region &region,
    int stride,
    const typename MapTraits<Tag>::type *data,
    typename MapTraits<Tag>::io_type *dummy)
{
    (void) dummy;

    Imf::FrameBuffer fb;
    MapTraits<Tag>::prepareFrameBuffer(
        fb,
        const_cast<typename MapTraits<Tag>::type *>(
            data - region.y0 * std::ptrdiff_t(stride) - region.x0),
        stride);
    writeFrameBuffer(filename, step, fb, region);
}

/**
 * Helper for @ref MemMap and @ref MemMapArray, for the case where the data needs conversion.
 *
 * @param in         Input file. The entire file is loaded.
 * @param[out] data  Read values, converted.
 * @param tmp        Scratch space with an element per pixel
 */
template<typename Tag>
void readSlice(
    typename std::enable_if<MapTraits<Tag>::needs_conversion::value, Imf::InputFile>::type &in,
    typename MapTraits<Tag>::type *data,
    typename MapTraits<Tag>::io_type *tmp)
{
    auto dw = in.header().dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    int h = dw.max.y - dw.min.y + 1;
    Imf::FrameBuffer fb;
    MapTraits<Tag>::prepareFrameBuffer(fb, tmp - dw.min.y * std::ptrdiff_t(w) - dw.min.x, w);
    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);

    for (std::size_t i = 0; i < (std::size_t) w * h; i++)
        MapTraits<Tag>::convert(tmp[i], data[i]);
}

/**
 * Helper for @ref MemMap and @ref MemMapArray, for the case where the data does need conversion.
 *
 * @param in         Input file. The entire file is loaded.
 * @param[out] data  Read values
 * @param tmp        Scratch space with an element per pixel
 */
template<typename Tag>
void readSlice(
    typename std::enable_if<!MapTraits<Tag>::needs_conversion::value, Imf::InputFile>::type &in,
    typename MapTraits<Tag>::type *data,
    typename MapTraits<Tag>::io_type *)
{
    auto dw = in.header().dataWindow();
    int w = dw.max.x - dw.min.x + 1;
    Imf::FrameBuffer fb;
    MapTraits<Tag>::prepareFrameBuffer(fb, data - dw.min.y * std::ptrdiff_t(w) - dw.min.x, w);
    in.setFrameBuffer(fb);
    in.readPixels(dw.min.y, dw.max.y);
}

/**
 * Generate the filenames used to store slices of an array.
 */
uts::string mapArrayFilename(const uts::string &prefix, int slice);

} // namespace detail

/// Base class shared by maps and map arrays
class MapRegion
{
private:
    Region region_;

protected:
    /**
     * Modify the region without allocating new data. The caller is expected to
     * have done any data allocation if necessary.
     */
    void setRegion(const Region &newRegion);

public:
    int width() const;   ///< Image width (equivalent to <code>region().width()</code>)
    int height() const;  ///< Image height (equivalent to <code>region().height()</code>)
    const Region &region() const;  ///< Image region

    /**
     * Change the region to place the start of the data at (x, y). This
     * preserves the existing storage and just changes the offset.
     */
    void translateTo(int x, int y);
};

/**
 * Base class for 2D maps. This base class does not assume memory ownership - it
 * could be a view of another map.
 */
class MapBase : public MapRegion
{
private:
    float step_ = 0.0f;

public:
    /**
     * Returns the horizontal separation between sample points, in metres
     * (or in whatever units are used to represent heights). If not
     * explicitly set, this will be 0 to indicate that it is unknown.
     *
     * @see @ref setStep().
     */
    float step() const;

    /**
     * Set the value returned by @ref step().
     *
     * @param step      New step value
     * @pre @a step &gt;= 0.
     */
    void setStep(float step);
};

/**
 * Base class for map arrays. This base class does not assume memory ownership - it
 * could be a view of another map.
 */
class MapArrayBase : public MapRegion
{
private:
    uts::vector<float> step_;

protected:
    /// Reserve memory to guarantee that a future @ref resizeStep cannot fail
    void reserveStep(int newSize);

    /// Shrink or grow the step array if necessary, preserving values
    void resizeStep(int newSize);

public:
    int arraySize() const;         ///< Number of array slices

    /**
     * Set the spatial resolution for a slice.
     *
     * @pre 0 &lt;= @a slice &lt; @ref arraySize()
     */
    void setStep(int slice, float step);

    /**
     * Copy the spatial resolution of slices from another array.
     *
     * @pre @a other.@ref arraySize() == @ref arraySize()
     */
    void setStep(const MapArrayBase &other);

    /**
     * Get the spatial resolution for a slice.
     *
     * @pre 0 &lt;= @a slice &lt; @ref arraySize()
     */
    float step(int slice) const;

    /**
     * Get the spatial resolution for all slices.
     */
    const uts::vector<float> &step() const;

    /**
     * Set all steps at once.
     */
    void setStep(uts::vector<float> &&step);
};

/**
 * Virtual base class for 2D array of arbitrary type. The concrete subclasses
 * will generally specify the type at compile time using a tag class e.g.
 * @ref height_tag. The actual type is then found from
 * <code>MapTraits&lt;tag&gt;</code>.
 *
 * A Map may have zero-sized dimensions, although they are not supported for
 * all uses e.g. they cannot be written to file. Also, empty images with one
 * zero and one non-zero dimension are not well-tested and should be avoided.
 *
 * A map also contains an indication of horizontal scale: see @ref step() and
 * @ref setStep().
 *
 * The map can represent any rectangular region of an abstract coordinate space
 * i.e. it doesn't need to be rooted at the origin.
 */
class Map : public MapBase
{
    friend class boost::serialization::access;
private:
    /**
     * Implemented by subclasses to allocate memory. The caller validates
     * the parameters and avoids reallocation if the width and height are
     * already suitable.
     */
    virtual void allocateImpl(int width, int height) = 0;

    template<typename Archive>
    void save(Archive &ar, const unsigned int) const
    {
        ar << region();
        auto s = step(); // boost::archive requires an lvalue
        ar << s;
    }

    template<typename Archive>
    void load(Archive &ar, const unsigned int)
    {
        Region r;
        ar >> r;
        allocate(r);
        float s;
        ar >> s;
        setStep(s);
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    virtual ~Map() = default;

    /**
     * Change the region represented, reallocating memory if necessary. It is
     * guaranteed that no reallocation will happen if the width and height of
     * the new region match those of the existing region.
     */
    void allocate(const Region &region);

    /// Make this an empty image
    void clear();

    /**
     * Retrieve the dimensions and step size of a file, without loading the
     * contents. An exception is thrown if the file header could not be read.
     *
     * @note This function only works on OpenEXR files. Conversely, it will work
     * on any OpenEXR file, even if it does not contain data that can actually
     * be read.
     */
    static std::tuple<Region, float> fileDimensions(const uts::string &filename);

    // Workarounds for http://llvm.org/bugs/show_bug.cgi?id=18005
    Map() = default;
    Map(Map &&) = default;
    Map &operator=(Map &&) = default;
};

/**
 * Virtual base class for array of identically-sized 2D arrays of arbitrary
 * type, which own their memory. This is similar to @ref Map but contains a
 * collection of images.  Each image in the collection has an independent
 * spatial resolution.
 */
class MapArray : public MapArrayBase
{
    friend class boost::serialization::access;
protected:
    /**
     * Implemented by subclasses to allocate memory. The caller validates
     * the parameters and avoids reallocation if the dimensions are
     * already suitable.
     */
    virtual void allocateImpl(int width, int height, int arraySize) = 0;

    template<typename Archive>
    void save(Archive &ar, const unsigned int) const
    {
        ar << region();
        ar << step();
    }

    template<typename Archive>
    void load(Archive &ar, const unsigned int)
    {
        Region r;
        uts::vector<float> step;
        ar >> r;
        ar >> step;
        allocate(r, step.size());
        setStep(std::move(step));
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

public:
    virtual ~MapArray() = default;
    /**
     * Allocate memory for the array (if necessary) and set the region.
     * It is guaranteed that no reallocation will be done if the dimensions
     * all match the original. Otherwise, just enough memory will be
     * allocated.
     *
     * If the size is increased, new slices have spatial resolution of zero.
     * Slices that remain retain their spatial resolution.
     */
    void allocate(const Region &region, int arraySize);

    /// Make this an empty image
    void clear();

    // Workarounds for http://llvm.org/bugs/show_bug.cgi?id=18005
    MapArray() = default;
    MapArray(MapArray &&) = default;
    MapArray &operator=(MapArray &&) = default;
};

template<typename Base, typename Tag, bool Serializable = detail::Serializable<Tag>::value>
class SerializableMap
{
private:
    class Dummy
    {
    };

public:
    /**
     * @name Dummy members so that subclasses can use @c using directives
     * @{
     */
    void write(Dummy) = delete;
    void read(Dummy) = delete;
    /**
     * @}
     */
};

template<typename Base, typename Tag>
class SerializableMap<Base, Tag, true>
{
public:
    /**
     * Write a region of the image to an OpenEXR file.
     *
     * @param filename       Output filename, including suffix
     * @param region         Region of image to write
     *
     * @note The @a region is interpolated in the same coordinate system
     * that is used by @ref Map::allocate, rather than being relative to the
     * first pixel.
     *
     * @pre
     * - 0 &lt;= @a region.x0 &lt; @a region.x1 &lt;= @ref Map::width()
     * - 0 &lt;= @a region.y0 &lt; @a region.y1 &lt;= @ref Map::height()
     */
    void write(const uts::string &filename, const Region &region) const;

    /**
     * Write the whole image to an OpenEXR file.
     *
     * @param filename       Output filename, including extension
     *
     * @pre The image is non-empty.
     */
    void write(const uts::string &filename) const;

    /**
     * Load the image from file, resizing it if necessary.
     *
     * @param filename       Input filename, including extension
     *
     * This method has exception safety: if there was an error reading
     * the file, the image is unmodified.
     */
    void read(const uts::string &filename);
};

/**
 * Mixin type (using CRTP) for a map that has a pointer to memory, and might or
 * might not own it. The base class must provide a @c get method (see @ref MemMap).
 */
template<typename Base, typename Tag>
class AddressableMap
{
public:
    /// Fill the contents with a specific value
    void fill(const typename MapTraits<Tag>::type &fillValue);

    detail::RowView<const Base> operator[](int y) const;   ///< Return a view of a row
    detail::RowView<Base>       operator[](int y);         ///< Return a view of a row
};

/**
 * Mixin type (using CRTP) for a map array that has a pointer to memory, and
 * might or might not own it. The base class must provide a @c get method (see
 * @ref MemMapArray).
 */
template<typename Base, typename Tag>
class AddressableMapArray
{
public:
    /// Fill the contents with a specific value
    void fill(const typename MapTraits<Tag>::type &fillValue);

    detail::SliceView<const Base> operator[](int y) const;   ///< Return a view of a slice
    detail::SliceView<Base>       operator[](int y);         ///< Return a view of a slice
};

/**
 * Concrete model of @ref Map that stores the data in dynamically-allocated
 * memory. It supports serialization via @ref SerializableMap.
 */
template<typename Tag>
class MemMap : public Map, public AddressableMap<MemMap<Tag>, Tag>, public SerializableMap<MemMap<Tag>, Tag>
{
public:
    typedef Tag tag_type;
    typedef typename MapTraits<Tag>::type value_type;

private:
    std::unique_ptr<value_type[]> data_;

    virtual void allocateImpl(int width, int height) override final;

    friend class boost::serialization::access;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & boost::serialization::base_object<Map>(*this);
        if (!region().empty())
            ar & boost::serialization::make_array<value_type>(data_.get(), region().pixels());
    }

public:
    const MemMap &toMemMap(const Region &region) const;
    void fromMemMap(MemMap &&in);

    /// Empty image constructor
    MemMap() {}
    /// Constructor for uninitialized data with arbitrary origin
    explicit MemMap(const Region &region);
    /// Construct from an OpenEXR file
    template<typename T = typename detail::Serializable<Tag> >
    explicit MemMap(const typename std::enable_if<T::value, uts::string>::type &filename);

    const value_type *get() const;  ///< Return a pointer to the raw data
    value_type *get();              ///< Return a pointer to the raw data
    std::size_t rowStride() const;  ///< Number of elements between rows in memory
};

template<typename Base, typename Tag, bool Serializable = detail::Serializable<Tag>::value>
class SerializableMapArray
{
private:
    class Dummy
    {
    };

public:
    /**
     * @name Dummy members so that subclasses can use @c using directives
     * @{
     */
    void write(Dummy) = delete;
    void read(Dummy) = delete;
    /**
     * @}
     */
};

template<typename Base, typename Tag>
class SerializableMapArray<Base, Tag, true>
{
public:
    /**
     * Write a region of the array to a sequence of OpenEXR files. The output files
     * are named <code><i>prefix</i>-<i>number</i>.exr</code>, where @a number
     * is the slice index starting from zero.
     *
     * @param filename       Filename prefix
     * @param region         Region of each image to write
     *
     * @note The @a region is interpolated in the same coordinate system
     * that is used by @ref MapArray::allocate, rather than being relative to the
     * first pixel.
     *
     * @pre
     * - 0 &lt;= @a region.x0 &lt; @a region.x1 &lt;= @ref MapArray::width()
     * - 0 &lt;= @a region.y0 &lt; @a region.y1 &lt;= @ref MapArray::height()
     * - The array has at least one slice.
     */
    void write(const uts::string &filename, const Region &region) const;

    /**
     * Write the array to a sequence of OpenEXR files. The output files
     * are named <code><i>prefix</i>-<i>number</i>.exr</code>, where @a number
     * is the slice index starting from zero.
     *
     * @param prefix       Filename prefix
     *
     * @pre The array is non-empty.
     */
    void write(const uts::string &prefix) const;

    /**
     * Load a sequence of images from files, replacing the current contents.
     * This method has a strong exception-safety guarantee. The files are named
     * as per @ref write, and are read until the next
     * filename is not an OpenEXR file.
     */
    void read(const uts::string &filename);
};

/**
 * Concrete model of @ref MapArray that stores the data in dynamically-allocated
 * memory.
 */
template<typename Tag>
class MemMapArray : public MapArray, public AddressableMapArray<MemMapArray<Tag>, Tag>,
    public SerializableMapArray<MemMapArray<Tag>, Tag>
{
    friend class boost::serialization::access;
public:
    typedef Tag tag_type;
    typedef typename MapTraits<Tag>::type value_type;

private:
    std::unique_ptr<value_type[]> data_;

    virtual void allocateImpl(int width, int height, int arraySize) override final;

    template<typename Archive>
    void serialize(Archive &ar, const unsigned int)
    {
        ar & boost::serialization::base_object<MapArray>(*this);
        const std::size_t pixels = region().pixels() * arraySize();
        if (pixels > 0)
            ar & boost::serialization::make_array<value_type>(data_.get(), pixels);
    }

public:
    using SerializableMapArray<MemMapArray<Tag>, Tag>::read;
    using SerializableMapArray<MemMapArray<Tag>, Tag>::write;

    const MemMapArray &toMemMapArray(const Region &region) const;
    void fromMemMapArray(MemMapArray &&in);

    /// Empty constructor
    MemMapArray() {}
    /// Constructor for uninitialized data with an arbitrary origin
    MemMapArray(const Region &region, int arraySize);
    /// Construct from an OpenEXR file
    template<typename T = typename detail::Serializable<Tag> >
    explicit MemMapArray(const typename std::enable_if<T::value, uts::string>::type &prefix)
    {
        this->read(prefix);
    }

    /**
     * Copy in to an existing slice.
     *
     * @pre
     * - 0 &lt;= @a slice &lt; @ref arraySize()
     * - @a region is entirely contained in both the source and target
     * - @c in.step() == #step(@a slice)
     */
    void read(int slice, const MemMap<Tag> &in, const Region &region);

    /**
     * Extract a single slice. The @a out map is resized and the
     * step size changed if necessary.
     *
     * @pre
     * - 0 &lt;= @a slice &lt; @ref arraySize()
     */
    void write(int slice, MemMap<Tag> &out) const;

    /**
     * Write a region of a single slice to a @ref MemMap. The target is not resized.
     *
     * @pre
     * - @a copyRegion is contained inside both the source and target
     * - The source and target have the same step
     * - 0 &lt;= @a slice &lt; @ref arraySize()
     */
    void write(int slice, MemMap<Tag> &out, const Region &region) const;

    const value_type *get() const;   ///< Return a pointer to the raw data
    value_type *get();               ///< Return a pointer to the raw data
    std::size_t rowStride() const;   ///< Number of elements between rows in memory
    std::size_t sliceStride() const; ///< Number of elements between slices in memory
};

/**********************************************************************/

template<typename Base, typename Tag>
void AddressableMap<Base, Tag>::fill(const typename MapTraits<Tag>::type &fillValue)
{
    Base *base = static_cast<Base *>(this);
    const auto data = base->get();
    if (data != nullptr)
    {
        auto rowPtr = data;
        const Region &r = base->region();
        for (int y = r.y0; y < r.y1; y++, rowPtr += base->rowStride())
            std::fill(rowPtr, rowPtr + r.width(), fillValue);
    }
}

template<typename Base, typename Tag>
detail::RowView<Base> AddressableMap<Base, Tag>::operator[](int y)
{
    Base *base = static_cast<Base *>(this);
    const Region &region = base->region();
    const auto data = base->get();
    assert(y >= region.y0 && y < region.y1);
    assert(data != nullptr);
    return {*base, data + (y - region.y0) * base->rowStride()};
}

template<typename Base, typename Tag>
detail::RowView<const Base> AddressableMap<Base, Tag>::operator[](int y) const
{
    const Base *base = static_cast<const Base *>(this);
    const Region &region = base->region();
    const auto data = base->get();
    assert(y >= region.y0 && y < region.y1);
    assert(data != nullptr);
    return {*base, data + (y - region.y0) * base->rowStride()};
}

/**********************************************************************/

template<typename Base, typename Tag>
void AddressableMapArray<Base, Tag>::fill(const typename MapTraits<Tag>::type &fillValue)

{
    Base *base = static_cast<Base *>(this);
    const auto data = base->get();
    const Region &r = base->region();
    for (int z = 0; z < base->arraySize(); z++)
    {
        auto rowPtr = data + z * base->sliceStride();
        for (int y = r.y0; y < r.y1; y++, rowPtr += base->rowStride())
        {
            std::fill(rowPtr, rowPtr + r.width(), fillValue);
        }
    }
}

template<typename Base, typename Tag>
detail::SliceView<Base> AddressableMapArray<Base, Tag>::operator[](int slice)
{
    Base *base = static_cast<Base *>(this);
    const auto data = base->get();
    assert(slice >= 0 && slice < base->arraySize());
    assert(data != nullptr);
    return {*base, slice, data + slice * base->sliceStride()};
}

template<typename Base, typename Tag>
detail::SliceView<const Base> AddressableMapArray<Base, Tag>::operator[](int slice) const
{
    const Base *base = static_cast<const Base *>(this);
    const auto data = base->get();
    assert(slice >= 0 && slice < base->arraySize());
    assert(data != nullptr);
    return {*base, slice, data + slice * base->sliceStride()};
}

/**********************************************************************/

template<typename Tag>
MemMap<Tag>::MemMap(const Region &region)
{
    allocate(region);
}

template<typename Tag>
template<typename T>
MemMap<Tag>::MemMap(const typename std::enable_if<T::value, uts::string>::type &filename)
{
    this->read(filename);
}

template<typename Tag>
const MemMap<Tag> &MemMap<Tag>::toMemMap(const Region &) const
{
    return *this;
}

template<typename Tag>
void MemMap<Tag>::fromMemMap(MemMap &&in)
{
    *this = std::move(in);
}

template<typename Tag>
void MemMap<Tag>::allocateImpl(int width, int height)
{
    assert(height == 0 || width <= std::numeric_limits<std::size_t>::max() / height / sizeof(value_type));
    std::size_t size = (std::size_t) width * height;
    if (size != region().pixels())
    {
        if (size > 0)
            data_.reset(new value_type[size]);
        else
            data_.reset();
    }
}

template<typename Tag>
auto MemMap<Tag>::get() -> value_type *
{
    return data_.get();
}

template<typename Tag>
auto MemMap<Tag>::get() const -> const value_type *
{
    return data_.get();
}

template<typename Tag>
std::size_t MemMap<Tag>::rowStride() const
{
    return region().width();
}

/**********************************************************************/

namespace detail
{
/**
 * Wraps the @c customWrite member of a map traits class, if any, and
 * provides a fallback that returns false if not.
 */
template<typename T>
inline bool customWrite(
    const MemMap<T> &in,
    const typename std::enable_if<HasCustomWrite<T>::value, uts::string>::type &filename,
    const Region &region)
{
    return MapTraits<T>::customWrite(in, filename, region);
}

template<typename T>
inline bool customWrite(
    const MemMap<T> &in,
    const typename std::enable_if<!HasCustomWrite<T>::value, uts::string>::type &filename,
    const Region &region)
{
    return false;
}

} // namespace detail

template<typename Base, typename Tag>
void SerializableMap<Base, Tag, true>::write(const uts::string &filename, const Region &region) const
{
    typedef typename Base::tag_type tag_type;
    typedef typename MapTraits<tag_type>::io_type io_type;
    const MemMap<tag_type> &mem = static_cast<const Base *>(this)->toMemMap(region);

    if (!detail::customWrite(mem, filename, region))
    {
        const Region &fullRegion = mem.region();
        assert(fullRegion.x0 <= region.x0 && region.x0 < region.x1 && region.x1 <= fullRegion.x1);
        assert(fullRegion.y0 <= region.y0 && region.y0 < region.y1 && region.y1 <= fullRegion.y1);
        std::unique_ptr<io_type[]> tmp;
        if (MapTraits<tag_type>::needs_conversion::value)
            tmp.reset(new io_type[region.pixels()]);

        std::ptrdiff_t stride = mem.width();
        detail::writeSlice<tag_type>(filename, mem.step(), region, stride,
                                     mem.get() + (region.y0 - fullRegion.y0) * stride + (region.x0 - fullRegion.x0),
                                     tmp.get());
    }
}

template<typename Base, typename Tag>
void SerializableMap<Base, Tag, true>::write(const uts::string &filename) const
{
    write(filename, static_cast<const Base *>(this)->region());
}

namespace detail
{
/**
 * Wraps the @c customRead member of a map traits class, if any, and
 * provides a fallback that returns false if not.
 */
template<typename T>
static inline bool customRead(MemMap<T> &out, const typename std::enable_if<HasCustomRead<T>::value, uts::string>::type &filename)
{
    return MapTraits<T>::customRead(out, filename);
}

template<typename T>
static inline bool customRead(MemMap<T> &out, const typename std::enable_if<!HasCustomRead<T>::value, uts::string>::type &filename)
{
    return false;
}

} // namespace detail

template<typename Base, typename Tag>
void SerializableMap<Base, Tag, true>::read(const uts::string &filename)
{
    typedef typename Base::tag_type tag_type;
    typedef typename MapTraits<tag_type>::io_type io_type;

    MemMap<tag_type> mem;
    if (!detail::customRead(mem, filename))
    {
        Imf::InputFile in(filename.c_str());
        auto dw = in.header().dataWindow();
        int w = dw.max.x - dw.min.x + 1;
        int h = dw.max.y - dw.min.y + 1;
        std::size_t size = (std::size_t) w * h;

        Region r(dw.min.x, dw.min.y, dw.max.x + 1, dw.max.y + 1);
        mem.allocate(r);

        std::unique_ptr<io_type[]> tmp;
        if (MapTraits<tag_type>::needs_conversion::value)
            tmp.reset(new io_type[size]);

        detail::readSlice<tag_type>(in, mem.get(), tmp.get());
        tmp.reset();

        mem.setStep(detail::getOpenEXRStep(in));
    }
    static_cast<Base *>(this)->fromMemMap(std::move(mem));
}

/**********************************************************************/

template<typename Tag>
MemMapArray<Tag>::MemMapArray(const Region &region, int arraySize)
{
    allocate(region, arraySize);
}

template<typename Tag>
const MemMapArray<Tag> &MemMapArray<Tag>::toMemMapArray(const Region &) const
{
    return *this;
}

template<typename Tag>
void MemMapArray<Tag>::fromMemMapArray(MemMapArray<Tag> &&in)
{
    *this = std::move(in);
}

template<typename Tag>
void MemMapArray<Tag>::allocateImpl(int width, int height, int arraySize)
{
    assert(height == 0 || arraySize == 0
           || width <= std::numeric_limits<std::size_t>::max() / height / arraySize / sizeof(value_type));
    std::size_t size = (std::size_t) width * height * arraySize;
    if (size != region().pixels() * this->arraySize())
    {
        if (size > 0)
            data_.reset(new value_type[size]);
        else
            data_.reset();
    }
}

template<typename Tag>
void MemMapArray<Tag>::read(int slice, const MemMap<Tag> &in, const Region &region)
{
    assert(this->region().contains(region));
    assert(in.region().contains(region));
    assert(slice >= 0 && slice < arraySize());
    assert(step(slice) == in.step());
    for (int y = region.y0; y < region.y1; y++)
    {
        const typename MapTraits<Tag>::type *inPtr = &in[y][region.x0];
        typename MapTraits<Tag>::type *outPtr = &(*this)[slice][y][region.x0];
        std::copy(inPtr, inPtr + region.width(), outPtr);
    }
}

template<typename Tag>
void MemMapArray<Tag>::write(int slice, MemMap<Tag> &out) const
{
    assert(slice >= 0 && slice < arraySize());
    out.allocate(region());
    out.setStep(step(slice));
    write(slice, out, region());
}

template<typename Tag>
void MemMapArray<Tag>::write(int slice, MemMap<Tag> &out, const Region &region) const
{
    assert(slice >= 0 && slice < arraySize());
    assert(this->region().contains(region));
    assert(out.region().contains(region));
    assert(out.step() == step(slice));
    for (int y = region.y0; y < region.y1; y++)
    {
        const typename MapTraits<Tag>::type *inPtr = &(*this)[slice][y][region.x0];
        typename MapTraits<Tag>::type *outPtr = &out[y][region.x0];
        std::copy(inPtr, inPtr + region.width(), outPtr);
    }
}

template<typename Tag>
auto MemMapArray<Tag>::get() -> value_type *
{
    return data_.get();
}

template<typename Tag>
auto MemMapArray<Tag>::get() const -> const value_type *
{
    return data_.get();
}

template<typename Tag>
std::size_t MemMapArray<Tag>::rowStride() const
{
    return region().width();
}

template<typename Tag>
std::size_t MemMapArray<Tag>::sliceStride() const
{
    return region().pixels();
}

/**********************************************************************/

template<typename Base, typename Tag>
void SerializableMapArray<Base, Tag, true>::write(const uts::string &prefix, const Region &region) const
{
    typedef typename Base::tag_type tag_type;
    typedef typename MapTraits<tag_type>::io_type io_type;
    const MemMapArray<tag_type> &mem = static_cast<const Base *>(this)->toMemMapArray(region);

    const Region &fullRegion = mem.region();
    assert(mem.arraySize() > 0);
    assert(fullRegion.x0 <= region.x0 && region.x0 < region.x1 && region.x1 <= fullRegion.x1);
    assert(fullRegion.y0 <= region.y0 && region.y0 < region.y1 && region.y1 <= fullRegion.y1);
    std::unique_ptr<io_type[]> tmp;
    std::size_t sliceSize = region.pixels();
    if (MapTraits<tag_type>::needs_conversion::value)
        tmp.reset(new io_type[sliceSize]);

    std::ptrdiff_t stride = mem.width();
    std::ptrdiff_t offset = (region.y0 - fullRegion.y0) * stride + (region.x0 - fullRegion.x0);
    for (int i = 0; i < mem.arraySize(); i++)
        detail::writeSlice<tag_type>(
            detail::mapArrayFilename(prefix, i), mem.step(i), region, stride,
            mem.get() + sliceSize * i + offset, tmp.get());
}

template<typename Base, typename Tag>
void SerializableMapArray<Base, Tag, true>::write(const uts::string &prefix) const
{
    write(prefix, static_cast<const Base *>(this)->region());
}

template<typename Base, typename Tag>
void SerializableMapArray<Base, Tag, true>::read(const uts::string &prefix)
{
    typedef typename Base::tag_type tag_type;
    typedef typename MapTraits<tag_type>::io_type io_type;

    // First check how many files there are
    int arraySize = 0;
    while (Imf::isOpenExrFile(detail::mapArrayFilename(prefix, arraySize).c_str()))
        arraySize++;
    if (arraySize == 0)
        throw std::ios::failure("no files found");

    int width, height;
    std::size_t sliceSize;
    MemMapArray<tag_type> mem;
    std::unique_ptr<io_type[]> tmp;
    uts::vector<float> step(arraySize);

    Imath::Box2i dw0;
    Region r;
    // Open the first file to determine dimensions etc
    {
        const uts::string filename = detail::mapArrayFilename(prefix, 0);
        Imf::InputFile in(filename.c_str());
        dw0 = in.header().dataWindow();
        width = dw0.max.x - dw0.min.x + 1;
        height = dw0.max.y - dw0.min.y + 1;
        sliceSize = (std::size_t) width * height;
        r = Region(dw0.min.x, dw0.min.y, dw0.max.x + 1, dw0.max.y + 1);
        mem.allocate(r, arraySize);

        if (MapTraits<tag_type>::needs_conversion::value)
            tmp.reset(new io_type[sliceSize]);
        detail::readSlice<tag_type>(in, mem.get(), tmp.get());
        step[0] = detail::getOpenEXRStep(in);
    }

    // Read remaining slices, checking for mismatches
    for (int slice = 1; slice < arraySize; slice++)
    {
        const uts::string filename = detail::mapArrayFilename(prefix, slice);
        Imf::InputFile in(filename.c_str());
        auto dw = in.header().dataWindow();
        if (dw != dw0)
            throw std::ios::failure("dimension mismatch in image array");
        detail::readSlice<tag_type>(in, mem.get() + sliceSize * slice, tmp.get());
        step[slice] = detail::getOpenEXRStep(in);
    }

    // Everything worked, so make it permanent
    mem.setStep(std::move(step));
    static_cast<Base *>(this)->fromMemMapArray(std::move(mem));
}

/**********************************************************************/

template<>
class MapTraits<height_tag>
{
public:
    typedef float type;
    typedef type io_type;
    typedef std::false_type needs_conversion;

    /// Extracts the height (for metaprogramming)
    static inline type getHeight(type h) { return h; }

    /// Reads from a Terragen file if the suffix is .ter
    static bool customRead(MemMap<height_tag> &out, const uts::string &filename);
    /// Writes to a Terragen file if the suffix is .ter
    static bool customWrite(const MemMap<height_tag> &out, const uts::string &filename, const Region &region);

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<distance_field_tag>
{
public:
    struct type
    {
        float dist;
        float param;
    };
    typedef type io_type;
    typedef std::false_type needs_conversion;

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<mask_tag>
{
public:
    typedef std::uint32_t type;   ///< Non-permitted/provided values
    typedef type io_type;
    typedef std::false_type needs_conversion;

    /// Extracts the mask (for metaprogramming)
    static inline type getMask(type x) { return x; }

    /// Mask value for anything permitted
    static constexpr type all = 0;
    /// Mask value for nothing permitted
    static constexpr type none = ~type(0);

    static constexpr int numTypes = std::numeric_limits<type>::digits;

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<height_and_mask_tag>
{
public:
    struct type
    {
        MapTraits<height_tag>::type height;
        MapTraits<mask_tag>::type mask;

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int)
        {
            ar & height;
            ar & mask;
        }
    };
    typedef type io_type;
    typedef std::false_type needs_conversion;

    /// Extracts the height (for metaprogramming)
    static inline MapTraits<height_tag>::type getHeight(const type &x) { return x.height; }
    /// Extracts the mask (for metaprogramming)
    static inline MapTraits<mask_tag>::type getMask(const type &x) { return x.mask; }

    /// Reads from a Terragen file if the suffix is .ter
    static bool customRead(MemMap<height_and_mask_tag> &out, const uts::string &filename);
    /// Writes to a Terragen file if the suffix is .ter
    static bool customWrite(const MemMap<height_and_mask_tag> &out, const uts::string &filename, const Region &region);

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<appearance_tag>
{
public:
    typedef std::array<std::uint16_t, APPEARANCE_MODES> type;
    typedef std::array<std::uint32_t, APPEARANCE_MODES> io_type;
    typedef std::true_type needs_conversion;

    static void convert(type in, io_type &out);
    static void convert(io_type in, type &out);
    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<coords_tag>
{
public:
    struct type
    {
        std::array<float, 2> flat;              ///< X, Y for flattened coordinates

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int)
        {
            ar & flat;
        }
    };
    typedef type io_type;
    typedef std::false_type needs_conversion;

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<coords_offset_tag>
{
public:
    struct alignas(16) type : public MapTraits<coords_tag>::type
    {
        float offset;                               ///< Value to be added to exemplar heights

        type() = default;
        type(const MapTraits<coords_tag>::type &base, float offset)
            : MapTraits<coords_tag>::type(base), offset(offset) {}

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int)
        {
            ar & boost::serialization::base_object<MapTraits<coords_tag>::type>(*this);
            ar & offset;
        }
    };
    static_assert(sizeof(type) == 16, "Incorrect size for type");
    typedef type io_type;

    typedef std::false_type needs_conversion;

    static void prepareFrameBuffer(Imf::FrameBuffer &fb, const io_type *base, std::size_t stride);
};

template<>
class MapTraits<height_constraint_tag>
{
public:
    struct type
    {
        float h;  ///< Target height
        float a;  ///< Cubic adjustment term (always negative)
        float c;  ///< Constant offset for linear sections
        float t;  ///< Cutover point from cubic to linear

        template<typename Archive>
        void serialize(Archive &ar, const unsigned int)
        {
            ar & h;
            ar & a;
            ar & c;
            ar & t;
        }
    };
    typedef type io_type;
    typedef std::false_type needs_conversion;
};

/**********************************************************************/

extern template class MemMap<height_tag>;
extern template class MemMap<distance_field_tag>;
extern template class MemMap<mask_tag>;
extern template class MemMap<height_and_mask_tag>;
extern template class MemMap<appearance_tag>;
extern template class MemMap<coords_tag>;
extern template class MemMap<coords_offset_tag>;
extern template class MemMap<height_constraint_tag>;

extern template class MemMapArray<height_tag>;
extern template class MemMapArray<appearance_tag>;
extern template class MemMapArray<coords_tag>;
extern template class MemMapArray<coords_offset_tag>;
// height_constraint_tag is not used in arrays

#endif /* !UTS_COMMON_MAP_H */
