/**
 * @file
 *
 * Report the image formats supported by the OpenCL implementation.
 */

#include "clhpp.h"
#include <common/debug_vector.h>
#include <common/debug_string.h>
#include <iostream>
#include <iomanip>
#include <sstream>

static uts::string orderToString(cl_channel_order order)
{
    switch (order)
    {
#define CASE(c) case c: return #c; break
        CASE(CL_R);
        CASE(CL_Rx);
        CASE(CL_A);
        CASE(CL_INTENSITY);
        CASE(CL_LUMINANCE);
        CASE(CL_RG);
        CASE(CL_RGx);
        CASE(CL_RA);
        CASE(CL_RGB);
        CASE(CL_RGBx);
        CASE(CL_RGBA);
        CASE(CL_ARGB);
        CASE(CL_BGRA);
#undef CASE
    default:
        {
            std::ostringstream out;
            out << std::hex << order;
            return out.str();
        }
    }
}

static uts::string typeToString(cl_channel_type type)
{
    switch (type)
    {
#define CASE(c) case c: return #c; break
        CASE(CL_SNORM_INT8);
        CASE(CL_SNORM_INT16);
        CASE(CL_UNORM_INT8);
        CASE(CL_UNORM_INT16);
        CASE(CL_UNORM_SHORT_565);
        CASE(CL_UNORM_SHORT_555);
        CASE(CL_UNORM_INT_101010);
        CASE(CL_SIGNED_INT8);
        CASE(CL_SIGNED_INT16);
        CASE(CL_SIGNED_INT32);
        CASE(CL_UNSIGNED_INT8);
        CASE(CL_UNSIGNED_INT16);
        CASE(CL_UNSIGNED_INT32);
        CASE(CL_HALF_FLOAT);
        CASE(CL_FLOAT);
#undef CASE
    default:
        {
            std::ostringstream out;
            out << std::hex << type;
            return out.str();
        }
    }
}

static void dumpFormats(const uts::vector<cl::ImageFormat> &formats)
{
    for (const auto &format : formats)
        std::cout << orderToString(format.image_channel_order) << " / "
            << typeToString(format.image_channel_data_type) << "\n";
}

int main()
{
    uts::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const auto &platform : platforms)
    {
        uts::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (const auto &device : devices)
        {
            std::cout << device.getInfo<CL_DEVICE_NAME>() << ":\n\n";

            cl_context_properties properties[] =
            {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties) platform(),
                0
            };
            cl::Context context(device, properties);

            uts::vector<cl::ImageFormat> formats;
            std::cout << "2D, read-write:\n";
            context.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &formats);
            dumpFormats(formats);
            std::cout << "\n3D, read-write:\n";
            context.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE3D, &formats);
            dumpFormats(formats);
            std::cout << "\n";
        }
    }
    return 0;
}
