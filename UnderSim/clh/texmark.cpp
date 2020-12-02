/**
 * @file
 *
 * Report the image formats supported by the OpenCL implementation.
 */

#include "clhpp.h"
#include <common/debug_vector.h>
#include <common/debug_unordered_map.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <boost/program_options.hpp>
#include "clh.h"

namespace po = boost::program_options;

static po::variables_map processOptions(int argc, char **argv)
{
    po::options_description desc("General options");
    desc.add_options()
        ("type", po::value<std::string>()->default_value("CL_FLOAT"), "image data type")
        ("modifier", po::value<std::string>()->default_value(""), "coordinate modifier");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
              .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
              .options(desc)
              .run(), vm);
    po::notify(vm);

    return vm;
}

int main(int argc, char **argv)
{
    auto vm = processOptions(argc, argv);
    cl::ImageFormat format(CL_RGBA, CL_FLOAT);
    uts::string type = vm["type"].as<std::string>();
#define DO_TYPE(name) else if (type == #name) format.image_channel_data_type = name
    if (false) {}
    DO_TYPE(CL_FLOAT);
    DO_TYPE(CL_HALF_FLOAT);
    DO_TYPE(CL_UNORM_INT8);
    DO_TYPE(CL_UNORM_INT16);
    else
    {
        std::cerr << "Unknown data type " << type << '\n';
        return 1;
    }

    uts::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const auto &platform : platforms)
    {
        uts::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (const auto &device : devices)
        {
            if (!(device.getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_GPU))
                continue;
            std::cout << device.getInfo<CL_DEVICE_NAME>() << ":\n\n";

            cl_context_properties properties[] =
            {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties) platform(),
                0
            };
            cl::Context context(device, properties);
            cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
            const int width = 1024;
            const int height = 1024;
            const int passes = 20;
            const int taps = 9;
            cl::Image2D image(context, CL_MEM_READ_ONLY, format, width, height);
            cl::Buffer output(context, CL_MEM_READ_WRITE, 4 * sizeof(cl_uint));
            uts::unordered_map<uts::string, uts::string> defines;
            defines["ADDR_MODIFIER"] = vm["modifier"].as<std::string>();
            cl::Program program = CLH::build(context, "texmark.cl", defines, "-cl-nv-maxrregcount=24");

            auto kernel = cl::make_kernel<const cl::Image2D &, const cl::Buffer &>(program, "benchmark2d");
            uts::vector<cl::Event> events;
            events.reserve(passes);
            for (int i = 0; i < passes; i++)
            {
                events.push_back(kernel({queue, {width, height}, {16, 16}}, image, output));
            }
            queue.finish();

            cl_ulong total = 0;
            for (int i = 0; i < passes; i++)
            {
                if (events[i].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE)
                {
                    std::cerr << "WARNING: pass " << i << " did not complete correctly\n";
                    continue;
                }
                cl_ulong start, end;
                start = events[i].getProfilingInfo<CL_PROFILING_COMMAND_START>();
                end = events[i].getProfilingInfo<CL_PROFILING_COMMAND_END>();
                total += end - start;
            }

            double mean = total * 1e-9 / passes;
            double rate = width * height * taps / mean;
            std::cout << rate * 1e-9 << '\n';
        }
    }
    return 0;
}
