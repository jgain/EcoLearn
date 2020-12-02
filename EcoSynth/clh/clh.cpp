/**
 * @file
 */

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include "clhpp.h"
#include <iostream>
#include <common/debug_vector.h>
#include <common/debug_string.h>
#include <common/debug_unordered_map.h>
#include <sstream>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <set>
#include <cstdlib>
#include "clh.h"
#include <common/source2cpp.h>
#include <common/stats.h>

namespace po = boost::program_options;

namespace CLH
{

void addOptions(boost::program_options::options_description &desc)
{
    desc.add_options()
        (Option::device, boost::program_options::value<std::string>(),
                         "OpenCL device name")
        (Option::cpu,    "Use only a CPU device")
        (Option::gpu,    "Use only a GPU device");
}

// How much we want to use a device: higher is better
static int deviceScore(const cl::Device &device)
{
    if (!device())
        return -1;
    else
    {
        cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
        if (type & CL_DEVICE_TYPE_GPU)
            return 1;
        else
            return 0;
    }
}

cl::Device findDevice(
    const boost::program_options::variables_map &vm,
    const std::function<uts::string(const cl::Device &)> &deviceFilter)
{
    cl::Device ans;

    const bool cpuOnly = vm.count(Option::cpu);
    const bool gpuOnly = vm.count(Option::gpu);
    // Parse device name, if one given
    bool haveRequired = false;
    uts::string requiredName;
    int requiredNum = 0;
    if (vm.count(Option::device))
    {
        requiredName = vm[Option::device].as<std::string>();
        uts::string::size_type colon = requiredName.rfind(':');
        if (colon != uts::string::npos)
        {
            // User may have specified a device number
            try
            {
                requiredNum = boost::lexical_cast<unsigned int>(requiredName.substr(colon + 1));
                requiredName = requiredName.substr(0, colon - 1);
            }
            catch (boost::bad_lexical_cast &e)
            {
                // Ignore
            }
        }
        haveRequired = true;
    }

    uts::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (const cl::Platform &platform : platforms)
    {
        uts::vector<cl::Device> devices;
        cl_device_type type = CL_DEVICE_TYPE_ALL;

        platform.getDevices(type, &devices);
        for (const cl::Device &device : devices)
        {
            bool match = true;

            const uts::string name = device.getInfo<CL_DEVICE_NAME>();
            cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();

            if (cpuOnly && !(type & CL_DEVICE_TYPE_CPU))
                match = false;
            if (gpuOnly && !(type & CL_DEVICE_TYPE_GPU))
                match = false;
            if (haveRequired)
            {
                if (name != requiredName)
                    match = false;
                else
                {
                    match = requiredNum == 0;
                    requiredNum--;
                }
            }

            uts::string reason = deviceFilter(device);
            if (!reason.empty())
            {
                // TODO: create a logging system
                stats::print("Skipping device ", device.getInfo<CL_DEVICE_NAME>(), ": ", reason, "\n");
                match = false;
            }

            if (match)
            {
                if (deviceScore(device) > deviceScore(ans))
                    ans = device;
            }
        }
    }

    return ans;
}

static void CL_CALLBACK contextCallback(const char *msg, const void *ptr, ::size_t cb, void *user)
{
    (void) ptr;
    (void) cb;
    (void) user;
    std::cerr << msg << "\n";
}

cl::Context makeContext(const cl::Device &device)
{
    const cl::Platform &platform = device.getInfo<CL_DEVICE_PLATFORM>();
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform(), 0};
    uts::vector<cl::Device> devices(1, device);
    return cl::Context(devices, props, contextCallback);
}

cl::Program build(const cl::Context &context, const uts::vector<cl::Device> &devices,
                  const uts::string &filename, const uts::unordered_map<uts::string, uts::string> &defines,
                  const uts::string &options)
{
    const uts::unordered_map<uts::string, uts::string> &sourceMap = getSourceMap();
    if (!sourceMap.count(filename))
        throw std::invalid_argument("No such program " + filename);
    const uts::string &source = sourceMap.find(filename)->second;

    std::ostringstream s;
    for (const auto &i : defines)
    {
        s << "#define " << i.first << " " << i.second << "\n";
    }
    s << "#line 1 \"" << filename << "\"\n";
    const uts::string header = s.str();
    cl::Program::Sources sources(2);
    sources[0] = std::make_pair(header.data(), header.length());
    sources[1] = std::make_pair(source.data(), source.length());
    cl::Program program(context, sources);

    try
    {
        program.build(devices, options.c_str());
    }
    catch (cl::Error &e)
    {
        for (const cl::Device &device : devices)
        {
            const uts::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            if (log != "" && log != "\n")
            {
                std::cerr << "Log for device " << device.getInfo<CL_DEVICE_NAME>() << '\n';
                std::cerr << log << '\n';
            }
        }
        throw;
    }

    return program;
}

cl::Program build(const cl::Context &context,
                  const uts::string &filename,
                  const uts::unordered_map<uts::string, uts::string> &defines,
                  const uts::string &options)
{
    uts::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    return build(context, devices, filename, defines, options);
}

/**
 * Return an event that is already signaled as @c CL_COMPLETE.
 * This is equivalent to the other form but uses the queue to determine the
 * context.
 */
void doneEvent(const cl::CommandQueue &queue, cl::Event *event)
{
    if (event != NULL)
    {
        cl::UserEvent signaled(queue.getInfo<CL_QUEUE_CONTEXT>());
        signaled.setStatus(CL_COMPLETE);
        *event = signaled;
    }
}

cl_int enqueueMarkerWithWaitList(const cl::CommandQueue &queue,
                                 const uts::vector<cl::Event> *events,
                                 cl::Event *event)
{
    if (events != NULL && events->empty())
        events = NULL; // to avoid having to check for both conditions later

    if (events == NULL && event == NULL)
        return CL_SUCCESS;
    else if (event == NULL)
        return queue.enqueueWaitForEvents(*events);
    else if (!(queue.getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
             || (events != NULL && events->size() > 1))
    {
        /* For the events->size() > 1, out-of-order case this is inefficient
         * but correct.  Alternatives would be to enqueue a dummy task (which
         * would have potentially large overhead to allocate a dummy buffer or
         * something), or to create a separate thread to wait for completion of
         * the events and signal a user event when done (which would force
         * scheduling to round trip via multiple CPU threads).
         *
         * The implementation in cl.hpp (version 1.2.1) leaks any previous
         * reference, so we use a temporary event.
         */
        cl::Event tmp;
        int status = queue.enqueueMarker(&tmp);
        if (status == CL_SUCCESS)
            *event = tmp;
        return status;
    }
    else if (events == NULL)
    {
        doneEvent(queue, event);
    }
    else
    {
        // Exactly one input event, so just copy it to the output
        if (event != NULL)
            *event = (*events)[0];
    }
    return CL_SUCCESS;
}

cl_int enqueueNDRangeKernel(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local,
    const uts::vector<cl::Event> *events,
    cl::Event *event)
{
    for (std::size_t i = 0; i < global.dimensions(); i++)
        if (static_cast<const std::size_t *>(global)[i] == 0)
        {
            return enqueueMarkerWithWaitList(queue, events, event);
        }

    int ret = queue.enqueueNDRangeKernel(kernel, offset, global, local, events, event);
    return ret;
}

static cl::NDRange makeNDRange(cl_uint dimensions, const std::size_t *sizes)
{
    switch (dimensions)
    {
    case 0: return cl::NDRange();
    case 1: return cl::NDRange(sizes[0]);
    case 2: return cl::NDRange(sizes[0], sizes[1]);
    case 3: return cl::NDRange(sizes[0], sizes[1], sizes[2]);
    default: std::abort(); // should never be reached
    }
    return cl::NDRange(); // should never be reached
}

cl_int enqueueNDRangeKernelSplit(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local,
    const uts::vector<cl::Event> *events,
    cl::Event *event)
{
    /* If no size given, pick a default.
     * TODO: get performance hint from CL_KERNEL_PREFERRED_KERNEL_WORK_GROUP_SIZE_MULTIPLE?
     */
    if (local.dimensions() == 0)
    {
        switch (global.dimensions())
        {
        case 1:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(256), events, event);
        case 2:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(16, 16), events, event);
        case 3:
            return enqueueNDRangeKernelSplit(queue, kernel, offset, global,
                                             cl::NDRange(8, 8, 8), events, event);
        default:
            return enqueueNDRangeKernel(queue, kernel, offset, global, local, events, event);
        }
    }

    const std::size_t *origGlobal = static_cast<const std::size_t *>(global);
    const std::size_t *origLocal = static_cast<const std::size_t *>(local);
    const std::size_t *origOffset = static_cast<const std::size_t *>(offset);
    const std::size_t dims = global.dimensions();

    std::size_t main[3], extra[3], extraOffset[3];

    for (std::size_t i = 0; i < dims; i++)
    {
        if (origLocal[i] == 0)
            throw cl::Error(CL_INVALID_WORK_GROUP_SIZE, "Local work group size is zero");
        main[i] = origGlobal[i] / origLocal[i] * origLocal[i];
        extra[i] = origGlobal[i] - main[i];
        extraOffset[i] = origOffset[i] + main[i];
    }

    uts::vector<cl::Event> wait;
    for (std::size_t mask = 0; mask < (1U << dims); mask++)
    {
        std::size_t curOffset[3] = {};
        std::size_t curGlobal[3] = {};
        std::size_t curLocal[3] = {};
        bool use = true;
        for (std::size_t i = 0; i < dims; i++)
        {
            if (mask & (1U << i))
            {
                curGlobal[i] = extra[i];
                curOffset[i] = extraOffset[i];
                curLocal[i] = extra[i];
            }
            else
            {
                curGlobal[i] = main[i];
                curOffset[i] = offset[i];
                curLocal[i] = origLocal[i];
            }
            use &= curGlobal[i] > 0;
        }
        if (use)
        {
            wait.push_back(cl::Event());
            queue.enqueueNDRangeKernel(kernel,
                                       makeNDRange(dims, curOffset),
                                       makeNDRange(dims, curGlobal),
                                       makeNDRange(dims, curLocal),
                                       events, &wait.back());
        }
    }

    return enqueueMarkerWithWaitList(queue, &wait, event);
}

bool imageFormatSupported(
    const cl::Context &context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl::ImageFormat format)
{
    uts::vector<cl::ImageFormat> formats;
    context.getSupportedImageFormats(flags, image_type, &formats);
    for (auto f : formats)
    {
        if (f.image_channel_order == format.image_channel_order
            && f.image_channel_data_type == format.image_channel_data_type)
            return true;
    }
    return false;
}

} // namespace CLH
