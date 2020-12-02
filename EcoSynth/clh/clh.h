/**
 * @file
 */

#ifndef CLH_H
#define CLH_H

#if HAVE_CONFIG_H
# include <config.h>
#endif
#include <cstdint>
#include <boost/program_options.hpp>
#include <common/debug_unordered_map.h>
#include <common/debug_vector.h>
#include <common/debug_string.h>
#include <functional>
#include <map>
#include "clhpp.h"

/// OpenCL helper functions
namespace CLH
{

/**
 * Exception thrown when an OpenCL device cannot be used.
 */
class invalid_device : public std::runtime_error
{
private:
    cl::Device device;
public:
    invalid_device(const cl::Device &device, const uts::string &msg)
        : std::runtime_error(device.getInfo<CL_DEVICE_NAME>() + ": " + msg) {}

    cl::Device getDevice() const
    {
        return device;
    }

    virtual ~invalid_device() throw()
    {
    }
};

/// %Option names for OpenCL options
namespace Option
{
const char * const device = "cl-device";
const char * const gpu = "cl-gpu";
const char * const cpu = "cl-cpu";
} // namespace Option

/**
 * Append program options for selecting an OpenCL device.
 *
 * The resulting variables map can be passed to @ref findDevice.
 */
void addOptions(boost::program_options::options_description &desc);

/**
 * Pick an OpenCL device based on command-line options. Each device is matched
 * against a number of criteria.
 * - <tt>--cl-device=name:n</tt> matches for the nth device with a prefix of @a
 *   name.
 * - <tt>--cl-device=name</tt> matches for all devices with a prefix of @a name.
 * - <tt>--cl-gpu</tt> causes only GPU devices to match.
 * - <tt>--cl-cpu</tt> causes only CPU devices to match.
 *
 * If <tt>--cl-device</tt> is not specified, then any device can match, subject to
 * <tt>--cl-gpu</tt> and <tt>--cl-cpu</tt> if specified. If multiple devices match,
 * GPUs are preferred, otherwise an arbitrary device is chosen.
 *
 * @param vm            Command-line options
 * @param deviceFilter  Function that returns a reason if a device is unusable, and an empty string otherwise.
 *
 * @return A devices matching the command-line options. If
 * no suitable device is found, returns a default-initialized @c cl::Device.
 */
cl::Device findDevice(
    const boost::program_options::variables_map &vm,
    const std::function<uts::string(const cl::Device &)> &deviceFilter);

/**
 * Create an OpenCL context suitable for use with a device.
 */
cl::Context makeContext(const cl::Device &device);

/**
 * Build a program for potentially multiple devices.
 *
 * If compilation fails, the build log will be emitted to the error log.
 *
 * @param context         Context to use for building.
 * @param devices         Devices to build for.
 * @param filename        File to load (relative to current directory).
 * @param defines         Defines that will be set before the source is preprocessed.
 * @param options         Extra compilation options.
 *
 * @throw std::invalid_argument if the file could not be opened.
 * @throw cl::Error if the program could not be compiled.
 */
cl::Program build(const cl::Context &context, const uts::vector<cl::Device> &devices,
                  const uts::string &filename, const uts::unordered_map<uts::string, uts::string> &defines = uts::unordered_map<uts::string, uts::string>(),
                  const uts::string &options = "");

/**
 * Build a program for all devices associated with a context.
 *
 * This is a convenience wrapper for the form that takes an explicit device
 * list.
 */
cl::Program build(const cl::Context &context,
                  const uts::string &filename, const uts::unordered_map<uts::string, uts::string> &defines = uts::unordered_map<uts::string, uts::string>(),
                  const uts::string &options = "");

/**
 * Implementation of clEnqueueMarkerWithWaitList which can be used in OpenCL
 * 1.1. It differs from the OpenCL 1.2 function in several ways:
 *  - If no input events are passed, it does not wait for anything (other than as
 *    constrained by an in-order queue), rather than waiting for all previous
 *    work.
 *  - If exactly one input event is passed, it may return this event (with
 *    an extra reference) instead of creating a new one.
 *  - It may choose to wait on all previous events in the command queue.
 *    Thus, if your algorithm depends on it not doing so (e.g. you've used
 *    user events to create dependencies backwards in time) it may cause
 *    a deadlock.
 *  - It is legal for @a event to be @c NULL (in which case the marker is
 *    still enqueued, so that if the wait list contained events from other
 *    queues then a barrier on this queue would happen-after those events).
 */
cl_int enqueueMarkerWithWaitList(const cl::CommandQueue &queue,
                                 const uts::vector<cl::Event> *events,
                                 cl::Event *event);

/**
 * Extension of @c cl::CommandQueue::enqueueNDRangeKernel that allows the
 * number of work-items to be zero.
 *
 * @param queue      Queue to enqueue on
 * @param kernel,offset,global,local,events,event As for @c cl::CommandQueue::enqueueNDRangeKernel
 */
cl_int enqueueNDRangeKernel(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local = cl::NullRange,
    const uts::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL);

/**
 * Extends kernel enqueuing by allowing the global size to not be a multiple of
 * the local size. Where necessary, multiple launches are used to handle the
 * left-over bits at the edges, adjusting the global offset to compensate.
 *
 * This does have some side-effects:
 *  - Different work-items will participate in workgroups of different sizes.
 *    Thus, the workgroup size cannot be baked into the kernel.
 *  - @c get_global_id will work as expected, but @c get_group_id and
 *    @c get_global_offset may not behave as expected.
 * In general this function is best suited to cases where the workitems
 * operate complete independently.
 *
 * The provided @a local is used both as the preferred work group size for the
 * bulk of the work, and as an upper bound on work group size.
 *
 * @param queue      Queue to enqueue on
 * @param kernel,offset,global,local,events,event As for @c cl::CommandQueue::enqueueNDRangeKernel
 */
cl_int enqueueNDRangeKernelSplit(
    const cl::CommandQueue &queue,
    const cl::Kernel &kernel,
    const cl::NDRange &offset,
    const cl::NDRange &global,
    const cl::NDRange &local = cl::NullRange,
    const uts::vector<cl::Event> *events = NULL,
    cl::Event *event = NULL);

namespace detail
{

/**
 * Sets a single kernel argument.
 *
 * @return The position of the next argument
 */
template<typename T>
inline cl_uint setKernelArg(cl::Kernel &kernel, cl_uint pos, const T &arg)
{
    kernel.setArg(pos, arg);
    return pos + 1;
}

/**
 * Sets a vector of consecutive kernel arguments (recursively).
 *
 * @return The position of the following kernel argument
 */
template<typename T>
inline cl_uint setKernelArg(cl::Kernel &kernel, cl_uint pos, const uts::vector<T> &arg)
{
    for (const auto &value : arg)
    {
        pos = setKernelArg(kernel, pos, value);
    }
    return pos;
}

/// Terminal case for recursive setArgs
inline cl_uint setKernelArgs(cl::Kernel &kernel, cl_uint pos)
{
    (void) kernel;
    (void) pos;
    return pos;
}

template<typename F, typename... T>
void setKernelArgs(cl::Kernel &kernel, cl_uint pos, F &&first, T&&... other)
{
    pos = setKernelArg(kernel, pos, std::forward<F>(first));
    setKernelArgs(kernel, pos, std::forward<T>(other)...);
}

};

/**
 * Type-safe wrapper around @ref enqueueNDRangeKernelSplit, modeled on
 * @c cl::make_kernel. It is a function object that takes a @c cl::EnqueueArgs to
 * specify launch parameters, followed by the kernel arguments.
 *
 * Unlike cl::make_kernel, it has special treatment for parameters of type uts::vector:
 * they are unpacked into individual arguments. This simplifies the case where the number
 * of arguments is determined based on a run-time parameter.
 *
 * @warning The function object is not reentrant: it cannot safely be called from
 * two threads at once.
 */
template<typename... T>
class SplitFunctor
{
private:
    cl::Kernel kernel;

public:
    typedef cl::Event result_type;

    SplitFunctor() = default;
    explicit SplitFunctor(const cl::Kernel &kernel) : kernel(kernel) {}
    explicit SplitFunctor(cl::Kernel &&kernel) : kernel(std::move(kernel)) {}

    cl::Event operator()(const cl::EnqueueArgs &args, T... params)
    {
        cl::Event out;
        detail::setKernelArgs(kernel, 0, std::forward<T>(params)...);
        enqueueNDRangeKernelSplit(args.queue_, kernel, args.offset_, args.global_, args.local_,
                                  &args.events_, &out);
        return std::move(out);
    }
};

/**
 * Checks whether a specific image format is supported for a given use.
 */
bool imageFormatSupported(
    const cl::Context &context,
    cl_mem_flags flags,
    cl_mem_object_type image_type,
    cl::ImageFormat format);

} // namespace CLH

#endif /* !CLH_H */
