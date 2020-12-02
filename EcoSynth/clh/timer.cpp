/**
 * @file
 *
 * Timing of OpenCL operations.
 */

#include <memory>
#include <utility>
#include <chrono>
#include <common/timer.h>
#include "clhpp.h"
#include "timer.h"

namespace CLH
{

Timer::Timer(const cl::CommandQueue &queue, const stats::TimeInit &t, double *out)
    : stats::TimerBase(t, out)
{
    if (stats::isTimingEnabled())
        queue.enqueueMarker(&startEvent);
}

Timer::Timer(const cl::CommandQueue &queue, const std::shared_ptr<stats::Time> &t, double *out)
    : stats::TimerBase(t, out)
{
    if (stats::isTimingEnabled())
        queue.enqueueMarker(&startEvent);
}

Timer::Timer(Timer &&other) : stats::TimerBase(std::move(other))
{
    if (this != &other)
    {
        startEvent = std::move(other.startEvent);
        stopEvent = std::move(other.stopEvent);
        other.startEvent = cl::Event();
        other.stopEvent = cl::Event();
    }
}

void Timer::stop()
{
    if (startEvent() && !stopEvent())
    {
        const cl::CommandQueue &queue = startEvent.getInfo<CL_EVENT_COMMAND_QUEUE>();
        queue.enqueueMarker(&stopEvent);
    }
}

Timer::~Timer()
{
    if (startEvent())
    {
        stop();
        stopEvent.wait();
        // Note: we measure the start for both the endpoint markers
        cl_ulong start = startEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = stopEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        std::chrono::duration<cl_ulong, std::nano> duration(end - start);
        done(duration);
    }
}

} // namespace CLH
