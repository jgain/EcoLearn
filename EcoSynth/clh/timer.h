/**
 * @file
 *
 * Timing of OpenCL operations.
 */

#ifndef UTS_CLH_TIMER_H
#define UTS_CLH_TIMER_H

#include <memory>
#include <common/timer.h>
#include "clhpp.h"

namespace CLH
{

/**
 * Timer class similar to @ref stats::Timer, but which measures time spent on the GPU.
 *
 * @see @ref stats::Timer, @ref stats::enableTimers.
 */
class Timer : public stats::TimerBase
{
private:
    cl::Event startEvent;
    cl::Event stopEvent;

public:
    Timer(const cl::CommandQueue &queue, const stats::TimeInit &t, double *out = nullptr);
    Timer(const cl::CommandQueue &queue, const std::shared_ptr<stats::Time> &t, double *out = nullptr);

    Timer(Timer &&other);

    void stop();

    ~Timer();
};

} // namespace CLH

#endif /* !UTS_CLH_TIMER_H */
