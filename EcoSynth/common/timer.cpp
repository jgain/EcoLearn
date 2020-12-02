/**
 * @file
 *
 * Utilities to simplify profiling.
 */

#include <chrono>
#include <utility>
#include <iostream>
#include <memory>
#include <thread>
#include <algorithm>
#include "debug_string.h"
#include "debug_vector.h"
#include "timer.h"
#include "stats.h"

namespace stats
{

namespace detail
{

static bool timersEnabled = false;    ///< Whether timers were enabled by @ref enableTimers

static uts::vector<std::shared_ptr<Time> > &getTimesRaw()
{
    /* This is wrapped into a function to give predictable initialization
     * order when a TimeInit is declared at file scope.
     */
    static uts::vector<std::shared_ptr<Time> > times;
    return times;
}

static std::mutex &getTimesMutex()
{
    // See comment in getTimes
    static std::mutex timesMutex;
    return timesMutex;
}

} // namespace detail

void enableTimers(bool enable)
{
    detail::timersEnabled = enable;
}

bool isTimingEnabled()
{
    return detail::timersEnabled;
}

uts::vector<std::shared_ptr<Time> > getTimes()
{
    const auto &times = detail::getTimesRaw();
    std::mutex &mutex = detail::getTimesMutex();

    uts::vector<std::shared_ptr<Time> > times2;
    {
        std::lock_guard<std::mutex> lock(mutex);
        times2 = times;
    }

    auto cmp = [](const std::shared_ptr<Time> &a, const std::shared_ptr<Time> &b)
    {
        return a->name() < b->name();
    };
    std::sort(times2.begin(), times2.end(), cmp);
    return times2;
}

void reportTimes()
{
    const auto &times = getTimes();
    for (const auto &p : times)
    {
        if (p->times() > 0)
        {
            std::chrono::duration<double> t = p->total();
            printAlways("TOTAL,", p->name(), ",", t.count(), ",", p->times(), "\n");
        }
    }
}


Time::Time(const uts::string &name) : name_(name), ticks_(0), times_(0)
{
}

const uts::string &Time::name() const
{
    return name_;
}

clock_type::duration Time::total() const
{
    return clock_type::duration(ticks_.load(std::memory_order_relaxed));
}

std::uint64_t Time::times() const
{
    return times_.load(std::memory_order_relaxed);
}

Time &Time::operator+=(const clock_type::duration &add)
{
    ticks_.fetch_add(add.count(), std::memory_order_relaxed);
    times_.fetch_add(1, std::memory_order_relaxed);
    return *this;
}


TimeInit::TimeInit(const uts::string &name)
{
    auto &times = detail::getTimesRaw();
    std::mutex &mutex = detail::getTimesMutex();

    std::lock_guard<std::mutex> lock(mutex);
    times.push_back(std::make_shared<Time>(name));
    time = times.back();
}

TimerBase::TimerBase(const TimeInit &t, double *out)
    : TimerBase(t.time.lock(), out)
{
}

TimerBase::TimerBase(const std::shared_ptr<Time> &t, double *out) :
    time(t),
    out(out)
{
}

TimerBase::TimerBase(TimerBase &&other)
{
    if (this != &other)
    {
        time = std::move(other.time);
        out = other.out;
        other.time = nullptr;
        other.out = nullptr;
    }
}

void TimerBase::doneDouble(double elapsed)
{
    if (time)
        printAlways("TIMER,", time->name(), ",", elapsed, '\n');
    if (out != nullptr)
        *out = elapsed;
}

Timer::Timer(const TimeInit &t, double *out) :
    TimerBase(t, out),
    start(detail::timersEnabled ? clock_type::now() : clock_type::time_point())
{
}

Timer::Timer(const std::shared_ptr<Time> &t, double *out) :
    TimerBase(t, out),
    start(detail::timersEnabled ? clock_type::now() : clock_type::time_point())
{
}

Timer::Timer(Timer &&other) : TimerBase(std::move(other))
{
    if (this != &other)
    {
        start = other.start;
        other.start = clock_type::time_point(); // mark as inactive
    }
}

void Timer::stop()
{
    if (detail::timersEnabled && start != clock_type::time_point())
    {
        auto end = clock_type::now();
        clock_type::duration elapsed = end - start;
        done(elapsed);
        start = clock_type::time_point(); // mark as stopped
    }
}

Timer::~Timer()
{
    stop();
}

} // namespace stats
