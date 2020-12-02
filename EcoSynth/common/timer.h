/**
 * @file
 *
 * Utilities to simplify profiling.
 */

#ifndef UTS_COMMON_TIMER_H
#define UTS_COMMON_TIMER_H

#include "debug_string.h"
#include <chrono>
#include <type_traits>
#include <memory>
#include <atomic>
#include "debug_vector.h"

namespace stats
{

// Use high-resolution timer if it is monotonic, otherwise steady_timer (which always is)
typedef std::conditional<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>::type clock_type;

/**
 * Keeps track of total accumulated time. It supports atomic increment of
 * elapsed time.
 */
class Time
{
public:
    static_assert(std::is_integral<clock_type::duration::rep>::value, "Duration type must be integral to use atomics");

    /// Initializes with zero total time
    explicit Time(const uts::string &name);

    const uts::string &name() const;
    clock_type::duration total() const;  ///< Total duration of all uses
    std::uint64_t times() const;         ///< Number of times called
    Time &operator+=(const clock_type::duration &add); ///< Add a new sample

private:
    uts::string name_;
    std::atomic<clock_type::duration::rep> ticks_;
    std::atomic<std::uint64_t> times_;
};

/**
 * Registration mechanism for times. A @ref TimeInit is safe to declare at
 * file scope, and this is the recommended usage. The constructor is
 * also thread-safe.
 */
class TimeInit
{
    friend class TimerBase;
private:
    std::weak_ptr<Time> time;

public:
    explicit TimeInit(const uts::string &name);
};

/**
 * Base class for all timers, which does not specify how the time is to be collected.
 */
class TimerBase
{
private:
    std::shared_ptr<Time> time;
    double *out;

    void doneDouble(double elapsed);

    // Make non-copyable
    TimerBase(const TimerBase &) = delete;
    TimerBase &operator=(const TimerBase &) = delete;

protected:
    template<typename Rep, typename Period>
    void done(std::chrono::duration<Rep, Period> elapsed)
    {
        if (time)
            *time += elapsed;
        double elapsedDouble = std::chrono::duration<double>(elapsed).count();
        doneDouble(elapsedDouble);
    }

public:
    explicit TimerBase(const TimeInit &t, double *out = nullptr);
    explicit TimerBase(const std::shared_ptr<Time> &t, double *out = nullptr);

    // Make movable, but not move assignable
    TimerBase(TimerBase &&);
};

/**
 * Timer that measures its own elapsed lifetime and reports it to stdout on
 * termination. It can also be stopped explicitly by calling @ref stop.
 *
 * The elapsed time is also accumulated into a @ref Time, and optionally
 * stored in a double.
 *
 * @see @ref stats::enableTimers.
 */
class Timer : public TimerBase
{
private:
    // Use high-resolution timer if it is monotonic, otherwise steady_timer (which always is)
    typedef std::conditional<
        std::chrono::high_resolution_clock::is_steady,
        std::chrono::high_resolution_clock,
        std::chrono::steady_clock>::type clock_type;

    std::chrono::time_point<clock_type> start;

public:
    explicit Timer(const TimeInit &t, double *out = nullptr);
    explicit Timer(const std::shared_ptr<Time> &t, double *out = nullptr);
    Timer(Timer &&);

    /// Stop the timer, print the information, and perform accumulations.
    void stop();
    ~Timer();
};

/**
 * Enable or disable reporting of elapsed time. Turning this on has a small
 * performance impact, and of course causes spam on stdout.
 *
 * @pre There are no currently live timers.
 */
void enableTimers(bool enable);

/**
 * Returns the value set by @ref enableTimers.
 */
bool isTimingEnabled();

/**
 * Retrieve all times. This function is thread-safe (it makes a copy of the
 * list of times), but nothing prevents the referenced times from being
 * updated while the list of walked.
 *
 * The list is sorted by name.
 */
uts::vector<std::shared_ptr<Time> > getTimes();

/**
 * Report total times for all registered timers. This function is thread-safe,
 * but it does not guarantee an atomic snapshot i.e., another thread may
 * increment some times part-way through the printout.
 */
void reportTimes();

} // namespace stats

#endif /* !UTS_COMMON_TIMER_H */
