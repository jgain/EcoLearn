/**
 * @file
 *
 * A thread-safe progress meter, modelled on boost::progress_display.
 *
 * The code is loosely based on that from mlsgpu.
 */

#ifndef UTS_COMMON_PROGRESS_H
#define UTS_COMMON_PROGRESS_H

#include <ostream>
#include <iostream>
#include <string>
#include <mutex>
#include <cstdint>
#include <atomic>
#include <utility>
#include <algorithm>

/**
 * A thread-safe progress meter which displays ASCII-art progress.
 */
template<typename T = std::uint64_t>
class ProgressDisplay
{
public:
    typedef T size_type;

    /**
     * Constructor.
     *
     * @param total     Amount of progress on completion
     * @param os        Output stream to show the progress bar
     * @param s1,s2,s3  Prefix to apply to each line of the progress bar
     */
    explicit ProgressDisplay(
        size_type total,
        std::ostream &os = std::cout,
        std::string s1 = "\n",
        std::string s2 = "",
        std::string s3 = "");

    /// Add a given amount to the progress
    void operator+=(size_type increment);
    /// Add 1 to the progress
    void operator++();

    size_type count() const;           ///< Current value
    size_type expected_count() const;  ///< Value at completion

private:
    // Make noncopyable
    ProgressDisplay(const ProgressDisplay &) = delete;
    ProgressDisplay &operator=(const ProgressDisplay &) = delete;

    std::atomic<size_type> current;
    size_type total;                   ///< Total amount of progress expected

    mutable std::mutex mutex;          ///< Lock protecting the stream
    std::ostream &os;                  ///< Output stream
    const std::string s1, s2, s3;

    static constexpr int totalTics = 51;  ///< Width of the ASCII art

    /// Return the number of tics corresponding to a given progress
    int ticsFor(size_type value) const;

    /// Print the header and initialize state
    void restart(size_type total);
};

template<typename T>
constexpr int ProgressDisplay<T>::totalTics;

template<typename T>
ProgressDisplay<T>::ProgressDisplay(
    size_type total,
    std::ostream &os,
    std::string s1,
    std::string s2,
    std::string s3)
: current(0), total(total), os(os),
    s1(std::move(s1)), s2(std::move(s2)), s3(std::move(s3))
{
    restart(total);
}

template<typename T>
T ProgressDisplay<T>::count() const
{
    return current;
}

template<typename T>
T ProgressDisplay<T>::expected_count() const
{
    return total;
}

template<typename T>
int ProgressDisplay<T>::ticsFor(size_type value) const
{
    return std::min(value, total) * totalTics / total; // TODO: can overflow
}

template<typename T>
void ProgressDisplay<T>::operator+=(size_type increment)
{
    size_type old = current.fetch_add(increment);
    size_type cur = old + increment;
    int oldTics = ticsFor(old);
    int curTics = ticsFor(cur);
    if (oldTics < curTics)
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = oldTics; i < curTics; i++)
            os << "*";
        if (curTics == totalTics)
            os << '\n';
        os.flush();
    }
}

template<typename T>
void ProgressDisplay<T>::operator++()
{
    *this += 1;
}

template<typename T>
void ProgressDisplay<T>::restart(size_type total)
{
    current = 0;
    this->total = total;
    os  << s1 << "0%   10   20   30   40   50   60   70   80   90   100%\n"
        << s2 << "|----|----|----|----|----|----|----|----|----|----|\n"
        << s3;
    os.flush();
}

extern template class ProgressDisplay<std::uint64_t>;

#endif /* !PROGRESS_H */
