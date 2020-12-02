/**
 * @file
 *
 * Utilities for recording statistics
 */

#include <mutex>
#include "stats.h"

namespace stats
{

namespace detail
{

static bool statsEnabled = false;   ///< Whether to print statistics information
static std::mutex statsPrintMutex;  ///< Mutex held by @ref printAlways.

bool getStatsEnabled()
{
    return statsEnabled;
}

std::mutex &getStatsPrintMutex()
{
    return statsPrintMutex;
}

} // namespace detail

void enableStats(bool enabled)
{
    detail::statsEnabled = true;
}

} // namespace stats
