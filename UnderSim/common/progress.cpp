/**
 * @file
 *
 * A thread-safe progress meter, modelled on boost::progress_display.
 */

#include "progress.h"

template class ProgressDisplay<std::uint64_t>;
