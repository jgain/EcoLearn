/**
 * @file
 *
 * Define uts::vector as either std::vector or __gnu_debug::vector depending on
 * UTS_DEBUG_CONTAINERS.
 */

#ifndef UTS_COMMON_DEBUG_VECTOR
#define UTS_COMMON_DEBUG_VECTOR

#include <vector>
#if defined(__GLIBCXX__) && defined(UTS_DEBUG_CONTAINERS)

#include <debug/vector>

namespace uts
{
    template<typename T, typename Alloc = std::allocator<T> >
    using vector = __gnu_debug::vector<T, Alloc>;
}

#else

namespace uts
{
    template<typename T, typename Alloc = std::allocator<T> >
    using vector = std::vector<T, Alloc>;
}

#endif

#endif /* !UTS_COMMON_DEBUG_VECTOR */
