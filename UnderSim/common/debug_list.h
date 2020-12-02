/**
 * @file
 *
 * Define uts::list as either std::list or __gnu_debug::list depending on
 * UTS_DEBUG_CONTAINERS.
 */

#ifndef UTS_COMMON_DEBUG_LIST
#define UTS_COMMON_DEBUG_LIST

#include <list>
#if defined(__GLIBCXX__) && defined(UTS_DEBUG_CONTAINERS)

#include <debug/list>

namespace uts
{
    template<typename T, typename Alloc = std::allocator<T> >
    using list = __gnu_debug::list<T, Alloc>;
}

#else

namespace uts
{
    template<typename T, typename Alloc = std::allocator<T> >
    using list = std::list<T, Alloc>;
}

#endif

#endif /* !UTS_COMMON_DEBUG_LIST */
