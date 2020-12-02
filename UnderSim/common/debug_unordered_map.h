/**
 * @file
 *
 * Define uts::unordered_map as either std::unordered_map or
 * __gnu_debug::unordered_map depending on UTS_DEBUG_CONTAINERS.
 */

#ifndef UTS_COMMON_DEBUG_UNORDERED_MAP
#define UTS_COMMON_DEBUG_UNORDERED_MAP

#include <unordered_map>
#if defined(__GLIBCXX__) && defined(UTS_DEBUG_CONTAINERS)

#include <debug/unordered_map>

namespace uts
{
    template<
        typename Key,
        typename T,
        typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key>,
        typename Allocator = std::allocator<std::pair<const Key, T> > >
    using unordered_map = __gnu_debug::unordered_map<Key, T, Hash, KeyEqual, Allocator>;
}

#else

namespace uts
{
    template<
        typename Key,
        typename T,
        typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key>,
        typename Allocator = std::allocator<std::pair<const Key, T> > >
    using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual, Allocator>;
}

#endif

#endif /* !UTS_COMMON_DEBUG_UNORDERED_MAP */
