/**
 * @file
 *
 * Define uts::string as either std::string or __gnu_debug::string depending on
 * UTS_DEBUG_CONTAINERS.
 */

#ifndef UTS_COMMON_DEBUG_STRING
#define UTS_COMMON_DEBUG_STRING

#include <string>
#if defined(__GLIBCXX__) && defined(UTS_DEBUG_CONTAINERS)

#include <debug/string>

namespace uts
{
    template<
        typename CharT,
        typename Traits = std::char_traits<CharT>,
        typename Alloc = std::allocator<CharT> >
    using basic_string = __gnu_debug::basic_string<CharT, Traits, Alloc>;
}

namespace std
{
    template<>
    struct hash<__gnu_debug::string>
    {
        size_t operator()(const __gnu_debug::string &s) const noexcept
        {
            return hash<std::string>()(s);
        }
    };
}

#else

namespace uts
{
    template<
        typename CharT,
        typename Traits = std::char_traits<CharT>,
        typename Alloc = std::allocator<CharT> >
    using basic_string = std::basic_string<CharT, Traits, Alloc>;
}

#endif

namespace uts
{
    typedef uts::basic_string<char> string;
    typedef basic_string<wchar_t> wstring;
    typedef basic_string<char16_t> u16string;
    typedef basic_string<char32_t> u32string;
}

#endif /* !UTS_COMMON_DEBUG_STRING */
