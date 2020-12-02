/**
 * @file
 *
 * Miscellaneous maths utilities.
 */

#ifndef UTS_COMMON_MATHUTILS_H
#define UTS_COMMON_MATHUTILS_H

#include <limits>
#include <cassert>

/// Tests whether @a x is a power of 2
static inline constexpr bool isPower2(int x)
{
    return x > 0 && (x & (x - 1)) == 0;
}

/**
 * Divides a value by 2<sup>@a shift</sup>, rounding to nearest. Ties are
 * broken by rounding towards +inf.
 *
 * @pre Assuming that @c int is a 32-bit type,
 * - 0 &lt;= @a shift &lt; 32
 * - -2<sup>30</sup> &lt; @a x &lt; 2<sup>30</sup>
 * The answer will be right in more cases than this, but it is a conservative estimate.
 *
 * @note The implementation assumes a twos-complement machine with an
 * arithmetic right shift.
 */
static inline constexpr int shrRound(int x, int shift)
{
    return (shift > 0) ? (x + (1 << (shift - 1))) >> shift : x;
}

/**
 * Divides a value by 2<sup>@a shift</sup> rounding towards negative infinity.
 *
 * @pre Assuming that @c int is a 32-bit type,
 * - 0 &lt;= @a shift &lt; 32
 *
 * @note The implementation assumes a twos-complement machine.
 */
static inline constexpr int shrDown(int x, int shift)
{
    return x >> shift;
}

/**
 * Divides a value by 2<sup>@a shift</sup> rounding towards positive infinity.
 *
 * @pre Assuming that @c int is a 32-bit type,
 * - 0 &lt;= @a shift &lt; 31
 * - -2<sup>30</sup> &lt; @a x &lt; 2<sup>30</sup>
 * The answer will be right in more cases than this, but it is a conservative estimate.
 *
 * @note The implementation assumes a twos-complement machine with an
 * arithmetic right shift.
 */
static inline constexpr int shrUp(int x, int shift)
{
    return (x + ((1 << shift) - 1)) >> shift;
}

/**
 * Round up @a x to the next multiple of @a ratio.
 *
 * @pre
 * - The result does not exceed the range of @a T.
 * - @a ratio is positive.
 * - @a x is non-negative.
 */
template<typename T>
static inline T roundUp(T x, T ratio)
{
    assert(ratio > 0);
    assert(x == 0 || (x > 0 && (x - 1) / ratio * ratio <= std::numeric_limits<T>::max() - ratio));
    return (x > 0) ? ((x - 1) / ratio + 1) * ratio : 0;
}

/**
 * Tests whether the ratio between two floating-point values is exactly
 * a power of 2. The return value is always false if either value is
 * negative or nonfinite.
 *
 * The case with 65536 is to reduce the number of recursive calls required,
 * making it usable in a constexpr setting.
 */
template<typename T>
constexpr bool isPower2Ratio(T a, T b)
{
    return (a > 0 && b > 0
            && a < std::numeric_limits<T>::infinity()
            && b < std::numeric_limits<T>::infinity())
        && ((a > b && isPower2Ratio(b, a))
            || a == b
            || (b >= 65536 * a && isPower2Ratio(65536 * a, b))
            || (b >= 2 * a && isPower2Ratio(2 * a, b)));
}

extern template bool isPower2Ratio<float>(float, float);

#endif /* !UTS_COMMON_MATHUTILS_H */
