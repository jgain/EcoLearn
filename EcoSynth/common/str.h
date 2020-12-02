/**
 * @file
 *
 * Misc string utilities.
 */

#ifndef COMMON_STR_H
#define COMMON_STR_H

#include "debug_string.h"

/// Tests whether @a suffix is a suffix of @a a
static inline bool endsWith(const uts::string &a, const uts::string &suffix)
{
    return a.size() >= suffix.size() && a.substr(a.size() - suffix.size()) == suffix;
}

#endif /* !COMMON_STR_H */
