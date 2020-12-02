/**
 * @file
 *
 * Wraps the include of cl.hpp to set appropriate preprocessor definitions.
 */

#ifndef UTS_CLH_CLHPP_H
#define UTS_CLH_CLHPP_H

#include <CL/opencl.h>
/* Prevent the OpenCL 1.2 functionality in cl.hpp from being exposed. This
 * prevents the binary from depending on an OpenCL 1.2 ICD.
 */
#undef CL_VERSION_1_2

/* Workaround for a cl.hpp bug on OSX: these macros are defined to specify
 * weak symbols, which seems to override the inline linkage and causes
 * duplicate symbol errors. Note that cl.hpp will redefine these as empty.
 */
#undef CL_EXT_PREFIX__VERSION_1_1_DEPRECATED
#undef CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED

#ifndef __CL_ENABLE_EXCEPTIONS
# define __CL_ENABLE_EXCEPTIONS
#endif
#ifdef CL_HPP_
# error "cl.hpp has already been included"
#endif

#ifdef UTS_DEBUG_CONTAINERS
# include <common/debug_vector.h>
# include <common/debug_string.h>
# define __USE_DEV_VECTOR
# define __USE_DEV_STRING
# define VECTOR_CLASS uts::vector
# define STRING_CLASS uts::string
#endif

#include <CL/cl.hpp>

#endif /* !UTS_CLH_CLHPP_H */
