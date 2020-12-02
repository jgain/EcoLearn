
/*******************************************************
 * OpenKODE Core extension: KD_KHR_float64
 *******************************************************/
/* Sample KD/KHR_float64.h for OpenKODE Core */
#ifndef __kd_KHR_float64_h_
#define __kd_KHR_float64_h_
#include <KD/kd.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef double KDfloat64KHR;
#define KD_E_KHR 2.718281828459045235
#define KD_PI_KHR 3.141592653589793239
#define KD_PI_2_KHR 1.570796326794896619
#define KD_2PI_KHR 6.283185307179586477
#define KD_LOG2E_KHR 1.442695040888963407
#define KD_LOG10E_KHR 0.4342944819032518276
#define KD_LN2_KHR 0.6931471805599453094
#define KD_LN10_KHR 2.302585092994045684
#define KD_PI_4_KHR 0.7853981633974483096
#define KD_1_PI_KHR 0.3183098861837906715
#define KD_2_PI_KHR 0.6366197723675813431
#define KD_2_SQRTPI_KHR 1.128379167095512574
#define KD_SQRT2_KHR 1.414213562373095049
#define KD_SQRT1_2_KHR 0.7071067811865475244
#define KD_DBL_EPSILON_KHR 2.2204460492503131e-16
#define KD_DBL_MAX_KHR 1.7976931348623157e+308
#define KD_DBL_MIN_KHR 2.2250738585072014e-308
#define KD_HUGE_VAL_KHR (1.0/0.0)
#define KD_DEG_TO_RAD_KHR 0.01745329251994329577
#define KD_RAD_TO_DEG_KHR 57.29577951308232088
KD_API KDfloat64KHR KD_APIENTRY kdAcosKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdAsinKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdAtanKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdAtan2KHR(KDfloat64KHR y, KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdCosKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdSinKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdTanKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdExpKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdLogKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdFabsKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdPowKHR(KDfloat64KHR x, KDfloat64KHR y);
KD_API KDfloat64KHR KD_APIENTRY kdSqrtKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdCeilKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdFloorKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdRoundKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdInvsqrtKHR(KDfloat64KHR x);
KD_API KDfloat64KHR KD_APIENTRY kdFmodKHR(KDfloat64KHR x, KDfloat64KHR y);

/* kdStrtodKHR: Convert a string to a 64-bit floating point number. */
KD_API KDfloat64KHR KD_APIENTRY kdStrtodKHR(const KDchar *s, KDchar **endptr);

/* kdDtostrKHR: Convert a 64-bit float to a string. */
#define KD_DTOSTR_MAXLEN_KHR 25
KD_API KDssize KD_APIENTRY kdDtostrKHR(KDchar *buffer, KDsize buflen, KDfloat64KHR number);

#ifdef __cplusplus
}
#endif

#endif /* __kd_KHR_float64_h_ */

