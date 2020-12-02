
/*******************************************************
 * OpenKODE Core extension: KD_KHR_perfcounter
 *******************************************************/
/* Sample KD/KHR_perfcounter.h for OpenKODE Core */
#ifndef __kd_KHR_perfcounter_h_
#define __kd_KHR_perfcounter_h_
#include <KD/kd.h>

#ifdef __cplusplus
extern "C" {
#endif


#define KD_INFINITE_COUNTER_VAL_KHR (KDINT64_MAX)
#define KD_UNKNOWN_COUNTER_VAL_KHR (-1)

/* KDCounterInfoKHR: Information on a single performance counter. */
typedef struct KDCounterInfoKHR {
    const KDchar *vendorName;
    const KDchar *name;
    const KDchar *description;
    KDint64 minValue;
    KDint64 maxValue;
    KDfloat32 defaultScale;
} KDCounterInfoKHR;

/* kdGetNumberOfCountersKHR: Return the number of performance counters. */
KD_API KDint KD_APIENTRY kdGetNumberOfCountersKHR(void);

/* kdGetCounterInformationKHR: Retrieve information on a performance counter. */
KD_API const KDCounterInfoKHR *KD_APIENTRY kdGetCounterInformationKHR(KDint index);

/* kdActivateCountersKHR: Make counters active. */
KD_API KDint KD_APIENTRY kdActivateCountersKHR(const KDint *indexes, KDint numindexes);

/* kdDeactivateCountersKHR: Makes counters inactive. */
KD_API KDint KD_APIENTRY kdDeactivateCountersKHR(const KDint *indexes, KDint numindexes);

/* kdStartSamplingKHR: Start the performance counters sampling. */
KD_API KDint KD_APIENTRY kdStartSamplingKHR(void);

/* kdStopSamplingKHR: Stop the performance counters sampling. */
KD_API KDint KD_APIENTRY kdStopSamplingKHR(void);

/* kdGetCounterValuesKHR: Retrieves list of counter values. */
KD_API KDint KD_APIENTRY kdGetCounterValuesKHR(const KDint *indexes, KDint numindexes, KDint64 *values);

#ifdef __cplusplus
}
#endif

#endif /* __kd_KHR_perfcounter_h_ */

