/**
 * @file
 */

#ifndef UTS_COMMON_CONSTRAINT_INTERFACE_H
#define UTS_COMMON_CONSTRAINT_INTERFACE_H

#include <functional>
#include <memory>
#include "region.h"
#include <stdio.h>
#include <iostream>

/**
 * Interface for holding the raw data in a curve, for passing between
 * the GUI and the synthesizer. It will obviously have private fields actually
 * holding the data and interfaces for the GUI to manipulate it.
 */

class CurveConstraintBase
{
public:

    virtual ~CurveConstraintBase(){}

    /**
     * Rasterize a subsection of the curve and its area of effect, calling a
     * function for each raster point.
     *
     * @param t0, t1      Portion of curve to consider. Pixels for which the
     *                    nearest point on the curve are outside of this range
     *                    will not trigger the callback.
     * @param step        Raster grid resolution, in units of the finest
     *                    synthesis level. Pixels are considered to exist at
     *                    (@a i+0.5)@a step, (@a j+0.5)@a step for all integral
     *                    @a i and @a j.
     * @param maxEffect   The distances of effect are clamped to this value when
     *                    determining which pixels are covered.
     * @param callback    The callback to be called per pixel (see above).
     *
     * @pre
     * - @a step is a power of 2.
     * - 0 &lt; @a maxEffect.
     *
     * The callback is passed the following parameters:
     * - The @a i and @a j referred to in the description of @a step
     *   (integers).
     * - The position on the curve as a 2-element array
     * - The height at the curve position
     * - The shortest distance from the curve to the raster point
     * - The gradient at the curve position
     * - The area of effect at the curve position
     * - The allowable variation in height at the curve position
     *
     * @note It is expected that each value of @a step will be used with a
     * single corresponding value of @a maxEffect, and there there will only
     * be about 10 different values of @a step (namely, one per synthesis
     * level). It is thus practical to cache some data.
     */
    virtual void forEachPixel(float t0, float t1, int step, float maxEffect, const std::function<void(int, int, const float [], float, float, float, float, float)> & callback) const = 0;

    /**
     * Return a bounding box (possibly a conservative one) for the (i, j)
     * values that would be enumerated by a call to @ref forEachPixel.
     *
     * @param t0, t1      Portion of curve to consider. Pixels for which the
     *                    nearest point on the curve are outside of this range
     *                    will not trigger the callback.
     * @param step        Raster grid resolution, in units of the finest
     *                    synthesis level. Pixels are considered to exist at
     *                    (@a i+0.5)@a step, (@a j+0.5)@a step for all integral
     *                    @a i and @a j.
     * @param maxEffect   The distances of effect are clamped to this value when
     *                    determining which pixels are covered.
     */
    virtual Region boundingBox(float t0, float t1, int step, float maxEffect) const = 0;
};

#endif /* !UTS_COMMON_CONSTRAINT_INTERFACE_H */
