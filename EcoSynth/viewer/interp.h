// interp.h: interpolate distributions using mass transport
// author: James Gain, from code supplied by Ulysse Vimont
// date: 7 August 2016

#include "synth.h"

typedef unsigned int uint;
typedef vector<float> Histogram;    // histograms are just arrays of float

// Tuning: determines accuracy of interpolation, but also effects efficiency.
#define MT_STEPS 100

class Interpolator
{
private:

    /**
     * @brief cumsum : computes the cumulated sum of the input
     * @param h : input
     * @param h_cumsum : output (size = input + 1)
     */
    void cumsum(const Histogram& h, Histogram& h_cumsum);

    /**
     * @brief invert : computes the invert of h as a nb_steps values histogram.
     * @param h : should be increasing.
     * @param h_inv
     * @param nb_steps
     */
    void invert(const Histogram& h, Histogram& h_inv, const uint nb_steps);

    /**
     * @brief normalize : in place normalization of h (h is guaranted to be valued in [0,1] after that).
     * @param h
     */
    void normalize(Histogram& h);

    /**
     * @brief denormalize : in place denormalization of h (h is guaranted to be valued in [min_val,max_val] after that IF it were originally normalized).
     * @param h
     */
    void denormalize(Histogram& h, float min_val, float max_val);

    /**
     * @brief get_fibre : computes y such that h(y) = x. Uses linear interpoation.
     * @param h : should be increasing.
     * @param x : should be in [min(h), max(h)]
     * @return y
     */
    float get_fibre(const Histogram& h, const float x);

    /**
     * @brief max
     * @param h
     * @return : the greater value in h
     */
    float max(const Histogram& h);

    /**
     * @brief min
     * @param h
     * @return : the smaller value in h
     */
    float min(const Histogram& h);

    /**
     * @brief diff : computes the difference between consecutive bins of the input
     * @param h : input
     * @param h_diff : output (size = input - 1)
     */
    void diff(const Histogram& h, Histogram& h_diff);

    /**
     * @brief interpolate_linear: per-value interpolation of histograms
     * @param h0
     * @param h1
     * @param ht
     * @param t
     */
    void interpolate_linear(const Histogram& h0, const Histogram& h1, Histogram& ht, const float t);

    /**
     * @brief interpolate_icdf : histogram interpolation using Invert Cumulated Density Function (1D optimal transport).
     * @param h0
     * @param h1
     * @param ht
     * @param t
     */
    void interpolate_icdf(const Histogram& h0, const Histogram& h1, Histogram& ht, const float t);

    /// Convert radial distribution to histogram
    void radToHist(AnalysisConfiguration &anconfig, RadialDistribution rad, Histogram & hist);

    /// Convert histogram to radial distribution
    void histToRad(AnalysisConfiguration &anconfig, Histogram hist, RadialDistribution &rad);

    void histPos(Histogram &hist);

public:

    void interp(std::vector<int> global_priority, Distribution &d0, Distribution &d1, Distribution &dout, const float t);
};
