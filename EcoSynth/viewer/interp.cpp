#include "interp.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <limits>
#include <math.h>

using namespace std;

float Interpolator::max(const Histogram& h)
{
    float ans = std::numeric_limits<float>::min();
    for(float f : h)
        ans = std::max(ans, f);
    return ans;
}

float Interpolator::min(const Histogram& h)
{
    float ans = std::numeric_limits<float>::max();
    for(float f : h)
        ans = std::min(ans, f);
    return ans;
}

void Interpolator::cumsum(const Histogram& h, Histogram& h_cumsum)
{
    h_cumsum.resize(h.size()+1);

    h_cumsum[0] = 0;
    for(uint i=0; i<h.size(); ++i)
        h_cumsum[i+1] = h_cumsum[i] + h[i];
}

void Interpolator::diff(const Histogram& h, Histogram& h_diff)
{
    h_diff.resize(h.size()-1);

    for(uint i=0; i<h.size()-1; ++i)
        h_diff[i] = h[i+1] - h[i];
}

float Interpolator::get_fibre(const Histogram& h, const float x)
{
    int imin = -1;
    int imax = -1;

    for(imax=h.size() - 1; imax>=0; --imax)
    {
        if(h[imax] <= x)
        {
            imax = imax+1;
            break;
        }
    }

    for(imin=0; imin<int(h.size()); ++imin)
    {
        if(h[imin] >= x)
        {
            imin = imin-1;
            break;
        }
    }

    if(imin == -1 || imax == -1)
    {
        return 0;
    }

    if(imin == int(h.size()) || imax == int(h.size()))
    {
        return h.size()-1;
    }

    if(imin == imax)
    {
        return imin; // Should not happend
    }


    float t = (x - h[imin]) / (h[imax] - h[imin]);
    float ans = (1.0 - t) * imin + t * imax;

    return ans;
}

void Interpolator::normalize(Histogram& h)
{
    float max_val = max(h);
    float min_val = min(h);

    for(float& f : h)
    {
        f = (f-min_val) / (max_val - min_val);
    }
}

void Interpolator::denormalize(Histogram& h, float min_val, float max_val)
{
    for(float& f : h)
    {
        f = f* (max_val - min_val) + min_val;
    }
}

void Interpolator::invert(const Histogram& h, Histogram& h_inv, const uint nb_steps)
{
    h_inv.resize(nb_steps);

    float max_val = max(h);
    float min_val = min(h);

    for(uint i=0; i<nb_steps; ++i)
    {
        float x = float(i) / (nb_steps-1);
        x = x*(max_val - min_val) + min_val;

        h_inv[i] = get_fibre(h, x);
    }
}

void Interpolator::interpolate_linear(const Histogram& h0, const Histogram& h1, Histogram& ht, const float t)
{
    assert(h0.size() == h1.size());

    uint size = h0.size();

    ht.resize(size);

    for(uint i = 0; i<size; ++i)
    {
        ht[i] = (1.0f - t) * h0[i] + t * h1[i];
    }
}

void Interpolator::interpolate_icdf(const Histogram& h0, const Histogram& h1, Histogram& ht, const float t)
{
    // Preliminary checks

    assert(h0.size() == h1.size());

    uint size = h0.size();

    // Cumulated sums computation

    Histogram h0_cumsum;
    cumsum(h0, h0_cumsum);
    Histogram h1_cumsum;
    cumsum(h1, h1_cumsum);

    // Cumulated sums extremal values backup (for further interpolation)

    float min_h0_cumsum = min(h0_cumsum);
    float max_h0_cumsum = max(h0_cumsum);
    float min_h1_cumsum = min(h1_cumsum);
    float max_h1_cumsum = max(h1_cumsum);

    // Cumulated sums normalization

    normalize(h0_cumsum);
    normalize(h1_cumsum);

    uint nb_steps = MT_STEPS*size; // Number of steps used for computing inverse. Note : This has a big influence on the final result precision.

    // Cumulated sums inversion

    Histogram h0_cumsum_invert;
    invert(h0_cumsum, h0_cumsum_invert, nb_steps);
    Histogram h1_cumsum_invert;
    invert(h1_cumsum, h1_cumsum_invert, nb_steps);

    // Inverted cumulated sums linear interpolation
    Histogram ht_cumsum_invert;
    interpolate_linear(h0_cumsum_invert, h1_cumsum_invert, ht_cumsum_invert, t);

    // Interpolated inverse inversion (back to regular representation)

    Histogram ht_cumsum;
    invert(ht_cumsum_invert, ht_cumsum, size+1);
    normalize(ht_cumsum);

    // Resulting interpolated cumulated sum de-normalization

    float ht_min = (1.0-t)*min_h0_cumsum + t*min_h1_cumsum;
    float ht_max = (1.0-t)*max_h0_cumsum + t*max_h1_cumsum;
    denormalize(ht_cumsum, ht_min, ht_max);

    // Result differentiation (i.e. de-accumulation)

    diff(ht_cumsum, ht);
}

void Interpolator::radToHist(AnalysisConfiguration &anconfig, RadialDistribution rad, Histogram & hist)
{
    // assumes bins in RadialDistribution histogram are indexed from 0
    int numbins = rad.m_data.size()+3;
    hist.clear();
    hist.resize(numbins, 0.0f);
    hist[0] = rad.m_less_than_half_shaded_distribution;
    hist[1] = rad.m_more_than_half_shaded_distribution;
    hist[2] = rad.m_fully_shaded_distribution;

    int i = 3;
    for(auto rit: rad.m_data)
    {
        hist[i] = rit.second;
        i++;
    }
}

void Interpolator::histToRad(AnalysisConfiguration &anconfig, Histogram hist, RadialDistribution & rad)
{
    // other params of for rad also need setting

    rad.m_less_than_half_shaded_distribution = hist[0];
    rad.m_more_than_half_shaded_distribution = hist[1];
    rad.m_fully_shaded_distribution = hist[2];
    int i = 3;
    for(int r = anconfig.r_min; r < anconfig.r_max; r += anconfig.r_diff)
    {
        rad.m_data.insert(std::make_pair(r, hist[i]));
        i++;
    }
}

void Interpolator::histPos(Histogram & hist)
{
    for(int i = 0; i < (int) hist.size(); i++)
    {
        if(hist[i] < 0.0f)
        {
            if(hist[i] < -1.0f)
                cerr << "WARNING: highly negative histogram bin" << endl;
            hist[i] = 0.0f;

        }
    }
}

void Interpolator::interp(std::vector<int> global_priority, Distribution &d0, Distribution &d1, Distribution &dout, const float t)
{
    std::vector<int> combined_priority; // combined category priority list

    dout.setEmpty(true);
    dout.getCategories().clear();
    dout.getCorrelations().clear();

    if(!d0.isEmpty())
        dout.setAnalysisConfig(d0.getAnalysisConfig());
    else
        dout.setAnalysisConfig(d1.getAnalysisConfig());

    // cerr << "GLOBAL PRIORITY LIST" << endl;
    // combine categories of d0 and d1 into a single sorted priority list, using global priority to determine order
    for(auto cit: global_priority)
    {
        // cerr << cit << " ";
        if(d0.getCategories().find(cit) != d0.getCategories().end() || d1.getCategories().find(cit) != d1.getCategories().end())
            combined_priority.push_back(cit);
    }
    // cerr << endl;

    // iterate in priority order
    for(std::vector<int>::iterator op = combined_priority.begin(); op != combined_priority.end(); op++)
    {
        std::vector<int>::iterator ip = combined_priority.begin();

        bool fin = false;
        // test correlations in priority order up to and including the current category
        while(!fin)
        {
            bool d0found = false, d1found = false;
            RadialDistribution r0, r1;

            // test for interaction
            std::pair<int, int> key = std::make_pair((* ip), (* op));
            if(d0.getCorrelations().find(key) != d0.getCorrelations().end())
            {
                r0 = d0.getCorrelations().find(key)->second;
                d0found = true;
            }

            if(d1.getCorrelations().find(key) != d1.getCorrelations().end())
            {
                r1 = d1.getCorrelations().find(key)->second;
                d1found = true;
            }

            RadialDistribution outrad;
            if(d0found && d1found)
            {
                // cerr << (*op) << ": " << (* ip) << " both found" << endl;
                // convert pairwise interaction to histogram
                Histogram h0; radToHist(d0.getAnalysisConfig(), r0, h0);
                Histogram h1; radToHist(d1.getAnalysisConfig(), r1, h1);

                // Optimal transport interpolation
                Histogram hout;
                interpolate_icdf(h0, h1, hout, t);
                histPos(hout);

                // convert interpolation back to radial distribution
                histToRad(d0.getAnalysisConfig(), hout, outrad);
                outrad.m_shaded_ratio = (int) (t * (float) r1.m_shaded_ratio + (1.0f-t) * (float) r0.m_shaded_ratio + 0.5f);
                outrad.m_header.reference_id = (* ip);
                outrad.m_header.destination_id = (* op);
                outrad.m_past_rmax_distribution = t * r1.m_past_rmax_distribution + (1.0f-t) * r0.m_past_rmax_distribution; // just interpolate this bin since it is a catch all category
                // ignore requires_optimization field of m_header since it doesn't seem to be used
                // cerr << "r0.pastr = " << r0.m_past_rmax_distribution << ", r1.pastr = " << r1.m_past_rmax_distribution << ", outr.pastr = " << outrad.m_past_rmax_distribution << endl;
                outrad.calculate_min_max();
            }
            else if(d0found)
            {
                // cerr << (*op) << ": " << (* ip) << " only d0 found" << endl;
                outrad = RadialDistribution(r0.m_header, r0.m_shaded_ratio, r0.m_less_than_half_shaded_distribution,
                                   r0.m_more_than_half_shaded_distribution,
                                   r0.m_fully_shaded_distribution,
                                   r0.m_past_rmax_distribution,
                                   r0.m_data);
                outrad.calculate_min_max();
            }
            else if(d1found)
            {
                // cerr << (*op) << ": " << (* ip) << " only d1 found" << endl;
                // outrad = r1;
                outrad = RadialDistribution(r1.m_header, r1.m_shaded_ratio, r1.m_less_than_half_shaded_distribution,
                                   r1.m_more_than_half_shaded_distribution,
                                   r1.m_fully_shaded_distribution,
                                   r1.m_past_rmax_distribution,
                                   r1.m_data);
                outrad.calculate_min_max();
            }

            if(d0found || d1found) // if missing in both then leave as empty
            {
                if(dout.getCorrelations().find(key) != dout.getCorrelations().end())
                {
                    cerr << "Error Interpolator::interp: correlation already exists" << endl;
                    exit(1);
                }
                dout.getCorrelations().insert(std::make_pair(key, outrad));
            }

            fin = (ip == op);
            ip++;
        }

        bool c0found, c1found;

        // interpolate category data here, principally the number of points
        c0found = d0.getCategories().find((*op)) != d0.getCategories().end();
        c1found = d1.getCategories().find((*op)) != d1.getCategories().end();
        int n0 = 0, n1 = 0;
        float hght0 = 0.0f, hght1 = 0.0f, root0 = 0.0f, root1 = 0.0f;
        float hmin0 = 0.0f, hmin1 = 0.0f, hmax0 = 0.0f, hmax1 = 0.0f;
        float havg0 = 0.0f, havg1 = 0.0f, hdev0 = 0.0f, hdev1 = 0.0f;
        CategoryProperties::Histogram hempty;

        if(c0found)
        {
            // create and populate dout category
            dout.getCategories()[(*op)] = d0.getCategories().find((* op))->second;
            for(auto dit: d0.getCategories().find((* op))->second.m_header.category_dependent_ids)
                dout.getCategories().find((*op))->second.m_header.category_dependent_ids.insert(dit);
            n0 = d0.getCategories().find((*op))->second.m_header.n_points;
            hght0 = d0.getCategories().find((*op))->second.m_header.height_to_radius_multiplier;
            root0 = d0.getCategories().find((*op))->second.m_header.height_to_root_size_multiplier;
            hmin0 = (float) d0.getCategories().find((*op))->second.m_header.height_properties.min;
            hmax0 = (float) d0.getCategories().find((*op))->second.m_header.height_properties.max;
            havg0 = d0.getCategories().find((*op))->second.m_header.height_properties.avg;
            hdev0 = d0.getCategories().find((*op))->second.m_header.height_properties.standard_dev;
        }

        if(c1found)
        {

            if(!c0found)
            {
                dout.getCategories()[(*op)] = d1.getCategories().find((* op))->second;
            }
            /*
            else // add any additional dependencies
            {
                for(auto dit: d1.getCategories().find((* op))->second.m_header.category_dependent_ids)
                  dout.getCategories().find((*op))->second.m_header.category_dependent_ids.insert(dit);
            }*/
            for(auto dit: d1.getCategories().find((* op))->second.m_header.category_dependent_ids)
                dout.getCategories().find((*op))->second.m_header.category_dependent_ids.insert(dit);

            n1 =  d1.getCategories().find((*op))->second.m_header.n_points;
            hght1 = d1.getCategories().find((*op))->second.m_header.height_to_radius_multiplier;
            root1 = d1.getCategories().find((*op))->second.m_header.height_to_root_size_multiplier;
            hmin1 = (float) d1.getCategories().find((*op))->second.m_header.height_properties.min;
            hmax1 = (float) d1.getCategories().find((*op))->second.m_header.height_properties.max;
            havg1 = d1.getCategories().find((*op))->second.m_header.height_properties.avg;
            hdev1 = d1.getCategories().find((*op))->second.m_header.height_properties.standard_dev;
        }

        if(c0found || c1found)
        {
            // intepolate number of points and other header properties

            dout.getCategories().find((*op))->second.m_header.n_points = (int) ((t * (float) n1) + (1-t) * (float) n0 + 0.5f);
            dout.getCategories().find((*op))->second.m_header.height_to_radius_multiplier = t * hght1 + (1-t) * hght0;
            dout.getCategories().find((*op))->second.m_header.height_to_root_size_multiplier = t * root1 + (1-t) * root0;
            dout.getCategories().find((*op))->second.m_header.height_properties.min = (int) (t * hmin1 + (1-t) * hmin0 + 0.5f);
            dout.getCategories().find((*op))->second.m_header.height_properties.max = (int) (t * hmax1 + (1-t) * hmax0 + 0.5f);
            dout.getCategories().find((*op))->second.m_header.height_properties.avg = t * havg1 + (1-t) * havg0;
            dout.getCategories().find((*op))->second.m_header.height_properties.standard_dev = t * hdev1 + (1-t) * hdev0;
            dout.setEmpty(false);
        }

        // no change to m_header.priority since this is not used anyway
        // TO DO this might cause problems on save and load?
    }
}
