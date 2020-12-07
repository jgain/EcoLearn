/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


//#define NDEBUG

#include "HistogramDistrib.h"
#include "common.h"
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <fstream>

using namespace std;

HistogramDistrib::HistogramDistrib(int hnumbins, int reservedbins, common_types::decimal min, common_types::decimal max, const std::vector<common_types::decimal>  &bin_data)
    : numbins(hnumbins), reserved(reservedbins), min(min), max(max), refcount(0), normal_factor(1.0f)
{
    if (numbins + reserved != bin_data.size())
    {
        throw invalid_argument("Number of indicated bins does not equal size of bin data");
    }
    bins = bin_data;
    float sum = std::accumulate(bins.begin(), bins.end(), 0.0f);
    if (sum > 1e-3f)
        normalize();
}

HistogramDistrib::HistogramDistrib(int hnumbins, int nreserved_bins, common_types::decimal min, common_types::decimal max)
    : HistogramDistrib(hnumbins, nreserved_bins, min, max, std::vector<common_types::decimal>(hnumbins + nreserved_bins, 0.0f))
{
}

HistogramDistrib::Metadata HistogramDistrib::get_metadata()
{
    Metadata meta;
    meta.binwidth = binwidth;
    meta.maxdist = max;
    meta.nreal_bins = numbins;
    meta.nreserved_bins = reserved;
    meta.ntotal_bins = numbins + reserved;

    return meta;

}

void HistogramDistrib::init(common_types::decimal minb, common_types::decimal maxb, int numb, int skip)
{
    if(maxb >= minb)
    {
        min = minb; max = maxb;
    }
    else
    {
        cerr << "Error HistogramDistrib::init - min exceeds max in range of histogram values" << endl;
    }
    if(numb > 0)
        numbins = numb;
    if(skip >= 0)
        reserved = skip;

    // intialize bins
    bins.clear();
    bins.resize(numbins+reserved, 0.0f);
}

int HistogramDistrib::maptobin(int val) const
{
    int pos = -1;
    if(numbins > 0)
    {
        if(val < 0) // data to be placed in reserved bins
        {
            pos = (val * -1) - 1;
            if(pos >= reserved)
                cerr << "Error HistogramDistrib::maptobin - reserved data out of range" << endl;
        }
        else
        {
            if(val >= min && val < max)
            {
                if(numbins == 1)
                    pos = 0;
                else
                {
                    pos = (int) ((((common_types::decimal) val - min)+pluszero_var) / (max - min) * numbins);
                }
                pos += reserved;
            }
            else
            {
                cerr << "Error HistogramDistrib::maptobin - data " << val << " out of range [" << min << ", " << max << "]" << endl;
            }

        }
    }
    if(pos < 0 || pos >= numbins+reserved)
    {
        throw std::runtime_error("Error HistogramDistrib::maptobin (int) - index out of range");
        //cerr << "Error HistogramDistrib::maptobin - index out of range" << endl;
        //exit(0);
    }

    return pos;
}

int HistogramDistrib::maptobin(common_types::decimal val) const
{
    int pos = -1;
    if(numbins > 0)
    {
        if(val < 0.0f) // data to be placed in reserved bins
        {
            pos = ((int) (val-pluszero_var) * -1) - 1;
            if(pos >= reserved)
                throw std::runtime_error("Error HistogramDistrib::maptobin (common_types::decimal) - index out of range");
                //cerr << "Error HistogramDistrib::maptobin - reserved data out of range" << endl;
        }
        else
        {
            if(val >= min && val <= max + pluszero_var)
            {
                if(numbins == 1)
                    pos = 0;
                else
                    pos = (int) (((val - min)+pluszero_var) / (max - min) * (common_types::decimal) numbins);
                pos += reserved;
            }
            else
            {
                cerr << "Error HistogramDistrib::maptobin - data " << val << " out of range [" << min << ", " << max << "]" << endl;
            }

        }
    }
    if(pos < 0 || pos >= numbins+reserved)
    {
        // if the value is equal to the max or slightly over it, we will get out of range error. This is a way to accommodate this situation (just assign pos = last bin if val is less than 0.1% over the maximum value)
        if (fabs(val - max) < (max - min) / 1000.0f)
        {
            pos = numbins + reserved - 1;
        }
        else
        {
            std::cout << "Distance " << val << " out of range" << std::endl;
            throw std::runtime_error("Error HistogramDistrib::maptobin (common_types::decimal) - index out of range");
        }
        //cerr << "Error HistogramDistrib::maptobin - index out of range" << endl;
        //exit(0);
    }

    return pos;
}

common_types::decimal HistogramDistrib::maptoval(int bin) const
{
    common_types::decimal val = -1.0f;

    if(bin < 0 || bin >= numbins+reserved)
    {
       cerr << "Error HistogramDistrib::maptoval - bin out of range at " << bin << endl;
    }
    else
    {
        // account for reserved histogram bins at the beginning
        if(bin < reserved)
        {
            val = (common_types::decimal) (bin - reserved);
        }
        else
        {
            if(numbins == 1)
                val = min;
            else
                val = min + ((common_types::decimal) (bin - reserved) / (common_types::decimal) numbins * (max - min));
        }
    }
    return val;
}

common_types::decimal HistogramDistrib::maptomaxval(int bin) const
{
    common_types::decimal val = -1.0f;

    if(bin < 0 || bin >= numbins+reserved)
    {
       cerr << "Error HistogramDistrib::maptoval - bin out of range at " << bin << endl;
    }
    else
    {
        // account for reserved histogram bins at the beginning
        if(bin < reserved) // no difference between max and min for reserved bins
        {
            val = (common_types::decimal) (bin - reserved);
        }
        else
        {
            if(numbins == 1)
            {
                val = min;
            }
            else
            {
                val = min + ((common_types::decimal) (bin + 1 - reserved) / (common_types::decimal) numbins * (max - min));
            }
        }
    }
    return val;
}

/*
common_types::decimal HistogramDistrib::probability(common_types::decimal val)
{
    return bins[maptobin(val)];
}*/

void HistogramDistrib::buildbins(std::vector<int> & data)
{
    int pos;

    for(auto d: data)
    {
        pos = maptobin((common_types::decimal) d);
        bins[pos] += 1.0f;
    }
    normalize();
    buildrnddistrib(1000);
    // print();
    validitytest();
}


/*XXX: Ensure that all bins are equal to zero first? */
void HistogramDistrib::buildannularbins(std::vector<common_types::decimal> & data, std::vector<common_types::decimal> & areas)
{
    int pos;

        // XXX: remove
        vector<vector<common_types::decimal> > bindists(bins.size());

    for(auto d: data)
    {
        pos = maptobin(d);
        bins[pos] += 1.0f;

                bindists[pos].push_back(d);
    }


    // check that bins and areas match in size
    if((int) bins.size() != (int) areas.size())
    {
        cerr << "Error HistogramDistrib::buildannularbins - number of bins and number of areas do not match" << endl;
        cerr << "histogram bins = " << (int) bins.size() << " and areas = " << (int) areas.size() << endl;
        cerr << "reported number bins = " << getnumtotbins() << endl;
    }

    for(int b = 0; b < (int) bins.size(); b++)
    {
        if (areas[b] > 0)
            bins[b] = bins[b] / areas[b];
        else
        {
            assert(areas[b] >= -pluszero_var);
            assert(bins[b] >= -pluszero_var && bins[b] <= pluszero_var);
        }
    }

}

void HistogramDistrib::unnormalize()
{
    for (auto &binval : bins)
        binval *= normal_factor;
    nmeth = normalize_method::NONE;
}

void HistogramDistrib::accumulatebins(HistogramDistrib & h)
{

    // histogram parameters must agree
    if(h.getmin() != getmin() || h.getmax() != getmax() || h.getnumreserved() != getnumreserved() || h.getnumtotbins() != getnumtotbins())
    {
        cerr << "Error HistogramDistrib::accumulatebins - parameters of combined and individual bins do not match" << endl;
        cerr << "min: " << getmin() << " to " << h.getmin() << "; max " << getmax() << " to " << h.getmax();
        cerr << "; reserved " << getnumreserved() << " to " << h.getnumreserved();
        cerr << "; tot bins " << getnumtotbins() << " to " << h.getnumtotbins() << endl;
    }


    for(int b = 0; b < getnumtotbins(); b++)
        bins[b] += h.getbin(b);
}

void HistogramDistrib::clean_bins()
{
    for (auto &el : bins)
    {
        if (el <= pluszero_var && el >= -pluszero_var)
            el = 0.0f;
    }
}

void HistogramDistrib::normalize_by_ref()
{
    assert(refcount >= 0);

    if (refcount == 0)
        return;

    for (auto &el : bins)
        el /= refcount;
    nmeth = normalize_method::BYREF;
}

void HistogramDistrib::unnormalize_by_ref()
{
    assert(refcount >= 0);

    if (refcount == 0)
        return;

    for (auto &el : bins)
        el *= refcount;

    nmeth = normalize_method::NONE;
}

void HistogramDistrib::normalize()
{
    common_types::decimal tot = 0.0f;

    // count the number of entries across all bins
    for(auto & el: bins)
    {
        if (el < std::min(pluszero_var * 1000, (common_types::decimal)1e-5f))
        {
            if (fabs(el) > 1e-8f)
                std::cout << "Setting almost zero bin to zero" << std::endl;
            el = 0.0f;
        }
        tot += el;
    }


    // divide through by total bin entries to express bins as a fraction
    if(tot > 0.0f)
    {
        float invtot = 1.0f / tot;
        for(auto & el: bins)
            el = el * invtot;
    }

    common_types::decimal binsum = calc_binsum();
    if (fabs(binsum) > 1e-4 && (binsum < 0.99f || binsum > 1.01f))
    {
        //std::cout << "histogram not normalized: " << binsum << std::endl;
    }

        normal_factor = tot;
        nmeth = COMPLETE;

        /*
        if (inFullAnalyse)
        {
                cerr << "Normalization factor in full analyse: " << normal_factor << endl;
        }
        if (inModifyHistogram)
        {
                cerr << "Normalization factor in optimised analyse: " << normal_factor << endl;
        }
        */
}

void HistogramDistrib::buildrnddistrib(int rndmax, int seed)
{
    common_types::decimal currfreq, sumfreq = 0.0f, prevfreq, bval_min, bval_max, bval = 0.0f;
    int b = 0;

    rnddistrib.clear();
    numrnd = 0;

    float binsum = std::accumulate(bins.begin(), bins.end(), 0.0f);
    if (binsum < pluszero_var)
    {
        // XXX: consider defaulting the distribution to a uniform distrib here?
        return;
    }

    if(rndmax > 0 && getnumtotbins() > 0)
    {
        numrnd = rndmax;
        prevfreq = sumfreq;
        sumfreq = bins[0];

        for(int i = 0; i < numrnd; i++)
        {
            currfreq = (common_types::decimal) i / (common_types::decimal) (numrnd-1);
            while (b != (getnumtotbins()-1) && currfreq > sumfreq)
            {
                b++;
                prevfreq = sumfreq;
                sumfreq += bins[b];
            }
            if (b == getnumtotbins() - 1 && fabs(sumfreq - 1.0f) > 1e-3f)
            {
                throw std::runtime_error("In buildrnddistrib, frequency sum should be very close to 1.0f. Currently it is " + std::to_string(sumfreq));
            }
            else if (b == getnumtotbins() - 1)
            {
                sumfreq = 1.0f;
                if (fabs(bins[b]) < pluszero_var)	// if the last bin is zero, and our rnddistrib vector hasn't been filled already, fill with last value (this is due to floating point imprecision)
                {
                    while (rnddistrib.size() < numrnd)
                    {
                        rnddistrib.push_back(rnddistrib.back());
                    }
                }
            }
            // account for reserved histogram bins at the beginning
            bval_min = (common_types::decimal) maptoval(b);
            bval_max = (common_types::decimal) maptomaxval(b);
            float div = sumfreq - prevfreq;
            float prop;
            if (div > 1e-5)
                prop = (currfreq - prevfreq) / (sumfreq - prevfreq);
            else
                prop = 0.5f;
            if (prop >= 1.0f + pluszero_var)
            {
                char errstr[256];
                sprintf(errstr, "prop in HistogramDistrib::buildrnddistrib above 1: bins[%d]: %f. sumfreq: %f, currfreq: %f", b, bins.at(b), sumfreq, currfreq);
                throw std::runtime_error(errstr);
            }
            bval = bval_min + (bval_max - bval_min) * prop;
            if (bval > bval_max)
            {
                throw std::runtime_error("bval higher than bval max in buildrnddistrib");
            }
            if (bval < 1e-2)
            {
                bval = bval_min + (bval_max - bval_min) * 0.5f;
            }
            if (rnddistrib.size() < numrnd)
            {
                rnddistrib.push_back(bval);
            }
        }

        // intialize
        roller.init(0, numrnd-1, seed);
    }
    else
    {
        cerr << "Error HistogramDistrib::buildrnddistrib - incorrect initial parameters" << endl;
    }

    if (rnddistrib.size() != numrnd)
    {
        throw std::runtime_error("rnddistrib != numrnd in HistogramDistrib::buildrnddistrib");
    }

}

void HistogramDistrib::setRndDistribSeed(int seed)
{
        roller.setSeed(seed);
}

common_types::decimal HistogramDistrib::rndgen()
{
    if(numrnd > 0) // if the rnddistrib array has been initialized
         // generate a random number in the range of the rnddistrib lookup array
        {
                int rand_idx = roller.gen();   /// XXX: remove this. send roller.gen() directly into rnddistrib as an index
                float rndval = rnddistrib.at(rand_idx);
                if (rndval > max)
                {
                    throw std::runtime_error("Maximum sampled value higher than maximum value in HistogramDistrib::rndgen");
                }
                return rndval;
        }
    else
    {
        throw std::runtime_error("rnddistrib array not initialized for HistogramDistrib");
        return 0.0f;
    }
}

common_types::decimal HistogramDistrib::rndgen_binmax()
{
    return rndgen() + (max - min) / (common_types::decimal)numbins;
}

float HistogramDistrib::sumbins() const
{
    float sum = 0.0f;
    for(int b = 0; b < getnumtotbins(); b++)
    {
        float thisbin = getbin(b);
        sum += thisbin;
    }
    return sum;
}

common_types::decimal HistogramDistrib::diff(const HistogramDistrib & cmp) const
{
    bool thiszero = true, cmpzero = true;
    common_types::decimal diffsum = 0.0f, diff;
    // can only compare histograms if their parameters are identical
    if(fabs(getmin() - cmp.getmin()) > 1e-5f || fabs(getmax() - cmp.getmax()) > 1e-5f || fabs(getnumreserved() - cmp.getnumreserved()) > 1e-5f || fabs(getnumbins() != cmp.getnumbins()) > 1e-5f)
    {
        string except_string = "Error HistogramDistrib::diff - comparing histograms with incompatible parameters\n";
        except_string += "min: " + to_string(getmin()) + " " + to_string(cmp.getmin()) + " max: " + to_string(getmax()) + " " + to_string(cmp.getmax()) + "\n";
        except_string += " numreserved: " + to_string(getnumreserved()) + " " + to_string(cmp.getnumreserved()) + " numbins: " + to_string(getnumbins()) + " " + to_string(cmp.getnumbins());
        throw std::runtime_error(except_string);
        //cerr << "Error HistogramDistrib::diff - comparing histograms with incompatible parameters" << endl;
        //cerr << "min: " << getmin() << " " << cmp.getmin() << " max: " << getmax() << " " << cmp.getmax();
        //cerr << " numreserved: " << getnumreserved() << " " << cmp.getnumreserved() << " numbins: " << getnumbins() << " " << cmp.getnumbins() << endl;
    }
    else
    {
        for(int b = 0; b < getnumtotbins(); b++)
        {
            float thisbin = getbin(b);
            float cmpbin = cmp.getbin(b);
            if (abs(thisbin) > 1e-4f)
                thiszero = false;
            if (abs(cmpbin) > 1e-4f)
                cmpzero = false;
            diff = getbin(b) - cmp.getbin(b);
            diffsum += diff * diff;
        }
    }

    return sqrt(diffsum);
    //return diffsum;
}

void HistogramDistrib::print()
{
    int ind = min;
    for(auto el: bins)
    {
        // cerr << ind << ": " << el << endl;
        cerr << el << " ";
        ind++;
    }
    cerr << endl;

    // cerr << "reserved = " << reserved << " min = " << min << " max = " << max << endl;
}

void HistogramDistrib::reset()
{
    numrnd = 0;
    std::fill(bins.begin(), bins.end(), 0.0f);
    rnddistrib.clear();
    normal_factor = 1.0f;
    refcount = 0;
}


bool HistogramDistrib::validitytest()
{
    bool valid = true;
    common_types::decimal val;

    // array sizes agree with store values
    if( ((int) bins.size() != numbins+reserved) || ((int) rnddistrib.size() != numrnd) )
    {
        valid = false;
        cerr << "Error HistogramDistrib::validitytest - internal array sizes incorrect" << endl;
    }

    // all elements in rnddistrib lookup fall into allowable bin ranges
    for(int i = 0; i < (int) rnddistrib.size(); i++)
    {
        val = rnddistrib[i];
        if(val < 0.0f) // possibly in reserved range
        {
            if(val < (common_types::decimal) (-1 * reserved))
            {
                valid = false;
                cerr << "Error HistogramDistrib::validitytest - reserved value in distribution accelerator out of range at index " << i << " with value " << val << endl;
            }
        }
        else if(val < (common_types::decimal) min || val > (common_types::decimal) max)
        {
            valid = false;
            cerr << "Error HistogramDistrib::validitytest - distribution accelerator has out of bin range values at index " << i << " with value " << val;
            cerr << " When max is " << max << " and min is " << min << endl;
        }

    }

    // bins total to unity
    common_types::decimal tot = 0.0f;
    for(auto el: bins)
       tot += el;

    if(tot > 1.0f+pluszero_var || tot < 1.0f-pluszero_var)
    {
        if(tot > 0.0f)
        {
            valid = false;
            cerr << "Error HistogramDistrib::validitytest - probabilities do not sum to unity" << endl;
        }
    }

    // bounds check on distribution accelerator, max with non-negative probability should end array
    int maxval = min;
    int maxbin = 0;
    for(int i = 0; i < reserved; i++)
        if(bins[i] > 0.0f)
        {
            maxval = i - reserved;
            maxbin = i;
        }

    for(int i = reserved; i < reserved+numbins; i++)
        if(bins[i] > 0.0f)
        {
            maxval = maptoval(i);
            maxbin = i;
        }


    return valid;
}

bool HistogramDistrib::unittest()
{
    bool valid = true;
    int i;

    // simple bounds tests on standard histogram
    init(0.0f, 100.0f, 100, 0);
    if(maptobin(99.0f) != 99 || maptoval(99) != 99.0f) // max value maps to last bin
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - incorrect maximal mapping on simple bounds tests" << endl;
    }
    if(maptobin(0.0f) != 0 || maptoval(0) != 0.0f) // min value maps to first bin
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - incorrect minimal mapping on simple bounds tests" << endl;
    }

    // simple bounds tests on reserved histogram
    init(0.0f, 100.0f, 100, 3);
    if(maptobin(99.0f) != 102 || maptoval(102.0f) != 99) // max value maps to last bin
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - incorrect maximal mapping on reserved bounds tests" << endl;
    }
    if(maptobin(0.0f) != 3 || maptoval(3.0f) != 0)
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - incorrect minimal real data mapping on reserved bounds tests" << endl;
    }
    if(maptobin(-1.0f) != 0 || maptoval(0.0f) != -1) // min value maps to first bin
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - incorrect minimal mapping on reserved bounds tests" << endl;
    }

    // simple test with element in every position
    std::vector<int> step;
    for(int i = 100; i < 200; i++)
        step.push_back(i);
    init(100.0f, 200.0f, 100, 0);
    buildbins(step);

    // uniform probability values throughout
    common_types::decimal val = 1.0f / 100.0f;
    i = 0;
    for(auto el: bins)
    {
        if(el > val+pluszero_var || el < val-pluszero_var)
        {
            valid = false;
            cerr << "Error HistogramDistrib::unittest - simple even partition test failed on element " << i << ": " << el << " should be " << val << endl;
        }
        i++;
    }

    // even steps in distribution accelerator
    i = 0;
    for(auto el: rnddistrib)
    {
        if(el != (i / 100) + 100)
        {
            valid = false;
            cerr << "Error HistogramDistrib::unittest - simple even partition accelerator test failed on element " << i << ": " << el << " should be " << i/100+100 << endl;
        }
        i++;
    }

    if(!validitytest())
    {
        valid = false;
        cerr << "Error HistogramDistrib::unittest - validity test fails on simple even partition" << endl;
    }

    return valid;
}

void HistogramDistrib::modify(const vector<int> &prev_bins, const vector<int> &new_bins, const vector<common_types::decimal> &prev_areas, const vector<common_types::decimal> &new_areas)
{
    for (auto &b : bins)
    {
            b *= normal_factor;
    }
    assert(prev_bins.size() == prev_areas.size());
    assert(new_bins.size() == new_areas.size());

    for (int i = 0; i < prev_bins.size(); i++)
    {
        int b = prev_bins[i];
        bins[b] -= 1.0f / prev_areas[i];
    }
    for (int i = 0; i < new_areas.size(); i++)
    {
        int b = new_bins[i];
        bins[b] += 1.0f / new_areas[i];
    }
    clean_bins();
    normalize();
}

void HistogramDistrib::modify(const vector<common_types::decimal> &prev_ds, const vector<common_types::decimal> &new_ds, const vector<common_types::decimal> &prev_areas, const vector<common_types::decimal> &new_areas, int repeat)
{
    for (auto &b : bins)
    {
        assert(!isnan(b));
        assert(b >= -pluszero_var);
    }

    for (auto &b : bins)
    {
            b *= normal_factor;
    }

    common_types::decimal area_sum = 0;

    // then, first remove the effect of the old position
    for (int i = 0; i < prev_ds.size(); i++)
    {
            int bin_idx = maptobin(prev_ds[i]);
            assert(bins[bin_idx] >= pluszero_var);
            bins[bin_idx] -= 1.0f/prev_areas[i];
            assert(bins[bin_idx] >= -pluszero_var);
            area_sum += prev_areas[i];
    }
    area_sum = 0.0f;

    // add the effect of the new position
    for (int i = 0; i < new_ds.size(); i++)
    {
            int bin_idx = maptobin(new_ds[i]);
            bins[bin_idx] += 1.0f/new_areas[i];
            area_sum += new_areas[i];
    }

    for (auto &b : bins)
    {
        assert(b >= -pluszero_var);
        assert(!isnan(b));
    }

    // normalize again
    normalize();
}

common_types::decimal HistogramDistrib::calc_binsum() const
{
    common_types::decimal sum = 0.0f;
    for (auto &b : bins)
    {
        sum += b;
    }
    return sum;
}

bool HistogramDistrib::bins_eqone() const
{
    return fabs(calc_binsum() - 1.0f) < 1e-3;
}

bool HistogramDistrib::bins_eqzero() const
{
    return fabs(calc_binsum()) < 1e-3;
}

common_types::decimal HistogramDistrib::get_binwidth() const
{
    return (max - min) / numbins;
}

float acos_custom(float x) {
    float negate = float(x < 0);
    x = fabs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0-x);
    ret = ret - 2 * negate * ret;
    return negate * 3.14159265358979 + ret;
}

std::vector<float> create_sinlookup()
{
    std::vector<float> lk(360);
    for (int i = 0; i < lk.size(); i++)
    {
        lk[i] = sin(i / 360.0f * 2.0f * M_PI);
    }
    return lk;
}

std::vector<float> HistogramDistrib::sinlookup = create_sinlookup();

float posangle(float x)
{
    x = fmod(x, 2.0f * M_PI);
    if (x < 0.0f)
        x += 2.0f * M_PI;
    return x;
}

int to_deg(float rad)
{
    int val = rad * HistogramDistrib::convmult;
    assert(val >= 0 && val < 360);
    return val;
}

common_types::decimal HistogramDistrib::calcQuadrantArea(common_types::decimal dv, common_types::decimal dh, common_types::decimal r) const
{
    common_types::decimal c, a, d, theta;

    c = 0.25 * M_PI * r * r; // circle area
    a = 0.0f;
    //d = sqrt(dv*dv + dh*dh); // distance to intersection of horizontal and vertical edges
    d = dv*dv + dh*dh; // distance to intersection of horizontal and vertical edges

    common_types::decimal rsq = r * r;

    if(dv > r && dh > r) // normal circle area
    {
        a = c;
    }
    else if(dh <= r && dv > r) // only horizontal edge intersects
    {
        theta = 2.0f * acos(dh/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
        assert(!isnan(theta));
    }
    else if(dv <= r && dh > r) // only vertical edge intersects
    {
        theta = 2.0f * acos(dv/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
        assert(!isnan(theta));
    }
    else if(d > rsq) // both edges intersect disk but their meeting point is outside the disk
    {
        theta = 2.0f * acos(dh/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
        theta = 2.0f * acos(dv/r);
        a = a - (r * r) / 4.0f * (theta - sin(theta));
        assert(!isnan(theta));
    }
    else if(d <= rsq) // both edges intersect disk and meet inside
    {
        a = dv * dh;
    }
    else // catch net - if this is met then there is an error
    {
        cerr << "Error HistogramDistrib::calcQuadrantArea - Not all branches satisfied" << endl;
    }
    return a;
}

common_types::decimal HistogramDistrib::calcDiskArea(float x, float y, common_types::decimal r, float width, float height) const
{
    common_types::decimal dv, dh, q, a = 0.0f;

    // by quadrant

    // quadrant 1
    dv = (common_types::decimal) (width - x);
    dh = (common_types::decimal) y;
    a += calcQuadrantArea(dv, dh, r);
    // cerr << "first quad = " << a << endl;

    // quadrant 2
    dv = (common_types::decimal) x;
    dh = (common_types::decimal) y;
    q = calcQuadrantArea(dv, dh, r);
    a += q;
    // cerr << "second quad = " << q << endl;

    // quadrant 3
    dv = (common_types::decimal) x;
    dh = (common_types::decimal) (height - y);
    q = calcQuadrantArea(dv, dh, r);
    a += q;
    // cerr << "third quad = " << q << endl;

    // quadrant 4
    dv = (common_types::decimal) (width - x);
    dh = (common_types::decimal) (height - y);
    q = calcQuadrantArea(dv, dh, r);
    a += q;
    // cerr << "fourth quad = " << q << endl;

    return a;
}

common_types::decimal HistogramDistrib::calcRingArea(float x, float y, common_types::decimal r1, common_types::decimal r2, float width, float height) const
{
    return calcDiskArea(x, y, r2, width, height) - calcDiskArea(x, y, r1, width, height);
}

common_types::decimal HistogramDistrib::calcRingArea(float x, float y, int binnum, float width, float height) const
{
    if (binnum < getnumreserved())
    {
        return -1.0;
    }
    float outerdist, innerdist;
    outerdist = maptomaxval(binnum);
    innerdist = maptoval(binnum);
    return calcRingArea(x, y, innerdist, outerdist, width, height);
}

common_types::decimal HistogramDistrib::calcReservedArea(float x, float y, float plnt_radius, float width, float height) const
{
    return calcDiskArea(x, y, plnt_radius, width, height);
}

common_types::decimal HistogramDistrib::distCode(float refrad, float otherrad, float sep, float diff) const
{
    float d;
    if(sep < 0.0f) // different reserved cases for overlapping plants
    {
        if(refrad > otherrad) // reference plant has greater radius so more than half intersect not possible
        {
            d = -3.0f-pluszero_var; // counts as less than half inclusion
        }
        else
        {
            if(diff > (common_types::decimal) otherrad) // less than half inclusion
            {
                d = -3.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) (otherrad - refrad)) // more than half inclusion
            {
                d = -2.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) refrad) // total inclusion
            {
                d = -1.0f-pluszero_var;
            }
            else // canopy intersects trunk
            {
                d = -1.0f-pluszero_var;
            }
        }
    }
    else // standard distance histogram bin
    {
        d = sep;
    }
    return d;

}


common_types::decimal HistogramDistrib::distCodeNew(float refrad, float otherrad, float sep, float diff) const
{
    if (sep < 0.0f && sep > -2.0f)
    {
        return -3 - pluszero_var;
    }
    else if (sep <= -2.0f && sep > -4.0f)
    {
        return -2 - pluszero_var;
    }
    else if (sep <= -4.0)
    {
        return -1 - pluszero_var;
    }
    else if (sep >= 0.0f)
        return sep;
}

common_types::decimal HistogramDistrib::distCodeNew(float refx, float refy, float refrad, float otherx, float othery, float otherrad) const
{
    common_types::decimal d, sep, diff;
    common_types::decimal dx, dy;

    dx = (refx - otherx) * (refx - otherx);
    dy = (refy - othery) * (refy - othery);

    diff = sqrt((common_types::decimal) dx + (common_types::decimal) dy);
    sep = diff - (common_types::decimal) refrad;
    sep -= (common_types::decimal) otherrad;

    return distCodeNew(refrad, otherrad, sep, diff);

}

common_types::decimal HistogramDistrib::distCode(float refx, float refy, float refrad, float otherx, float othery, float otherrad) const
{
    common_types::decimal d, sep, diff;
    common_types::decimal dx, dy;

    dx = (refx - otherx) * (refx - otherx);
    dy = (refy - othery) * (refy - othery);

    diff = sqrt((common_types::decimal) dx + (common_types::decimal) dy);
    sep = diff - (common_types::decimal) refrad;
    sep -= (common_types::decimal) otherrad;

    if(sep < 0.0f) // different reserved cases for overlapping plants
    {
        if(refrad > otherrad) // reference plant has greater radius so more than half intersect not possible
        {
            d = -3.0f-pluszero_var; // counts as less than half inclusion
        }
        else
        {
            if(diff > (common_types::decimal) otherrad) // less than half inclusion
            {
                d = -3.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) (otherrad - refrad)) // more than half inclusion
            {
                d = -2.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) refrad) // total inclusion
            {
                d = -1.0f-pluszero_var;
            }
            else // canopy intersects trunk
            {
                d = -1.0f-pluszero_var;
            }
        }
    }
    else // standard distance histogram bin
    {
        d = sep;
    }
    return d;
}

void HistogramDistrib::normalize(normalize_method nmeth)
{
    switch (nmeth)
    {
        case NONE:
            break;
        case BYREF:
            normalize_by_ref();
            break;
        case COMPLETE:
            normalize();
            break;
        default:
            break;
    }
}

void HistogramDistrib::unnormalize(normalize_method nmeth)
{
    if (this->nmeth == normalize_method::NONE)
        return;
    else if (this->nmeth != nmeth)
        nmeth = this->nmeth;

    switch (nmeth)
    {
        case NONE:
            break;
        case BYREF:
            unnormalize_by_ref();
            break;
        case COMPLETE:
            unnormalize();
            break;
        default:
            break;
    }
}

int HistogramDistrib::maptobin_raw(float refrad, float otherrad, float sep, float diff)
{
    return maptobin(distCodeNew(refrad, otherrad, sep, diff));
}


void HistogramDistrib::calc_div_area(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, int &binnum, common_types::decimal &divarea)
{
    float dx = refplnt.x - oplnt.x;
    float dy = refplnt.y - oplnt.y;
    float dsq = dx * dx + dy * dy;
    float dist = sqrt(dsq);
    float adj_dist = dist - refplnt.radius - oplnt.radius;

    if (adj_dist >= 0.0f && adj_dist <= 1e-3f)
    {
        adj_dist = 1e-3f + pluszero_var;
    }
    if (adj_dist + pluszero_var >= max)
    {
        divarea = -1.0f;
        return;
    }
    if (adj_dist > 0.0f && adj_dist < min)
    {
        adj_dist = min + pluszero_var;
    }

    float distcode = distCodeNew(refplnt.radius, oplnt.radius, adj_dist, dist);		// why did I make this an int...?
    binnum = maptobin(distcode);
    assert(!isinf(bins.at(binnum)));
    assert(!isnan(bins.at(binnum)));
    if (adj_dist < 0.0)
    {
        if (refplnt.radius == 0)
            std::cout << "refplnt.radius == 0" << std::endl;
        divarea = calcReservedArea(refplnt.x, refplnt.y, refplnt.radius, width, height);
    }
    else
    {
        divarea = calcRingArea(refplnt.x, refplnt.y, binnum, width, height);
    }
}

bool HistogramDistrib::is_equal(const HistogramDistrib &other) const
{
    if (this->bins.size() != other.bins.size())
        return false;

    for (int i = 0; i < this->bins.size(); i++)
    {
        if (fabs(this->bins[i] - other.bins[i]) > 1e-2f)
        {
            std::cout << "normal factor 1: " << this->normal_factor << std::endl;
            std::cout << "normal factor 2: " << other.normal_factor << std::endl;
            std::cout << "binnum: " << i << std::endl;
            std::cout << this->bins[i] << " != " << other.bins[i] << std::endl;
            return false;
        }
    }
    return true;
}

void HistogramDistrib::setbins(const std::vector<common_types::decimal> &bins)
{
    this->bins = bins;
}

bool HistogramDistrib::add_plantdist(const basic_tree &refplnt, const basic_tree &oplnt,
                                float width, float height, normalize_method nmeth)
{
    common_types::decimal divarea;
    int binnum;
    calc_div_area(refplnt, oplnt, width, height, binnum, divarea);
    if (divarea < 0.0f)
    {
        return false;
    }

    common_types::decimal add = common_types::decimal(1.0) / divarea;

    assert(!isnan(divarea));
    assert(!isinf(add));
    assert(!isnan(add));
    unnormalize(nmeth);
    bins.at(binnum) += add;
    normalize(nmeth);
    common_types::decimal value = bins.at(binnum);
    std::cout << "";
    refcount++;

    return true;
}

bool HistogramDistrib::remove_plantdist(const basic_tree &refplnt, const basic_tree &oplnt, float width, float height, normalize_method nmeth)
{

    common_types::decimal divarea;
    int binnum;
    calc_div_area(refplnt, oplnt, width, height, binnum, divarea);
    if (divarea < 0.0f)
    {
        return false;
    }

    common_types::decimal add = common_types::decimal(1.0) / divarea;
    unnormalize(nmeth);
    bins.at(binnum) -= add;
    if (bins.at(binnum) < 1e-5f) bins.at(binnum) = 0.0f;
    normalize(nmeth);
    common_types::decimal value = bins.at(binnum);
    std::cout << "";
    refcount--;
    if (refcount < 0)
        throw std::runtime_error("refcount cannot be negative");
    return true;
}

bool HistogramDistrib::add_dist(float dist, normalize_method nmeth)
{
    int binnum = maptobin(dist);
    unnormalize(nmeth);
    bins.at(binnum) += 1.0f;
    normalize(nmeth);
    refcount++;
}

void HistogramDistrib::remove_dist(float dist, normalize_method nmeth)
{
    int binnum = maptobin(dist);
    unnormalize(nmeth);
    bins.at(binnum) -= 1.0f;
    normalize(nmeth);
    refcount--;
}

int HistogramDistrib::get_refcount() const
{
    return refcount;
}
