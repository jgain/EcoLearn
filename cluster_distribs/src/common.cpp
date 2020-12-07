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


#include "common.h"
//#include "MinimalPlant.h"
#include <common/basic_types.h>
#include "HistogramDistrib.h"
#include "generic_rng.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cmath>

using namespace common_types;

namespace common_vars
{
    generic_rng unif_rng;
    unsigned float_round, double_round;
}

bool common_funcs::read_pdb(std::string filename, std::map<int, std::vector<MinimalPlant> > &retvec)
{
    //std::vector< std::vector<basic_plant> > retvec;
    std::ifstream infile;
    int numcat, skip;

    infile.open(filename, std::ios_base::in);
    if(infile.is_open())
    {
        // list of prioritized categories, not all of which are used in a particular sandbox
        infile >> numcat;
        //retvec.resize(numcat);
        for(int c = 0; c < numcat; c++)
        {
            common_types::decimal junk;
            int cat;
            int nplants;

            infile >> cat;
            for (int i = 0; i < 3; i++)
                infile >> junk;	// skip minheight, maxheight, and avgCanopyRadToHeightRatio

            infile >> nplants;
            retvec[cat].resize(nplants);
            for (int plnt_idx = 0; plnt_idx < nplants; plnt_idx++)
            {
                common_types::decimal x, y, z, radius, height;
                infile >> x >> y >> z;
                infile >> height;
                infile >> radius;
                MinimalPlant plnt = {(int)x, (int)y, (int)height, (int)radius, false};
                retvec[cat][plnt_idx] = plnt;
                /*
                int height_bin = get_height_bin(height, height_bins);
                if (height_bin < 0)
                    continue;
                else if (height_bin == height_bins.size() - 1)
                {
                    std::cerr << "Warning: height is higher than supposed maximum height. Assigning to maximum height bin" << std::endl;
                }
                */
            }
        }
        std::cerr << std::endl;


        infile.close();
        return true;
    }
    else
        return false;
}

bool common_funcs::write_pdb(std::string filename, std::vector< std::vector<MinimalPlant> > &plants)
{
    std::ofstream outfile;
    int skip;
    auto calc_plant_stats = [](std::vector<MinimalPlant> &plnts, common_types::decimal &min_height, common_types::decimal &max_height, common_types::decimal &avg_canopy_height_ratio){
        min_height = std::numeric_limits<common_types::decimal>::max();
        max_height = -std::numeric_limits<common_types::decimal>::max();
        common_types::decimal canopy_sum = 0.0f;
        common_types::decimal height_sum = 0.0f;
        for (auto &plnt : plnts)
        {
            if (plnt.h < min_height)
                min_height = plnt.h;
            if (plnt.h > max_height)
                max_height = plnt.h;
            height_sum += plnt.h;
            canopy_sum += plnt.r;
        }
        avg_canopy_height_ratio = canopy_sum / height_sum;
    };

    outfile.open(filename, std::ofstream::out | std::ofstream::trunc);
    if(outfile.is_open())
    {
        // list of prioritized categories, not all of which are used in a particular sandbox
        outfile << (int) plants.size() << std::endl;

        // write out individual plants
        for (int plnttype_idx = 0; plnttype_idx < plants.size(); plnttype_idx++)
        {
            int pid = plnttype_idx;
            common_types::decimal min_height, max_height, avg_canopy_height_ratio;
            calc_plant_stats(plants[plnttype_idx], min_height, max_height, avg_canopy_height_ratio);
            int nplants = plants[plnttype_idx].size();
            outfile << pid << " " << min_height << " " << max_height << " " << avg_canopy_height_ratio << std::endl;
            outfile << nplants << std::endl;
            for(int p = 0; p < nplants; p++)
            {
                int x, y, rad, z = -1, height;
                x = plants[plnttype_idx][p].x;
                y = plants[plnttype_idx][p].y;
                rad = plants[plnttype_idx][p].r;
                height = plants[plnttype_idx][p].h;
                //outfile << pid << " " << x << " " << y << " " << rad << endl;
                outfile << x << " " << y << " " << z << " " << height << " " << rad << std::endl;
            }
        }

        outfile.close();
        return true;
    }
    else
        return false;

}


void common_funcs::categorise_to_heights(std::vector<MinimalPlant> &plants, const std::vector<common_types::decimal> &height_bins,
                                                                              std::map<int, std::vector<MinimalPlant> > &plant_map,
                                                                              std::map<int, std::pair<int, int> > &class_height_map)
{
    for (int i = 1; i < height_bins.size(); i++)
    {
        class_height_map[height_bins.size() - 1 - i] = {height_bins[i - 1], height_bins[i]};
    }
    for (auto &plnt : plants)
    {
        int cat = -1;
        for (int hi = 1; hi < height_bins.size(); hi++)
        {
            if (plnt.h <= height_bins[hi])
            {
                cat = (height_bins.size() - 1) - hi;
                break;
            }
        }
        if (cat < 0)
        {
            continue;
        }
        plant_map[cat].push_back(plnt);
    }
}

std::vector<int> common_funcs::get_surr_idxes(int idx, int w, int h, int extent_x, int extent_y, bool include_self)
{
    return get_surr_idxes(idx % w, idx / w, w, h, extent_x, extent_y, include_self);
}

std::vector<int> common_funcs::get_surr_idxes(int x, int y, int w, int h, int extent_x, int extent_y, bool include_self)
{
    std::vector<int> idxes;
    for (int cy = y - extent_y; cy <= y + extent_y; cy++)
    {
        if (cy >= 0 && cy < h)
        {
            for (int cx = x - extent_x; cx <= x + extent_x; cx++)
            {
                if (cx >= 0 && cx < w)
                    if (include_self || !include_self && (cx != x || cy != y))
                        idxes.push_back(cy * w + cx);
            }
        }
    }
    return idxes;
}

float common_funcs::get_height_from_radius(float radius, common_types::decimal a, common_types::decimal b)
{
    return exp((log(radius) - a) / b);
}

float common_funcs::get_radius_from_height(float h, common_types::decimal a, common_types::decimal b)
{
    return exp(a + b * log(h));
}

common_types::decimal common_funcs::get_radius_from_height_convert(common_types::decimal height, common_types::decimal a, common_types::decimal b)
{
    common_types::decimal m_per_foot = 0.3048;
    common_types::decimal radius = common_funcs::get_radius_from_height(height, a, b);
    return radius / (m_per_foot * 3);
}

bool common_funcs::is_this_higher_priority(int thistype, int othertype)
{
    return thistype < othertype;
}

common_types::decimal common_funcs::calc_ring_area(MinimalPlant &refplnt, MinimalPlant &cmp, HistogramDistrib &hst, const rectangle &valid_area)
{
    bool xsect;
    common_types::decimal radius_sum = refplnt.r + cmp.r;
    common_types::decimal distance = common_funcs::dist_code(refplnt, cmp, xsect);
    if (radius_sum > distance)
    {
        return calc_disk_area(refplnt, refplnt.r, valid_area);
    }
    else
    {
        distance -= radius_sum;
        if (distance > hst.getmax())	// XXX: check that getmax does not return the start of the last bin
            return 0.0f;
        int bin = hst.maptobin(distance);
        return common_funcs::calc_ring_area(refplnt, bin, hst, valid_area);
        /*
        common_types::decimal high_bdist = hst.maptomaxval(bin);
        common_types::decimal low_bdist = hst.maptoval(bin);

        return calc_disk_area(refplnt, high_bdist, valid_area) - calc_disk_area(refplnt, low_bdist, valid_area);
        */
    }
}

common_types::decimal common_funcs::calc_ring_area(MinimalPlant &refplnt, int bin, HistogramDistrib &hst, const rectangle &valid_area)
{
    if (bin < hst.getnumreserved())
    {
        return calc_disk_area(refplnt, refplnt.r, valid_area);
    }

    common_types::decimal high_bdist = refplnt.r + hst.maptomaxval(bin);
    common_types::decimal low_bdist = refplnt.r + hst.maptoval(bin);

    return calc_disk_area(refplnt, high_bdist, valid_area)
            - calc_disk_area(refplnt, low_bdist, valid_area);
}

common_types::decimal common_funcs::calc_disk_area(const MinimalPlant & refplnt, common_types::decimal r, const rectangle &area)
{
    common_types::decimal dv, dh, q, a = 0.0f;

    // by quadrant

    // quadrant 1
    dv = (common_types::decimal) (area.xmax - refplnt.x);
    dh = (common_types::decimal) refplnt.y;
    a += calc_quadrant_area(dv, dh, r);
    // cerr << "first quad = " << a << endl;

    // quadrant 2
    dv = (common_types::decimal) refplnt.x;
    dh = (common_types::decimal) refplnt.y;
    q = calc_quadrant_area(dv, dh, r);
    a += q;
    // cerr << "second quad = " << q << endl;

    // quadrant 3
    dv = (common_types::decimal) refplnt.x;
    dh = (common_types::decimal) (area.ymax - refplnt.y);
    q = calc_quadrant_area(dv, dh, r);
    a += q;
    // cerr << "third quad = " << q << endl;

    // quadrant 4
    dv = (common_types::decimal) (area.xmax - refplnt.x);
    dh = (common_types::decimal) (area.ymax - refplnt.y);
    q = calc_quadrant_area(dv, dh, r);
    a += q;
    // cerr << "fourth quad = " << q << endl;

    return a;
}

common_types::decimal common_funcs::calc_quadrant_area(common_types::decimal dv, common_types::decimal dh, common_types::decimal r)
{
    common_types::decimal c, a, d, theta;

    c = 0.25 * M_PI * r * r; // circle area
    a = 0.0f;
    d = sqrt(dv*dv + dh*dh); // distance to intersection of horizontal and vertical edges

    if(dv > r && dh > r) // normal circle area
    {
        a = c;
    }
    else if(dh <= r && dv > r) // only horizontal edge intersects
    {
        theta = 2.0f * acos(dh/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
    }
    else if(dv <= r && dh > r) // only vertical edge intersects
    {
        theta = 2.0f * acos(dv/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
    }
    else if(d > r) // both edges intersect disk but their meeting point is outside the disk
    {
        theta = 2.0f * acos(dh/r);
        a = c - (r * r) / 4.0f * (theta - sin(theta));
        theta = 2.0f * acos(dv/r);
        a = a - (r * r) / 4.0f * (theta - sin(theta));
    }
    else if(d <= r) // both edges intersect disk and meet inside
    {
        a = dv * dh;
    }
    else // catch net - if this is met then there is an error
    {
        std::cerr << "Error common_funcs::calcQuadrantArea - Not all branches satisfied" << std::endl;
    }
    return a;
}

common_types::decimal common_funcs::dist_code(MinimalPlant & refplnt, MinimalPlant & cmpplnt, bool &xsect)
{
    common_types::decimal d, sep, diff;
    int dx, dy;

    xsect = false;
    dx = (refplnt.x - cmpplnt.x) * (refplnt.x - cmpplnt.x);
    dy = (refplnt.y - cmpplnt.y) * (refplnt.y - cmpplnt.y);

    diff = sqrt((common_types::decimal) dx + (common_types::decimal) dy);
    sep = diff - (common_types::decimal) refplnt.r;
    sep -= (common_types::decimal) cmpplnt.r;

    if(sep < 0.0f) // different reserved cases for overlapping plants
    {
        if(refplnt.r > cmpplnt.r) // reference plant has greater radius so more than half intersect not possible
        {
            d = -3.0f-pluszero_var; // counts as less than half inclusion
        }
        else
        {
            if(diff > (common_types::decimal) cmpplnt.r) // less than half inclusion
            {
                d = -3.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) (cmpplnt.r - refplnt.r)) // more than half inclusion
            {
                d = -2.0f-pluszero_var;
            }
            else if(diff > (common_types::decimal) refplnt.r) // total inclusion
            {
                d = -1.0f-pluszero_var;
            }
            else // canopy intersects trunk
            {
                // cerr << "Error PlantTypeDistrib::extractSingletonHistogram - canopy and trunk intersect" << endl;
                d = -1.0f-pluszero_var;
                xsect = true;
            }
        }
    }
    else // standard distance histogram bin
    {
        d = sep;
    }
    return d;
}

bool common_funcs::files_are_equal(std::string filename1, std::string filename2, bool show_unequal, bool verbose)
{
    std::ifstream instream1(filename1), instream2(filename2);

    while (instream1.good() && instream2.good())
    {
        std::string orig_str, copy_str;
        std::getline(instream1, orig_str);
        std::getline(instream2, copy_str);
        if (orig_str != copy_str)
        {
            if (show_unequal)
            {
                std::cout << "Files not equal. The following lines were not equal: " << std::endl;
                std::cout << orig_str << "   !=    " << copy_str << std::endl;
                std::cout << "Subsequent lines may also not be equal" << std::endl;
            }
            return false;
        }
        if (verbose)
        {
            std::cout << orig_str << std::endl;
        }
    }

    if (instream1.good() || instream2.good())
    {
        if (show_unequal)
            std::cout << "Files not equal. One file was smaller than the other" << std::endl;
        return false;
    }
    return true;
}

