/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include "moisture.h"
#include "waterfill.h"

#include <iostream>

using namespace std;

float MoistureSim::slopeImpact(float slope, float slopethresh, float slopemax, float runofflimit)
{
    float revrunoff;

    if(slope < slopethresh) // max absorption for slopes below threshold
        revrunoff = runofflimit;
    else if(slope > slopemax) // complete runoff for slopes above slope max
        revrunoff = 0.0f;
    else // otherwise linearly interpolate
    {
        revrunoff = (1.0f - (slope - slopethresh) / (slopemax - slopethresh)) * runofflimit;
    }
    return revrunoff;
}

void MoistureSim::simSoilCycle(Terrain * ter, MapFloat * terslope, std::vector<float> precipitation, float slopethresh, float slopemax, float evaporation, float runofflimit, float soilsaturation, float riverlevel, std::vector<MapFloat> & wsh)
{
    /* Algorithm:
     * Per pixel:
     *  Init resorvoir to 50% capacity by soil type
     *  Init excess_m to 0
     *  Iterate 2 or 3 times (to achieve equilibrium)
     *      Per month (m)
     *          res_p += (1-w_e) * P_m - w_t
     *          run flow simulation with excess_m
     *          if res_p < w_s
     *              res_p = min(w_s, res_p + flow_p)
     *              if res_p >= w_s
     *                  mark as standing water
     *          else
     *              excess_m+1,m+2,m+3 += 1/3 (res_p - w_s)
     *              res_p = w_s
     *              if flow_p > 0
     *                  mark as standing water
     *          moisture_p = min(w_s, res_p + w_t)
     * where w_e = proportion of evaporation, P_m is precipitation per month, w_s = maximum available water saturation by soil type
     * w_t = monthly transpiration
     */

    std::vector<float> reservoir, transpiration;
    std::vector<std::vector<uint>> excess;
    int dx, dy;
    terslope->getDim(dx, dy);
    cerr << "image size = " << dx << " by " << dy << endl;
    QImage wfimg = QImage(dx, dy, QImage::Format_RGB32);
    QImage fpimg = QImage(dx, dy, QImage::Format_RGB32);
    WaterFill wf, fp; // water network, flood plain
    int i, j, x, y, t, p;
    bool flood;
    float wval, flow, onsoil, insoil, runoff, plain, groundwater, avgexcess, slope, sloperunoff, avgslope, avgrunoff;
    int minmnth[12], maxmnth[12], floodcount, excesscount;

    // calculate transpiration levels to achieve equilibrium for each terrain pixel, because of dependence on slope
    // sum of min runofflimit, monthly non-evaporated precipitation
    transpiration.resize(dy*dy, 0.0f);
    avgrunoff = 0.0f; avgslope = 0.0f;
    cerr << "slopeImpact at " << slopethresh << " = " << slopeImpact(slopethresh, slopethresh, slopemax, runofflimit) << endl;
    cerr << "slopeImpact at " << slopemax << " = " << slopeImpact(slopemax, slopethresh, slopemax, runofflimit) << endl;
    cerr << "slopeImpact at " << 25.0f << " = " << slopeImpact(25.0f, slopethresh, slopemax, runofflimit) << endl;

    cerr << "PRECIPITATION: ";
    for(i = 0; i < 12; i++)
        cerr << precipitation[i] << " ";
    cerr << endl;

    float minslope = 90.0f, maxslope = 0.0f;
    for(p = 0; p < dx * dy; p++)
    {
        x = p%dx;
        y = p/dx;
        slope = terslope->get(x, y);
        if(slope < minslope)
            minslope = slope;
        if(slope > maxslope)
            maxslope = slope;
        sloperunoff = slopeImpact(slope, slopethresh, slopemax, runofflimit);
        avgrunoff += sloperunoff;
        avgslope += slope;
        for(i = 0; i < 12; i++)
            transpiration[p] += min(sloperunoff, (1.0f - evaporation) * precipitation[i]);
    }

    cerr << "average slope = " << avgslope / ((float) dx * dy) << endl;
    cerr << "average runoff = " << avgrunoff / ((float) dx * dy) << endl;


    // water fill parameters
    float riverreach = 50.0f; // weighting for how far water table expands out around rivers
    float slopeweight = 1.0f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to widen
    float flowpowerterm = 0.5f; // power term applied to flow result
/*
    // for alpine
#ifdef ALPINEBIOME
    riverreach = 50.0f; // weighting for how far water table expands out around rivers
    slopeweight = 1.0f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to widen
#endif

    // for savannah
#ifdef SAVANNAHBIOME
    riverreach = 100.0f; // weighting for how far water table expands out around rivers
    slopeweight = 0.9f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to widen
#endif

    // for canyon
#ifdef CANYONBIOME
    riverreach = 50.0f; // weighting for how far water table expands out around rivers
    slopeweight = 1.4f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to wide
#endif

    // for mediterrainean style canyon
#ifdef CANYONWETBIOME
    riverreach = 50.0f; // weighting for how far water table expands out around rivers
    slopeweight = 1.4f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to wide
#endif

    // for med
#ifdef MEDBIOME
    riverreach = 50.0f; // weighting for how far water table expands out around rivers
    slopeweight = 1.0f; // how much impact slope has on river expansion, > 1 to narrow, < 1 to widen
#endif
*/
    // initialise water flow field
    cerr << "waterflow: set terrain" << endl;
    wf.setTerrain(ter);

    // initialise terrain-based moisture values
    reservoir.resize(dx*dy, 0.2 * soilsaturation);
    // reservoir.resize(dx*dy, 0.0f); // no initial water
    for(i = 0; i < 12; i++)
    {
        std::vector<uint> noexcess;
        noexcess.resize(dx*dy, 0);
        excess.push_back(noexcess);
        minmnth[i] = 1000; maxmnth[i] = 0;
    }

    for(t = 0; t < 2; t++) // simulation needs to be run twice to ensure proper settings for early months
    {
        cerr << "iteration " << t << endl;
        floodcount = 0; excesscount = 0; avgexcess = 0.0f;

        for(i = 0; i < 12; i++) // calculate soil moisture per month
        {
            wf.setAbsorbsion(riverlevel);
            wf.reset();

            wf.compute(); // seed flow computation with water seepage values
            wf.compute();

            wf.expandRivers(riverreach, slopeweight); // only flood plain has expanded river influence

            for(p = 0; p < dx * dy; p++)
            {
                onsoil = (1.0f - evaporation) * precipitation[i];

                x = p%dx;
                y = p/dx;

                slope = terslope->get(x, y);
                sloperunoff = slopeImpact(slope, slopethresh, slopemax, runofflimit);
                insoil = min(onsoil, sloperunoff);
                runoff = max(0.0f, onsoil - insoil);
                reservoir[p] += insoil - (transpiration[p] / 12.0f); // monthly rainfall-transpiration cycle
                // reservoir[p] = max(0.0f, reservoir[p]);

                // get flow sim values
                // infiltrate extra water from the flood plain

                flood = wf.isFlowingWater(y, x); // x, y
                if(flood)
                {
                    floodcount++;
                    plain = 0.0f;
                }
                else
                {
                    plain = wf.riverMoisture(y, x); // x, y
                }

                groundwater = reservoir[p];       
                if(plain > 0.5f) // pixel is in the expansion area of the river so increase soil moisture
                {
                    excesscount++;
                    avgexcess += plain;
                    // groundwater = 1000.0f; // for visual feedback on floodplain area
                    groundwater = min(soilsaturation, groundwater + plain * soilsaturation);
                }

                if(runoff > 0.0f) // rainfall exceeds soil capacity so spread exceess over subsequent months
                {
                    for(j = 1; j < 6; j++)
                        excess[(i+j)%12][p] += (int) (runoff / 5.0f * 0.1); // 0.1 scale factor for GC sim convert to metres
                    excess[i][p] = 0;
                }

                // set moisture
                if(flood)
                    wval = 2000.0f; // standing water so set to very high saturation value
                else
                    wval = max(0.0f, groundwater);
                if(t == 1) // store on second run
                {
                    wsh[i].set(x, y, wval);

                    // assign wval
                    if(wval < minmnth[i])
                        minmnth[i] = wval;
                    if(wval > maxmnth[i])
                        maxmnth[i] = wval;

                }
            }
            /*
            if(t == 1)
            {
                wsh.setMin(i+1, minmnth[i]);
                wsh.setMax(i+1, maxmnth[i]);
            }*/
            cerr << "Month " << i << " Finished" << endl;
            cerr << "Flooded proportion = " << (float) floodcount / (float) (dx * dy) << endl;
        }
        cerr << "Avg Flooding Proportion = " << (float) floodcount / (12.0f * (float) (dx * dy)) << endl;
        cerr << "Avg Excess Proportion = " << (float) excesscount / (12.0f * (float) (dx * dy)) << endl;
        cerr << "Avg Excess = " << avgexcess / (float) excesscount << endl;
    }
}

bool MoistureSim::constantValidation(MapFloat * terslope, std::vector<MapFloat> &wsh, int precipitation)
{
    int dx, dy;
    terslope->getDim(dx, dy);

    for(int i = 0; i < 12; i++)
        for(int x = 0; x < dx; x++)
            for(int y = 0; y < dy; y++)
                if( wsh[i].get(x, y) != precipitation)
                {
                    cerr << "MoisuterSim::constantValidation: moisture value " <<  wsh[i].get(x, y) << " at " << x << ", " << y << " instead of " << precipitation << endl;
                    return false;
                }
    return true;
}
