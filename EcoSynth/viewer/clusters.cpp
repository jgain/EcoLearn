#include "clusters.h"
#include "grass.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <list>

using namespace std;

void Clusters::mapColours()
{
    int maxval, maxind;

    colmapping.resize((int) climits.size(), 0);

    // put all cluster indices in a list
    std::list<int> indices;
    for(int i = 0; i < (int) climits.size(); i++)
        indices.push_back(i);


    // cluster 0 always maps to water
    colmapping[0] = 1;
    indices.remove(0);
    /*
    maxval = 0;
    for(auto ind: indices)
    {
        if(climits[ind].avg.moisture[0] > maxval)
        {
            maxval = climits[ind].avg.moisture[0];
            maxind = ind;
        }
    }
    if(maxval > 1000)
    {
        colmapping[maxind] = 1;
        indices.remove(maxind);
    }*/

    // A moisture rich cluster with avg moisture exceeding 200
    maxval = 0;
    for(auto ind: indices)
    {
        if(climits[ind].avg.moisture[0] > maxval)
        {
            maxval = climits[ind].avg.moisture[0];
            maxind = ind;
        }
    }
    if(maxval > 200)
    {
        colmapping[maxind] = 2;
        indices.remove(maxind);
    }

    // The cluster with the steepest average slope
    maxval = 0;
    for(auto ind: indices)
    {
        if(climits[ind].avg.slope > maxval)
        {
            maxval = climits[ind].avg.slope;
            maxind = ind;
        }
    }
    if(maxval > 0)
    {
        colmapping[maxind] = 3;
        indices.remove(maxind);
    }

    // The cluster with the highest average sunlight
    maxval = 0;
    for(auto ind: indices)
    {
        if(climits[ind].avg.sunlight[0] > maxval)
        {
            maxval = climits[ind].avg.sunlight[0];
            maxind = ind;
        }
    }
    if(maxval > 0)
    {
        colmapping[maxind] = 4;
        indices.remove(maxind);
    }

    // The cluster with the lowest average sunlight
    maxval = 20;
    for(auto ind: indices)
    {
        if(climits[ind].avg.sunlight[0] < maxval)
        {
            maxval = climits[ind].avg.sunlight[0];
            maxind = ind;
        }
    }
    if(maxval < 5)
    {
        colmapping[maxind] = 5;
        indices.remove(maxind);
    }

    // The remaining cluster with the highest temperature
    maxval = 0;
    for(auto ind: indices)
    {
        if(climits[ind].avg.temperature[1] > maxval)
        {
            maxval = climits[ind].avg.temperature[1];
            maxind = ind;
        }
    }
    if(maxval > 10)
    {
        colmapping[maxind] = 6;
        indices.remove(maxind);
    }

    /*
    // alpine colour assignment for X clusters
    colmapping[0] = ;
    colmapping[1] = ;
    colmapping[2] = ;
    etc
    */

    // assign all other clusters with sequential values
    int i = 7;
    for(auto ind: indices)
    {
        colmapping[ind] = i;
        i++;
    }

    // print out mapping table
    cerr << "Colour Mapping Table" << endl;
    for(auto col: colmapping)
        cerr << col << " ";
    cerr << endl;

}

bool Clusters::inBounds(int cidx, SampleDescriptor sd)
{
    if(sd.slope < climits[cidx].min.slope || sd.slope > climits[cidx].max.slope)
        return false;
    if(sd.moisture[0] < climits[cidx].min.moisture[0] || sd.moisture[0] > climits[cidx].max.moisture[0])
        return false;
    if(sd.moisture[1] < climits[cidx].min.moisture[1] || sd.moisture[1] > climits[cidx].max.moisture[1])
        return false;
    if(sd.temperature[0] < climits[cidx].min.temperature[0] || sd.temperature[0] > climits[cidx].max.temperature[0])
        return false;
    if(sd.temperature[1] < climits[cidx].min.temperature[1] || sd.temperature[1] > climits[cidx].max.temperature[1])
        return false;
    if(sd.sunlight[0] < climits[cidx].min.sunlight[0] || sd.sunlight[0] > climits[cidx].max.sunlight[0])
        return false;
    if(sd.sunlight[1] < climits[cidx].min.sunlight[1] || sd.sunlight[1] > climits[cidx].max.sunlight[1])
        return false;

    // ignore age

    return true;
}

bool Clusters::clipToBounds(int cidx, SampleDescriptor &sd)
{
    if(inBounds(cidx, sd))
        return true;
    else
    {
        // clip
        if(sd.slope < climits[cidx].min.slope)
            sd.slope = climits[cidx].min.slope;
        if(sd.slope > climits[cidx].max.slope)
            sd.slope = climits[cidx].max.slope;

        // TO DO: moisture can be more permissive due to local variation
        if(sd.moisture[0] < climits[cidx].min.moisture[0])
            sd.moisture[0] = climits[cidx].min.moisture[0];
        if(sd.moisture[0] > climits[cidx].max.moisture[0])
            sd.moisture[0] = climits[cidx].max.moisture[0];
        if(sd.moisture[1] < climits[cidx].min.moisture[1])
            sd.moisture[1] = climits[cidx].min.moisture[1];
        if(sd.moisture[1] > climits[cidx].max.moisture[1])
            sd.moisture[1] = climits[cidx].max.moisture[1];

        if(sd.temperature[0] < climits[cidx].min.temperature[0])
            sd.temperature[0] = climits[cidx].min.temperature[0];
        if(sd.temperature[0] > climits[cidx].max.temperature[0])
            sd.temperature[0] = climits[cidx].max.temperature[0];
        if(sd.temperature[1] < climits[cidx].min.temperature[1])
            sd.temperature[1] = climits[cidx].min.temperature[1];
        if(sd.temperature[1] > climits[cidx].max.temperature[1])
            sd.temperature[1] = climits[cidx].max.temperature[1];

        if(sd.sunlight[0] < climits[cidx].min.sunlight[0])
            sd.sunlight[0] = climits[cidx].min.sunlight[0];
        if(sd.sunlight[0] > climits[cidx].max.sunlight[0])
            sd.sunlight[0] = climits[cidx].max.sunlight[0];
        if(sd.sunlight[1] < climits[cidx].min.sunlight[1])
            sd.sunlight[1] = climits[cidx].min.sunlight[1];
        if(sd.sunlight[1] > climits[cidx].max.sunlight[1])
            sd.sunlight[1] = climits[cidx].max.sunlight[1];
    }
    return false;
}

bool Clusters::clipToMapBounds(int x, int y, SampleDescriptor &sd)
{
    int cidx = clsmap->get(x, y); // lookup cluster index
    clipToBounds(cidx, sd);
}

void Clusters::display()
{
    int i = 0;

    for(auto cit: climits)
    {
        cerr << "Cluster: " << i << endl;
        cerr << "Slope: min " << cit.min.slope << ", max " << cit.max.slope << ", avg " << cit.avg.slope << endl;
        cerr << "June Temp: min " << cit.min.temperature[1] << ", max " << cit.max.temperature[1] << ", avg " << cit.avg.temperature[1] << endl;
        cerr << "Dec Temp: min " << cit.min.temperature[0] << ", max " << cit.max.temperature[0] << ", avg " << cit.avg.temperature[0] << endl;
        cerr << "June Moisture: min " << cit.min.moisture[1] << ", max " << cit.max.moisture[1] << ", avg " << cit.avg.moisture[1] << endl;
        cerr << "Dec Moisture: min " << cit.min.moisture[0] << ", max " << cit.max.moisture[0] << ", avg " << cit.avg.moisture[0] << endl;
        cerr << "June Illumination: min " << cit.min.sunlight[1] << ", max " << cit.max.sunlight[1] << ", avg " << cit.avg.sunlight[1] << endl;
        cerr << "Dec Illumination: min " << cit.min.sunlight[0] << ", max " << cit.max.sunlight[0] << ", avg " << cit.avg.sunlight[0] << endl;
        cerr << endl;
        i++;
    }
}




bool Clusters::read(Terrain * ter, std::string clsname, std::string mapname)
{
    ifstream infile;
    int cidx;

    // cluster 0 for open water
    ConditionBounds cw;
    cw.min.moisture[0] = 2000; cw.max.moisture[0] = 2000; cw.avg.moisture[0] = 2000;
    cw.min.moisture[1] = 2000; cw.max.moisture[1] = 2000; cw.avg.moisture[1] = 2000;
    climits.push_back(cw);

    infile.open((char *) clsname.c_str(), ios_base::in);
    if(infile.is_open())
    {
        int numclusters;
        infile >> numclusters;
        for(int i = 0; i < numclusters; i++)
        {
            ConditionBounds cb;
            infile >> cidx;
            if(i != cidx)
            {
                cerr << "Error ClusterLimits::read: corrupted file " << clsname << endl;
                infile.close();
                return false;
            }
            infile >> cb.min.temperature[1] >> cb.max.temperature[1] >> cb.avg.temperature[1];
            infile >> cb.min.temperature[0] >> cb.max.temperature[0] >> cb.avg.temperature[0];
            infile >> cb.min.moisture[1] >> cb.max.moisture[1] >> cb.avg.moisture[1];
            infile >> cb.min.moisture[0] >> cb.max.moisture[0] >> cb.avg.moisture[0];
            infile >> cb.min.sunlight[1] >> cb.max.sunlight[1] >> cb.avg.sunlight[1];
            infile >> cb.min.sunlight[0] >> cb.max.sunlight[0] >> cb.avg.sunlight[0];
            infile >> cb.min.slope >> cb.max.slope >> cb.avg.slope;

            cb.min.age = 0; cb.max.age = 1000; cb.avg.age = 300; // even though these values should not be used
            /*
            if(i == 3)
            {
                cb.avg.slope = 0.0f;
                cb.avg.temperature[0] = 12;
                cb.avg.temperature[1] = 28;
                cb.avg.moisture[0] = 119;
                cb.avg.moisture[1] = 160;
                cb.avg.sunlight[0] = 6;
                cb.avg.sunlight[1] = 14;
                cb.avg.age = 100;
            }*/
            climits.push_back(cb);
        }
        mapColours();
        infile.close();
            
        // now also read map
        cerr << "mapname = " << mapname << endl;
        // set dimensions to match terrain
        int dx, dy;
        ter->getGridDim(dx, dy);
        clsmap->matchDim(dx, dy);
        clsmap->load(mapname, TypeMapType::CLUSTER);
        return true;
    }
    else
        return false;
}
