// clusters.h: provides bounds on variation in ecosystems based on terrain conditions
// author: James Gain
// date: 19 August 2016

#ifndef _clusters_h
#define _clusters_h

#include "descriptor.h"
#include "typemap.h"
#include "terrain.h"

struct ConditionBounds
{
    SampleDescriptor min;
    SampleDescriptor max;
    SampleDescriptor avg;
};

class Clusters
{
private:
    std::vector<ConditionBounds> climits;   //< bounds by cluster
    std::vector<int> colmapping;    //< mapping table for typemap colours
    TypeMap * clsmap;     //< local cluster lookup map

    /**
     * @brief mapColours    Create a colour mapping table for clusters
     */
    void mapColours();

public:

    Clusters(int dx, int dy){ clsmap = new TypeMap(dx, dy, TypeMapType::CLUSTER); }

    ~Clusters(){ climits.clear(); delete clsmap; }

    /// getter for mapwidth, mapheight, and map values
    int mapwidth(){ return clsmap->width(); }
    int mapheight(){ return clsmap->height(); }
    int get(int x, int y){ return clsmap->get(x, y); }
    int set(int x, int y, int c){ clsmap->set(x, y, c); }
    int getNumCluster(){ return (int) colmapping.size(); }
    TypeMap * getMap(){ return clsmap; }


    /**
     * @brief getClusterCol provide the colour lookup that corresponds to a particular cluster
     * @param cidx  cluster index
     * @return colour index that corresponds to the cluster
     */
    int getClusterCol(int cidx){ return colmapping[cidx]; }

    /**
     * @brief inBounds  Test ecosystem generating parameters against the condition bounds
     * @param cidx  Index of cluster to test against
     * @param sd    Parameters for conditions
     * @return      true if the descriptor is within the min, max bounds, false otherwise
     */
    bool inBounds(int cidx, SampleDescriptor sd);

    /**
     * @brief clipToBounds Force ecoystem generating parameters back into condition bounds of a particular cluster
     * @param cidx  Index of cluster to limit against
     * @param sd    Parameters for conditions
     * @return      true if the descriptor has not been clipped, false otherwise
     */
    bool clipToBounds(int cidx, SampleDescriptor &sd);
    
    /**
     * @brief clipToBounds Force ecoystem generating parameters back into condition bounds of a particular cluster
     * @param cidx  Index of cluster to limit against
     * @param sd    Parameters for conditions
     * @return      true if the descriptor has not been clipped, false otherwise
     */
    bool clipToMapBounds(int x, int y, SampleDescriptor &sd);

    /**
     * @brief getClusterMean    provide the mean sample descriptor for a particular cluster
     * @param cidx  Index of cluster
     * @return      average descriptor for the cluster
     */
    SampleDescriptor & getClusterMean(int cidx){ return climits[cidx].avg; }
    SampleDescriptor & getClusterMean(int x, int y){ return getClusterMean(get(x,y)); }

    /**
     * @brief display Print out the cluster limits
     */
    void display();

    /**
     * @brief read 	Read cluster stats from a text file produced by CubeInSpace
     * @param clsname  name of file containing cluster limits
     * @param mapname  name of file with cluster mapping
     * @return      	true if the read succeeded, false otherwise
     */
    bool read(Terrain * ter, std::string clsname, std::string mapname);
};

#endif
