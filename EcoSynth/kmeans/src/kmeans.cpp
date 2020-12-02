#include "kmeans.h"

// #include <QObject>
// #include <QTimer>
//#include <QVector2D>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
// uncomment to disable assert()
// #define NDEBUG
#include <cassert>

using namespace std;

bool kMeans::close(int k, std::vector<std::array<float,kMeans::ndim>> &clusters, float tol)
{
    //QVector2D w1, w2;
    
    for(int i = 0; i < k; i++)
        for(int j = i+1; j < k; j++)
        {
            if(sqrtf(kdist(clusters[i], clusters[j])) < tol)
                return true;
        }
    return false;
}

void kMeans::search(std::vector<std::array<float,kMeans::ndim>> &features, int &k, std::vector<int> &assigns, std::vector<std::array<float,kMeans::ndim>> &clusters)
{
    bool fin = false;
    float currdiff, bestdiff, globaldiff;
    std::vector<int> bestassigns, currassigns;
    std::vector<std::array<float,kMeans::ndim>> bestclusters, currclusters;
    
    globaldiff = numeric_limits<float>::max();
    
    // test for all k in the allowable range
    for(int c = 1; c < kmax; c++)
    {
        bestdiff = numeric_limits<float>::max();
        // run k-means repeatedly choosing the best result
        for(int r = 0; r < kreps; r++)
        {
            currdiff = cluster(features, c, currassigns, currclusters);
            
            if(currdiff < bestdiff && !close(c, currclusters, 0.06f)) // ignores clusters that are too close together
            {
                bestdiff = currdiff;
                bestassigns = currassigns;
                bestclusters = currclusters;
            }
            
        }
        
        if(bestdiff < globaldiff)
        {
            k = c;
            assigns = bestassigns;
            clusters = bestclusters;
        }
    }
}

float kMeans::kdist(std::array<float,kMeans::ndim> f1, std::array<float,kMeans::ndim> f2)
{
    float diff = 0.0f;
    
    for(int e = 0; e < kMeans::ndim; e++)
        diff += powf(f1[e] - f2[e], 2);
    return diff;
}

float kMeans::cluster(std::vector<std::array<float,kMeans::ndim>> &features, int k, std::vector<int> &assigns, std::vector<std::array<float,kMeans::ndim>> &clusters)
{
    // randomly choose k-elements as means, avoiding repetition
    // Forgy initialization
    std::vector<int> kindices;
    std::default_random_engine generator(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> distribution(std::uniform_int_distribution<int>(0, (int) features.size()-1));
    std::vector<int> prevassigns;
    float diff = 0.0f, bestdist;
    int i = 0;
    bool converged;
    std::vector<int> rndseeds;
    
    // cerr << "feature size = " << (int) features.size() << endl;
    
    clusters.clear();
    rndseeds.assign(knumseeds, -1);
    
    assert((int) features.size() > knumseeds);  // otherwise we cannot draw a unique sufficient seeds
    
    // randomly choose starting cluster
    // uses a variant of kmeans++: choose knumseeds random samples, from these select the one furthest from any of the existing seeds
    for(int c = 0; c < k; c++)
    {
        int rnd, bidx;
        // cerr << "k = " << k << endl;
        
        for(int r = 0; r < knumseeds; r++)
        {
            bool valid = false;
            while(!valid) // repeat draw until we get a number that isn't an existing seed
            {
                rnd = distribution.operator()(generator);
                valid = true;
                for(int e = 0; e < c; e++)
                    if(kindices[e] == rnd)
                        valid = false;
            }
            rndseeds[r] = rnd;
            // cerr << rnd << endl;
        }
        
        // find furthest candidate
        bestdist = 0.0f; bidx = rndseeds[0];
        for(int r = 0; r < knumseeds; r++)
        {
            float shortdist = numeric_limits<float>::max();
            
            // what is the shortest distance to an existing seed?
            for(int i = 0; i < c; i++)
            {
                float dist = sqrtf(kdist(features[rndseeds[r]], clusters[i]));
                if(dist < shortdist)
                    shortdist = dist;
            }
            
            if(shortdist > bestdist)
            {
                bidx = rndseeds[r];
                bestdist = shortdist;
            }
        }
        
        kindices.push_back(bidx);
        clusters.push_back(features[bidx]);
    }
    
    assigns.clear();
    assigns.assign(features.size(), 0);
    prevassigns.assign(features.size(), 0);
    
    // repeat until convergence threshold is reached or a maximum number of iterations
    converged = false;
    while(!converged && i < kmaxiter)
    {
        std::cout << "Iteration " << i << " of " << kmaxiter << std::endl;
        // partition by assigning to closest mean
        for(int f = 0; f < (int) features.size(); f++)
        {
            float bestdist = numeric_limits<float>::max(), currdist;
            int cidx = 0;
            
            for(int c = 0; c < k; c++)
            {
                currdist = kdist(features[f], clusters[c]);
                if(currdist < bestdist)
                {
                    bestdist = currdist;
                    cidx = c;
                }
            }
            assigns[f] = cidx;
        }
        
        // update the new means
        for(int c = 0; c < k; c++)
        {
            int kcount = 0;
            for(int e = 0; e < kMeans::ndim; e++)
                clusters[c][e] = 0.0f;
            for(int f = 0; f < (int) features.size(); f++)
            {
                if(assigns[f] == c) // member of the current cluster so include in centroid
                {
                    for(int e = 0; e < kMeans::ndim; e++)
                        clusters[c][e] += features[f][e];
                    kcount++;
                }
            }
            for(int e = 0; e < kMeans::ndim; e++)
                clusters[c][e] /= (float) kcount;
        }
        
        // check whether assignment of features to clusters is stable
        converged = true;
        for(int f = 0; f < (int) features.size(); f++)
        {
            if(prevassigns[f] != assigns[f])
                converged = false;
            prevassigns[f] = assigns[f];
        }
        
        i++;
    }
    // std::cerr << "converged after " << i << " iterations" << std::endl;
    for(int f = 0; f < features.size(); f++)
        diff += kdist(features[f], clusters[assigns[f]]);
    diff /= (float) features.size();
    // std::cerr << "with diff " << diff << std::endl;
    return diff;
}

bool kMeans::unitTest(int numclusters)
{
    std::default_random_engine generator(std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_int_distribution<int> distribution(std::uniform_int_distribution<int>(0, 100000));
    std::vector<std::array<float,kMeans::ndim>> features, seeds, clusters;
    std::vector<int> assigns;
    float diff = 0.0f;
    int k;
    bool fin;
    
    // randomly generate numclusters seeds, each with a seperation of at least 0.1
    for(int c = 0; c < numclusters; c++)
    {
        int i = 0;
        
        std::array<float,kMeans::ndim> currseed;
        fin = false;
        while(!fin && i < 100)
        {
            // float dist, mindist, rnd;
            
            // generate random seed in [0,2] X 4
            for(int e = 0; e < kMeans::ndim; e++)
                currseed[e] = 2.0f * ((float) distribution.operator()(generator) / 100000.0f);
            
            seeds.push_back(currseed);
            if(close(c, seeds, 0.1f))
                seeds.pop_back();
            else
                fin = true;
            i++;
        }
        if(i == 100)
        {
            std::cerr << "Error kMeansUnitTest: random seed assignment failed for seed " << c << std::endl;
            return false;
        }
    }
    
    // create a feature vector with points randomly offset from the initial seeds
    for(int c = 0; c < numclusters; c++)
        for(int f = 0; f < 100; f++) // number set sufficiencly high that random distribution will tend to the correct mean
        {
            std::array<float, kMeans::ndim> rndf = seeds[c];
            for(int e = 0; e < kMeans::ndim; e++)
            {
                // generate random number in range [-0.01, 0.01]
                float rndnum = 0.02f * ((float) distribution.operator()(generator) / 100000.0f) - 0.01f;
                rndf[e] = seeds[c][e] + rndnum;
            }
            features.push_back(rndf);
        }
    
    // run kmeans
    search(features, k, assigns, clusters);
    
    // print out seeds
    std::cerr << "SEEDS (#" << numclusters << ")" << std::endl;
    for(int c = 0; c < numclusters; c++)
        cerr << seeds[c][0] << " " << seeds[c][1] << " " << seeds[c][2] << " " << seeds[c][3] << std::endl;
    
    // print out clusters
    std::cerr << std::endl << "CLUSTERS (#" << k << ")" << std::endl;
    
    for(int c = 0; c < k; c++)
        std::cerr << clusters[c][0] << " " << clusters[c][1] << " " << clusters[c][2] << " " << clusters[c][3] << std::endl;
    
    std::cerr << std::endl;
    
    /*
     for(int i = 0; i < k; i++)
     for(int j = i+1; j < k; j++)
     {
     std::cerr << "dist c" << i << " to c" << j << " = " << sqrtf(kdist(clusters[i], clusters[j])) << std::endl;
     }
     */
    
    // test difference between cluster means and seeds
    if(k != numclusters)
    {
        std::cerr << "Error kMeansUnitTest: number of found clusters " << k << " != actual clusters " << numclusters << std::endl;
        return false;
    }
    else // test difference between seeds and clusters
    {
        int idx;
        float cdiff, diff, bestdiff;
        
        std::vector<bool> picked;
        picked.assign(k, false);
        
        // seeds and clusters may be in any order, so find closest match by distance
        cdiff = 0.0f;
        
        for(int c = 0; c < k; c++)
        {
            bestdiff = numeric_limits<float>::max(); idx = -1;
            for(int s = 0; s < k; s++)
                if(!picked[s])
                {
                    diff = sqrtf(kdist(seeds[s], clusters[c]));
                    if(diff < bestdiff)
                    {
                        bestdiff = diff;
                        idx = s;
                    }
                }
            if(idx == -1)
                std::cerr << "kMeansUnitTest: error in seed to cluster matching" << std::endl;
            picked[idx] = true;
            cdiff += sqrtf(kdist(seeds[idx], clusters[c]));
        }
        
        cdiff = cdiff / (float) k;
        std::cerr << "kMeansUnitTest: average distance from true cluster centers = " << cdiff << std::endl;
        if(cdiff > ktesttol)
        {
            std::cerr << "Error kMeansUnitTest: average dist " << cdiff << " above threshold of " << ktesttol << std::endl;
            return false;
        }
    }
    
    return true;
}
