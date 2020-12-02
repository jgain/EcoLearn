//
//  kmeans.h
//  
//
//  Created by James Gain on 2020/01/17.
//

#ifndef kmeans_h
#define kmeans_h

#include <stdio.h>
#include <vector>
#include <array>

using namespace std;

class kMeans
{
private:
    
    const int kmax = 15; // maximum number of seperate clouds per layer
    const float kdifftol = 0.00001f;   // tolerance on average change in kmeans to terminate iteration
    const int kmaxiter = 50;        // maximum number of iterations during kmeans calculation
    const float ktesttol = 0.25f;   // threshold for kmeans unit test
    const int kreps = 5;           // number of times a k-means is randomly intialized
    const int knumseeds = 30;       // number of random seeds for determining intial kmeans start

    /**
     * @brief close   Determine whether any clusters are too close together, in the sense that they no are no longer sufficiently distinct
     * @param k         number of clusters
     * @param clusters  the mean values for each cluster
     * @param tol       tolerance threshold on permissible distance
     * @return true if elements of the cluster are too close, otherwise false
     */
    bool close(int k, std::vector<std::array<float,4>> &clusters, float tol);
    
    /**
     * @brief kdist Evaluate and return the sum of squared differences between elements of a feature array
     * @param f1    first feature
     * @param f2    second feature
     * @return  sum of squares differences between features
     */
    float kdist(std::array<float,4> f1, std::array<float,4> f2);

public:
    
    kMeans(){};
    
    /**
     * @brief cluster    Perform a k-means clustering based on a 4-element feature vector
     * @param features  input elements on which to perform clustering (4-element float arrays)
     * @param k         number of clusters
     * @param assigns   assignment of input elements to clusters
     * @param clusters  the mean values for each cluster
     * @return the average spread of the clusters (i.e., distance of features in a cluster to its mean). A measure of clustering quality
     */
    float cluster(std::vector<std::array<float,4>> &features, int k, std::vector<int> &assigns, std::vector<std::array<float,4>> &clusters);
    
    /**
     * @brief search  Apply cluster repeatedly with increasing k, to find an optimal number of clusters and their assignment to features
     * @param features  input elements on which to perform clustering (4-element float arrays)
     * @param k         number of clusters
     * @param assigns   assignment of input elements to clusters
     * @param clusters  the mean values for each cluster
     */
    void search(std::vector<std::array<float,4>> &features, int &k, std::vector<int> &assigns, std::vector<std::array<float,4>> &clusters);
    
    /**
     * @brief kmeansunittest    Simple unit test of k-means clustering
     * @param numclusters   number of clusters to create during test
     * @return true, if the correct number of clusters and their corresponding centers are found
     */
    bool unitTest(int numclusters);
};

#endif /* kmeans_h */
