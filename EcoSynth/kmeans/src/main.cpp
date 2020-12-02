#include "kmeans.h"

int main(int argc, char **argv)
{
    kMeans * km = new kMeans();
    
    km->unitTest(5);
    
    // km->search(); // to perform an actual clustering, will need to pass in a feature vector with data to be clustered. The assigns (mapping from each feature to a cluster id) and clusters (cluster centers) and num cluster k will be returned.

    return 1;
}
