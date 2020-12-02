// synth.h: generate plant distributions interactively. Recoded from Harry Longs original generator.
// author: James Gain
// date: 1 August 2016

#ifndef _synth_h
#define _synth_h

//#define VALIDATE

#include "view.h"
#include "terrain.h"

#include <radialDistribution/reproducer/reproduction_configuration.h>
#include <radialDistribution/reproducer/radial_distribution.h>
#include <radialDistribution/reproducer/point_spatial_hashmap.h>
#include <radialDistribution/reproducer/category_properties.h>
#include <radialDistribution/reproducer/dice_roller.h>
#include <radialDistribution/reproducer/point_map.h>
#include <radialDistribution/analyser/analysis_configuration.h>

class PlantGrid;
class ConditionsMap;

struct AggregateSizeProperties
{
    float totalHeight;
    float heightToCanopyRadius;
    int n;
};

struct SubSynthArea
{
    int startx, starty, extentx, extenty;
};

typedef std::map<int, AggregateSizeProperties> CategorySizes;
typedef std::map<int,std::vector<AnalysisPoint> > GeneratedPoints;


const int placethresh = 1000;
// const int movesthresh = 2;
const float jumpchance = 0.5f;

class MultiDistributionReproducer
{
public:

    struct SizeProperties
    {
        int minHeight;
        int maxHeight;
        float heightToCanopyRadius;
    };

    typedef std::map<std::pair<int,int>, RadialDistribution> PairCorrelations;
    typedef std::map<int,CategoryProperties> CategoryPropertiesContainer;
    typedef std::map<int, SizeProperties> GeneratedPointsProperties;


    ~MultiDistributionReproducer(){}

    // void resetTiming(){ inittime = 0.0f; movetime = 0.0f; rndcount = 0; t1time = 0.0f; t2time = 0.0f; t3time = 0.0f; t4time = 0.0f; t5time = 0.0f; }

    /// Print out profiling
    // void reportTiming();

    /// print time required for a specified number of calls to generate_points
    void profileRndGen(int count);

    /// Generate points according to dimensions provided in reproduction_config
    void startPointGeneration(GeneratedPoints & genpnts, int numcycles);

    /// Access previously generated points
    // GeneratedPoints & getGeneratedPoints();

    /// Remove points in the provided set from spatial tracking
    void clearPoints(std::vector<AnalysisPoint> * toclear);

    bool testCleared(int startx, int starty, int extentx, int extenty);

    /// setter for aspects of reproduction configuration information
    void setRepConfig(ReproductionConfiguration &rep);

    /// setter for sub area to be synthesized
    void setSubSynthArea(int startx, int starty, int extentx, int extenty);

    MultiDistributionReproducer(ConditionsMap * cmap, ReproductionConfiguration reproduction_config, AnalysisConfiguration analysis_configuration,
                                GeneratedPointsProperties * outGeneratedPointProperties);

private:

    // float inittime, movetime, t1time, t2time, t3time, t4time, t5time;
    long rndcount;
    SubSynthArea subsynth;
    int movecycles;

    std::default_random_engine generator; // random number generator

    // SubSynthArea locsynth;
#ifdef VALIDATE
    float strsum;
    int strcnt;
    int xsect;
    int zerocnt;
    int quickcnt;
    int radsum;
    int avgsum;
    int devsum;
    int minsum;
    int maxsum;
    int radcnt;
#endif
    int matching_density_initialize(int cid);

    void accelerated_point_validity_check(const AnalysisPoint & reference_point, bool & needs_check);
    void accelerated_point_validity_check(const AnalysisPoint & reference_point, int queried_category, bool & needs_check);
    float calculate_strength(const AnalysisPoint & reference_point, const std::vector<AnalysisPoint> & reachable_points);
    float calculate_strength(const AnalysisPoint & reference_point);

    void generate_points_through_random_moves(int cid);

    void move_point(AnalysisPoint & point, AnalysisPoint & new_location);
    void add_destination_point(const AnalysisPoint & point);
    void remove_destination_point(const AnalysisPoint & point, int destination_points_position);
    void generate_points(int cid);

    const RadialDistribution * get_radial_distribution(const AnalysisPoint & reference_point, int reference_category, int target_category);

    /// Variations on matching_density_initialize
    /// Only check whether another trunk is in exactly the same location
    int variantPlaceNoHits(int cid);
    /// As above but also looks for canopy-trunk intersects with higher categories
    int variantPlaceNoTrunks(int cid);
    /// The full strength test with allowance made for missing plants in the surroundings
    int variantPlaceFull(int cid);

    /// Variations on generate_points_through_random_moves
    /// Combine small local moves with larger jumps according to a probability mix
    void variantMoveStep(int cid);
    /// As above but also seperate out phases of moving and changing tree heights
    void variantMoveHeightSep(int cid);


    /// test whether a plants canopy intersects the trunk of any existing plants
    int canopyIntersect(const AnalysisPoint & refpnt);

    /// store reachable points so as to avoid hitting the nearest point accel structure more than once
    void cacheReachablePoints(const AnalysisPoint & refpnt, std::vector<AnalysisPoint> & reachpnts);

    /// count the expected number of points in the currently active category that is expected over the reproduced area
    int areaBasedPointCount(int cid);

    /// get a random point within the reproduction area. This is take from the point_factory but specialised to work with lookups on the condition map.
    /// cid is the currently active category
    AnalysisPoint getRandomPoint(int cid);

    /// return a point moved by a small increment in a random direction and simultaneously alter its height by a small amount
    AnalysisPoint getRandomStep(const AnalysisPoint & refpnt, int cid);

    /// necessary pre-check to make sure that canopies don't intersect trunks
    bool canopyIntersectFree(const AnalysisPoint & refpnt, std::vector<AnalysisPoint> & reachpnts);

    ConditionsMap * m_cmap; //< map of the terrain conditions, used to look up local distributions during generation
    ReproductionConfiguration m_reproduction_configuration;
    PointSpatialHashmap m_spatial_point_storage;
    AnalysisConfiguration m_analysis_configuration;
    GeneratedPointsProperties * m_generated_points_properties;

    std::vector<AnalysisPoint> m_active_category_points;
    // GeneratedPoints m_all_generated_points;

    std::set<int> m_placed_categories; //< for accelerated point validity check through already placed categories rather than every point

    /// For generating random points
    DiceRoller m_dice_roller;
    PointMap m_taken_points;
};

class Distribution
{
private:

    QString inputdir;                       //< directory holding distribution files
    // ReproductionConfiguration * repconfig;  //< container for distribution parameters
    AnalysisConfiguration analyzeconfig;  //< container for analysis parameters
    MultiDistributionReproducer::PairCorrelations loaded_pair_correlations;  //< distribution correlations between and within species
    MultiDistributionReproducer::CategoryPropertiesContainer loaded_category_properties; //< priority categories between distributions
    bool empty;

public:

    Distribution(){ empty = true; }

    ~Distribution(){}

    /// copy constructor
    Distribution(Distribution const & other);

    friend void swap(Distribution & first, Distribution & second)
    {
        using std::swap;
        swap(first.empty, second.empty);
        swap(first.inputdir, second.inputdir);
        swap(first.analyzeconfig, second.analyzeconfig);
        swap(first.loaded_pair_correlations, second.loaded_pair_correlations);
        swap(first.loaded_category_properties, second.loaded_category_properties);
    }

    /// assignment overloading - canonical copy-swap approach
    Distribution& operator=(Distribution copy)
    {
        // standard copy-swap idiom
        swap(*this, copy);
        return * this;
    }

    /// overloaded equivalence operator
    bool operator==(Distribution &other);

    /// Test equivalence of floating point numbers to within a certain number of digits of precision
    bool equiv(const float &a, const float &b, int precision=1);

    /// Is the current distribution a subset of another
    bool subsetEqual(Distribution & other, bool subset = false);

    /// test equivalence of pair correlations with 3 digits of precision
    bool correlationsEqual(Distribution &other, bool subset = false);

    /// getter for corellation histograms
    MultiDistributionReproducer::PairCorrelations & getCorrelations(){ return loaded_pair_correlations; }

    /// getter for category properties
    MultiDistributionReproducer::CategoryPropertiesContainer & getCategories(){ return loaded_category_properties; }

    /// setter for analysis configuration information
    void setAnalysisConfig(AnalysisConfiguration &inconfig){ analyzeconfig = inconfig; }

    /// getter for analysis configuration information
    AnalysisConfiguration & getAnalysisConfig(){ return analyzeconfig; }

    /// get a count of the number of major plants
    int getNumMajorPlants();

    /// setter for whether distribution has data or not
    void setEmpty(bool isempty){ empty = isempty; }

    /// test to see if the distribution contains any data
    bool isEmpty(){ return empty; }

    /// print high-level distribution attributes to console
    void summaryDisplay();

    /// print distribution to console
    void display();

    /**
     * @brief Read in plant distributions from file
     * @param rootdir   name of the root directory that contains all the distributions
     * @param ter       terrain onto which the plants will be placed
     * @param catsizes  various plant size statistics by category to be updated
     * @retval true  if load succeeds,
     * @retval false otherwise.
     */
    bool read(string rootdir, Terrain * ter, CategorySizes & catsizes);

    /**
     * @brief write Write out plant distributions to file
     * @param writedir  name of the root directory containing all the distributions
     * @return true if save succeeds, false otherwise
     */
    bool write(string writedir);
};

#endif
