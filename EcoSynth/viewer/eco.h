// eco.h: core classes for controlling ecosystems and plant placement
// author: James Gain
// date: 27 February 2016

#ifndef _eco_h
#define _eco_h

// #define MEDBIOME
// #define ALPINEBIOME
//#define SAVANNAHBIOME
//#define LOWRES

#define HIGHRES

#include "common/basic_types.h"

#include "terrain.h"
#include "pft.h"
// #include "synth.h"
// #include "palette.h"
#include "grass.h"
// #include "clusters.h"
#include "unordered_map"
#include "boost/functional/hash.hpp"


// const float trunkradius = 0.07f;
// const float trunkheight = 0.4f;
// const float canopyradius = 1.0f;
// const float canopyheight = 0.6f;

#ifdef MEDBIOME
// MEDITERRANEAN BIOME
enum class FunctionalPlantType // Broad Categories of Plant Types
{
    MEDMNE,     //< Mediterranean Needle-Leaved Evergreen Shade Intolerant Tree
    MEDTBS,     //< Mediterranean Temperate Broad-Leaved Summergreen Shade Tolerant Tree
    MEDIBS,     //< Mediterranean Temperate Broad-Leaved Summergreen Shade Intolerant Tree
    MEDTBE,     //< Mediterranean Temperate Broad-Leaved Evergreen Shade Tolerant Tree
    MEDMSEB,    //< Mediterranean Broad-leaved Evergreen Shade Tolerrant Shrub
    MEDMSEN,    //< Mediterranean Needle-leaved Evergreen Shade Intermediate Shrub
 //   MEDITBS,    //< Mediterranean Temperate Broad-Leaved Summergreen Shade Intolerant Invader Tree
    FPTYPEEND
};
// **
const std::array<FunctionalPlantType, 6> all_functionalplants = {
    FunctionalPlantType::MEDMNE,
    FunctionalPlantType::MEDTBS,
    FunctionalPlantType::MEDIBS,
    FunctionalPlantType::MEDTBE,
    FunctionalPlantType::MEDMSEB,
    FunctionalPlantType::MEDMSEN,
 //   FunctionalPlantType::MEDITBS,
}; // to allow iteration over the functional plants
#endif

#ifdef ALPINEBIOME
// ALPINE BIOME
enum class FunctionalPlantType // Broad Categories of Plant Types
{
    ALPTNE,     //< Alpine Temperate Needle-leaved Evergreen
    ALPTBS,     //< Alpine Temperate Broad-leaved Summergreen
    NONE,
    ALPTBE,     //< Alpine Temperate Broad-leaved Evergreen
    ALPTS,      //< Alpine Temperate Broad-leaved Evergreen Shrub
    ALPBNE,     //< Alpine Boreal Needle-leaved Evergreen
    ALPBNS,     //< Alpline Boreal Needle-leaved Summergreen
    ALPBBS,     //< Alpine Boreal Broad-leaved Summergreen
    ALPBS,      //< Alpine Boreal Broad-leaved Summergreen Shrub
    FPTYPEEND
};

const std::array<FunctionalPlantType, 9> all_functionalplants = {
    FunctionalPlantType::ALPTNE,
    FunctionalPlantType::ALPTBS,
    FunctionalPlantType::NONE,
    FunctionalPlantType::ALPTBE,
    FunctionalPlantType::ALPTS,
    FunctionalPlantType::ALPBNE,
    FunctionalPlantType::ALPBNS,
    FunctionalPlantType::ALPBBS,
    FunctionalPlantType::ALPBS,
}; // to allow iteration over the functional plants
#endif

#ifdef SAVANNAHBIOME
// SAVANNAH BIOME
enum class FunctionalPlantType // Broad Categories of Plant Types
{
    PBE,    //< Tropical Broad-leaved Evergreen
    PBR,    //< Tropical Broad-leaved Raingreen
    PBES,   //< Tropical Broad-leaved Evergreen Shrub
    PBRS,   //< Tropical Broad-leaved Raingreen Shrub
    AE,     //< Arboreal Broad-leaved Evergreen
    FPTYPEEND
};

const std::array<FunctionalPlantType, 5> all_functionalplants = {
    FunctionalPlantType::PBE,
    FunctionalPlantType::PBR,
    FunctionalPlantType::PBES,
    FunctionalPlantType::PBRS,
    FunctionalPlantType::AE,
}; // to allow iteration over the functional plants
#endif

const int specOffset = 0; //< conversion from HL database - 12 in HL case, 0 later
const int maxNiches = 10;  //< maximum number of initial terrain niches from HL system
const int maxSpecies = 16*3; // multiplier is for three age categories
const int pgdim = 50;

// const int pgdim = 3;

struct Plant
{
    vpPoint pos;    //< position on terrain in world units
    float height;   //< plant height in metres
    float canopy;   //< canopy radius in metres
    glm::vec4 col;  //< colour variation randomly assigned to plant
    bool iscanopy;	//< is plant part of canopy? (otherwise, part of undergrowth)
};

struct SubSpecies
{
    std::string name;   //< subspecies name
    int chance;         //< chance of selecting this species, out of 1000
};

struct PlantPopulation
{
    std::vector<std::vector<Plant>> pop;    //< list of all plants in an ecosystem by type
};

class GrassSim;

class PlantGrid
{
private:
    std::vector<PlantPopulation> pgrid; //< flattened grid holding plant populations
    // std::vector<std::vector<AnalysisPoint>> sgrid; //< flattened grid holding synthesized points that match up to the pgrid
    // std::vector<std::string> speciesTable;  //< names of species
    std::vector<std::vector<SubSpecies>> speciesTable; //< name and probabilities for subspecies assignment
    int numSubSpecies;

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// reset grid to empty state
    void initGrid();

    /**
     * @brief cellLocate    Find the cell index of location in map coordinates
     * @param ter           Terrain onto which the ecosystem is placed
     * @param mx            x-location in map coordinates
     * @param my            y-location in map coordinates
     * @param cx            x-index in grid
     * @param cy            y-index in grid
     */
    void cellLocate(Terrain * ter, int mx, int my, int &cx, int &cy);

    /// return the number of plant species in the plant population
    int numSpecies()
    {
        if((int) pgrid.size() > 0)
            return (int) pgrid[0].pop.size();
        else
            return 0;
    }

    /// Initialise the list of species according to Biome
    void initSpeciesTable();

public:
    int gx, gy;     //< grid dimensions

    PlantGrid(){ gx = 0; gy = 0; initGrid(); initSpeciesTable(); }

    PlantGrid(int dx, int dy){ gx = dx; gy = dy; initGrid(); initSpeciesTable(); }

    ~PlantGrid(){ delGrid(); speciesTable.clear(); }

    /// completely delete grid
    void delGrid();

    /// clear the contents of the grid to empty
    void clear(){ initGrid(); }

    /// Check whether the grid contains any plants
    bool isEmpty();

    /**
     * @brief clearCell Clear all plants in a specified grid cell
     * @param x     x-location in grid
     * @param y     y-location in grid
     */
    void clearCell(int x, int y);

    /**
     * @brief clearRegion   Remove plants within a contiguous block of cells
     * @param ter           Terrain onto which plants are placed
     * @param region        Region to clear
     */
    void clearRegion(Terrain * ter, Region region);

    /**
     * @brief placePlant    Insert a plant into the correct species population localised with the grid
     * @param ter       Terrain onto which the plant is to be placed
     * @param species   Species type
     * @param plant     Plant information
     */
    void placePlant(Terrain * ter, int species, Plant plant);

    /**
     * @brief placePlant    Insert a plant into the correct species population at a given location in the grid
     * @param ter       Terrain onto which the plant is to be placed
     * @param species   Species type
     * @param plant     Plant information
     * @param x         x-location
     * @param y         y-location
     */
    void placePlantExactly(Terrain * ter, int species, Plant plant, int x, int y);

    /// Output current sate of the PlantGrid
    void reportNumPlants();

    /**
     * @brief pickPlants    Select plants that match the current niche and place in an output grid
     * @param ter           Terrain onto which the plants will be placed
     * @param clusters      Map of ecosystem niches
     * @param niche         Required niche that must be matched during plant selection
     * @param outgrid       Place where plants will be placed
     */
    void pickPlants(Terrain * ter, TypeMap * clusters, int niche, PlantGrid & outgrid);

    /**
     * @brief pickAllPlants Select all plants from the current grid and copy to an output grid indiscriminately
     * @param ter           Terrain onto which the plants will be placed
     * @param offx          x-coordinate offset added to plant position
     * @param offy          y-coordinate offset added to plant position
     * @param scf           scale factor for initial position and size before offset
     * @param outgrid       Place where plants will be placed
     */
    void pickAllPlants(Terrain * ter, float offx, float offy, float scf, PlantGrid & outgrid);

    /**
     * @brief setPopulation copy a plant population into the grid
     * @param x     grid location in the x direction
     * @param y     grid location in the y direction
     * @param pop   plant population to place in the grid
     */
    void setPopulation(int x, int y, PlantPopulation & pop);

    /**
     * @brief getPopulation get a plant population from the grid
     * @param x     grid location in the x direction
     * @param y     grid location in the y direction
     * @return      population at the specified grid position
     */
    PlantPopulation * getPopulation(int x, int y);

    /**
     * @brief getSynthPoints get a collection of synthesized points from the grid for possible modification
     * @param x     grid location in x
     * @param y     grid location in y
     * @return      access to synthesized points at the specified grid position
     */
    // std::vector<AnalysisPoint> * getSynthPoints(int x, int y);

    /**
     * @brief getRegionIndices  Find the grid bounds of a map region
     * @param ter       Terrain onto which the ecosystem is placed
     * @param region    Bounding region in map coordinates
     * @param sx        starting x-index
     * @param sy        starting y-index
     * @param ex        ending x-index
     * @param ey        ending y-index
     */
    void getRegionIndices(Terrain * ter, Region region, int &sx, int &sy, int &ex, int &ey);

    /**
     * @brief vectoriseByPFT    Select all plants that match a specific plant functional type
     * @param pft       Plant functional type for selection
     * @param pftPlnts  Vector in which to place extracted plants
     */
    void vectoriseByPFT(int pft, std::vector<Plant> &pftPlnts);

    /**
     * @brief burnGrass Apply a radial decrease in grant height around the locations of all plants individually
     * @param grass Grass simulator
     * @param ter   Underlying terrain
     * @param scale Extent of terrain
     */
    void burnGrass(GrassSim * grass, Terrain * ter, float scale);

    /**
     * Read in plant positions from a PDB format text file
     * @param filename  name of file to load (PDB format)
     * @param ter  terrain onto which trees are projected
     * @param maxtree   update height if it exceeds the currently tallest tree
     * @retval true  if load succeeds,
     * @retval false otherwise.
     */
    bool readPDB(string filename, Terrain * ter, float & maxtree);

    /**
     * Write plant positions to a PDB format text file
     * @param filename  name of file to load (PDB format)
     * @retval true  if load succeeds,
     * @retval false otherwise.
     */
    bool writePDB(string filename);
    void clearAllPlants(Terrain *ter);
};

/// Plant Rendering
class ShapeGrid
{
private:
    std::vector<std::vector<Shape>> shapes; //< shape templates for each grid cell and each species
    Biome * biome;                          //< biome determines shape and colour of trees
    int gx, gy;                             //< grid dimensions to match a plant grid

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// reset grid to empty state
    void initGrid();


    /**
      * Create geometry for PFT with sphere top
      * @param trunkheight  proportion of height devoted to bare trunk
      * @param trunkradius  radius of main trunk
      * @param shape        geometry for PFT
      */
    void genSpherePlant(float trunkheight, float trunkradius, Shape &shape);

    /**
      * Create geometry for PFT with tapered box top
      * @param trunkheight  proportion of height devoted to bare trunk
      * @param trunkradius  radius of main trunk
      * @param taper        proportion of box base to which box top tapers
      * @param scale        scale factor to reduce visual impact, otherwise this tree is visually dominant
      * @param shape        geometry for PFT
      */
    void genBoxPlant(float trunkheight, float trunkradius, float taper, float scale, Shape &shape);

    /**
      * Create geometry for PFT with cone top
      * @param trunkheight  proportion of height devoted to bare trunk
      * @param trunkradius  radius of main trunk
      * @param shape        geometry for PFT
      */
    void genConePlant(float trunkheight, float trunkradius, Shape &shape);

    /**
      * Create geometry for PFT with inverted cone top
      * @param trunkheight  proportion of height devoted to bare trunk
      * @param trunkradius  radius of main trunk
      * @param shape        geometry for PFT
      */
    void genInvConePlant(float trunkheight, float trunkradius, Shape &shape);

    /**
      * Create geometry for PFT with inversted cone top
      * @param trunkheight  proportion of height devoted to bare trunk
      * @param trunkradius  radius of main trunk
      * @param shape        geometry for PFT
      */
    void genUmbrellaPlant(float trunkheight, float trunkradius, Shape &shape);

public:

    ShapeGrid(){ gx = 0; gy = 0; }

    ShapeGrid(int dx, int dy, Biome * shpbiome){ gx = dx; gy = dy; biome = shpbiome; initGrid(); }

    ~ShapeGrid(){ delGrid(); }

    /// completely delete grid
    void delGrid();

    void genOpenglTextures();

    /// clear the contents of the grid to empty
    void clear(){ initGrid(); }

    /**
     * Create geometry to represent each of the Functional Plant Types over a grid
     */
    void genPlants();

    /**
     * @brief bindPlants    Update positioning information for plants within certain location bounds
     * @param view          The current viewpoint
     * @param ter           Terrain onto which plants will be bound
     * @param plantvis      Flags for which plant species are visible
     * @param esys          The ecosystem grid
     * @param region        A bound on the region to be updated
     */
    void bindPlants(View * view, Terrain * ter, bool * plantvis, PlantGrid * esys, Region region);
    void bindPlantsSimplified(Terrain *ter, PlantGrid *esys);

    /**
     * @brief drawPlants    Bundle rendering parameters for instancing lists
     * @param drawParams    Rendering parameters for the different plant species
     */
    void drawPlants(std::vector<ShapeDrawData> &drawParams);
    void genPlants(std::string model_filename);
};


enum class SamplingDimensions // Sampling Dimensions in the Database
{
    ANGLE,      //<
    WATER,      //< Average Soil Moisture per month over the course of a year
    SUN,        //< Average Sun Exposure per month over the course of a year
    TEMP,       //< Average Temperature per month over the course of a year
    OLD,        //< Age of the Ecosystem Simulation
    SDEND
};

/// Coordinates in the sampling database
struct SampleCoord
{
    int slope;
    int water;
    int sun;
    int temp;
    int age;
};

class EcoSystem
{
private:

    ShapeGrid eshapes;                //< graphical representation of ecosystem
    // SamplingDatabase edb;             //< ecosystem samples
    // VizGrid visdb;                    //< sampling grid for database visualization
    // ConditionsMap cmap;               //< ecosystem conditions used to derive output ecosystem
    std::vector<PlantGrid> niches;    //< individual ecosystems for each niche
    bool dirtyPlants;                       //< flag that the plant positions have changed since last binding
    bool drawPlants;                        //< flag that plants are ready to draw
    float maxtreehght;                      //< largest tree height among all loaded species
    PlantGrid esys;                   //< combined output ecosystem
    Biome * biome;                      //< biome matching ecosystem
    std::map<int, GLfloat *> speccols;	//< colours for each species

    /// Multi-distribution synthesis data
    // MultiDistributionReproducer::GeneratedPointsProperties pointproperties;
    // MultiDistributionReproducer * synthesizer;

public:

    EcoSystem();

    EcoSystem(Biome * ecobiome);

    ~EcoSystem();

    /// Initialize ecosystem in empty state
    void init();

    /// Remove all plants and reset ecosystem
    void clear();

    /// Set flag to rebind plants
    void redrawPlants(){ dirtyPlants = true; }

    void genOpenglTextures();

    /// Getter for maximum tree height
    float getMaxTreeHeight(){ return maxtreehght; }

    /// getPlants: return a pointer to the actual plants in the ecosystem. Assumes pickAllPlants has been called previously.
    PlantGrid * getPlants(){ return &esys; }

    /// getNiche: return a pointer to a particular ecosystem niche (n)
    PlantGrid * getNiche(int n){ return &niches[n]; }

    void setBiome(Biome * ecobiome){ biome = ecobiome; clear(); }

    /**
       * Read in plant positions from a PDB format text file
       * @param filename  name of file to load (PDB format)
       * @param ter       terrain onto which the plants will be placed
       * @param niche     particular layer for which plants are active, 0 means all layers
       * @retval true  if load succeeds,
       * @retval false otherwise.
       */
    bool loadNichePDB(string filename, Terrain * ter, int niche = 0);

    /**
       * Write plant positions to a PDB format text file
       * @param filename  name of file to load (PDB format)
       * @param niche     particular cluster for which plants are active, 0 means all clusters
       * @retval true  if save succeeds,
       * @retval false otherwise.
       */
    bool saveNichePDB(string filename, int niche = 0);

    /**
     * @brief loadSamplingDB
     * @param dirpath   directory containing the database
     * @param ter       terrain onto which samples will be distributed
     * @param scale     extent of the plants in metres
     * @return true if load succeeds, false otherwise
     */
    // bool loadSamplingDB(string dirpath, Terrain * ter, float scale);
    // bool loadClusterDB(string dirpath, Terrain * ter, float scale);

    /// getter for sampling database
    // SamplingDatabase * getSamplingDB(){ return &edb; }

    /// getter for conditions map
    // ConditionsMap * getConditionsMap(){ return &cmap; }

    /**
     * @brief pickSubSamplesDB    Select a range of samples from database and place onto the terrain
     * @param ter   terrain onto which samples will be placed
     * @param tmap  texture map for grouping samples in the visualization
     * @param start start of sampling coordinate range
     * @param end   inclusive end of sampling coordinate range
     * @param scale size of palette in metres
     */
    // void pickSubSamplesDB(Terrain * ter, TypeMap * tmap, SampleCoord start, SampleCoord end, float scale);

    /**
     * @brief validateSubSampleRange    Determine whether the sample range is valid for the current database
     * @param start     start of the sampling coordinate range
     * @param end       inclusive end of the sampling coordinate range
     * @return          return true if samples betweeen start and end can be generated
     */
    // bool validateSubSampleRange(SampleCoord start, SampleCoord end){ return visdb.validRange(&edb, start, end); }

    /**
     * @brief displayCursorSampleDB Print out the ecosystem conditions for the relevant sample at a particular position on the terrain
     * @param ter   underlying terrain onto which samples have been placed
     * @param pnt   cursor position on terrain
     */
    // void displayCursorSampleDB(Terrain * ter, vpPoint pnt);

    /**
     * @brief getCursorSampleDB Get the descriptor for the relevant sample at a particular position on the terrain
     * @param ter   underlying terrain onto which samples have been placed
     * @param pnt   cursor position on terrain
     * @param sd    sample descriptor at this location if it exists
     * @return      true if a descriptor is under the cursor, false otherwise
     */
    // bool getCursorSampleDB(Terrain * ter, vpPoint pnt, SampleDescriptor & sd);

    /**
     * Pick plants from different niche ecosystems based on the cluster map to form a final ecosystem
     * @param ter       terrain onto which the plants will be placed
     * @param clusters  resource cluster texture map
     */
    void pickPlants(Terrain * ter, TypeMap * clusters);

    /**
     * Pick plants from all niche ecosystems to form a final ecosystem
     * @param ter       terrain onto which the plants will be placed
     */
    void pickAllPlants(Terrain * ter);

    /**
     * Position plants instances on terrain for subsequent rendering. Needs to be run after plant positions have changed.
     * @param view          current viewpoint
     * @param ter           terrain onto which plants will be placed
     * @param clusters      resource cluster texture map
     * @param plantvis      boolean array of which plant species are to be rendered and which not
     * @param drawParams    parameters for drawing plant species, appended to the current drawing parameters
     */
    void bindPlants(View * view, Terrain * ter, TypeMap * clusters, bool * plantvis, std::vector<ShapeDrawData> &drawParams);
    void bindPlantsSimplified(Terrain *ter, std::vector<ShapeDrawData> &drawParams);

    /**
     * @brief synth Synthesize distribution to create plant positions
     * @param ter           terrain onto which the plants will be placed
     * @param scale         overall scale of the terrain
     * @param numcycles     number of iterations to do in refinement of plant positions
     * @param paintmap      the map being painted by the user in order to get limits on the area to resynthesize, if nothing is passed in the whole terrain is synthesized
     */
    // void synth(Terrain * ter, float scale, int numcycles = 2, TypeMap * paintmap = nullptr);
    //void slowsynth(Terrain * ter, float scale, int numcycles, TypeMap * paintmap = nullptr);
    void placePlant(Terrain *ter, const Plant &plant, int species);
    void placePlant(Terrain *ter, const basic_tree &tree, bool canopy);
    void placeManyPlants(Terrain *ter, const std::vector<basic_tree> &trees, bool canopy);
    void clearAllPlants(Terrain *ter);
};

#endif
