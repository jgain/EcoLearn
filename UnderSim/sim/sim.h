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


#ifndef Sim
#define Sim
/* file: sim.h
   author: (c) James Gain, 2018
   notes: ecosystem simulation
*/

#include "sun.h"
#include "moisture.h"
#include "dice_roller.h"
#include "common/basic_types.h"
#include <list>

// simulation parameters
// #define STEPFILE
#define lapserate 9.8f // in dry air conditions decrease is on average 9.8C per 1000m
#define def_reservecapacity 3.0f
#define def_moisturedemand 4.0f
#define def_seeddistmult 5.0f
#define def_seedprob 0.00004f
#define def_mortalitybase 0.05f
#define def_viabilityneg -0.2f
#define def_stresswght 1.0f
#define def_hghtthreshold 3.0f // kill plants above this height because they are too tall for undergrowth

enum class PlantCompartment {
    ROOT,   //< address root compartment of plant
    CANOPY, //< address canopy compartment of plant
    PCEND
};

enum class PlantSimState {
    ALIVE, //< plant is active and can grow
    DEAD, //< plant has died
    STATIC, //< canopy plant that is fixed in size over the course of the simulation
    PSSEND
};

struct SimPlant
{
    PlantSimState state;     //< tree status
    int age;        //< age in years
    vpPoint pos;    //< position on terrain in world units
    float gx, gy;   //< position on terrain in terrain grid units
    float height;   //< plant height in metres
    float canopy;   //< canopy radius in metres
    float root;     //< roots radius in metres
    float reserves; //< plant biomass pool that sees it through short periods of poor conditions
    float stress;   //< stress factor on plant due to poor conditions
    glm::vec4 col;  //< colour variation randomly assigned to plant, but keep consistent to track growth
    int pft;        //< plant functional type to which tree belongs
    float water;    //< accumulator for water contribution
    float sunlight; //< accumulator for sunlight contribution
    int watercnt;   //< number of cells intersected by roots
    int sunlightcnt; //< number of cells intersected by canopy
};

struct PlntInCell
{
    SimPlant * plnt;            //< index of plant in plant population
    float hght;         //< height of plant in metres, used for sorting
};

struct SimParams
{
    float reservecapacity;  //< no. of months in plants carbohydrate reserves
    float moisturedemand;   //< minimum amount of moisture taken by dominant plant
    float seeddistmult;     //< number of times the canopy radius to which seeding extends - not currently used?
    float seedprob;         //< probability in [0,1] of a seed sprouting in a given cell per year (originally 0.00001, or 1e-5)
    float stresswght;       //< weighting of stress as a role in mortality
    float mortalitybase;    //< base term in mortality function
    float viabilityneg;     //< cap on how negative viability can be in a given month
    // also viability c&r values for each species
};

float cmpHeight(PlntInCell i, PlntInCell j);

struct SimCell
{
    std::vector<PlntInCell> canopies;    //< all plants whose canopies intersect the cell
    std::vector<PlntInCell> roots;       //< all plnts whose roots intersect the cell
    bool growing;               //< cell is under the cover of static trees and so can grow
    bool available;				//< cell does not intersect with any canopy or undergrowth tree trunks
    float leftoversun;          //< avg. remaining sunlight per growing month for seeding
    float leftoverwet;          //< avg. remaining moisture per growing month for seeding
    bool canopysorted;          //< is the list of canopies sorted by decreasing tree height
    bool rootsorted;            //< is the list of roots sorted by decreasing tree height
    std::vector<int> seedbank;  //< list of subbiomes that can provide seeds for this cell
    float seed_chance;			//< chance of seeding occurring in the current cell, given that it is possible to seed. Determined by distance to mother trees
};



class Simulation;

class MapSimCell
{
private:
    int step;                       //< multiple for conversion from terrain grid to simulation grid
    int gx, gy;                     //< grid dimensions
    std::vector<SimCell> smap;      //< grid of simcells
    DiceRoller * dice;              //< random number generator
    const float radius_mult = 6.0f;
    const float seed_radius_mult = 6.0f;
    std::map<int, ValueGridMap<float> > closest_distances;		// the distance of the closest canopy tree of a given species, to each pixel. Negative for within canopy
    std::map<int, ValueGridMap<int> > species_counts;			// the count of each species that reaches to each pixel in influence (seeding)

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// check if coordinates are within the grid
    inline bool ingrid(int x, int y){ return x >= 0 && x < gx && y >= 0 && y < gy; }

    /**
     * @brief notInSeedbank check if a particular subbiome is NOT found in the seedbank for a particular cell
     * @param sbidx subbiome index
     * @param x     x-coord grid position (sub cell accuracy)
     * @param y     y-coord grid position (sub cell accuracy)
     * @return      true if the subbiome is not present, false otherwise
     */
    bool notInSeedbank(int sbidx, int x, int y);

    /**
     * @brief inscribeSeeding  Add a seeding area into the simulation grid
     * @param sbidx     index of subbiome being placed
     * @param x         x-coord grid position (sub cell accuracy)
     * @param y         y-coord grid position (sub cell accuracy)
     * @param rcanopy   radius of seeding
     */
    void inscribeSeeding(int sbidx, int spec_idx, float px, float py, float rcanopy, Terrain * ter);

    /**
     * @brief singleSeed    Instantiate a single sapling randomly from the seed bank
     *   int x, y; @param x         x-coord grid position
     * @param y         y-coord grid position
     * @param plntpop   Plant population
     * @param sim       Ongoing simulation state
     * @param ter       Underlying terrain
     * @param biome     Biome statistics for plant species
     */
    bool singleSeed(int x, int y, std::list<SimPlant *> * plntpop, Simulation * sim, Terrain * ter, Biome * biome, std::vector<int> &noseed_count);


public:

    MapSimCell(){ gx = 0; gy = 0; initMap(); dice = new DiceRoller(0,10000);}

    ~MapSimCell(){ delMap(); }

    void writeSeedBank(std::string outfile);		// added by KKapp
    void readSeedBank(string infile);				// added by KKapp

    /// getter for grid dimensions
    void getDim(int &dx, int &dy){ dx = gx; dy = gy; }

    /// setter for grid dimensions
    void setDim(int dx, int dy, int mult){ gx = dx; gy = dy; step = mult; initMap(); }

    /// gett for step
    int getStep(){ return step; }

    /**
     * @brief toTerGrid Convert from simulation grid coordinates to terrain grid coordinates, taking care to check for out of bounds issues
     * @param mx    simgrid x
     * @param my    simgrid y
     * @param tx    terrain grid x
     * @param ty    terrain grid y
     * @param ter   underlying terrain
     */
    void toTerGrid(int mx, int my, int &tx, int &ty, Terrain * ter);


    /// clear the contents of the grid to empty
    void initMap();

    /// completely delete map
    void delMap();

    /// reset per year seeding suitability to zero
    void resetSeeding();

    /// convert from terrain grid position to simulation grid position
    inline float convert(float pos){ return (float) step * pos; }

    /// convert from simulation grid position to terrain grid position
    inline float convert_to_tergrid(float pos) { return (float) pos / step;}

    /// clamp to simgrid domain
    void clamp(int &x, int &y);

    /// getter and setter for map elements
    SimCell * get(int x, int y){ return &smap[flatten(x, y)]; }
    void set(int x, int y, SimCell &val){ smap[flatten(x, y)] = val; }

    const std::vector<SimCell> &get_smap()
    {
        return smap;
    }

    /**
     * @brief inscribe  Add a new plant into the simulation grid
     * @param plntidx   index of plant being placed
     * @param px         x-coord grid position (sub cell accuracy)
     * @param py         y-coord grid position (sub cell accuracy)
     * @param rcanopy   diameter of plant canopy
     * @param rroot     diameter of plant root
     * @param isStatic  true if a canopy plant is being inscribed
     */
    void inscribe(std::list<SimPlant *>::iterator plntidx, float px, float py, float rcanopy, float rroot, bool isStatic, Terrain *ter, Simulation * sim);

    /**
     * @brief expand  increase radius of plant in the simulation grid
     * @param plntidx   index of plant being placed
     * @param x         x-coord grid position (sub cell accuracy)
     * @param y         y-coord grid position (sub cell accuracy)
     * @param prevrcanopy   previous diameter of plant canopy
     * @param prevrroot     previous diameter of plant root
     * @param newrcanopy    new expanded diameter of plant canopy
     * @param newrroot      new expanded diameter of plant root
     */
    void expand(std::list<SimPlant *>::iterator plntidx, float px, float py, float prevrcanopy, float prevrroot, float newrcanopy, float newrroot);

    /**
     * @brief uproot Remove a plant from the simulation grid
     * @param plntidx   index of plant being uprooted
     * @param px        x-coord grid position (sub cell accuracy)
     * @param py        y-coord grid position (sub cell accuracy)
     * @param rcanopy   diameter of plant canopy
     * @param rroot     diameter of plant root
     * @param ter       terrain containing ecosystem
     */
    void uproot(std::list<SimPlant *>::iterator plntidx, float px, float py, float rcanopy, float rroot, Terrain * ter);

    /// plantHeight: return the height in metres of a plant with a given index
    float plantHeight(int idx);

    /**
     * @brief traverse  Traverse the simulation grid and collate sunlight and soil moisture contributions to each plant
     * @param seedable  Does this traversal contribute to seeding determination due to growing season
     */
    void traverse(std::list<SimPlant *> * plntpop, Simulation * sim, Biome * biome, MapFloat * sun, MapFloat * wet, bool seedable);

    /**
     * @brief establishSeedBank Render canopy plants into the seedbank
     * @param plntpop   Plant population
     * @param biome     Biome statistics for plant species
     * @param ter       Terrain onto which the plants are placed
     */
    void establishSeedBank(std::list<SimPlant *> * plntpop, int plntpopsize, Biome * biome, Terrain * ter);

    /**
     * @brief seeding Execute once-yearly seeding test
     * @param plntpop   Existing list of plants to which new seedlings will be added
     * @param sim       Ongoing simulation state
     * @param ter       Terrain onto which plants are placed
     * @param biome     Biome statistics for plant species
     */
    void seeding(std::list<SimPlant *> * plntpop, int plntpopsize, Simulation * sim, Terrain * ter, Biome * biome);

    /**
     * @brief testSeeding   Test seeding of a single plant from mixed sub-biomes
     * @param pos       position on terrain to force plant seeding for test purposes
     * @param sim       Ongoing simulation state
     * @param ter       Terrain onto which plants are placed
     * @param biome     Biome statistics for plant species
     */
    // void testSeeding(vpPoint pos, Simulation * sim, Terrain * ter, Biome * biome);

    /**
     * @brief visualize Display the canopy grid as an image, with plants coded by colour
     * @param visimg    visualisation image
     * @param plnts     list of plants in simulation
     */
    // void visualize(QImage * visimg, std::vector<SimPlant* > *plnts);

    /**
     * @brief unitTests A set of basic unit tests for the MapSimCell class
     * @param visimg     image into which visualisation of unit tests is written
     * @return  true if the unit tests pass, false otherwise
     */
    // bool unitTests(QImage * visimg, Terrain *ter, Simulation * sim);

    /**
     * @brief validate  check validity of mapsim structure
     * @param plnts     list of plants in simulation
     * @return          true if the mapsim is valid, false otherwise
     */
    // bool validate(std::vector<SimPlant *> *plnts);
    void init_countmaps(const std::set<int> &species, int gw, int gh, int tw, int th);
};

// other map inputs: slope, temperature

class Simulation
{

private:
    MapSimCell simcells;            //< flattened sub-cell grid structure storing intersecting roots and canopies
    std::list<SimPlant *> plntpop;  //< all alive plants in the simulation (dead ones are deleted, hence the list data structure)
    int plntpopsize;                //< count of number of plants in simualation
    std::vector<MapFloat> sunlight; //< local per cell illumination for each month
    std::vector<MapFloat> moisture; //< local per cell moisture for each month
    MapFloat slope;                 //< local per cell slope derived from terrain
    std::vector<float> temperature; //< average monthly temperature
    std::vector<float> cloudiness;  //< average monthly cloud cover
    std::vector<float> rainfall;    //< average monthly rainfall
    Terrain * ter;                  //< terrain over which ecosystem is being simulated
    Biome * biome;                  //< biome within which plants are simulated
    float time;                     //< internal ecosystem clock in years
    SunLight * sunsim;              //< sunlight simulator
    DiceRoller * dice;              //< random number generator
    std::vector<float> hghts;       //< store pre- and post- simulation heights for comparison

    /** @brief initSim    Initialize internal data for the simulation
    * @param dx, dy       Main void Simulation::checkpointedSim(EcoSystem * eco, std::string seedbank_file, std::string seedchance_filename, int delYears, std::string out_filename)terrain dimensions
    * @param subcellfactor  number of sub-cells per main simulation cell
    */
    void initSim(int dx, int dy, int subcellfactor);

    /**
     * @brief readMonthlyMap Read a set of 12 maps from file
     * @param filename   name of file to be read
     * @param monthly    content will be loaded into this vector of maps
     */
    bool readMonthlyMap(std::string filename, std::vector<MapFloat> &monthly);

    /**
     * @brief writeMonthlyMap    Write a set of 12 maps to file
     * @param filename   name of file to be written
     * @param monthly    map content to be written
     */
    bool writeMonthlyMap(std::string filename, std::vector<MapFloat> &monthly);

    /// clear internal simulation data
    void delSim();

    /**
     * @brief clearPass Reset between simulation passes
     */
    void clearPass();

    /**
     * @brief death Test to see whether a plant dies, based on current viability
     * @param plntind   Index of the plant being tested
     * @param stress    Plants stress from environmental factors
     */
    bool death(std::list<SimPlant *>::iterator pind, float stress);

    /**
     * @brief growth    Apply monthly growth equation to plant moderated by current vitality
     * @param pind      Index of the plant being grown
     * @param vitality  Plant vitality from environmental factors
     */
    void growth(std::list<SimPlant *>::iterator pind, float vitality);

    // seeding

    /// Simstep: a single simulation step, equal to one month
    void simStep(int month);

    /// printParams: display the current state of the simulation input parameters
    void printParams();

    /**
     * @brief averageViability  Calculate and report the average viability of each species over growing areas of the current terrain.
     *                          This is the same basis that is used for seeding probability.
     * @param targetnums        Average number of expected undergrowth plants per canopy tree for each species
     */
    void averageViability(std::vector<float> &targetnums);

    /**
     * @brief plantTarget   Compute the distance from the target species numbers as a proportion of canopy trees
     * @param targetnums    Average number of expected undergrowth plants per canopy tree for each species
     * @return  sum of absolute differences between actual and target numbers of undergrowth plants across all species
     */
    float plantTarget(std::vector<float> &targetnums);

public:

    SimParams sparams;              //< simulation parameters for testing outcomes

    Simulation(){ ter = NULL; initSim(1000, 1000, 5); }

    ~Simulation(){ ter = NULL; delete dice; delSim(); }

   /** @brief Simulation    Initialize the simulation using an input terrain
   *                        position on the earth, average cloud cover per month, a per cell elevation, canopy height, and
   *                        canopy density. Store output in 12 sunlight maps, one for each month
   * @param terrain        Terrain onto which the plants are placed
   * @param subcellfactor  number of sub-cells per main simulation cell
   */
   Simulation(Terrain * terrain, Biome * simbiome, int subcellfactor);

   void incrPlantPop(){ plntpopsize++; }

   Terrain * getTerrain(){ return ter; }

   void writeSeedBank(std::string outfile);
   /**
    * @brief calcSunlight    Calculate per cell understory photosynthetically active radiation for each month, factoring in
    *                        position on the earth, average cloud cover per month, a per cell elevation, canopy height, and
    *                        canopy density. Store output in 12 sunlight maps, one for each month
    * @param glsun           OpenGL sunlight simulation engine
    * @param inclCanopy      if true, account for filtering effect of canopy plants
    */
   void calcSunlight(GLSun * glsun, int minstep, int nsamples, bool inclCanopy);
   void reportSunAverages();

   /**
    * @brief calcSlope    Calculate per cell ground slope
    */
   void calcSlope();

   /**
    * @brief calcMoisture    Calculate per cell and per month moisture values, based on rainfall values and calculation of stream power
    */
   void calcMoisture();

   /**
    * @brief importCanopy    Import initial canopy trees from an external ecosystem. These will be treated as static by the simulation
    * @param        Ecosystem storing canopy trees
    */
   void importCanopy(EcoSystem * eco, string seedbank_file = "", string seedchance_filename = "");

   /**
    * @brief exportUnderstory   Export current state of understory ecosystem plants for rendering and further processing. The existing
    *                           ecosystem is overwritten.
    * @param eco    Ecosystem to hold export
    */
   void exportUnderstory(EcoSystem * eco);

   /// read and write simulation condition maps
   bool readSun(std::string filename){ return readMonthlyMap(filename, sunlight); }
   bool writeSun(std::string filename){ return writeMonthlyMap(filename, sunlight); }
   void writeSun(std::string filename, int month) { writeMap(filename, sunlight.at(month - 1)); }
   void writeSunCopy(std::string filename) { write_monthly_map_copy(filename, sunlight); }
   bool writeAssignSun(std::string filename);
   bool readMoisture(std::string filename){ return readMonthlyMap(filename, moisture); }
   bool writeMoisture(std::string filename){ return writeMonthlyMap(filename, moisture); }

   /**
    * @brief readClimate    read in average monthly climatic variables
    * @param filename   name of the file containing temperature values
    * @return   false, if the file does not exist, true otherwise
    */
   bool readClimate(std::string filename);

   /**
    * @brief getTemperature Return the temperature at a given position on the terrain
    * @param x      x grid location
    * @param y      y grid location
    * @param mth    month of the year
    * @return       temperature in degrees celsius
    */
   float getTemperature(int x, int y, int mth);

   /**
    * @brief pickInfo   Provide console print of information at a particular location
    * @param x          x-position in terrain grid coordinates
    * @param y          y-position in terrain grid coordinates
    */
   void pickInfo(int x, int y);

   /// getters for condition maps
   MapFloat * getSunlightMap(int mth){ return &sunlight[mth]; }
   MapFloat * getMoistureMap(int mth){ return &moisture[mth]; }
   MapFloat * getSlopeMap(){ return &slope; }
   SunLight * getSun(){ return sunsim; }

   /// getter for simulation map data structure
   MapSimCell * getSimMap(){ return &simcells; }

   /**
    * @brief simulate   Run the simulation for a certain number of iterations
    * @param eco        ecosystem for storing plant outputs
    * @param seedbank_file  precomputed placement of seeds
    * @param seedchance_filename precomputed probability of sprouting
    * @param delYears   number of years to simulate
    */
   void simulate(EcoSystem * eco, std::string seedbank_file, std::string seedchance_filename, int delYears);

   void setTemperature(std::array<float, 12> temp);
   void setRainfall(std::array<float, 12> rain);
   void setCloudiness(std::array<float, 12> cloud);
   void writeMap(std::string filename, const MapFloat &map);
   void calcCanopyDensity(EcoSystem *eco, MapFloat *density, string outfilename = "");
   void write_monthly_map_copy(std::string filename, std::vector<MapFloat> &monthly);
};

#endif
