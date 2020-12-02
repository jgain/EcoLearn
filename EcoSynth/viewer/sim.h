/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
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

#ifndef Sim
#define Sim
/* file: sim.h
   author: (c) James Gain, 2018
   notes: ecosystem simulation
*/

#include "sun.h"
#include "moisture.h"
#include "dice_roller.h"

#include "cluster_distribs/src/ClusterMatrices.h"

#define lapserate 9.8f // in dry air conditions decrease is on average 9.8C per 1000m
#define reservecapacity 3.0f // no. of months in plants carbohydrate reserves

enum class PlantCompartment {
    ROOT,   //< address root compartment of plant
    CANOPY, //< address canopy compartment of plant
    PCEND
};

struct PlntInCell
{
    int idx;            //< index of plant in plant population
    float hght;         //< height of plant in metres, used for sorting
};

float cmpHeight(PlntInCell i, PlntInCell j);

struct SimCell
{
    std::vector<PlntInCell> canopies;    //< all plants whose canopies intersect the cell
    std::vector<PlntInCell> roots;       //< all plnts whose roots intersect the cell
    bool growing;               //< cell is under the cover of static trees and so can grow
    bool canopysorted;          //< is the list of canopies sorted by decreasing tree height
    bool rootsorted;            //< is the list of roots sorted by decreasing tree height
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
    int age;        //< age in months
    vpPoint pos;    //< position on terrain in world units
    float gx, gy;   //< position on terrain in terrain grid units
    float height;   //< plant height in metres
    float canopy;   //< canopy radius in metres
    float root;     //< roots radius in metres
    float reserves; //< plant biomass pool that sees it through short periods of poor conditions
    glm::vec4 col;  //< colour variation randomly assigned to plant, but keep consistent to track growth
    int pft;        //< plant functional type to which tree belongs
    float water;    //< accumulator for water contribution
    float sunlight; //< accumulator for sunlight contribution
    int watercnt;   //< number of cells intersected by roots
    int sunlightcnt; //< number of cells intersected by canopy
};

class MapSimCell
{
private:
    int step;                       //< multiple for conversion from terrain grid to simulation grid
    int gx, gy;                     //< grid dimensions
    std::vector<SimCell> smap;      //< grid of simcells
    std::vector<SimPlant> * plntpop;  //< pointer to plant population

    /// return the row-major linearized value of a grid position
    inline int flatten(int dx, int dy){ return dx * gy + dy; }

    /// convert from terrain grid position to simulation grid position
    inline float convert(float pos){ return (float) step * pos; }

    /// check if coordinates are within the grid
    inline bool ingrid(int x, int y){ return x >= 0 && x < gx && y >= 0 && y < gy; }

public:

    MapSimCell(){ gx = 0; gy = 0; initMap(); }

    ~MapSimCell(){ delMap(); }

    /// getter for grid dimensions
    void getDim(int &dx, int &dy){ dx = gx; dy = gy; }

    /// setter for grid dimensions
    void setDim(int dx, int dy, int mult){ gx = dx; gy = dy; step = mult; initMap(); }

    /// clear the contents of the grid to empty
    void initMap();

    /// completely delete map
    void delMap(){ smap.clear(); }

    /// getter and setter for map elements
    SimCell * get(int x, int y){ return &smap[flatten(x, y)]; }
    void set(int x, int y, SimCell &val){ smap[flatten(x, y)] = val; }

    /**
     * @brief inscribe  Add a new plant into the simulation grid
     * @param plntidx   index of plant being placed
     * @param x         x-coord grid position (sub cell accuracy)
     * @param y         y-coord grid position (sub cell accuracy)
     * @param rcanopy   radius of plant canopy
     * @param rroot     radius of plant root
     */
    void inscribe(int plntidx, float px, float py, float rcanopy, float rroot);

    /**
     * @brief inscribe  Add a new plant into the simulation grid
     * @param plntidx   index of plant being placed
     * @param x         x-coord grid position (sub cell accuracy)
     * @param y         y-coord grid position (sub cell accuracy)
     * @param prevrcanopy   previous radius of plant canopy
     * @param prevrroot     previous radius of plant root
     * @param newrcanopy    new expanded radius of plant canopy
     * @param newrroot      new expanded radius of plant root
     */
    void expand(int plntidx, float px, float py, float prevrcanopy, float prevrroot, float newrcanopy, float newrroot);

    /// plantHeight: return the height in metres of a plant with a given index
    float plantHeight(int idx);

    /**
     * @brief traverse  Traverse the simulation grid and collate sunlight and soil moisture contributions to each plant
     * @param plntpop   Plant population indexed by elements in the simulation grid
     */
    void traverse(std::vector<SimPlant> * plnts, Biome * biome, MapFloat * sun, MapFloat * wet);

    /**
     * @brief visualize Display the canopy grid as an image, with plants coded by colour
     * @param visimg    visualisation image
     * @param plnts     list of plants in simulation
     */
    void visualize(QImage * visimg, std::vector<SimPlant> *plnts);

    /**
     * @brief unitTests A set of basic unit tests for the MapSimCell class
     * @param visimg     image into which visualisation of unit tests is written
     * @return  true if the unit tests pass, false otherwise
     */
    bool unitTests(QImage * visimg);

    /**
     * @brief validate  check validity of mapsim structure
     * @param plnts     list of plants in simulation
     * @return          true if the mapsim is valid, false otherwise
     */
    bool validate(std::vector<SimPlant> *plnts);
};

// other map inputs: slope, temperature

class Simulation
{

private:
    MapSimCell simcells;            //< flattened sub-cell grid structure storing intersecting roots and canopies
    std::vector<SimPlant> plntpop;  //< all plants, alive and dead, in the simulation
    std::vector<MapFloat> landsun;	//< sunlight based only on landscape (no vegetation)
    std::vector<MapFloat> sunlight; //< local per cell illumination for each month
    std::vector<MapFloat> moisture; //< local per cell moisture for each month
    MapFloat average_sunlight;		//< average sunlight over all months
    MapFloat average_moisture;		//< ditto for moisture
    MapFloat average_landsun;		//< ditto for sun shading due to landscape only
    MapFloat average_adaptsun;		//< sunlight map for on-the-fly calculation of sunlight in interface
    MapFloat slope;                 //< local per cell slope derived from terrain
    MapFloat rocks;
    std::vector<float> temperature; //< average monthly temperature
    std::vector<float> cloudiness;  //< average monthly cloud cover
    std::vector<float> rainfall;    //< average monthly rainfall
    Terrain * ter;                  //< terrain over which ecosystem is being simulated
    Biome * biome;                  //< biome within which plants are simulated
    float time;                     //< internal ecosystem clock in years
    SunLight * sunsim;              //< sunlight simulator
    DiceRoller * dice;              //< random number generator
    MapFloat temperate_mapfloat;	//< another holder for temperature, for compatibility with other parts of app (specifically the GrassSim class' setConditions func

    /** @brief initSim    Initialize internal data for the simulation
    * @param dx, dy       Main terrain dimensions
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
     * @brief death Test to see whether a plant dies, based on current viability
     * @param plntind   Index of the plant being tested
     * @param stress    Plants stress from environmental factors
     */
    bool death(int pind, float stress);

    /**
     * @brief growth    Apply monthly growth equation to plant moderated by current vitality
     * @param pind      Index of the plant being grown
     * @param vitality  Plant vitality from environmental factors
     */
    void growth(int pind, float vitality);

    // seeding

    /// Simstep: a single simulation step, equal to one month
    void simStep(int month);

public:

    Simulation(){ ter = NULL; initSim(1000, 1000, 5); }

    ~Simulation(){ ter = NULL; delete dice; delSim(); }

   /** @brief Simulation    Initialize the simulation using an input terrain
   *                        position on the earth, average cloud cover per month, a per cell elevation, canopy height, and
   *                        canopy density. Store output in 12 sunlight maps, one for each month
   * @param terrain        Terrain onto which the plants are placed
   * @param subcellfactor  number of sub-cells per main simulation cell
   */
   Simulation(Terrain * terrain, Biome * simbiome, int subcellfactor);

   /**
    * @brief calcSunlight    Calculate per cell understory photosynthetically active radiation for each month, factoring in
    *                        position on the earth, average cloud cover per month, a per cell elevation, canopy height, and
    *                        canopy density. Store output in 12 sunlight maps, one for each month
    * @param glsun           OpenGL sunlight simulation engine
    */
   void calcSunlight(GLSun * glsun, int minutestep, int nsamples);

   void calc_adaptsun(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata);
   MapFloat *get_adaptsun() { return &average_adaptsun; }

   /**
    * @brief calcSlope    Calculate per cell ground slope
    */
   void calcSlope();

   bool hasTerrain()
   {
       return ter;
   }

   /**
    * @brief calcMoisture    Calculate per cell and per month moisture values, based on rainfall values and calculation of stream power
    */
   void calcMoisture();

   /**
    * @brief importCanopy    Import initial canopy trees from an external ecosystem. These will be treated as static by the simulation
    * @param        Ecosystem storing canopy trees
    */
   void importCanopy(EcoSystem * eco);

   /**
    * @brief exportUnderstory   Export current state of understory ecosystem plants for rendering and further processing. The existing
    *                           ecosystem is overwritten.
    * @param eco    Ecosystem to hold export
    */
   void exportUnderstory(EcoSystem * eco);

   /// read and write simulation condition maps
   bool readSun(std::string filename);
   bool writeSun(std::string filename){ return writeMonthlyMap(filename, sunlight); }
   bool writeAssignSun(std::string filename);
   bool readMoisture(std::string filename);
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

   static void calc_average_monthly_map(std::vector<MapFloat> &mmap, MapFloat &avgmap);

   /// getters for condition maps
   MapFloat * getSunlightMap(int mth){ return &sunlight[mth]; }
   MapFloat * getMoistureMap(int mth){ return &moisture[mth]; }
   MapFloat * getSlopeMap(){ return &slope; }
   SunLight * getSun(){ return sunsim; }
   MapFloat * get_average_sunlight_map() { return &average_sunlight; }
   MapFloat * get_average_landsun_map() { return &average_landsun; }
   MapFloat * get_average_moisture_map() { return &average_moisture; }
   MapFloat * get_average_adaptsun_map() { return &average_adaptsun; }
   MapFloat * get_temperature_map() { return &temperate_mapfloat; }

   /**
    * @brief simulate   Run the simulation for a certain number of iterations
    * @param delYears   number of years to simulate
    */
   void simulate(int delYears);
   void calcSunlightSelfShadowOnly(GLSun *glsun);
   void calc_average_sunlight_map();
   void calc_average_moisture_map();
   void calc_average_landsun_map();
   void calc_temperature_map();
   void set_terrain(Terrain *terrain);
   MapFloat *get_rocks();
   void set_rocks();
   bool readLandscapeSun(std::string filename);
   void copy_temperature_map(const ValueGridMap<float> &tempmap);
   void copy_map(const ValueGridMap<float> &srcmap, abiotic_factor f);
   ValueGridMap<float> create_alphamap(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float rw, float rh);
   void calc_adaptsun(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, float rw, float rh);
   int inscribe_alpha(const basic_tree &plnt, float plntalpha, ValueGridMap<float> &alphamap);
};

#endif
