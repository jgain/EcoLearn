#ifndef PFT
#define PFT
/* file: pft.h
   author: (c) James Gain, 2018
   notes: plant functional type database
*/

#include "terrain.h"

namespace data_importer
{
    struct common_data;		// forward declaration for function prototype in Biome class
}

#define maxpftypes 10

enum TreeShapeType
{
    SPHR,   //< sphere shape, potentially elongated
    BOX,    //< cuboid shape
    CONE,    //< cone shape
    INVCONE
};


class Viability
{
public:
    float o1, o2;   //< outer values for which the function returns -1
    float z1, z2;   //< first and second zero crossing points
    float i1, i2;   //< inner values for between which the function returns 1

    Viability(){ o1 = o2 = z1 = z2 = i1 = i2 = 0.0f; }
    ~Viability(){}

    /// Set critical point values for the viability function
    void setValues(float outer1, float outer2, float zero1, float zero2, float inner1, float inner2)
    {
        o1 = outer1; o2 = outer2;
        z1 = zero1; z2 = zero2;
        i1 = inner1; i2 = inner2;
    }

    /// Set critical point values for the viability function
    void getValues(float &outer1, float &outer2, float &zero1, float &zero2, float &inner1, float &inner2)
    {
        outer1 = o1; outer2 = o2;
        zero1 = z1; zero2 = z2;
        inner1 = i1; inner2 = i2;
    }

    /**
     * @brief eval  Evaluate the viability function at a particular value
     * @param val   input value to be evaluated
     * @return      viability result
     */
    float eval(float val);
};

struct PFType
{
    string code;        //< a mnemonic for the plant type
    //< rendering parameters
    GLfloat basecol[4];  //< base colour for the PFT, individual plants will vary
    float draw_hght;    //< canopy height scaling
    float draw_radius;  //< canopy radius scaling
    float draw_box1;    //< box aspect ratio scaling
    float draw_box2;    //< box aspect ration scaling
    TreeShapeType shapetype; //< shape for canopy: sphere, box, cone
    //< simulation parameters
    Viability sun, wet, temp, slope; //< viability functions for different environmental factors
    float alpha;        //< sunlight attenuation multiplier
    int maxage;         //< expected maximum age of the species in months

    //< growth parameters
    int grow_months;    //< number of months of the year in the plants growing season
    float grow_m, grow_c1, grow_c2; // terms in growth equation: m * (c1 + exp(c2))

    //< allometry parameters
    float alm_m, alm_c1; //< terms in allometry equation to convert height to canopy radius: r = e ** (c1 + m ln(h))
    float alm_rootmult; //< multiplier to convert canopy radius to root radius
};

class Biome
{
private:
    std::vector<PFType> pftypes; //< vector of plant functional types in the biome
    std::vector<std::string> catTable;  //< lookup of category names corresponding to category numbers
    std::string name; //< biome name

public:

    // soil moisture infiltration parameters
    float slopethresh;       //< slope at which runoff starts increasing linearly
    float slopemax;          //< slope at which water stops being absorbed altogether
    float evaporation;       //< proportion of rainfall that is evaporated
    float runofflevel;       //< cap on mm per month that can be asorbed by the soil before runoff
    float soilsaturation;    //< cap in mm on amount of water that soil can hold
    float waterlevel;        //< surface water level above which a location is marked as a river

    Biome(){}

    ~Biome(){ pftypes.clear(); }

    /// numPFTypes: returns the number of plant functional types in the biome
    int numPFTypes(){ return (int) pftypes.size(); }

    /// getPFType: get the ith plant functional type in the biome
    PFType * getPFType(int i){ return &pftypes[i]; }

    /// getAlpha: return the alpha canopy factor for PFT type i
    float getAlpha(int i){ return pftypes[i].alpha; }

    /// getMinIdealMoisture: return the start of the ideal moisture range for PFT type i
    float getMinIdealMoisture(int i){ return pftypes[i].wet.i1; }

    /// categoryNameLookup: get the category name corresponding to a category number
    void categoryNameLookup(int idx, std::string &catName);

    /**
     * @brief viability Calculate the viability score of a particular plant
     * @param pft       the plant functional type index of the plant
     * @param sunlight  avg. hours of sunlight per day in the given month
     * @param moisture  total water available in mm
     * @param temperature   avg. temperature over the given month
     * @param slope     incline in degrees where the plant is located
     * @return          viability in [-1, 1]
     */
    float viability(int pft, float sunlight, float moisture, float temperature, float slope);

    /**
     * @brief read  read plant functional type data for a biome from file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool read(const std::string &filename);

    /**
     * @brief write  write plant functional type data for a biome to a file
     * @param filename  name of file to be read
     * @return true if the file is found and has the correct format, false otherwise
     */
    bool write(const std::string &filename);
    bool read_dataimporter(data_importer::common_data &cdata);
    bool read_dataimporter(std::string cdata_fpath);

    GLfloat *get_species_colour(int specid);
};



#endif
