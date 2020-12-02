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


//
// TypeMap
//

#include "typemap.h"
#include "vecpnt.h"
#include "grass.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <QFileInfo>
#include <QLabel>

/*
Perceptually uniform colourmaps from:
http://peterkovesi.com/projects/colourmaps/
*/

using namespace std;

// Sonoma County Colours
float hardwood[] = {0.749f, 0.815f, 0.611f, 1.0f};
float conifer[] = {0.812f, 0.789f, 0.55f, 1.0f};
float mixed[] = {0.552f, 0.662f, 0.533f, 1.0f};
float riparian[] = {0.4f, 0.6f, 0.6f, 1.0f};
float nonnative[] = {0.7, 0.6, 0.4, 1.0f};
float sliver[] = {0.652f, 0.762f, 0.633f, 1.0f};
float shrubery[] = {0.882f, 0.843f, 0.713f, 1.0f};
float ripshrub[] = {0.509f, 0.67f, 0.584f, 1.0f};
float herb[] = {0.75f, 0.7f, 0.7f, 1.0f};
float herbwet[] = {0.623f, 0.741f, 0.825f, 1.0f};
float aquatic[] = {0.537f, 0.623f, 0.752f, 1.0f};
float salt[] = {0.727f, 0.763f, 0.534f, 1.0f};
float barrenland[] = {0.818f, 0.801f, 0.723f, 1.0f};
float agriculture[] = {0.894f, 0.913f, 0.639f, 1.0f};
float wet[] = {0.737f, 0.823f, 0.952f, 1.0f};
float developed[] = {0.5f, 0.4f, 0.5f, 1.0f};


// palette colours

float freecol[] = {0.755f, 0.645f, 0.538f, 1.0f};
float sparseshrub[] = {0.814f, 0.853f, 0.969f, 1.0f};
float sparsemed[] = {0.727f, 0.763f, 0.834f, 1.0f};
float sparsetall[] = {0.537f, 0.623f, 0.752f, 1.0f};
float denseshrub[] = {0.749f, 0.815f, 0.611f, 1.0f};
float densemed[] = {0.552f, 0.662f, 0.533f, 1.0f};
float densetall[] = {0.300f, 0.515f, 0.1f, 1.0f};


// default colours
float barren[] = {0.818f, 0.801f, 0.723f, 1.0f};            // 1
float ravine[] = {0.755f, 0.645f, 0.538f, 1.0f};            // 2
float canyon[] = {0.771f, 0.431f, 0.351f, 1.0f};            // 3
float grassland[] = {0.552f, 0.662f, 0.533f, 1.0f};         // 4
float pasture[] = {0.894f, 0.913f, 0.639f, 1.0f};           // 5
float foldhills[] = {0.727f, 0.763f, 0.534f, 1.0f};         // 6
float orchard[] = {0.749f, 0.815f, 0.611f, 1.0f};           // 7
float evergreenforest[] = {0.300f, 0.515f, 0.0f, 1.0f};     // 8
float otherforest[] = {0.552f, 0.662f, 0.533f, 1.0f};       // 9
float woodywetland[] = {0.509f, 0.67f, 0.584f, 1.0f};       // 10
float herbwetland[] = {0.623f, 0.741f, 0.825f, 1.0f};       // 11
float frillbank[] = {0.4f, 0.6f, 0.6f, 1.0f};               // 12
float shrub[] = {0.882f, 0.843f, 0.713f, 1.0f};             // 13
float flatinterest[] = {0.812f, 0.789f, 0.55f, 1.0f};      // 14
float water[] = {0.737f, 0.823f, 0.952f, 1.0f};            // 15
float special[] = {0.4f, 0.4f, 0.4f, 1.0f};                 // 16
float extra[] = {0.5f, 0.4f, 0.5f, 1.0f};                   // 17
float realwater[] = {0.537f, 0.623f, 0.752f, 1.0f};         // 18
float boulders[] = {0.671f, 0.331f, 0.221f, 1.0f};          // 19

TypeMap::TypeMap(TypeMapType purpose)
{
    tmap = new MemMap<int>;
    setPurpose(purpose);
}

TypeMap::TypeMap(int w, int h, TypeMapType purpose)
{
    tmap = new MemMap<int>;
    matchDim(w, h);
    setPurpose(purpose);
}

TypeMap::~TypeMap()
{
    delete tmap;
    for(int i = 0; i < (int) colmap.size(); i++)
        delete [] colmap[i];
    colmap.clear();
}

void TypeMap::clear()
{
    tmap->fill(0);
}

void TypeMap::initPaletteColTable()
{
    GLfloat *col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    numSamples = 7;

    colmap[0] = freecol;
    colmap[1] = sparseshrub;
    colmap[2] = sparsemed;
    colmap[3] = sparsetall;
    colmap[4] = denseshrub;
    colmap[5] = densemed;
    colmap[6] = densetall;
}

void TypeMap::initCategoryColTable()
{
    GLfloat *col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // entry 0 is reserved as transparent
    numSamples = 16;
    colmap[1] = hardwood;
    colmap[2] = conifer;
    colmap[3] = mixed;
    colmap[4] = riparian;
    colmap[5] = nonnative;
    colmap[6] = sliver;
    colmap[7] = shrubery;
    colmap[8] = ripshrub;
    colmap[9] = herb;
    colmap[10] = herbwet;
    colmap[11] = aquatic;
    colmap[12] = salt;
    colmap[13] = barrenland;
    colmap[14] = agriculture;
    colmap[15] = wet;
    colmap[16] = developed;
}

void TypeMap::initNaturalColTable()
{
    GLfloat *col;

    for(int i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // saturated prime colours and combos
    /*
     (colmap[1])[0] = 1.0f; // red
     (colmap[2])[1] = 1.0f; // green
     (colmap[3])[2] = 1.0f; // blue
     (colmap[4])[1] = 1.0f; (colmap[4])[2] = 1.0f; // cyan
     (colmap[5])[0] = 1.0f; (colmap[5])[1] = 1.0f; // yellow
     (colmap[6])[0] = 1.0f; (colmap[6])[2] = 1.0f; // magenta
     (colmap[7])[0] = 0.5f;  (colmap[7])[1] = 0.5f; (colmap[7])[2] = 0.5f; // grey
     (colmap[8])[1] = 0.5f; (colmap[8])[2] = 0.5f; // teal
     */

    numSamples = 20;

    // default
    //colmap[0] = c0;
    colmap[1] = barren;
    colmap[2] = ravine;
    colmap[3] = canyon;
    colmap[4] = grassland;
    colmap[5] = pasture;
    colmap[6] = foldhills;
    colmap[7] = orchard;
    colmap[8] = woodywetland;
    colmap[9] = otherforest;
    colmap[10] = woodywetland;
    colmap[11] = herbwetland;
    colmap[12] = frillbank;
    colmap[13] = shrub;
    colmap[14] = flatinterest;
    colmap[15] = water;
    colmap[16] = special;
    colmap[17] = extra;
    colmap[18] = realwater;
    colmap[19] = boulders;
}

void TypeMap::initPerceptualColTable(std::string colmapfile, int samples, float truncend)
{
    GLfloat *col;
    float r[256], g[256], b[256];
    ifstream infile;
    string valstr, line;
    int i, pos, step;

    if(samples < 3 || samples > 32)
        cerr << "Error: sampling of colour map must be in the range [3,32]" << endl;

    for(i = 0; i < 32; i++) // set all colours in table to black initially
    {
        col = new GLfloat[4];
        col[0] = col[1] = col[2] = 0.0f; col[3] = 1.0f;
        colmap.push_back(col);
    }

    // input is a csv file, with 256 RGB entries, one on each line
    // note that this is not robust to format errors in the input file
    infile.open((char *) colmapfile.c_str(), ios_base::in);

    if(infile.is_open())
    {
        i = 0;
        while(std::getline(infile, line))
        {
            std::size_t prev = 0, pos;

            // red component
            pos = line.find_first_of(",", prev);
            valstr = line.substr(prev, pos-prev);
            istringstream isr(valstr);
            isr >> r[i];
            prev = pos+1;

            // green component
            pos = line.find_first_of(",", prev);
            valstr = line.substr(prev, pos-prev);
            istringstream isg(valstr);
            isg >> g[i];
            prev = pos+1;

            // blue component
            valstr = line.substr(prev, std::string::npos);
            istringstream isb(valstr);
            isb >> b[i];

            i++;
        }
        infile.close();
    }

    // now sample the colour map at even intervals according to the number of samples
    // first and last samples map to the beginning and end of the scale
    step = (int) ((256.0f * truncend) / (float) (samples-1));
    pos = 0;
    for(i = 1; i <= samples; i++)
    {
        colmap[i][0] = (GLfloat) r[pos]; colmap[i][1] = (GLfloat) g[pos]; colmap[i][2] = (GLfloat) b[pos];
        pos += step;
    }
    numSamples = samples+1;
}

void TypeMap::clipRegion(Region &reg)
{
    if(reg.x0 < 0) reg.x0 = 0;
    if(reg.y0 < 0) reg.y0 = 0;
    if(reg.x1 > width()) reg.x1 = width();
    if(reg.y1 > height()) reg.y1 = height();
}

void TypeMap::matchDim(int w, int h)
{
    int mx, my;

    mx = tmap->width();
    my = tmap->height();

    // if dimensions don't match then reallocate
    if(w != mx || h != my)
    {
        dirtyreg = Region(0, 0, w, h);
        tmap->allocate(Region(0, 0, w, h));
        tmap->fill(0); // set to empty type
    }
}

void TypeMap::replaceMap(MemMap<int> * newmap)
{
    assert(tmap->width() == newmap->width());
    assert(tmap->height() == newmap->height());
    for (int y = 0; y < tmap->height(); y++)
        for (int x = 0; x < tmap->width(); x++)
            (* tmap)[y][x] = (* newmap)[y][x];
}

void TypeMap::bandCHMMap(MapFloat * chm, float mint, float maxt)
{
    int tp;
    float val;

    if(maxt > mint)
    {
        for (int x = 0; x < tmap->width(); x++)
            for (int y = 0; y < tmap->height(); y++)
            {
                val = chm->get(y, x);

                // discretise into ranges of height values
                if(val <= 0.0f) // transparent
                {
                    tp = 1;
                }
                else if(val <= mint) // black
                {
                    tp = 2;
                }
                else if(val >= maxt) // red
                {
                    tp = numSamples+2;
                }
                else // green range
                {
                    tp = (int) ((val-mint) / (maxt-mint+pluszero) * (numSamples-1))+2;
                }
                (* tmap)[y][x] = tp;
            }
    }
}

int TypeMap::load(const uts::string &filename, TypeMapType purpose)
{
    MemMap<mask_tag> mask;
    int tp, maxtp = 0; // mintp = 100;
    int width, height;
    ifstream infile;
    float val, maxval = 0.0f, range;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> width >> height;
        // cerr << "width = " << width << " height = " << height << endl;
        matchDim(width, height);
        // convert to internal type map format

        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                switch(purpose)
                {
                    case TypeMapType::EMPTY: // do nothing
                        break;
                    case TypeMapType::PAINT:
                        infile >> tp;
                        break;
                    case TypeMapType::CATEGORY:
                        infile >> tp;
                        tp++;
                        break;
                    case TypeMapType::SLOPE:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                         // discretise into ranges of slope values
                        range = 90.0f; // maximum slope is 90 degrees
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::WATER:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of water values
                        range = 100.0f;
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::SUNLIGHT:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of illumination values
                        range = 12.0f; // hours of sunlight
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::TEMPERATURE:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of temperature values

                        range = 20.0f; //10
                        // clamp values to range, temperature is bidrectional
                        if(val < -range) val = -range;
                        if(val > range) val = range;
                        tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-1))+1;

                        break;
                    case TypeMapType::CHM:
                        infile >> val;
                        if(val > maxval)
                            maxval = val;

                        // discretise into ranges of height values
                        range = 75.0f; // maximum tree height in feet
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::CDM:
                       infile >> val;
                       if(val > maxval)
                            maxval = val;

                        // discretise into ranges of illumination values
                        range = 1.0f; // maximum density
                        // clamp values to range
                        if(val < 0.0f) val = 0.0f;
                        if(val > range) val = range;
                        tp = (int) (val / (range+pluszero) * (numSamples-1))+1;
                        break;
                    case TypeMapType::SUITABILITY:
                        break;
                    default:
                        break;
                }
                (* tmap)[y][x] = tp;

                if(tp > maxtp)
                    maxtp = tp;
                /*
                if(tp < mintp)
                    mintp = tp;
                */

            }
        }
        infile.close();
        // cerr << "maxtp = " << maxtp << endl;
        // cerr << "mintp = " << mintp << endl;
    }
    else
    {
        cerr << "Error TypeMap::loadTxt: unable to open file" << filename << endl;
    }
    return maxtp;
}

bool TypeMap::loadCategoryImage(const uts::string &filename)
{
    int width, height;
    QImage img(QString::fromStdString(filename)); // load image from file

    QFileInfo check_file(QString::fromStdString(filename));

    if(!(check_file.exists() && check_file.isFile()))
        return false;

    // set internal storage dimensions
    width = img.width();
    height = img.height();
    matchDim(width, height);

    // convert to internal type map format
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++)
        {
            QColor col = img.pixelColor(x, y);
            int r, g, b;
            col.getRgb(&r, &g, &b); // all channels store the same info so just use red
            (* tmap)[y][x] = r - 100; // convert greyscale colour to category index
        }
    return true;
}

void TypeMap::setWater(MapFloat * wet, float wetthresh)
{
    int gx, gy;

    wet->getDim(gx, gy);
    for(int x = 0; x < gx; x++)
        for(int y = 0; y < gy; y++)
        {
            if(wet->get(x, y) >= wetthresh)
            {
                (* tmap)[y][x] = 0;
            }
        }
}

int TypeMap::convert(MapFloat * map, TypeMapType purpose, float range)
{
    int tp, maxtp = 0;
    int width, height;
    float val, maxval = 0.0f;

    map->getDim(width, height);
    matchDim(width, height);
    // convert to internal type map format
    int mincm, maxcm;
    mincm = 100; maxcm = -1;

    for(int x = 0; x < width; x++)
        for(int y = 0; y < height; y++)
        {
            tp = 0;
            switch(purpose)
            {
                case TypeMapType::EMPTY: // do nothing
                    break;
                case TypeMapType::PAINT: // do nothing
                    break;
                case TypeMapType::CATEGORY: // do nothing, since categories are integers not floats
                    break;
                case TypeMapType::SLOPE:
                    val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::WATER:
                    val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of water values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::SUNLIGHT:
                     val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                case TypeMapType::TEMPERATURE:
                    val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of temperature values
                    // clamp values to range, temperature is bidrectional
                    if(val < -range) val = -range;
                    if(val > range) val = range;
                    tp = (int) ((val+range) / (2.0f*range+pluszero) * (numSamples-2)) + 1;

                    break;
                case TypeMapType::CHM:
                     val = map->get(y, x);
                     if(val > maxval)
                        maxval = val;

                    // discretise into ranges of tree height values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    if(tp < mincm)
                        mincm = tp;
                    if(tp > maxcm)
                        maxcm = tp;
                    break;         
                case TypeMapType::CDM:
                    val = map->get(y, x);
                    if(val > maxval)
                        maxval = val;

                    // discretise into ranges of tree density values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    if(tp < mincm)
                        mincm = tp;
                    if(tp > maxcm)
                        maxcm = tp;
                    break;
                case TypeMapType::SUITABILITY:
                    val = map->get(x, y);
                    if(val > maxval)
                        maxval = val;
                    // discretise into ranges of illumination values
                    // clamp values to range
                    if(val < 0.0f) val = 0.0f;
                    if(val > range) val = range;
                    tp = (int) (val / (range+pluszero) * (numSamples-2)) + 1;
                    break;
                default:
                    break;
            }
            (* tmap)[y][x] = tp;

            if(tp > maxtp)
                maxtp = tp;
        }
    /*
    if(purpose == TypeMapType::CDM)
    {
        cerr << "Minimum colour value = " << mincm << endl;
        cerr << "Maxiumum colour value = " << maxcm << endl;
    }*/
    return maxtp;
}

void TypeMap::save(const uts::string &filename)
{
    ofstream outfile;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        outfile << width() << " " << height() << endl;

        // dimensions
        for (int x = 0; x < width(); x++)
            for (int y = 0; y < height(); y++)
            {
                outfile << get(x, y) << " ";
            }
        outfile.close();
    }
    else
    {
        cerr << "Error TypeMap::save: unable to write to file" << endl;
    }
}

void TypeMap::saveToPaintImage(const uts::string &filename)
{
    unsigned char * mask = new unsigned char[tmap->width()*tmap->height()];
    int i = 0;

    cerr << "paint file: " << filename << endl;

    //mask.resize(tmap->width()*tmap->height(), 0.0f);
    for (int x = 0; x < tmap->width(); x++)
        for (int y = 0; y < tmap->height(); y++)
        {
            switch((*tmap)[x][y]) // check order
            {
            case 0:
                mask[i] = 0;
                break;
            case 1: // sparse low
                mask[i] = 38;
                break;
            case 2: // sparse med
                mask[i] = 76;
                break;
            case 3: // sparse tall
                mask[i] = 115;
                break;
            case 4: // dense low
                mask[i] = 153;
                break;
            case 5: // dense med
                mask[i] = 191;
                break;
            case 6: // dense tall
                mask[i] = 230;
                break;
            default:
                mask[i] = 0;
            }
            i++;
        }

    // use QT image save functions
    QImage img;
    img = QImage(mask, tmap->width(), tmap->height(), QImage::Format_Grayscale8);
    img.save(QString::fromStdString(filename), "PNG", 100);
    delete [] mask;
}

void TypeMap::setPurpose(TypeMapType purpose)
{
    usage = purpose;
    switch(usage)
    {
        case TypeMapType::EMPTY:
            initPaletteColTable();
            break;
        case TypeMapType::PAINT:
            initPaletteColTable();
            break;
        case TypeMapType::CATEGORY:
            initCategoryColTable();
            break;
        case TypeMapType::SLOPE:
            initPerceptualColTable("../../../../data/colourmaps/linear_kry_5-95_c72_n256.csv", 10);
            break;
        case TypeMapType::WATER:
            initPerceptualColTable("../../../../data/colourmaps/linear_blue_95-50_c20_n256.csv", 10);
            break;
        case TypeMapType::SUNLIGHT:
            initPerceptualColTable("../../../../data/colourmaps/linear_kry_5-95_c72_n256.csv", 10);
            break;
        case TypeMapType::TEMPERATURE:
            initPerceptualColTable("../../../../data/colourmaps/diverging_bwr_55-98_c37_n256.csv", 10);
            break;
        case TypeMapType::CHM:
            // initPerceptualColTable("../colourmaps/linear_ternary-green_0-46_c42_n256.csv", 20);
            initPerceptualColTable("../../../../data/colourmaps/linear_green_5-95_c69_n256.csv", 20);
            // replace 0 with natural terrain colour
            colmap[1][0] = 0.7f; colmap[1][1] = 0.6f; colmap[1][2] = 0.5f; // transparent
            colmap[2][0] = 0.0f; colmap[2][1] = 0.0f; colmap[2][2] = 1.0f; // black
            colmap[numSamples+2][0] = 1.0f; colmap[numSamples+2][1] = 0.0f; colmap[numSamples+2][2] = 0.0f; // red
            break;
        case TypeMapType::CDM:
            initPerceptualColTable("../../../../data/colourmaps/linear_green_5-95_c69_n256.csv", 20);
            // replace 0 with natural terrain colour
            colmap[1][0] = 0.7f; colmap[1][1] = 0.6f; colmap[1][2] = 0.5f;
            break;
        case TypeMapType::SUITABILITY:
            initPerceptualColTable("../../../../data/colourmaps/linear_gow_60-85_c27_n256.csv", 20, 0.8f);
            // initPerceptualColTable("../../colourmaps/isoluminant_cgo_70_c39_n256.csv", 10);
            break;
        default:
            break;
    }
}

void TypeMap::resetType(int ind)
{
    // wipe all previous occurrences of ind
    #pragma omp parallel for
    for(int j = 0; j < tmap->height(); j++)
        for(int i = 0; i < tmap->width(); i++)
            if((* tmap)[j][i] == ind)
                (* tmap)[j][i] = 0;
    dirtyreg.x0 = 0; dirtyreg.y0 = 0;
    dirtyreg.x1 = tmap->width(); dirtyreg.y1 = tmap->height();
}

void TypeMap::setColour(int ind, GLfloat * col)
{
    for(int i = 0; i < 4; i++)
        colmap[ind][i] = col[i];
}
