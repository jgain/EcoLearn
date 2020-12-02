// synth.h: generate plant distributions interactively. Recoded from Harry Longs original generator.
// author: James Gain
// date: 1 August 2016

#include "synth.h"
#include "eco.h"
#include <radialDistribution/reproducer/file_utils.h>
#include <radialDistribution/reproducer/analysis_point.h>
#include <radialDistribution/reproducer/utils.h>
#include <radialDistribution/reproducer/distribution_factory.h>

#include <math.h>
#include <iostream>
#include <chrono>
#include <QStringList>
#include <thread>
#include <QDebug>


#include <cstdint>
#include <random>
using namespace std;

random_device rd;
/* The state must be seeded so that it is not everywhere zero. */
uint64_t s[2] = { (uint64_t(rd()) << 32) ^ (rd()),
    (uint64_t(rd()) << 32) ^ (rd()) };
uint64_t curRand;
uint8_t bit = 63;

uint64_t xorshift128plus(void) {
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23; // a
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
    return s[1] + y;
}


/// copy constructor
Distribution::Distribution(Distribution const& other)
{
    inputdir = other.inputdir;
    analyzeconfig = other.analyzeconfig;
    loaded_pair_correlations = other.loaded_pair_correlations;
    loaded_category_properties = other.loaded_category_properties;
    empty = other.empty;
}

bool Distribution::operator==(Distribution &other)
{
    return subsetEqual(other);
}

bool Distribution::equiv(const float &a, const float &b, int precision)
{
    return fabs(a - b) < max(a, 1.0f) * pow(0.1,precision);
}

bool Distribution::subsetEqual(Distribution & other, bool subset)
{
    // compare headers
    if(empty != other.empty) return false;
    if(analyzeconfig.r_min != other.analyzeconfig.r_min) return false;
    if(analyzeconfig.r_max != other.analyzeconfig.r_max) return false;
    if(analyzeconfig.r_diff != other.analyzeconfig.r_diff) return false;
    if(analyzeconfig.analysis_window_height != other.analyzeconfig.analysis_window_height) return false;
    if(analyzeconfig.analysis_window_width != other.analyzeconfig.analysis_window_width) return false;

    if(subset)
    {
        if((int) analyzeconfig.priority_sorted_category_ids.size() > (int) other.analyzeconfig.priority_sorted_category_ids.size())
            return false;
    }
    else
    {
        if((int) analyzeconfig.priority_sorted_category_ids.size() != (int) other.analyzeconfig.priority_sorted_category_ids.size())
            return false;
        for(int i = 0; i < (int) analyzeconfig.priority_sorted_category_ids.size(); i++)
            if(analyzeconfig.priority_sorted_category_ids[i] != other.analyzeconfig.priority_sorted_category_ids[i])
                return false;
    }

    // compare categories
    if(subset)
    {
        if((int) getCategories().size() > (int) other.getCategories().size())
            return false;
    }
    else
    {
        if((int) getCategories().size() != (int) other.getCategories().size())
            return false;
    }
    for(auto it: getCategories())
    {
        if(other.getCategories().find(it.first) == other.getCategories().end())
            return false;
        CategoryProperties cat = other.getCategories().find(it.first)->second;

        if(it.second.m_header.category_id != cat.m_header.category_id)
            return false;
        if(it.second.m_header.bin_size != cat.m_header.bin_size)
            return false;
        if(it.second.m_header.n_points != cat.m_header.n_points)
            return false;
        // priority not used so ignore
        // if(it.second.m_header.priority != cat.m_header.priority)
        //    return false;
        if(it.second.m_header.height_to_radius_multiplier != cat.m_header.height_to_radius_multiplier)
            return false;
        if(it.second.m_header.height_to_root_size_multiplier != cat.m_header.height_to_root_size_multiplier)
            return false;

/*
         cerr << "DEPENDENT IDS" << endl;
         for(auto in: it.second.m_header.category_dependent_ids)
             cerr << in << " ";

         cerr << "RADIUS: " << it.second.m_header.radius_properties.min << " " << it.second.m_header.radius_properties.max << " " << it.second.m_header.radius_properties.avg << " " <<  it.second.m_header.radius_properties.standard_dev << endl;
         cerr << "HEIGHT: " << it.second.m_header.height_properties.min << " " << it.second.m_header.height_properties.max << " " << it.second.m_header.height_properties.avg << " " <<  it.second.m_header.height_properties.standard_dev << endl;
         cerr << "ROOT: " << it.second.m_header.root_size_properties.min << " " << it.second.m_header.root_size_properties.max << " " << it.second.m_header.root_size_properties.avg << " " <<  it.second.m_header.root_size_properties.standard_dev << endl;
*/
    }

    // compare correlations
    if(!correlationsEqual(other, subset))
        return false;
    return true;

}

bool Distribution::correlationsEqual(Distribution &other, bool subset)
{
    if(subset)
    {
        if((int) loaded_pair_correlations.size() > (int) other.loaded_pair_correlations.size())
            return false;
    }
    else
    {
        if((int) loaded_pair_correlations.size() != (int) other.loaded_pair_correlations.size())
            return false;
    }

    for(auto it: loaded_pair_correlations)
    {
        if(other.getCorrelations().find(it.first) == other.getCorrelations().end())
            return false;

        const RadialDistribution corr = other.getCorrelations().find(it.first)->second;
        if(!equiv(it.second.getMinimum(), corr.getMinimum()))
            return false;
        if(!equiv(it.second.getMaximum(), corr.getMaximum()))
            return false;
        // if(!equiv(it.second.m_within_radius_distribution, corr.m_within_radius_distribution))
        //    return false;
        if(!equiv(it.second.m_less_than_half_shaded_distribution, corr.m_less_than_half_shaded_distribution))
            return false;
        if(!equiv(it.second.m_more_than_half_shaded_distribution, corr.m_more_than_half_shaded_distribution))
            return false;
        if(!equiv(it.second.m_fully_shaded_distribution, corr.m_fully_shaded_distribution))
            return false;
        if(!equiv(it.second.m_past_rmax_distribution, corr.m_past_rmax_distribution))
            return false;
        if((int) it.second.m_data.size() != (int) corr.m_data.size())
            return false;
        for(auto in: it.second.m_data)
        {
            float a = it.second.m_data.find(in.first)->second;
            float b = corr.m_data.find(in.first)->second;
            if(!equiv(a, b))
                return false;
        }
    }
    return true;
}

void Distribution::summaryDisplay()
{
    cerr << "CATEGORIES: ";
    for(auto it: analyzeconfig.priority_sorted_category_ids)
        cerr << it << " ";
    cerr << endl;
    for(auto it: loaded_category_properties)
    {
         cerr << it.first << ": ";
         cerr << "NUM PLANTS = " << it.second.m_header.n_points;
         cerr << ", HEIGHT = " << it.second.m_header.height_properties.avg << endl;
    }
    cerr << endl;
}

int Distribution::getNumMajorPlants()
{
    int numplnts = 0;

    // summaryDisplay();

    if(!empty)
    {
        for(auto it: loaded_category_properties)
        {
            if(it.second.m_header.height_properties.avg > 1000.0f)
                numplnts += it.second.m_header.n_points;
        }
    }
    return numplnts;
}

void Distribution::display()
{

    cerr << "EMPTY = " << empty << endl;
    cerr << inputdir.toStdString() << endl;
    cerr << analyzeconfig.r_min << " " << analyzeconfig.r_max << " " << analyzeconfig.r_diff << endl;
    cerr << analyzeconfig.analysis_window_width << " " << analyzeconfig.analysis_window_height << endl;
    for(auto it: analyzeconfig.priority_sorted_category_ids)
        cerr << it << " ";
    cerr << endl;

    cerr << "NUM CORRELATIONS = " << (int) loaded_pair_correlations.size() << endl;
    for(auto it: loaded_pair_correlations)
    {
        cerr << "(" << it.first.first << ", " << it.first.second << ") ";
        cerr << "min " << it.second.getMinimum() << " max " << it.second.getMaximum() << " ";
        cerr << " outrad " << it.second.m_past_rmax_distribution << endl;
        cerr << "( " << it.second.m_header.reference_id << ", " << it.second.m_header.destination_id << ") " << it.second.m_header.requires_optimization << endl;
        cerr << (int) it.second.m_data.size() << ": ";
        for(auto in: it.second.m_data)
            cerr << in.second << " ";
        cerr << endl;
    }

    cerr << "NUM CATEGORIES = " << (int) loaded_category_properties.size() << endl;
    for(auto it: loaded_category_properties)
    {
         cerr << it.first << ": " << it.second.m_header.category_id << " " << it.second.m_header.bin_size << " ";
         cerr << "n points = " << it.second.m_header.n_points;
         cerr << " priority = " << it.second.m_header.priority << " htor = " << it.second.m_header.height_to_radius_multiplier << " " << endl;
         cerr << it.second.m_header.height_to_root_size_multiplier << endl;
         cerr << "DEPENDENT IDS" << endl;
         for(auto in: it.second.m_header.category_dependent_ids)
              cerr << in << " ";
         cerr << endl;

         cerr << "RADIUS: " << it.second.m_header.radius_properties.min << " " << it.second.m_header.radius_properties.max << " " << it.second.m_header.radius_properties.avg << " " <<  it.second.m_header.radius_properties.standard_dev << endl;
         cerr << "HEIGHT: " << it.second.m_header.height_properties.min << " " << it.second.m_header.height_properties.max << " " << it.second.m_header.height_properties.avg << " " <<  it.second.m_header.height_properties.standard_dev << endl;
         cerr << "ROOT: " << it.second.m_header.root_size_properties.min << " " << it.second.m_header.root_size_properties.max << " " << it.second.m_header.root_size_properties.avg << " " <<  it.second.m_header.root_size_properties.standard_dev << endl;
    }
    cerr << endl;
}

bool Distribution::write(string writedir)
{
    // create directory structure
    QString base_directory = QString::fromStdString(writedir);
    // Init the base directory
    if(!base_directory.endsWith("/"))
        base_directory.append("/");
    QString m_base_dir = base_directory;
    QString m_radial_distribution_dir, m_category_properties_dir, m_csv_dir;

    if(!FileUtils::init_directory_structure(m_base_dir, m_radial_distribution_dir, m_category_properties_dir, m_csv_dir))
    {
        std::cerr << "An error occured whilst creating the directory structure..." << std::endl;
        return false;
    }

    // analysis configuration file
    QString generic_filename(FileUtils::_CONFIGURATION_FILENAME);
    // Write bin file
    {
        QString output_filename(m_base_dir);
        output_filename.append(generic_filename);
        analyzeconfig.write(output_filename.toStdString());
    }
    // Write CSV file
    {
        QString output_filename(m_csv_dir);
        output_filename.append(generic_filename).append(".csv");
        analyzeconfig.writeToCSV(output_filename.toStdString());
    }

    // category properties
    QString m_output_bin_dir = m_category_properties_dir;
    QString m_output_hr_dir = m_csv_dir;
    for(auto cprop: loaded_category_properties)
    {
        CategoryProperties category_properties = cprop.second;
        int category_id(category_properties.m_header.category_id);

        QString filename("category_");
        filename.append(QString::number(category_id));

        // category_properties.m_header.category_dependent_ids = m_category_dependencies;

        // Write category properties file
        {
            QString output_filename(m_output_bin_dir);
            output_filename.append(filename).append(FileUtils::_CATEGORY_PROPERTIES_EXT);
            category_properties.write(output_filename.toStdString());
        }
        // human readable
        {
            QString output_filename(m_output_hr_dir);
            output_filename.append(filename).append("_properties").append(".csv");
            category_properties.writeToCSV(output_filename.toStdString());
        }
    }

    m_output_bin_dir = m_radial_distribution_dir;
    m_output_hr_dir = m_csv_dir;
    // radial distributions
    for(auto lpair: loaded_pair_correlations)
    {
        RadialDistribution radial_distribution = lpair.second;
        int reference_category_id(radial_distribution.m_header.reference_id);
        int target_category_id(radial_distribution.m_header.destination_id);
        // int reference_category_id(lpair.first.first);
        // int target_category_id(lpair.first.second);
        // cerr << "ref:targ " << reference_category_id << ": " << target_category_id << " in lpair " << lpair.first.first << ": " << lpair.first.second << endl;

        // std::pair<int,int> pair(radial_distribution.m_header.reference_id, radial_distribution.m_header.destination_id);
        // loaded_pair_correlations.insert(std::pair<std::pair<int,int>, RadialDistribution>(pair, radial_distribution));

        QString filename("category_");
        filename.append(QString::number(reference_category_id)).append("_and_").append(QString::number(target_category_id));

        // check for empty radial distribution
        // radial_distribution.printToConsole();

        // Write rad file
        {
            QString output_filename(m_output_bin_dir);
            output_filename.append(filename).append(FileUtils::_RADIAL_DISTRIBUTION_EXT);
            radial_distribution.write(output_filename.toStdString());
        }


        // Write CSV file
        {
            QString output_filename(m_output_hr_dir);
            output_filename.append(filename).append("_pair_correlation").append(".csv");
            radial_distribution.writeToCSV(output_filename.toStdString());
        }

    }

    std::string filename(m_base_dir.append(FileUtils::_TIMESTAMP_FILENAME).toStdString());
    unsigned long m_timestamp( std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());

    // Write timestamp file
    std::ofstream file;
    file.open(filename);
    if(file.is_open())
        file << m_timestamp << "\n";
    file.close();

    return true;
}

bool Distribution::read(string rootdir, Terrain * ter, CategorySizes & catsizes)
{
    // cerr << "rootdir is " << rootdir << endl;
    inputdir = rootdir.c_str();

    // Load various aspects of a distribution from the source directory, including histograms within and between categories and all category metadata
    if(!FileUtils::check_directory_structure(inputdir))
    {
        cerr << "Distribution::read: Incorrect directory structure in " << inputdir.toStdString() << endl;
        return false;
    }

    analyzeconfig = AnalysisConfiguration(FileUtils::get_configuration_file(inputdir).toStdString());

    // Load category properties
    std::vector<QString> category_properties_files(FileUtils::get_category_properties_files(inputdir));
    for(QString category_property_file : category_properties_files)
    {
        CategoryProperties category_property(category_property_file.toStdString());
        loaded_category_properties.insert(std::pair<int, CategoryProperties>(category_property.m_header.category_id, category_property));
    }

#ifdef VALIDATE
    if(rootdir == string("/home/james/Desktop/medResample/7"))
        cerr << "Priority: ";
#endif
    // Check all necessary category properties files were there
    for(auto it(analyzeconfig.priority_sorted_category_ids.begin()); it != analyzeconfig.priority_sorted_category_ids.end(); it++)
    {

        if(loaded_category_properties.find(*it) == loaded_category_properties.end())
        {
            std::cerr << "Distribution::read: Missing category properties file for category " << *it << std::endl;
            return false;
        }
#ifdef VALIDATE
        if(rootdir == string("/home/james/Desktop/medResample/7"))
            cerr << (* it) << " ";
#endif
        CategoryProperties &prop = loaded_category_properties.find(*it)->second;
        // update size header info
        // cerr << "hght avg = " << prop.m_header.height_properties.avg << " sdv = " << prop.m_header.height_properties.standard_dev << endl;
        // cerr << "hght min = " << prop.m_header.height_properties.min << " max = " << prop.m_header.height_properties.max << endl;
        bool ignore = (prop.m_header.height_properties.avg <= 2); // && (prop.m_header.height_properties.standard_dev < pluszero);
        if(!ignore) // has at least one element
        {
            if(catsizes.find(* it) == catsizes.end()) // category not found so add it
            {
                AggregateSizeProperties aggsize;
                aggsize.n = 1; aggsize.totalHeight = (float) prop.m_header.height_properties.avg;
                aggsize.heightToCanopyRadius = prop.m_header.height_to_radius_multiplier;
#ifdef VALIDATE
                if(rootdir == string("/home/james/Desktop/medResample/7"))
                {
                cerr << "adding cateogory " << (* it) << " with CanopyRatio " << aggsize.heightToCanopyRadius << endl;
                cerr << "standard deviation = " << prop.m_header.height_properties.standard_dev << endl;
                }
#endif
                catsizes[(* it)] = aggsize;
            }
            else // category exists so update average plant size
            {
                catsizes[(* it)].n += 1;
                catsizes[(* it)].totalHeight += (float) prop.m_header.height_properties.avg;
                catsizes[(* it)].heightToCanopyRadius += prop.m_header.height_to_radius_multiplier;
            }
        }
    }
#ifdef VALIDATE
    if(rootdir == string("/home/james/Desktop/medResample/7"))
        cerr << endl;
#endif
    // create pair correlations
    std::vector<std::pair<int,int> > required_pair_correlations;
    for(auto reference_category(analyzeconfig.priority_sorted_category_ids.begin()); reference_category != analyzeconfig.priority_sorted_category_ids.end(); reference_category++)
    {

        for(auto target_category(std::find(analyzeconfig.priority_sorted_category_ids.begin(), analyzeconfig.priority_sorted_category_ids.end(), *reference_category));
                            target_category != analyzeconfig.priority_sorted_category_ids.end(); target_category++)
        {
            required_pair_correlations.push_back(std::pair<int,int>(*reference_category, *target_category));
#ifdef VALIDATE
            if(rootdir == string("/home/james/Desktop/medResample/7"))
                cerr << *reference_category << ": " << *target_category << endl;
#endif
        }
    }

    // Load radial distributions
    std::vector<QString> radial_distribution_files(FileUtils::get_radial_distribution_files(inputdir));
    for(QString radial_distribution_file : radial_distribution_files)
    {
        RadialDistribution radial_distribution(radial_distribution_file.toStdString());
//        if(rootdir == string("/home/james/Desktop/medResample/7"))
//            radial_distribution.printToConsole();

        std::pair<int,int> pair(radial_distribution.m_header.reference_id, radial_distribution.m_header.destination_id);
        loaded_pair_correlations.insert(std::pair<std::pair<int,int>, RadialDistribution>(pair, radial_distribution));
    }

    // Check all necessary pair correlations were loaded
    for(auto it(required_pair_correlations.begin()); it != required_pair_correlations.end(); it++)
    {
        if(loaded_pair_correlations.find(*it) == loaded_pair_correlations.end())
        {
            std::cerr << "Distribution::read: Missing pair correlation for category pair [" << it->first << "," << it->second << "]" << std::endl;
            // exit(1);
            // create empty pair correlation to compensate
        }
    }

    empty = false;
    return true;
}


/// MultiDistributionReproducer

#define SPATIAL_HASHMAP_CELL_WIDTH 10
#define SPATIAL_HASHMAP_CELL_HEIGHT 10
/////////////////////////////////////////////////////////////////////////////////
MultiDistributionReproducer::MultiDistributionReproducer(ConditionsMap * cmap,
                                                           ReproductionConfiguration reproduction_settings, AnalysisConfiguration analysis_configuration,
                                                           GeneratedPointsProperties * outGeneratedPointProperties) :
    m_cmap(cmap), m_reproduction_configuration(reproduction_settings),
    m_spatial_point_storage(m_reproduction_configuration.width, m_reproduction_configuration.height), m_analysis_configuration(analysis_configuration),
    m_generated_points_properties(outGeneratedPointProperties), m_dice_roller(0,RAND_MAX)
{
    // Initialize the point properties tracker
    SizeProperties size_properties;
    // placeholder for now. Will need to create a global categories hierarchy TO DO
    for(auto it(cmap->getSamplingDatabase()->getCategorySizes().begin()); it != cmap->getSamplingDatabase()->getCategorySizes().end(); it++)
    {
        size_properties.heightToCanopyRadius = it->second.heightToCanopyRadius;
        // std::cerr << "CR " << it->first << " " << size_properties.heightToCanopyRadius << std::endl;
        size_properties.minHeight = -1;
        size_properties.maxHeight = -1;
        m_generated_points_properties->emplace(it->first, size_properties);
    }
    movecycles = 2;
}

void MultiDistributionReproducer::setRepConfig(ReproductionConfiguration &rep)
{
    m_reproduction_configuration.height = rep.height;
    m_reproduction_configuration.width = rep.width;
}


void MultiDistributionReproducer::setSubSynthArea(int startx, int starty, int extentx, int extenty)
{
    subsynth.startx = startx;
    subsynth.starty = starty;
    subsynth.extentx = extentx;
    subsynth.extenty = extenty;
}


void MultiDistributionReproducer::clearPoints(std::vector<AnalysisPoint> * toclear)
{
    for(auto pnt: (* toclear))
    {
          m_taken_points.removePoint(pnt.getCenter());
          m_spatial_point_storage.removePoint(pnt);
    }

    m_placed_categories.clear();
}

bool MultiDistributionReproducer::testCleared(int startx, int starty, int extentx, int extenty)
{
    bool isempty = true;
    int takencount = 0;
    // look for points in the identified area in m_taken_points and m_spatial_point_storage
    // should be empty

    // check if taken point falls in region
    for(auto msp = m_taken_points.begin(); msp != m_taken_points.end(); msp++)
    {
        if(msp->first.first >= startx && msp->first.first < startx+extentx && msp->first.second >= starty && msp->first.second < starty+extenty)
        {
            // in region so count
            takencount++;
            isempty = false;
        }
    }

    if(!isempty)
        cerr << "Error MultiDistributionReproducer::testCleared: " << takencount << " points remain in cleared area" << endl;
    return isempty;
}

void MultiDistributionReproducer::profileRndGen(int count)
{
    Timer t;

    t.start();
    for(int i = 0; i < count; i++)
    {
        // getRandomPoint(15002);
    }
    t.stop();
    cerr << "Time for " << count << " random point generations = " << t.peek() << "s" << endl;
}

AnalysisPoint MultiDistributionReproducer::getRandomStep(const AnalysisPoint & refpnt, int cid)
{
    int x, y;
    int dx, dy, dh;
    QPoint position;
    bool fin = false;
    int max_attempts = 100, attempts = 0;
    int maxstep = 6; // * binsize;

    while(!fin)
    {
        dx = rand() % maxstep; dy = rand() % maxstep;
        // x = refpnt.x()+dx; y = refpnt.y()+dy;
        if(!m_cmap->lookupSynthDistrib(x, y)->isEmpty()) // hit an empty part so skip
        {
            position = QPoint(x, y);
            fin = !m_taken_points.containsPoint(position) && (m_cmap->lookupSynthDistrib(x, y)->getCategories().find(cid) != m_cmap->lookupSynthDistrib(x, y)->getCategories().end());
        }

        attempts++; rndcount++;
        if(attempts > max_attempts)
            fin = true;
    }

    if(attempts > max_attempts)
    {
        cerr << "Error getRandomStep: too many attempts at placement" << endl;
        // exit(0);
    }

    // now look up height range in category properties
    CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(x, y)->getCategories().find(cid)->second;

    // set height randomly in range, with uniform distribution
    int minrange = m_active_cp.m_header.height_properties.min;
    int maxrange = m_active_cp.m_header.height_properties.max;

    int range = maxrange-minrange;
    int height;

    if(range == 0) // to avoid the error that results from div by zero
        height = minrange;
    else
    {
        // dh = ;
        // height = refpnt.height() + dh; minrange + m_dice_roller.generate() % range; // calculate small increment in +- 10% range
    }

    return AnalysisPoint(m_active_cp.m_header.category_id, position, height*m_active_cp.m_header.height_to_radius_multiplier,
                         height*m_active_cp.m_header.height_to_root_size_multiplier, height);
}

AnalysisPoint MultiDistributionReproducer::getRandomPoint(int cid)
{
    // replaces PointFactory functions, which require the category to be set before the position is determined
    // This is not possible under the ConditionMap approach because categories are looked up based on position

    int x, y;
    // int m_width = m_reproduction_configuration.width;
    // int m_height = m_reproduction_configuration.height;

    // get random position
    QPoint position;
    bool fin = false;
    int max_attempts = 10000, attempts = 0;

    while(!fin)
    {
        // position = QPoint(subsynth.startx+m_dice_roller.generate()%(subsynth.extentx-1), subsynth.starty+m_dice_roller.generate()%(subsynth.extenty-1));
        position = QPoint(subsynth.startx+rand()%(subsynth.extentx-1), subsynth.starty+rand()%(subsynth.extenty-1));
        // position = QPoint(subsynth.startx+xorshift128plus()%(subsynth.extentx-1), subsynth.starty+xorshift128plus()%(subsynth.extenty-1));

        x = position.x(); y = position.y();
        //cerr << "pos = " << x << ", " << y << endl;

        if(!m_cmap->lookupSynthDistrib(x, y)->isEmpty()) // hit an empty part so skip
        {
            fin = !m_taken_points.containsPoint(position) && (m_cmap->lookupSynthDistrib(x, y)->getCategories().find(cid) != m_cmap->lookupSynthDistrib(x, y)->getCategories().end());
        }

        attempts++; rndcount++;
        if(attempts > max_attempts)
            fin = true;
    }

    if(attempts > max_attempts)
    {
        cerr << "Too many attempts at placement" << endl;
        // exit(0);
    }


    // now look up height range in category properties
    //if(m_cmap->lookupSynthDistrib(x, y)->getCategories().find(cid) == m_cmap->lookupSynthDistrib(x, y)->getCategories().end())
    //    cerr << "CID " << cid << " not found in categories" << endl;
    CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(x, y)->getCategories().find(cid)->second;

    // set height randomly in range, with uniform distribution
    int minrange = m_active_cp.m_header.height_properties.min;
    int maxrange = m_active_cp.m_header.height_properties.max;

    int range = maxrange-minrange;
    int height;

    /*
    if(range == 0) // to avoid the error that results from div by zero
        height = minrange;
    else
        // height = minrange + m_dice_roller.generate() % range;
        height = minrange + rand() % range;
        // height = minrange + xorshift128plus() % range;

    // set to average
    // height = m_active_cp.m_header.height_properties.avg;
    */

    if(range == 0) // to avoid the error that results from div by zero
    {
        height = minrange;
    }
    else // generate height according to mean and standard deviation, using a normal distribution
    {
        std::normal_distribution<double> distribution(m_active_cp.m_header.height_properties.avg, m_active_cp.m_header.height_properties.standard_dev);
        height = distribution(generator);

        // clamp to range
        if(height < minrange)
            height = minrange;
        if(height > maxrange)
            height = maxrange;
        // cerr << "avg = " << m_active_cp.m_header.height_properties.avg << " dev = " << m_active_cp.m_header.height_properties.standard_dev << " res = " << height << endl;
        // cerr << "min = " << m_active_cp.m_header.height_properties.min << " max = " << m_active_cp.m_header.height_properties.max << endl;
    }

    return AnalysisPoint(m_active_cp.m_header.category_id, position, height*m_active_cp.m_header.height_to_radius_multiplier,
                         height*m_active_cp.m_header.height_to_root_size_multiplier, height);

    // The point_factory seems to use a histogram based approach for height but it may not be correctly hooked up on the analysis side
    // Need to ask HL about this
}

/*
void MultiDistributionReproducer::reportTiming()
{
    cerr << "Density Initialisation = " << inittime << "s" << endl;
    cerr << "Moves = " << movetime << "s" << endl;
    cerr << "Random Gen Calls = " << rndcount << endl;

    cerr << "Area based time = " << t1time << "s" << endl;
    cerr << "Random Point time = " << t2time << "s" << endl;
    cerr << "Accel point validity time = " << t3time << "s" << endl;
    cerr << "Calc strength time = " << t4time << "s" << endl;
    cerr << "Add Dest time = " << t5time << "s" << endl;
}*/

void MultiDistributionReproducer::startPointGeneration(GeneratedPoints & genpnts, int numcycles)
{
    // std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    // std::chrono::high_resolution_clock::time_point end_time;

    movecycles = numcycles;

    int total_point_count(0);
#ifdef VALIDATE
    strcnt = 0;
    strsum = 0.0f;
    xsect = 0;
    zerocnt = 0;
    quickcnt = 0;
#endif

    // m_cmap->summaryDisplayDescriptor(0,0);
    // m_cmap->summaryDisplayDescriptor(99700, 84200);
    // m_cmap->summaryDisplayDescriptor(50000, 50000);

    // for(auto it(m_analysis_configuration.priority_sorted_category_ids.begin()); it != m_analysis_configuration.priority_sorted_category_ids.end(); it++)
    for(auto it: m_cmap->getSamplingDatabase()->getGlobalPriority())
    {
        //if(it >= 18000)
        //    std::cerr << "Inserting points for category " << it << "..." << std::endl;

        generate_points(it);

        if(m_active_category_points.size() > 0)
        {
            genpnts[m_active_category_points.at(0).getCategoryId()] = m_active_category_points;
            total_point_count += m_active_category_points.size();
        }

        m_active_category_points.clear();
    }
#ifdef VALIDATE
    if(strcnt > 0)
    {
        cerr << strcnt << " avg synthqual = " << strsum / (float) strcnt << endl;
        cerr << "number of trunk intersects = " << xsect << " / " << strcnt << "; zero strength intersects = " << zerocnt << " / " << xsect;
        cerr << "; quick check passes = " << quickcnt << " / " << zerocnt << endl;
    }
#endif


    // end_time = std::chrono::high_resolution_clock::now();

    // auto duration (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

 //   std::cout << "Simulation over! " << std::endl
//              << "\t Source density: " << (((float)m_radial_distribution.m_properties.n_reference_points)/m_radial_distribution.m_properties.analysed_area) << std::endl
//              << "\t Output density: " << ((float)m_destination_points->size())/(m_settings.width*m_settings.height) << std::endl
//              << "\t Calculation time: " << duration << " ms." << std::endl
//              << "\t Total points: " << total_point_count << std::endl;
}

void MultiDistributionReproducer::generate_points(int cid)
{
    Timer t;
    int sx, sy, ex, ey;

    // divide subsynth area up into local cells
    m_cmap->getCoord(subsynth.startx, subsynth.starty, sx, sy);
    m_cmap->getCoord(subsynth.startx+subsynth.extentx, subsynth.starty+subsynth.extenty, ex, ey);

    // locsynth = subsynth;
    /*
    int cellextent = (int) (subsynth.extentx / (ex-sx));
    cerr << "subsynth" << endl;
    cerr << "start = " << subsynth.startx << ", " << subsynth.starty << " extent = " << subsynth.extentx << ", " << subsynth.extenty << endl;
    cerr << "map start = " << sx << ", " << sy << " map extent = " << ex-sx << ", " << ey-sy << endl;
    cerr << "cell extent = " << cellextent << endl;
*/
    // t.start();
    matching_density_initialize(cid);
    // t.stop(); inittime += t.peek();
    // t.start();
    generate_points_through_random_moves(cid);
    // t.stop(); movetime += t.peek();

    /*
    for(int x = 0; x < sx-ex; x++)
        for(int y = 0; y < sy-ey; y++)
        {
            // map to local start and extent
            locsynth.startx = subsynth.startx + x * cellextent;
            locsynth.starty = subsynth.starty + y * cellextent;
            locsynth.extentx = cellextent;
            locsynth.extenty = cellextent;

            // t.start();
            matching_density_initialize(cid);
            // t.stop(); inittime += t.peek();
            // t.start();
            generate_points_through_random_moves(cid);
            // t.stop(); movetime += t.peek();
        }
    */
}

int MultiDistributionReproducer::canopyIntersect(const AnalysisPoint & refpnt)
{
    std::vector<AnalysisPoint> reachpnts(m_spatial_point_storage.getPossibleReachablePoints(refpnt, m_analysis_configuration.r_max));
    int xsectcnt = 0;

    QLineF line;

    line.setP1(refpnt.getCenter());

    std::vector<AnalysisPoint> existing_points;
    // select out only those points that have the same distribution, otherwise creates issues with order of plant generation
    for(auto it: reachpnts)
    {
        // if(m_cmap->equivDescriptor(m_cmap->getSynthDescriptor(refpnt.getCenter().x(), refpnt.getCenter().y()),
        //                            m_cmap->getSynthDescriptor(it.getCenter().x(), it.getCenter().y())))
            existing_points.push_back(it);
    }

    const float refpntrad = refpnt.getRadius();

    for(const AnalysisPoint & existpnt : existing_points)
    {

            line.setP2(existpnt.getCenter());
            // const float existpntrad = existpnt.getRadius();
            float distance = line.length();

            if(distance < refpntrad)
                xsectcnt++;
    }
#ifdef VALIDATE
    if(xsectcnt > 0) // now test if strength is zero
    {
        bool discard;
        if(calculate_strength(refpnt) < pluszero)
        {
            zerocnt++;
            bool needs_check;
            accelerated_point_validity_check(refpnt, needs_check);
            if(!needs_check)
                quickcnt++;
        }
    }
#endif
    return xsectcnt;
}

void MultiDistributionReproducer::cacheReachablePoints(const AnalysisPoint & refpnt, std::vector<AnalysisPoint> & reachpnts)
{
    reachpnts = m_spatial_point_storage.getPossibleReachablePoints(refpnt, m_analysis_configuration.r_max);
}

bool MultiDistributionReproducer::canopyIntersectFree(const AnalysisPoint & refpnt, std::vector<AnalysisPoint> & reachpnts)
{
    QLineF line;
    line.setP1(refpnt.getCenter());
    const float refpntrad = refpnt.getRadius();

    for(const AnalysisPoint & existpnt : reachpnts)
    {
        line.setP2(existpnt.getCenter());
        float distance = line.length();

        if(distance < refpntrad)
            return false;
    }
    return true;
}

void MultiDistributionReproducer::generate_points_through_random_moves(int cid)
{
    DiceRoller dice_roller(0,1000);

    int n_accepted_moves(0), n_refused_moves(0);

 #ifdef VALIDATE
    radcnt = 0;
    radsum = 0;
    avgsum = 0;
    devsum = 0;
    minsum = 0;
    maxsum = 0;
 #endif

    for(int selected_point_index(0); selected_point_index < (int) m_active_category_points.size(); selected_point_index++)
    {
        // std::cerr << "s = " << selected_point_index << " / " << m_active_category_points.size() << std::endl;
        AnalysisPoint & selected_point(m_active_category_points.at(selected_point_index));

        /*
        if(selected_point_index == 0)
        {
            if(!m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->isEmpty())
            {
                m_cmap->summaryDisplayDescriptor(selected_point.getCenter().x(), selected_point.getCenter().y());
                m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->summaryDisplay();
            }
        }*/
#ifdef VALIDATE
        if(selected_point_index == 0)
        {
            // cerr << "CID " << cid << " distribution" << endl;
            // m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->display();
        }
#endif
        // double source_point_strength = std::max(0.000001, calculate_strength(selected_point) );
        float source_point_strength = calculate_strength(selected_point);

        AnalysisPoint highest_scoring_dest_point;
        float highest_strength(-1);

        for(int i(0); i < movecycles; i++)
        {
            /*********************
             * DESTINATION POINT *
             *********************/
            // AnalysisPoint random_dest_point = getRandomPoint(selected_point.getHeight());
            AnalysisPoint random_dest_point = getRandomPoint(cid); // vary both position and height, necessary because species height ranges change with position

            // will need to check validity of new position, move only within current category boundaries? TO DO

            // Calculate strength
            float destination_point_strength = calculate_strength(random_dest_point);
            // std::cout << "Random destination point strength: " << destination_point_strength << std::endl;

            if(destination_point_strength > highest_strength)
            {
                highest_scoring_dest_point = random_dest_point;
                highest_strength = destination_point_strength;
            }
        }

        float acceptance_ratio(highest_strength/source_point_strength);
        // std::cerr << "Acceptance ratio: " << acceptance_ratio << std::endl;

        if(ProbabilisticUtils::returnTrueWithProbability(acceptance_ratio, dice_roller))
        //if(ProbabilisticUtils::returnTrueWithProbabilitySupplied(acceptance_ratio, rand()%1000))
        {
            n_accepted_moves++;
            move_point(selected_point, highest_scoring_dest_point);
#ifdef VALIDATE
            strsum -= source_point_strength;
            strsum += highest_strength;

            radsum += highest_scoring_dest_point.getHeight();
            CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->getCategories().find(cid)->second;
            avgsum +=  m_active_cp.m_header.height_properties.avg;
            devsum +=  m_active_cp.m_header.height_properties.standard_dev;
            minsum +=  m_active_cp.m_header.height_properties.min;
            maxsum +=  m_active_cp.m_header.height_properties.max;
            radcnt++;
#endif
        }
        else
        {
            n_refused_moves++;
#ifdef VALIDATE
            radsum += selected_point.getHeight();
            CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->getCategories().find(cid)->second;
            avgsum +=  m_active_cp.m_header.height_properties.avg;
            devsum +=  m_active_cp.m_header.height_properties.standard_dev;
            minsum +=  m_active_cp.m_header.height_properties.min;
            maxsum +=  m_active_cp.m_header.height_properties.max;
            radcnt++;
#endif
        }

        // std::cerr << ((((float)points_processed)/m_active_category_points.size()) * 100) << "%" << std::endl;
    }

#ifdef VALIDATE
    if(radcnt > 0)
    {
        cerr << "CID " << cid << " : num plants = " << radcnt << " avg height = " << (float) radsum / (float) radcnt << " vs. expected height = " << (float) minsum / (float) radcnt <<  " " << (float) avgsum / (float) radcnt << " " << (float) maxsum / (float) radcnt << endl;
        cerr << "      avg stdev = " << (float) devsum / (float) radcnt << endl;
    }
#endif

    // std::cerr << "Accepted moves: " << n_accepted_moves << " | Refused moves: " << n_refused_moves << std::endl;
}

int MultiDistributionReproducer::areaBasedPointCount(int cid)
{
    int a_points = 0, num_areas = 0, num_empty = 0;
    int sx, sy, ex, ey;
    Distribution * distrib;

    // over region in question and for particular category, find point proportionality using area based approach
    // m_cmap->getCoord(m_reproduction_configuration.width, m_reproduction_configuration.height, ex, ey);

    // TO DO - remove iteration since this should only be a single cell
    m_cmap->getCoord(subsynth.startx, subsynth.starty, sx, sy);
    m_cmap->getCoord(subsynth.startx+subsynth.extentx, subsynth.starty+subsynth.extenty, ex, ey);

    for(int x = sx; x < ex; x++)
        for(int y = sy; y < ey; y++)
        {
            // distrib = m_cmap->lookupDirectDistribFast(x, y);
            distrib = m_cmap->lookupDirectDistrib(x, y);
            if(!distrib->isEmpty())
            {
                auto cat = distrib->getCategories().find(cid);
                if(cat != distrib->getCategories().end())
                {
                    a_points += cat->second.m_header.n_points;

                    //if(x == 0 && y == 0 && cid == 17002)
                    //if(cid == 17002 &&  m_cmap->getDescriptor(x, y).age == 50)
                    //    cerr << cat->second.m_header.n_points << endl;
                }
            }
            num_areas++;
        }

/*
    if(cid == 18000)
    {
        cerr << "CID: " << cid << endl;
        cerr << "a_point = " << a_points << endl;
        cerr << "subsynth extent = " << subsynth.extentx << " X " << subsynth.extenty << endl;
        cerr << "window extent = " << m_analysis_configuration.analysis_window_width << " X " << m_analysis_configuration.analysis_window_height << endl;
        cerr << "analysis config = " << m_analysis_configuration.r_min << ", " << m_analysis_configuration.r_max << ", " << m_analysis_configuration.r_diff << endl;
        cerr << "calc points = " << a_points * (((float) (subsynth.extentx*subsynth.extenty)) / ((float) num_areas * m_analysis_configuration.analysis_window_width * m_analysis_configuration.analysis_window_height)) << endl;
        cerr << "ANALYSIS AREA = " << m_analysis_configuration.analysis_window_width * m_analysis_configuration.analysis_window_height << endl;
        cerr << "SYNTHESIS AREA = " << ((long) subsynth.extentx* (long) subsynth.extenty) << endl;
        cerr << "SCALE RATIO = " << 1000.0f * (((float) (subsynth.extentx*subsynth.extenty)) / ((float) num_areas * m_analysis_configuration.analysis_window_width * m_analysis_configuration.analysis_window_height)) << endl;
        cerr << "num areas = " << num_areas << endl;
        cerr << endl;
    }
*/

    float syntharea = (float) subsynth.extentx * (float) subsynth.extenty;
    float analysisarea = (float) num_areas * (float) m_analysis_configuration.analysis_window_width * (float) m_analysis_configuration.analysis_window_height;

    return (int) ((float) a_points * syntharea / analysisarea);
}

////
//// VARIATION METHODS FOR TESTING SYNTHESIS ALTERNATIVES
///

int MultiDistributionReproducer::variantPlaceNoHits(int cid)
{
    int n_points = areaBasedPointCount(cid);
    m_active_category_points.clear();

    while((int) m_active_category_points.size() < n_points)
    {
        AnalysisPoint random_point = getRandomPoint(cid);
        add_destination_point(random_point);
    }
    return n_points;
}

int MultiDistributionReproducer::variantPlaceNoTrunks(int cid)
{
    int i = 0, fails = 0;

    int n_points = areaBasedPointCount(cid);

    m_active_category_points.clear();

    while((int) m_active_category_points.size() < n_points)
    {
        AnalysisPoint random_point = getRandomPoint(cid);
        std::vector<AnalysisPoint> reachpnts;
        cacheReachablePoints(random_point, reachpnts);

        bool placed = false;
        if(canopyIntersectFree(random_point, reachpnts))
        {
            add_destination_point(random_point);
            i = 0; // reset failure counter
        }
        else
        {
            i++;
            if(i >= placethresh) // have failed to find a valid placement after threshold attempts
            { // so just place

#ifdef VALIDATE
                if(canopyIntersect(random_point) > 0)
                    xsect++;
                strsum += calculate_strength(random_point);
                strcnt++;
#endif
                add_destination_point(random_point);
                i = 0; // reset failure counter
                fails++;
            }
        }
    }


    if(fails > 0)
        std::cerr << "Failed to achieve placement " << fails << " / " << n_points << std::endl;

    return n_points;
}

int MultiDistributionReproducer::variantPlaceFull(int cid)
{
    int i = 0, fails = 0;
    bool needs_check;

    int n_points = areaBasedPointCount(cid);

    m_active_category_points.clear();

    while((int) m_active_category_points.size() < n_points)
    {
        AnalysisPoint random_point = getRandomPoint(cid);
        std::vector<AnalysisPoint> reachpnts;
        cacheReachablePoints(random_point, reachpnts);

        bool placed = false;
        if(canopyIntersectFree(random_point, reachpnts))
        {
            // Attempt accelerated point validity check
            accelerated_point_validity_check(random_point, needs_check);

            // note that this version of calculate_strength should be robust to not finding point within radius
            if((!needs_check) || (needs_check && calculate_strength(random_point, reachpnts) > 0.0)) // pluzero might be better
            {
#ifdef VALIDATE
                if(canopyIntersect(random_point) > 0)
                    xsect++;
                strsum += calculate_strength(random_point);
                strcnt++;
#endif
                add_destination_point(random_point);

                i = 0; // reset failure counter
                placed = true;
            }
        }
        if(!placed)
        {
            i++;
            if(i >= placethresh) // have failed to find a nonzero strength or valid placement
            {   // so just place

#ifdef VALIDATE
                if(canopyIntersect(random_point) > 0)
                    xsect++;
                strsum += calculate_strength(random_point);
                strcnt++;
#endif
                add_destination_point(random_point);

                i = 0; // reset failure counter
                fails++;
            }
        }
    }


    if(fails > 0)
        std::cerr << "Failed to achieve placement " << fails << " / " << n_points << std::endl;

    return n_points;
}

void MultiDistributionReproducer::variantMoveStep(int cid)
{
    DiceRoller dice_roller(0,1000);

    int n_accepted_moves(0), n_refused_moves(0);

 #ifdef VALIDATE
    radcnt = 0;
    radsum = 0;
    avgsum = 0;
    devsum = 0;
    minsum = 0;
    maxsum = 0;
 #endif

    for(int selected_point_index(0); selected_point_index < (int) m_active_category_points.size(); selected_point_index++)
    {
        AnalysisPoint & selected_point(m_active_category_points.at(selected_point_index));

        float source_point_strength = calculate_strength(selected_point);

        AnalysisPoint highest_scoring_dest_point;
        float highest_strength(-1);

        for(int i(0); i < movecycles; i++)
        {
            /*********************
             * DESTINATION POINT *
             *********************/
            // AnalysisPoint random_dest_point = getRandomPoint(selected_point.getHeight());

            AnalysisPoint random_dest_point;
            // choose between jump or step here
            if(ProbabilisticUtils::returnTrueWithProbability(jumpchance, dice_roller))
            {
                // jump
                random_dest_point = getRandomPoint(cid);
            }
            else
            {
                // step in random direction by two bin widths and alter radius up or down by 20% of height range
                random_dest_point = getRandomStep(selected_point, cid);
            }

            float destination_point_strength = calculate_strength(random_dest_point);

            if(destination_point_strength > highest_strength)
            {
                highest_scoring_dest_point = random_dest_point;
                highest_strength = destination_point_strength;
            }
        }

        float acceptance_ratio(highest_strength/source_point_strength);

        if(ProbabilisticUtils::returnTrueWithProbability(acceptance_ratio, dice_roller))
        //if(ProbabilisticUtils::returnTrueWithProbabilitySupplied(acceptance_ratio, rand()%1000))
        {
            n_accepted_moves++;
            move_point(selected_point, highest_scoring_dest_point);
#ifdef VALIDATE
            strsum -= source_point_strength;
            strsum += highest_strength;

            radsum += highest_scoring_dest_point.getHeight();
            CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->getCategories().find(cid)->second;
            avgsum +=  m_active_cp.m_header.height_properties.avg;
            devsum +=  m_active_cp.m_header.height_properties.standard_dev;
            minsum +=  m_active_cp.m_header.height_properties.min;
            maxsum +=  m_active_cp.m_header.height_properties.max;
            radcnt++;
#endif
        }
        else
        {
            n_refused_moves++;
#ifdef VALIDATE
            radsum += selected_point.getHeight();
            CategoryProperties m_active_cp = m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->getCategories().find(cid)->second;
            avgsum +=  m_active_cp.m_header.height_properties.avg;
            devsum +=  m_active_cp.m_header.height_properties.standard_dev;
            minsum +=  m_active_cp.m_header.height_properties.min;
            maxsum +=  m_active_cp.m_header.height_properties.max;
            radcnt++;
#endif
        }
    }
}

void MultiDistributionReproducer::variantMoveHeightSep(int cid)
{

}

////
////

int MultiDistributionReproducer::matching_density_initialize(int cid)
{
    int i(0);
    bool needs_check;
    int fails = 0;

    int n_points = areaBasedPointCount(cid);

    /*
    if(cid == 15000)
    {
        cerr << "cat " << cid << " expected points = " << n_points << endl;

    }*/

    /*
    if(!m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->isEmpty())
    {
        m_cmap->summaryDisplayDescriptor(selected_point.getCenter().x(), selected_point.getCenter().y());
        m_cmap->lookupSynthDistrib(selected_point.getCenter().x(), selected_point.getCenter().y())->summaryDisplay();
    }*/

    // m_placed_categories.clear();
    // m_generated_points_properties->clear();
    m_active_category_points.clear();

    // std::cerr << "n point = " << n_points << std::endl;
    while((int) m_active_category_points.size() < n_points)
    {
        AnalysisPoint random_point = getRandomPoint(cid);
        std::vector<AnalysisPoint> reachpnts;
        cacheReachablePoints(random_point, reachpnts);

        bool placed = false;
        if(canopyIntersectFree(random_point, reachpnts))
        {
            // Attempt accelerated point validity check
            accelerated_point_validity_check(random_point, needs_check);

            if((!needs_check) || (needs_check && calculate_strength(random_point, reachpnts) > 0.0)) // pluzero might be better
            {
#ifdef VALIDATE
                if(canopyIntersect(random_point) > 0)
                    xsect++;
                strsum += calculate_strength(random_point);
                strcnt++;
#endif
                add_destination_point(random_point);

                i = 0; // reset failure counter
                placed = true;
            }
        }
        if(!placed)
        {
            i++;
            if(i >= placethresh) // have failed to find a nonzero strength or valid placement
            {   // so just place

#ifdef VALIDATE
                if(canopyIntersect(random_point) > 0)
                    xsect++;
                strsum += calculate_strength(random_point);
                strcnt++;
#endif
                add_destination_point(random_point);

                i = 0; // reset failure counter
                fails++;
            }
        }
    }


    if(fails > 0)
    {
        std::cerr << "Failed to achieve placement " << fails << " / " << n_points << std::endl;
        // AnalysisPoint random_point = getRandomPoint(cid);
        // calculate_strength_verbose(random_point);
    }

    return n_points;
}

void MultiDistributionReproducer::move_point(AnalysisPoint & point, AnalysisPoint & new_location)
{    
    m_taken_points.removePoint(point.getCenter());
    m_spatial_point_storage.removePoint(point);

    point = new_location;

    // Add
    m_taken_points.insertPoint(point.getCenter());
    m_spatial_point_storage.addPoint(point);
}

void MultiDistributionReproducer::add_destination_point(const AnalysisPoint & point)
{
    m_taken_points.insertPoint(point.getCenter());
    m_active_category_points.push_back(point); // Update the active category points vector
    m_spatial_point_storage.addPoint(point);

    // Check min/max
    SizeProperties & props((*m_generated_points_properties)[point.getCategoryId()]);
    int point_height(point.getHeight());
    if(props.maxHeight == -1 || point.getHeight() > props.maxHeight)
        props.maxHeight = point_height;
    if(props.minHeight == -1 || point.getHeight() < props.minHeight)
        props.minHeight = point_height;
}

void MultiDistributionReproducer::remove_destination_point(const AnalysisPoint & point, int this_category_points_index)
{
    m_taken_points.removePoint(point.getCenter());
    m_active_category_points.erase(m_active_category_points.begin()+this_category_points_index); // Update the active category points vector

    m_spatial_point_storage.removePoint(point);
}

void MultiDistributionReproducer::accelerated_point_validity_check(const AnalysisPoint & reference_point, bool & needs_check)
{
    // add current category to set if it isn't already there
    m_placed_categories.insert(reference_point.getCategoryId()); // assuming it has the correct category at this point

    for(auto cat: m_placed_categories) // iterate over categories belonging to point already placed, although not all categories may actually appear in surrounding points
    {
        accelerated_point_validity_check(reference_point, cat, needs_check);

        if(needs_check)
            return;
    }

/*
    // Check pair correlations for this point and other points present and check if any are equal to zero
    for(auto it(m_all_generated_points.begin()); it != m_all_generated_points.end(); it++)
    {
        accelerated_point_validity_check(reference_point, it->first, needs_check);

        if(needs_check)
            return;
    }

    // With itself
    accelerated_point_validity_check(reference_point, m_point_factory.getActiveCategoryId(), needs_check);*/
}

void MultiDistributionReproducer::accelerated_point_validity_check(const AnalysisPoint & reference_point, int queried_category, bool & needs_check)
{
    needs_check = true;

    if(m_cmap->lookupSynthDistrib(reference_point.getCenter().x(), reference_point.getCenter().y())->getCategories().find(reference_point.getCategoryId())->second.m_header.category_dependent_ids.size() > 0) // dependent categories
    {
        // std::cerr << "is dependent on other categories" << std::endl;
        return;
    }

    const RadialDistribution * distribution  = get_radial_distribution(reference_point, queried_category, reference_point.getCategoryId());
    if(distribution != nullptr && distribution->getMinimum() == 0) // < pluszero) // tolerance is introduced here because mass transport interpolation introduces some numerical error
    {
        return;
    }

    // If we got here, point is valid
    needs_check = false;
}

float MultiDistributionReproducer::calculate_strength(const AnalysisPoint & reference_point)
{
    //std::vector<AnalysisPoint> possible_reachable_points(m_spatial_point_storage.getPossibleReachablePoints(reference_point, m_analysis_configuration.r_max));

    std::vector<AnalysisPoint> possible_reachable_points;
    cacheReachablePoints(reference_point, possible_reachable_points);
    return calculate_strength(reference_point, possible_reachable_points);
}

float MultiDistributionReproducer::calculate_strength(const AnalysisPoint & candidate_point, const std::vector<AnalysisPoint> & reachable_points)
{   
    float strength(1.0f);

    QLineF line;
    line.setP1(candidate_point.getCenter());

    std::vector<AnalysisPoint> existing_points;

    // select out only those points that have the same distribution, otherwise causes issues with matches not being satisfied
    for(auto it: reachable_points)
    {
        if(m_cmap->equivDescriptor(m_cmap->getSynthDescriptor(candidate_point.getCenter().x(), candidate_point.getCenter().y()),
                                   m_cmap->getSynthDescriptor(it.getCenter().x(), it.getCenter().y())))
            existing_points.push_back(it);
    }
    // if((int) existing_points.size() != (int) reachable_points.size())
    //    boundcount++;

    CategoryProperties category_properties (m_cmap->lookupSynthDistrib(candidate_point.getCenter().x(), candidate_point.getCenter().y())->getCategories().find(candidate_point.getCategoryId())->second);
    std::set<int> dependent_category_ids(category_properties.m_header.category_dependent_ids);
    bool dependencyMet = (dependent_category_ids.size() == 0);

    const float candidatePointRadius = candidate_point.getRadius();

    for(const AnalysisPoint & existing_point : existing_points)
    {
        line.setP2(existing_point.getCenter());
        float distance = line.length();

        // perform strong canopy intersection test here
        if(distance < candidatePointRadius)
        {
            strength = 0;
            return strength;
        }

        // Get the pair correlation
        const RadialDistribution * radial_distribution( get_radial_distribution(candidate_point, existing_point.getCategoryId(), candidate_point.getCategoryId()) );

        if(radial_distribution != nullptr) // if there is no correlation then strength is ignored
        {
            const float existingPointRadius = existing_point.getRadius();

            if(distance > existingPointRadius+candidatePointRadius)
            {
                distance -= (existingPointRadius+candidatePointRadius);
                // distance = (float) sqrt(distance); //?
                int r_bracket ( RadialDistributionUtils::getRBracket(distance, m_analysis_configuration.r_min, m_analysis_configuration.r_diff) );
                if(r_bracket < m_analysis_configuration.r_max)
                {
                    float r(radial_distribution->m_data.find(r_bracket)->second);
                    strength *= r;
                }
            }
            else // Shaded
            {
                // assumes candidatePointRadius < existingPointsRadius because it is a smaller plant
                // But actually this may not be the case, depending on canopy spread

                dependencyMet = dependencyMet || std::find(dependent_category_ids.begin(), dependent_category_ids.end(), existing_point.getCategoryId()) != dependent_category_ids.end();
                //?

                if(distance > existingPointRadius) // Less than half shaded
                {
                    strength *= radial_distribution->m_less_than_half_shaded_distribution;
                }
                else if (distance > existingPointRadius - candidatePointRadius) // More than half, less than full
                {
                    strength *= radial_distribution->m_more_than_half_shaded_distribution;
                }
                else // Fully shaded
                {
                    strength *= radial_distribution->m_fully_shaded_distribution;
                }
            }

            if(strength == 0) // Optimization. It will always be zero
                return strength;
        }
    }

    //return dependencyMet ? strength : 0;
    return strength;
}

const RadialDistribution * MultiDistributionReproducer::get_radial_distribution(const AnalysisPoint & reference_point, int reference_category, int target_category)
{

    // auto pair_correlation_it (m_pair_correlations.find(std::pair<int,int>(reference_category, target_category)));
    auto pair_correlation_it (m_cmap->lookupSynthDistrib(reference_point.getCenter().x(), reference_point.getCenter().y())->getCorrelations().find(std::pair<int,int>(reference_category, target_category)));

    // if( pair_correlation_it == m_pair_correlations.end())
    if( pair_correlation_it == m_cmap->lookupSynthDistrib(reference_point.getCenter().x(), reference_point.getCenter().y())->getCorrelations().end())
    {
        // std::cerr << "FAILED TO FIND PAIR CORRELATION DATA! between " << reference_category << " and " << target_category << std::endl;
        // now allowed simply ignore
        return nullptr;
    }
    return &pair_correlation_it->second;
}

