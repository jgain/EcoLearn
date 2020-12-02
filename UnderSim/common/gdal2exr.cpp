#include <iostream>
#include <locale>
#include <common/debug_string.h>
#include <common/debug_vector.h>
#include <stack>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstring>
#include <cmath>
#include <boost/program_options.hpp>
#include "map.h"
#include "initialize.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

namespace po = boost::program_options;

namespace
{
namespace Option
{
    static const char *inputFile() { return "input"; }
    static const char *outputFile() { return "output"; }
    static const char *forceStep() { return "force-step"; }
    static const char *noClip() { return "no-clip"; }
    static const char *noOrigin() { return "no-origin"; }
    static const char *region() { return "region"; }
    static const char *help() { return "help"; }
} // namespace Option
} // anonymous namespace

class GDALDeleter
{
public:
    void operator()(GDALDataset *dataset) const
    {
        if (dataset != nullptr)
            GDALClose(dataset);
    }
};

static MemMap<height_tag> loadMap(const uts::string &filename, const po::variables_map &vm, float &noData)
{
    const std::unique_ptr<GDALDataset, GDALDeleter> dataset((GDALDataset *) GDALOpen(filename.c_str(), GA_ReadOnly));
    if (!dataset)
        throw std::runtime_error("Could not open " + filename);

    MemMap<height_tag> ans;
    try
    {
        // importFromWkt uses a char** instead of const char **
        char * projectionName = const_cast<char *>(dataset->GetProjectionRef());
        OGRSpatialReference srs;
        // Note: this updates projectionName to point past-the-end of the WKT
        srs.importFromWkt(&projectionName);

        float step;
        if (!vm.count(Option::forceStep()))
        {
            double transform[6];
            if (dataset->GetGeoTransform(transform) != CE_None)
                throw std::runtime_error("No step size found in file or on command line");

            double scale;
            if (srs.IsGeographic())
            {
                /* For now assume this means degrees, and ignore the fact that
                 * X will have a different scale to Y. The value of 111120 is based
                 * on a nautical mile being 1852m.
                 */
                scale = 111120;
            }
            else
            {
                char *unitNameChar;
                scale = srs.GetLinearUnits(&unitNameChar);
                uts::string unitName(unitNameChar);
                if (unitName.empty() || unitName == "unknown")
                    throw std::runtime_error("No units provided");
            }

            double xstep = hypot(transform[1], transform[4]) * scale;
            double ystep = hypot(transform[2], transform[5]) * scale;
            if (std::fabs(xstep - ystep) > 1e-6 * std::max(xstep, ystep))
                throw std::runtime_error("Pixels are not square: " + std::to_string(xstep) + " x " + std::to_string(ystep));
            step = xstep;
        }
        else
            step = vm[Option::forceStep()].as<float>();

        const int width = dataset->GetRasterXSize();
        const int height = dataset->GetRasterYSize();
        const int numBands = dataset->GetRasterCount();
        if (numBands < 1)
            throw std::runtime_error("No data");

        GDALRasterBand *band = dataset->GetRasterBand(1);
        uts::string unitType = band->GetUnitType();
        if (unitType != "" && unitType != "m")
            throw std::runtime_error("Unknown unit type: " + unitType);

        ans.allocate({0, 0, width, height});
        ans.setStep(step);
        CPLErr status = band->RasterIO(GF_Read, 0, 0, width, height, (void *) ans.get(),
                                       width, height, GDT_Float32, 0, 0);
        if (status != CE_None)
            throw std::runtime_error("RasterIO failed");
        noData = band->GetNoDataValue();

        // RasterIO gives raw values, prior to scale+bias
        double scale = band->GetScale();
        double offset = band->GetOffset();
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                ans[y][x] = float(ans[y][x] * scale + offset);

        // Compute x value of central meridian
        if (srs.GetUTMZone() != 0)
        {
            double falseEasting = srs.GetProjParm(SRS_PP_FALSE_EASTING, 500000.0);
            double transform[6];
            if (dataset->GetGeoTransform(transform) == CE_None)
            {
                double x = (falseEasting - transform[0]) / transform[1];
                std::cout << "Central meridian is at X = " << int(std::round(x)) << '\n';
            }
        }
    }
    catch (std::runtime_error &e)
    {
        throw std::runtime_error("Failed to read " + filename + ": " + e.what());
    }
    return ans;
}

namespace { struct count_tag{}; }

static Region dataRegion(const MemMap<height_tag> &heights, float noData)
{
    // Code hasn't been adapted to handle arbitrary origins yet
    assert(heights.region().x0 == 0);
    assert(heights.region().y0 == 0);

    int W = heights.width();
    int H = heights.height();
    uts::vector<int> upCount(W + 1, 0); // number of valid entries upwards from here
    Region best;
    std::size_t bestArea = 0;
    std::stack<std::pair<int, int> > st;

    for (int y1 = 0; y1 < H; y1++)
    {
        while (!st.empty())
            st.pop();
        st.emplace(-1, -1);
        for (int x = 0; x <= W; x++)
        {
            int u;
            if (x == W || heights[y1][x] == noData)
                u = 0;
            else
                u = upCount[x] + 1;
            upCount[x] = u;
            while (u < st.top().first)
            {
                Region test(st.top().second, y1 - st.top().first + 1, x, y1 + 1);
                std::size_t area = std::size_t(test.width()) * test.height();
                if (area > bestArea)
                {
                    bestArea = area;
                    best = test;
                }
                st.pop();
            }
            st.emplace(u, x);
        }
    }
    return best;
}

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: gdal2exr [options] input.tif output.exr\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile(), 1);
    positional.add(Option::outputFile(), 1);

    po::options_description desc("General options");
    desc.add_options()
        (Option::forceStep(),   po::value<float>(), "override step size extracted from file")
        (Option::noClip(),      "do not clip the image to exclude missing data")
        (Option::noOrigin(),    "do not move data to start at the origin")
        (Option::region(),      po::value<std::string>(), "Subregion to extract (x:y:w:h)")
        (Option::help(),        "show help");

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::inputFile(),   po::value<std::string>()->required(), "input file")
        (Option::outputFile(),  po::value<std::string>()->required(), "output file");

    po::options_description all("All options");
    all.add(desc);
    all.add(hidden);

    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
                  .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
                  .options(all)
                  .positional(positional)
                  .run(), vm);
        po::notify(vm);

        if (vm.count(Option::help()))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n";
        usage(std::cerr, desc);
        std::exit(1);
    }
}

int main(int argc, char **argv)
{
    utsInitialize();
    try
    {
        GDALAllRegister();
        po::variables_map vm = processOptions(argc, argv);
        float noData;
        MemMap<height_tag> m = loadMap(vm[Option::inputFile()].as<std::string>(), vm, noData);
        Region region = m.region();
        if (!vm.count(Option::noClip()))
            region = dataRegion(m, noData);
        if (vm.count(Option::region()))
        {
            // TODO: make robust!
            int x, y, w, h;
            std::sscanf(vm[Option::region()].as<std::string>().c_str(),
                        "%d:%d:%d:%d", &x, &y, &w, &h);
            if (x < 0 || w <= 0 || x + w > region.width()
                || y < 0 || h <= 0 || y + h > region.height())
            {
                std::cerr << "--" << Option::region() << " values are out of range\n";
                return 1;
            }
            region.x0 += x;
            region.y0 += y;
            region.x1 = region.x0 + w;
            region.y1 = region.y0 + h;
        }
        if (!vm.count(Option::noOrigin()))
        {
            m.translateTo(m.region().x0 - region.x0,
                          m.region().y0 - region.y0);

            // Translate the region to the origin
            region.x1 -= region.x0;
            region.y1 -= region.y0;
            region.x0 = 0;
            region.y0 = 0;
        }
        m.write(vm[Option::outputFile()].as<std::string>(), region);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
