#include <stdexcept>
#include <cstddef>
#include <map>
#include <common/debug_string.h>
#include <common/debug_vector.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/spirit/include/qi.hpp>
#include "str.h"
#include "map.h"
#include "map_rgba.h"
#include "initialize.h"

namespace po = boost::program_options;

namespace
{
namespace Option
{
    /** @name Tokens for the option parser @{ */
    static const char *inputFile() { return "input"; }
    static const char *outputFile() { return "output"; }
    static const char *resolution() { return "resolution"; }
    static const char *addHeight() { return "add-height"; }
    static const char *setMask() { return "set-mask"; }
    static const char *getMask() { return "get-mask"; }
    static const char *filterType() { return "filter-type"; }
    static const char *zeroType() { return "zero-type"; }
    static const char *addBorder() { return "add-border"; }
    static const char *shade() { return "shade"; }
    static const char *help() { return "help"; }
    /** @} */
} // namespace Option

} // anonymous namespace

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: heighttool [options] input.exr output.exr\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::inputFile(), 1);
    positional.add(Option::outputFile(), 1);

    po::options_description desc("General options");
    desc.add_options()
        (Option::resolution(),  po::value<float>(), "set spatial resolution (metres per sample)")
        (Option::addHeight(),   po::value<float>(), "increment all heights by this amount")
        (Option::getMask(),     po::value<std::string>(), "generate visualization of the type mask")
        (Option::setMask(),     po::value<std::string>(), "adds terrain type information")
        (Option::filterType(),  po::value<int>(),   "flatten all terrain except for one type")
        (Option::zeroType(),    po::value<int>(),   "set a specific type for all pixels with zero height")
        (Option::addBorder(),   po::value<std::string>(), "add zero-height borders (left,top,right,bottom)")
        (Option::shade(),       po::value<std::string>(), "render the terrain to file")
        (Option::help(),        "show help");

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::inputFile(),   po::value<std::string>(), "input file")
        (Option::outputFile(),  po::value<std::string>(), "output file");

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

        if (!vm.count(Option::inputFile()))
            throw po::error("Input file must be specified");
        bool needOutput =
            vm.count(Option::resolution())
            || vm.count(Option::addHeight())
            || vm.count(Option::setMask());
        if (needOutput && !vm.count(Option::outputFile()))
            throw po::error("Output file must be specified");

        return vm;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << "\n\n";
        usage(std::cerr, desc);
        std::exit(1);
    }
}

static MemMap<height_and_mask_tag> addBorder(
    const MemMap<height_and_mask_tag> &map,
    const std::string &spec)
{
    namespace qi = boost::spirit::qi;
    auto p = spec.begin();
    std::vector<int> borders;
    bool success = qi::parse(p, spec.end(), (qi::uint_ >> ',' >> qi::uint_ >> ',' >> qi::uint_ >> ',' >> qi::uint_ ), borders);
    if (!success || p != spec.end())
        throw std::runtime_error("Unexpected token at '" + std::string(p, spec.end()) + "'");

    Region r = map.region();
    Region outRegion(r.x0 - borders[0], r.y0 - borders[1], r.x1 + borders[2], r.y1 + borders[3]);
    MemMap<height_and_mask_tag> out(outRegion);
    out.setStep(map.step());
    out.fill(MapTraits<height_and_mask_tag>::type{0.0f, MapTraits<mask_tag>::all});
    for (int y = r.y0; y < r.y1; y++)
        for (int x = r.x0; x < r.x1; x++)
            out[y][x] = map[y][x];
    return out;
}

static void getMask(const MemMap<height_and_mask_tag> &map, const std::string &filename)
{
    typedef MapTraits<rgba_tag>::type color;
    typedef MapTraits<mask_tag>::type mask_t;

    const color empty = {{0.0f, 0.0f, 0.0f, 0.0f}};
    const uts::vector<color> &palette = MapTraits<rgba_tag>::colorPalette();
    const mask_t valid = (mask_t(1) << palette.size()) - 1;
    const Region &r = map.region();

    MemMap<rgba_tag> out(r);
    std::size_t rangeErrors = 0;
#pragma omp parallel for schedule(static) reduction(+:rangeErrors)
    for (int y = r.y0; y < r.y1; y++)
        for (int x = r.x0; x < r.x1; x++)
        {
            auto present = ~map[y][x].mask;
            if (~present == 0)
                out[y][x] = empty;
            else
            {
                if (present & ~valid)
                    rangeErrors++;

                // Average together the palette entries
                std::array<float, 4> sum = {};
                int n = 0;
                for (std::size_t i = 0; i < palette.size(); i++)
                {
                    if ((present >> i) & 1)
                    {
                        n++;
                        for (int j = 0; j < 4; j++)
                            sum[j] += palette[i][j];
                    }
                }
                if (n > 1)
                    for (int j = 0; j < 4; j++)
                        sum[j] /= n;
                out[y][x] = sum;
            }
        }

    if (rangeErrors > 0)
        std::cerr << "Warning: unsupported terrain type. Extend the palette.\n";

    out.write(filename);
}

static void setMask(MemMap<height_and_mask_tag> &map, const std::string &filename)
{
    if (endsWith(filename, ".exr"))
    {
        MemMap<mask_tag> mask(filename);
        Region update = mask.region() & map.region();
        if (update.empty())
            throw std::runtime_error("Mask image does not intersect the exemplar");
        if (!mask.region().contains(map.region()))
            std::cerr << "Warning: mask image does not completely cover the exemplar\n";

#pragma omp parallel for
        for (int y = update.y0; y < update.y1; y++)
            for (int x = update.x0; x < update.x1; x++)
                map[y][x].mask = mask[y][x];
    }
    else
    {
        typedef MapTraits<mask_tag>::type mask_t;

        MemMap<rgba_tag> paint(filename);
        if (!paint.region().contains(map.region()))
        {
            /* Special case: if the sizes match but not the offset, ignore the offset */
            if (paint.region().width() == map.region().width()
                && paint.region().height() == map.region().height())
                paint.translateTo(map.region().x0, map.region().y0);
            else
                throw std::runtime_error("Mask image does not cover the exemplar");
        }

        colorsToMasks(map.region(), paint,
                      [&map] (int x, int y, mask_t mask) { map[y][x].mask = mask; });
    }
}

static void filterType(MemMap<height_and_mask_tag> &map, const int type)
{
    typedef MapTraits<mask_tag>::type mask_t;

#pragma omp parallel for
    for (int y = map.region().y0; y < map.region().y1; y++)
        for (int x = map.region().x0; x < map.region().x1; x++)
        {
            mask_t present = ~map[y][x].mask;
            if (!(present & (mask_t(1) << type)))
                map[y][x].height = 0.0f;
        }
}

static void zeroType(MemMap<height_and_mask_tag> &map, const int type)
{
    if (type < 0 || type >= MapTraits<mask_tag>::numTypes)
        throw std::runtime_error("Type " + std::to_string(type) + " is out of range");

    const Region r = map.region();
    typedef MapTraits<mask_tag>::type mask_t;
    mask_t mask = ~((mask_t(1) << type) - 1);
#pragma omp parallel for
    for (int y = r.y0; y < r.y1; y++)
        for (int x = r.x0; x < r.x1; x++)
            if (map[y][x].height == 0.0f)
                map[y][x].mask = mask;
}

static MemMap<gray_tag> shade(const MemMap<height_and_mask_tag> &in)
{
    const double azimuth = 1.75 * M_PI;
    const double altitude = 0.25 * M_PI;
    const double sinAltitude = sin(altitude);
    const double cosAltitude = cos(altitude);
    // The 8 is a normalization factor for the filter kernel
    const double scale = 1.0 / (8.0 * in.step());

    const Region r = in.region();
    MemMap<gray_tag> out(r);
    out.setStep(in.step());

    out.fill(0.0f);
#pragma omp parallel for
    for (int y = r.y0 + 1; y < r.y1 - 1; y++)
        for (int x = r.x0 + 1; x < r.x1 - 1; x++)
        {
            double dx = (in[y - 1][x - 1].height + 2 * in[y][x - 1].height + in[y + 1][x - 1].height)
                - (in[y + 1][x - 1].height + 2 * in[y][x + 1].height + in[y + 1][x + 1].height);
            dx *= scale;
            double dy = (in[y - 1][x - 1].height + 2 * in[y - 1][x].height + in[y - 1][x + 1].height)
                - (in[y + 1][x - 1].height + 2 * in[y + 1][x].height + in[y + 1][x + 1].height);
            dy *= scale;

            double g2 = dx * dx + dy * dy;
            double aspect = atan2(dy, dx);
            double cang = (sinAltitude - cosAltitude * sqrt(g2) * sin(aspect - azimuth))
                / sqrt(1.0 + g2);
            if (cang < 0.0)
                cang = 0.0;
            out[y][x] = cang;
        }
    return out;
}

int main(int argc, char **argv)
{
    utsInitialize();
    try
    {
        po::variables_map vm = processOptions(argc, argv);
        MemMap<height_and_mask_tag> map;
        const uts::string inputFile = vm[Option::inputFile()].as<std::string>();
        map.read(inputFile);

        if (vm.count(Option::resolution()))
            map.setStep(vm[Option::resolution()].as<float>());
        if (vm.count(Option::addBorder()))
            map = addBorder(map, vm[Option::addBorder()].as<std::string>());
        if (vm.count(Option::addHeight()))
        {
            float add = vm[Option::addHeight()].as<float>();
            for (int y = 0; y < map.height(); y++)
                for (int x = 0; x < map.width(); x++)
                    map[y][x].height += add;
        }
        if (vm.count(Option::setMask()))
            setMask(map, vm[Option::setMask()].as<std::string>());
        if (vm.count(Option::getMask()))
            getMask(map, vm[Option::getMask()].as<std::string>());
        if (vm.count(Option::filterType()))
            filterType(map, vm[Option::filterType()].as<int>());
        if (vm.count(Option::zeroType()))
            zeroType(map, vm[Option::zeroType()].as<int>());
        if (vm.count(Option::shade()))
        {
            const uts::string filename = vm[Option::shade()].as<std::string>();
            MemMap<gray_tag> shaded = shade(map);
            shaded.write(filename);
        }

        if (vm.count(Option::outputFile()))
        {
            const uts::string outputFile = vm[Option::outputFile()].as<std::string>();
            map.write(outputFile);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
