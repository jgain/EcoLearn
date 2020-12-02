#include <iostream>
#include <algorithm>
#include <common/debug_vector.h>
#include <utility>
#include <memory>
#include <cstddef>
#include <thread>
#include <boost/program_options.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <ImfFrameBuffer.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfChannelList.h>
//#include <omp.h>
#include "region.h"

namespace po = boost::program_options;

namespace
{
namespace Option
{
    /** @name Tokens for the option parser @{ */
    static const char *inputFile() { return "input"; }
    static const char *outputFile() { return "output"; }
    static const char *resolution() { return "resolution"; }
    static const char *region() { return "region"; }
    static const char *origin() { return "origin"; }
    static const char *help() { return "help"; }
    /** @} */
} // namespace Option

} // anonymous namespace

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: exrrewrite [options] input.exr output.exr\n\n";
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
        (Option::region(),      po::value<std::string>(), "pick out a region (x0,y0,width,height)")
        (Option::origin(),      po::value<std::string>(), "set new region origin (x,y)")
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
        if (!vm.count(Option::outputFile()))
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

static Region parseRegion(const std::string &spec)
{
    namespace qi = boost::spirit::qi;
    auto p = spec.begin();
    Region region;
    bool success = qi::parse(p, spec.end(), (qi::int_ >> ',' >> qi::int_ >> ',' >> qi::uint_ >> ',' >> qi::uint_ ), region);
    if (!success || p != spec.end())
        throw std::runtime_error("Unexpected token at '" + std::string(p, spec.end()) + "'");
    // Spec contains width and height, not x1 and y1
    region.x1 += region.x0;
    region.y1 += region.y0;
    return region;
}

static std::pair<int, int> parseCorner(const std::string &spec)
{
    namespace qi = boost::spirit::qi;
    auto p = spec.begin();
    std::pair<int, int> corner;
    bool success = qi::parse(p, spec.end(), (qi::int_ >> ',' >> qi::int_), corner);
    if (!success || p != spec.end())
        throw std::runtime_error("Unexpected token at '" + std::string(p, spec.end()) + "'");
    return corner;
}

static std::size_t typeToSize(Imf::PixelType type)
{
    switch (type)
    {
    case Imf::UINT: return sizeof(std::uint32_t);
    case Imf::HALF: return sizeof(std::uint16_t);
    case Imf::FLOAT: return sizeof(float);
    default:
        throw std::runtime_error("Unknown pixel type");
    }
}

int main(int argc, char **argv)
{
    try
    {
        po::variables_map vm = processOptions(argc, argv);
        //Imf::setGlobalThreadCount(omp_get_max_threads());

        Imf::Header header;
        std::ptrdiff_t stride;
        Region inRegion;
        Region keepRegion;
        uts::vector<std::unique_ptr<char[]> > buffers;

        {
            Imf::InputFile in(vm[Option::inputFile()].as<std::string>().c_str());
            Imf::FrameBuffer fb;
            header = in.header();
            Imath::Box2i &dw = header.dataWindow();
            inRegion = Region(dw.min.x, dw.min.y, dw.max.x + 1, dw.max.y + 1);
            stride = inRegion.width();
            const std::size_t pixels = inRegion.pixels();
            const std::ptrdiff_t bias = -(inRegion.y0 * stride + inRegion.x0);

            if (vm.count(Option::region()))
            {
                keepRegion = parseRegion(vm[Option::region()].as<std::string>());
                if (keepRegion.empty())
                    throw std::runtime_error("Cannot specify an empty region to keep");
                if (!inRegion.contains(keepRegion))
                    throw std::runtime_error("Region to keep extends outside data window");
            }
            else
                keepRegion = inRegion;

            const auto &channels = header.channels();
            for (auto i = channels.begin(); i != channels.end(); ++i)
            {
                const auto &channel = i.channel();
                std::size_t pixelSize = typeToSize(channel.type);
                std::unique_ptr<char[]> data(new char[pixelSize * pixels]);
                fb.insert(i.name(), Imf::Slice(
                        channel.type,
                        data.get() + bias * pixelSize,
                        pixelSize,
                        pixelSize * stride));
                buffers.emplace_back(std::move(data));
            }
            in.setFrameBuffer(fb);
            in.readPixels(keepRegion.y0, keepRegion.y1 - 1);
        }

        Region outRegion = keepRegion;
        if (vm.count(Option::origin()))
        {
            std::pair<int, int> corner = parseCorner(vm[Option::origin()].as<std::string>());
            outRegion.x0 = corner.first;
            outRegion.y0 = corner.second;
            outRegion.x1 = corner.first + keepRegion.width();
            outRegion.y1 = corner.second + keepRegion.height();
        }

        {
            Imf::FrameBuffer fb;
            header.compression() = Imf::ZIP_COMPRESSION;
            Imath::Box2i &dw = header.dataWindow();
            dw.min.x = outRegion.x0;
            dw.min.y = outRegion.y0;
            dw.max.x = outRegion.x1 - 1;
            dw.max.y = outRegion.y1 - 1;
            header.displayWindow() = dw;

            Imf::FrameBuffer outFb;
            const auto &channels = header.channels();
            int pos = 0;
            // coordinates of first element in each buffer
            int x0 = inRegion.x0 + (outRegion.x0 - keepRegion.x0);
            int y0 = inRegion.y0 + (outRegion.y0 - keepRegion.y0);
            const std::ptrdiff_t bias = -(y0 * stride + x0);
            for (auto i = channels.begin(); i != channels.end(); ++i, ++pos)
            {
                const auto &channel = i.channel();
                std::size_t pixelSize = typeToSize(channel.type);
                fb.insert(i.name(), Imf::Slice(
                        channel.type,
                        buffers[pos].get() + bias * pixelSize,
                        pixelSize,
                        pixelSize * stride));
            }

            Imf::OutputFile out(vm[Option::outputFile()].as<std::string>().c_str(), header);
            out.setFrameBuffer(fb);
            out.writePixels(outRegion.height());
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
