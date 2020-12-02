/**
 * @file
 *
 * Automatically classifies terrain types.
 */

#include <boost/program_options.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <random>
#include <cmath>
#include <sstream>
#include <svm.h>
#include <eigen3/Eigen/Core>
#include <common/map.h>
#include <common/maputils.h>
#include <common/debug_vector.h>
#include <common/initialize.h>
#include <common/progress.h>
#include <common/mathutils.h>

namespace po = boost::program_options;

static constexpr int NTYPES = std::numeric_limits<MapTraits<mask_tag>::type>::digits;
typedef MapTraits<mask_tag>::type mask_t;
typedef MapTraits<height_tag>::type height_t;

namespace
{
namespace Option
{
    /** @name Tokens for the option parser @{ */
    static const char *exemplar() { return "exemplar"; }
    static const char *inputFile() { return "input"; }
    static const char *outputFile() { return "output"; }
    static const char *featureSize() { return "feature-size"; }
    static const char *maxSamples() { return "max-samples"; }
    static const char *cross() { return "cross"; }
    static const char *explore() { return "explore"; }
    static const char *regions() { return "regions"; }
    static const char *width() { return "width"; }
    static const char *height() { return "height"; }
    static const char *help() { return "help"; }
    /** @} */
} // namespace Option
} // anonymous namespace

static void usage(std::ostream &o, const po::options_description desc)
{
    o << "Usage: utsclassify [options] exemplar.exr [...] -i input.exr [-o output.exr]\n\n";
    o << desc;
}

static po::variables_map processOptions(int argc, char **argv)
{
    po::positional_options_description positional;
    positional.add(Option::exemplar(), -1);

    po::options_description desc("General options");
    desc.add_options()
        ("input,i",         po::value<std::string>()->required(), "input file")
        ("output,o",        po::value<std::string>(), "output file [same as input]")
        (Option::featureSize(), po::value<float>()->default_value(2000.0f), "maximum feature size (metres)")
        (Option::maxSamples(), po::value<int>()->default_value(1000), "maximum samples per type for training")
        (Option::cross(),   po::value<int>(), "report result of N-way cross-validation")
        (Option::explore(),                   "explore the parameter space")
        (Option::regions(), po::value<std::string>(), "list of corners for regions to predict (x,y+x,y...)")
        (Option::width(),   po::value<int>()->default_value(1024), "width for regions specified with --regions")
        (Option::height(),  po::value<int>()->default_value(1024), "height for regions specified with --regions")
        (Option::help(),    "show help");

    po::options_description hidden("Hidden options");
    hidden.add_options()
        (Option::exemplar(), po::value<std::vector<std::string>>()->composing(), "exemplars");

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

        /* Using ->required() on the option gives an unhelpful message */
        if (!vm.count(Option::exemplar()))
        {
            std::cerr << "At least one exemplar file must be specified.\n\n";
            usage(std::cerr, desc);
            std::exit(1);
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

static int reflect(int x, int size)
{
    while (x < 0) x += 2 * size;
    while (x >= 2 * size)
        x -= 2 * size;
    if (x >= size)
        x = 2 * size - 1 - x;
    return x;
}

// Weights taken from http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=pyrup#pyrup
// TODO: unify with code in database.cpp
static constexpr float gaussian5[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};

/**
 * Apply a 1D 5-tap Gaussian smoothing filter. Borders are handled by
 * reflection.
 */
static void smooth1D(
    const MapTraits<height_tag>::type *in,
    MapTraits<height_tag>::type *out,
    int size,
    int step)
{
    for (int x = 0; x < size; x++)
    {
        int xi = x - 2 * step;
        MapTraits<height_tag>::type sum = 0;
        for (int i = 0; i < 5; i++, xi += step)
            sum += gaussian5[i] * in[reflect(xi, size)];
        out[x] = sum;
    }
}

/**
 * Applies a 5x5 Gaussian smoothing filter. Borders are handled by
 * reflection. Samples are spaced @a step apart.
 *
 * @note It is tempting to consider making this an in-place transformation (or
 * at least allowing in-place transformations. It will @em almost work, but
 * the vertical reflection will cause issues.
 */
static MemMap<height_tag> smooth(const MemMap<height_tag> &in, int step)
{
    typedef MapTraits<height_tag>::type height_t;
    Region r = in.region();
    MemMap<height_tag> out(r);
    out.setStep(in.step());
    int W = r.width();
    int H = r.height();
    constexpr int WINDOW = 5;

    // Horizontally filtered rows
    uts::vector<height_t> rows[WINDOW];
    for (int i = 0; i < WINDOW; i++)
        rows[i].resize(W);

    for (int phase = 0; phase < step; phase++)
    {
        // Prime the pipeline
        for (int y = 0; y < WINDOW - 1; y++)
        {
            int yi = reflect(phase + (y - WINDOW / 2) * step, H);
            const height_t *inRow = in.get() + yi * W;
            smooth1D(inRow, rows[y].data(), W, step);
        }
        // Do vertical smoothing
        for (int y = phase; y < H; y += step)
        {
            int yi = reflect(y + WINDOW / 2 * step, H);
            const height_t *inRow = in.get() + yi * W;
            smooth1D(inRow, rows[WINDOW - 1].data(), W, step);

            height_t *outRow = out.get() + y * r.width();
            for (int x = 0; x < W; x++)
            {
                height_t sum = 0;
                for (int i = 0; i < WINDOW; i++)
                    sum += gaussian5[i] * rows[i][x];
                outRow[x] = sum;
            }

            // Move up the rows
            for (int i = 1; i < WINDOW; i++)
                rows[i - 1].swap(rows[i]);
        }
    }

    return out;
}

/**
 * Generate multiresolution noise measures from a heightmap. The returned
 * matrix has one column per element of @a in, and @a levels rows. The
 * elements are ordered from fine to coarse. The first @a skip octaves
 * are skipped.
 *
 * @note The input map is destroyed in the process.
 */
static Eigen::MatrixXf featureVectors(
    MemMap<height_tag> &&in,
    const uts::vector<Region> &regions,
    int levels, int skip)
{
    std::size_t pixels = 0;
    for (const Region &r : regions)
        pixels += r.pixels();

    Eigen::MatrixXf features(levels, pixels);
    for (int i = 0; i < skip; i++)
        in = smooth(in, 1 << i);

    for (int i = 0; i < levels; i++)
    {
        MemMap<height_tag> smoothed = smooth(in, 1 << (i + skip));
        std::size_t startPixel = 0;
        for (const Region &r : regions)
        {
#pragma omp parallel for schedule(static)
            for (int y = r.y0; y < r.y1; y++)
            {
                std::size_t p = startPixel + (y - r.y0) * r.width();
                for (int x = r.x0; x < r.x1; x++, p++)
                {
                    float delta = in[y][x] - smoothed[y][x];
                    features(i, p) = delta * delta;
                }
            }
            startPixel += r.pixels();
        }
        in = std::move(smoothed);
    }
    in.clear(); // contains garbage, might as well free the memory

    return features;
}

struct Exemplar
{
    MemMap<height_tag> heights;
    MemMap<mask_tag> masks;

    explicit Exemplar(const MemMap<height_and_mask_tag> &hm)
    {
        std::tie(heights, masks) = demultiplex(hm);
    }
};

struct TerrainType
{
    int index;
    std::size_t samples = 0;
    std::size_t keepSamples = 0;
};

/// Deletes an svm_model
struct ModelDelete
{
    void operator()(svm_model *model) const
    {
        svm_free_and_destroy_model(&model);
    }
};

/**
 * Loads the exemplars from disk and demultiplexes them.
 *
 * @param first, last   A forward iterator range of strings containing filenames.
 * @param step          Step size of the input file.
 *
 * @throw std::runtime_error if any of the exemplars has a bad step size relative to @a step.
 * @throw std::exception if an exception is thrown on loading
 */
template<typename ForwardIterator>
static uts::vector<Exemplar> loadExemplars(
    ForwardIterator first, ForwardIterator last, float step)
{
    uts::vector<Exemplar> exemplars;
    exemplars.reserve(std::distance(first, last));
    for (auto i = first; i != last; ++i)
    {
        const std::string &filename = *i;
        MemMap<height_and_mask_tag> hm(filename);
        if (hm.step() <= 0)
            throw std::runtime_error(filename + ": unknown resolution");
        if (!isPower2Ratio(hm.step(), step))
            throw std::runtime_error(filename + ": step size is not a power of two times input step size");
        exemplars.emplace_back(hm);
    }
    return exemplars;
}

/**
 * Returns a list of usable terrain types based on the exemplars.
 *
 * @param first, last    A forward iterator range of exemplars.
 * @param maxSamples     Upper bound for @c keepSamples
 */
template<typename ForwardIterator>
uts::vector<TerrainType> getTypes(
    ForwardIterator first, ForwardIterator last, std::size_t maxSamples)
{
    TerrainType types[NTYPES];
    for (int i = 0; i < NTYPES; i++)
        types[i].index = i;

    for (auto i = first; i != last; ++i)
    {
        const Exemplar &exemplar = *i;
        std::size_t pixels = exemplar.heights.region().pixels();
        for (std::size_t i = 0; i < pixels; i++)
        {
            mask_t mask = exemplar.masks.get()[i];
            if (mask != MapTraits<mask_tag>::all)
            {
                mask_t present = ~mask;
                for (int j = 0; j < NTYPES; j++)
                    if ((present >> j) & 1)
                    {
                        types[j].samples++;
                    }
            }
        }
    }

    uts::vector<TerrainType> ans;
    for (int i = 0; i < NTYPES; i++)
        if (types[i].samples >= 20)
        {
            types[i].keepSamples = std::min(maxSamples, types[i].samples);
            ans.push_back(types[i]);
        }
    return ans;
}

/**
 * Extract feature vectors and label from the exemplars.
 *
 * @param first, last      Iterator range over the exemplars
 * @param levels           Size of each feature vector
 * @param maxStep          Largest step size amongst exemplars
 * @param types            Valid terrain types and their counts
 * @param[out] featurePtrs Feature vectors per sample
 * @param[out] labels      Terrain type id corresponding to each feature
 *
 * @return Scaling factors per feature
 *
 * @note The exemplars are destroyed in the process.
 */
template<typename ForwardIterator>
static Eigen::VectorXf getTrainData(
    ForwardIterator first, ForwardIterator last,
    int levels,
    float maxStep,
    uts::vector<TerrainType> types,
    const uts::vector<struct svm_node *> &featurePtrs,
    uts::vector<double> &labels)
{
    int nextSample = 0;
    std::mt19937 engine;
    Eigen::VectorXf biggest = Eigen::VectorXf::Zero(levels);
    for (auto e = first; e != last; ++e)
    {
        Exemplar &exemplar = *e;
        const Region full = exemplar.masks.region();
        std::size_t pixels = full.pixels();
        int skip = (int) std::round(std::log2(maxStep / exemplar.masks.step()));
        Eigen::MatrixXf f = featureVectors(std::move(exemplar.heights), {full}, levels, skip);
        for (std::size_t i = 0; i < pixels; i++)
        {
            mask_t mask = exemplar.masks.get()[i];
            if (mask != MapTraits<mask_tag>::all)
            {
                biggest = biggest.cwiseMax(f.col(i));
                mask_t present = ~mask;
                for (TerrainType &t : types)
                    if ((present >> t.index) & 1)
                    {
                        // Decide whether to retain this sample
                        bool keep;
                        if (t.keepSamples == t.samples)
                            keep = true;
                        else
                        {
                            std::bernoulli_distribution dist(t.keepSamples / (double) t.samples);
                            keep = dist(engine);
                        }

                        if (keep)
                        {
                            for (int k = 0; k < levels; k++)
                            {
                                featurePtrs[nextSample][k].index = k;
                                featurePtrs[nextSample][k].value = f(k, i);
                            }
                            labels[nextSample] = t.index;
                            nextSample++;
                            t.keepSamples--;
                        }
                        t.samples--;
                    }
            }
        }
    }

    // Scale all training features to [0, 1] range
    Eigen::VectorXf scale = Eigen::VectorXf::Constant(levels, 1.0f).cwiseQuotient(biggest);
    for (auto feature : featurePtrs)
        for (int j = 0; j < levels; j++)
            feature[j].value *= scale(j);

    return scale;
}

static struct svm_parameter makeParameters(const svm_problem &problem, int numTypes)
{
    struct svm_parameter parameter = {};
    parameter.svm_type = C_SVC;
    parameter.kernel_type = RBF;
    parameter.gamma = 256.0 / numTypes; // tuned
    parameter.cache_size = 512; // in MB
    parameter.eps = 0.001;   // svm-train default
    parameter.C = 1024.0;    // tuned
    parameter.shrinking = 1; // svm-train default

    const char *error = svm_check_parameter(&problem, &parameter);
    if (error != nullptr)
    {
        throw std::runtime_error(error);
    }

    return parameter;
}

static uts::vector<Region> parseRegions(const std::string &spec, int width, int height)
{
    namespace qi = boost::spirit::qi;

    uts::vector<Region> out;
    uts::vector<std::pair<int, int> > positions;
    auto first = spec.begin();
    bool success = qi::parse(first, spec.end(), (qi::int_ >> ',' >> qi::int_) % '+', positions);
    if (!success || first != spec.end())
        throw std::runtime_error("Unexpected token at '" + std::string(first, spec.end()) + "'");

    for (const auto &pos : positions)
    {
        Region r(pos.first, pos.second, pos.first + width, pos.second + height);
        out.push_back(r);
    }
    return out;
}

static MemMap<height_and_mask_tag> predict(
    MemMap<height_tag> &&in,
    const uts::vector<Region> &regions,
    struct svm_model *model,
    int levels,
    float maxStep,
    const Eigen::VectorXf &scale)
{
    const Region full = in.region();
    MemMap<height_and_mask_tag> out(full);
    out.setStep(in.step());

    // Copy the heights, before featureVectors destroys them
#pragma omp parallel for schedule(static)
    for (int y = full.y0; y < full.y1; y++)
        for (int x = full.x0; x < full.x1; x++)
            out[y][x].height = in[y][x];

    int skip = (int) std::round(std::log2(maxStep / in.step()));
    Eigen::MatrixXf f = featureVectors(std::move(in), regions, levels, skip);

    std::size_t pixels = 0;
    for (const Region &r : regions)
        pixels += r.pixels();
    ProgressDisplay<> progress(pixels);
    std::size_t startPixel = 0;
    for (const Region &r : regions)
    {
#pragma omp parallel for schedule(dynamic,1)
        for (int y = r.y0; y < r.y1; y++)
        {
            std::size_t p = startPixel + (y - r.y0) * r.width();
            uts::vector<svm_node> node(levels + 1);
            node[levels].index = -1;
            for (int i = 0; i < levels; i++)
                node[i].index = i;
            for (int x = r.x0; x < r.x1; x++, p++)
            {
                for (int i = 0; i < levels; i++)
                    node[i].value = f(i, p) * scale(i);
                int t = (int) svm_predict(model, node.data());
                assert(t >= 0 && t < NTYPES);
                out[y][x].mask = ~(mask_t(1) << t);
            }
            progress += r.width();
        }
        startPixel += r.pixels();
    }

    return out;
}

static void print_svm_empty(const char *string)
{
    // Do nothing, to suppress output
}

static double crossValidate(const svm_problem &problem, const svm_parameter &parameter, int N)
{
    std::size_t numSamples = problem.l;
    std::vector<double> target(numSamples);

    svm_cross_validation(&problem, &parameter, N, target.data());
    std::size_t right = 0;
    for (std::size_t i = 0; i < numSamples; i++)
    {
        if (target[i] == problem.y[i])
            right++;
    }
    return (double) right / (double) numSamples;
}

static void explore(const svm_problem &problem, svm_parameter parameter, int N, int numTypes)
{
    for (int lC = 10; lC <= 16; lC++)
        for (int lG = 3; lG <= 9; lG++)
        {
            parameter.C = std::exp2(lC);
            parameter.gamma = std::exp2(lG) / numTypes;
            double ratio = crossValidate(problem, parameter, N);
            std::cout << "C = 2^" << lC << " g = 2^" << lG << " / " << numTypes << ": " << ratio * 100.0 << "%\n";
        }
}

int main(int argc, char **argv)
{
    utsInitialize();
    try
    {
        po::variables_map vm = processOptions(argc, argv);
        svm_set_print_string_function(print_svm_empty);
        float featureSize = vm[Option::featureSize()].as<float>();
        std::size_t maxSamples = vm[Option::maxSamples()].as<int>();

        MemMap<height_tag> in(vm[Option::inputFile()].as<std::string>());
        if (in.step() <= 0)
            throw std::runtime_error("Input file has unknown resolution");

        Region full = in.region();
        uts::vector<Region> regions;
        if (vm.count(Option::regions()))
        {
            regions = parseRegions(
                vm[Option::regions()].as<std::string>(),
                vm[Option::width()].as<int>(),
                vm[Option::height()].as<int>());
            for (const Region &r : regions)
            {
                if (!full.contains(r))
                {
                    std::ostringstream msg;
                    msg << "Region (" << r.x0 << ", " << r.y0 << ")(" << r.x1 << ", " << r.y1
                        << ") not contained in (" << full.x0 << ", " << full.y0 << ")(" << full.x1 << ", " << full.y1 << ")";
                    throw std::runtime_error(msg.str());
                }
            }
        }
        else
            regions.push_back(full);

        const std::vector<std::string> &exemplarFilenames = vm[Option::exemplar()].as<std::vector<std::string> >();
        uts::vector<Exemplar> exemplars = loadExemplars(exemplarFilenames.begin(), exemplarFilenames.end(), in.step());

        // Determine the coarsest of the exemplars and the input, which
        // determines which octaves can be used in training.
        float maxStep = in.step();
        for (const auto &e : exemplars)
            maxStep = std::max(maxStep, e.heights.step());
        int levels = std::floor(std::log2(featureSize / maxStep)) + 1;
        levels = std::max(levels, 1);
        levels = std::min(levels, 24); // danger of integer overflow beyond this, and no good reason for it

        std::vector<TerrainType> types = getTypes(exemplars.begin(), exemplars.end(), maxSamples);
        std::size_t numSamples = 0;
        for (const auto &t : types)
            numSamples += t.keepSamples;
        // libsvm takes number of training elements as an int
        if (numSamples > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("Too many samples for libsvm");
        }

        uts::vector<double> labels(numSamples);
        uts::vector<struct svm_node> features(numSamples * (levels + 1));
        uts::vector<struct svm_node *> featurePtrs(numSamples);
        for (std::size_t i = 0; i < numSamples; i++)
        {
            featurePtrs[i] = &features[i * (levels + 1)];
            featurePtrs[i][levels].index = -1;
        }
        Eigen::VectorXf scale =
            getTrainData(exemplars.begin(), exemplars.end(), levels, maxStep,
                         types, featurePtrs, labels);

        struct svm_problem problem;
        problem.l = numSamples;
        problem.y = labels.data();
        problem.x = featurePtrs.data();
        struct svm_parameter parameter = makeParameters(problem, types.size());

        if (vm.count(Option::cross()))
        {
            int N = vm[Option::cross()].as<int>();
            if (vm.count(Option::explore()))
            {
                explore(problem, parameter, N, types.size());
            }
            else
            {
                double ratio = crossValidate(problem, parameter, N);
                std::cout << "Cross-validation: " << ratio * 100.0 << "% accuracy\n";
            }
            return 0;
        }
        else if (vm.count(Option::explore()))
        {
            throw std::runtime_error("--explore requires --cross");
        }

        struct svm_model *modelPtr = svm_train(&problem, &parameter);
        std::unique_ptr<svm_model, ModelDelete> model(modelPtr);
        std::cout << "Training complete, nSV = " << model->l << '\n';

        MemMap<height_and_mask_tag> out =
            predict(std::move(in), regions, model.get(), levels, maxStep, scale);
        std::string outFilename;
        if (vm.count(Option::outputFile()))
            outFilename = vm[Option::outputFile()].as<std::string>();
        else
            outFilename = vm[Option::inputFile()].as<std::string>();
        out.write(outFilename);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
