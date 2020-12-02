#include <iostream>
#include <locale>
#include <stdexcept>
#include "debug_string.h"
#include "map.h"
#include "initialize.h"

static void usage()
{
    std::cerr << "Usage: map2exr input.map output.exr\n";
}

static MemMap<height_tag> loadMap(const uts::string &filename)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Could not open " + filename);

    MemMap<height_tag> ans;
    try
    {
        in.imbue(std::locale::classic());
        in.exceptions(std::ios::failbit | std::ios::badbit);
        int R, C;
        in >> C >> R;
        if (R <= 0 || C <= 0)
            throw std::runtime_error("dimensions are invalid");
        ans.allocate({0, 0, R, C});
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                in >> ans[r][c];
        in.close();
    }
    catch (std::runtime_error &e)
    {
        throw std::runtime_error("Failed to read " + filename + ": " + e.what());
    }
    return std::move(ans);
}

int main(int argc, char **argv)
{
    utsInitialize();
    try
    {
        if (argc != 3)
        {
            usage();
            return 1;
        }
        MemMap<height_tag> m = loadMap(argv[1]);
        m.write(argv[2]);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
