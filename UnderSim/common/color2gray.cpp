#include <stdexcept>
#include <array>
#include <cstddef>
#include <type_traits>
#include <ImfFrameBuffer.h>
#include "map.h"
#include "map_rgba.h"
#include "initialize.h"

int main(int argc, char **argv)
{
    utsInitialize();
    if (argc != 3)
    {
        std::cerr << "Usage: color2gray input.exr output.exr\n";
        return 1;
    }

    try
    {
        MemMap<gray_tag> in;
        in.read(argv[1]);
        in.write(argv[2]);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
