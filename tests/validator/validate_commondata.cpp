#include "data_importer.h"

#include <fstream>
#include <chrono>

int main(int argc, char * argv [])
{

    if (argc != 3)
    {
        std::cout << "Usage: validate_commondata <db pathname> <output_pathname>" << std::endl;
        return 1;
    }
    std::string dbpathname = argv[1];
    std::string output_pathname = argv[2];

    data_importer::common_data cdata(dbpathname);

    std::ofstream ofs(output_pathname);

    srand(std::chrono::steady_clock::now().time_since_epoch().count());

    for (int i = 0; i < 10000; i++)
    {
        int specid = rand() % 16;
        float height = (rand() % 9500 + 500) / 10000.0f * cdata.canopy_and_under_species.at(specid).maxhght;

        int vueid;
        float radius = cdata.modelsamplers.at(specid).sample_rh_ratio(height, &vueid) * height;

        std::string buf;
        buf.resize(256);
        int nchar = sprintf((char *)buf.data(), "%d %f %f %d %f", specid, height, radius, vueid, radius * 2 / height);
        buf.resize(nchar);
        //printf("--------------------------------------\n");
        ofs << buf << std::endl;
    }

    return 0;
}
