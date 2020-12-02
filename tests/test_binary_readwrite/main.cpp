#include "../data_importer/data_importer.h"

int main(int argc, char * argv [])
{

	std::string mapname = "/home/konrad/PhDStuff/abioticfixed/S1000-1000-3072/S1000-1000-3072_wet.txt";

	std::string outname = "/home/konrad/PhDStuff/binwet_out.bin";

	auto mmap = data_importer::read_monthly_map<ValueGridMap<float> >(mapname);

	data_importer::write_monthly_map_binary(outname, mmap);

	return 0;
}
