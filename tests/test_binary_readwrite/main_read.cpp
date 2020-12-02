#include "../data_importer/data_importer.h"
#include <chrono>

int main(int argc, char * argv [])
{

	std::string mapname = "/home/konrad/PhDStuff/S1000-1000-3072_sun_landscape.txt";

	std::string mapold = "/home/konrad/PhDStuff/abioticfixed/S1000-1000-3072/S1000-1000-3072_wet.txt";

	std::string bininname = "/home/konrad/PhDStuff/binwet_out.bin";

	auto bt = std::chrono::steady_clock::now().time_since_epoch();

	auto mmap = data_importer::read_monthly_map_binary<ValueGridMap<float> >(bininname);

	auto et = std::chrono::steady_clock::now().time_since_epoch();

	std::cout << "Time to read file " << bininname << " (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << std::endl;

	bt = std::chrono::steady_clock::now().time_since_epoch();

	auto mmap_avg = data_importer::average_mmap<ValueGridMap<float> >(mmap);

	et = std::chrono::steady_clock::now().time_since_epoch();

	std::cout << "Time to average monthly map: " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << std::endl;

	/*
	bt = std::chrono::steady_clock::now().time_since_epoch();

	auto mmap2 = data_importer::read_monthly_map<ValueGridMap<float> >(mapold);

	et = std::chrono::steady_clock::now().time_since_epoch();

	std::cout << "Time to read file " << mapold << " (ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << std::endl;
	*/

	//data_importer::write_monthly_map(mapname, mmap);

	return 0;
}
