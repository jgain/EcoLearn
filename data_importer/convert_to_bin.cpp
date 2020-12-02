/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include "AbioticMapper.h"
#include "data_importer.h"

void convert_ifnot_bin(std::string fname)
{
	if (!data_importer::data_dir::is_binary(fname))
	{
		auto mmap = data_importer::read_monthly_map<ValueGridMap<float> >(fname);

		std::string binfname = fname.substr(0, fname.size() - 3);
		binfname += "bin";
		data_importer::write_monthly_map_binary(binfname, mmap);
	}
}

int main(int argc, char * argv [])
{
	if (argc != 2)
	{
		std::cout << "Usage: convert_to_bin <dirname>" << std::endl;
		return 1;
	}

	data_importer::data_dir ddir(argv[1]);

	convert_ifnot_bin(ddir.temp_fname);
	convert_ifnot_bin(ddir.sun_fname);
	convert_ifnot_bin(ddir.wet_fname);

	return 0;
}
