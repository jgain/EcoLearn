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
#include "common/basic_types.h"
#include "data_importer/data_importer.h"
#include "data_importer/map_procs.h"
#include <chrono>

abiotic_maps_package::abiotic_maps_package(const ValueGridMap<float> &wet,
             const ValueGridMap<float> &sun,
             const ValueGridMap<float> &temp,
             const ValueGridMap<float> &slope)
    : wet(wet), sun(sun), temp(temp), slope(slope)
{
    validate_maps();
}

// TODO: let the caller optionally give a monthly, or average filename for moisture and sunlight
abiotic_maps_package::abiotic_maps_package(std::string wet_fname, std::string sun_fname, std::string temp_fname, std::string slope_fname, aggr_type aggr)
{
    if (aggr == aggr_type::AVERAGE)
    {
        slope = data_importer::load_txt<ValueGridMap<float> >(slope_fname);

        std::map<std::string, ValueGridMap<float> *> datas;
        std::map<std::string, std::string> fnames;

        std::vector<std::string> types = {"temp", "sun", "wet"};
        fnames["temp"] = temp_fname;
        fnames["sun"] = sun_fname;
        fnames["wet"] = wet_fname;
        datas["temp"] = &temp;
        datas["sun"] = &sun;
        datas["wet"] = &wet;

        for (auto &tp : types)
        {
            std::string fname = fnames[tp];
            auto bt = std::chrono::steady_clock::now().time_since_epoch();
            std::vector<ValueGridMap<float> > vec;
            if (fname.substr(fname.size() - 3, 3) == "bin")
                vec = data_importer::read_monthly_map_binary<ValueGridMap<float> >(fname);
            else if (fname.substr(fname.size() - 3, 3) == "txt")
                vec = data_importer::read_monthly_map<ValueGridMap<float> >(fname);
            else
                throw std::invalid_argument(tp + " name argument " + fname + " has an unknown extension");
            auto et = std::chrono::steady_clock::now().time_since_epoch();
            std::cout << "Time for reading " << tp <<  " data: " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << std::endl;
            //temp = average_monthly_data_hostcall(tempvec, w, h, rw, rh);
            *datas[tp] = average_monthly_data_hostcall(vec);
            auto finet = std::chrono::steady_clock::now().time_since_epoch();
            //last_rw = rw, last_rh = rh, last_w = w, last_h = h;

            std::cout << "Time for averaging " << tp << " data: " << std::chrono::duration_cast<std::chrono::milliseconds>(finet - et).count() << std::endl;
        }
    }
    else
    {
        int monthidx = static_cast<int>(aggr);
        slope = data_importer::load_txt<ValueGridMap<float> >(slope_fname);
        temp = data_importer::read_monthly_map<ValueGridMap<float>>(temp_fname).at(monthidx);
        wet = data_importer::read_monthly_map<ValueGridMap<float>>(wet_fname).at(monthidx);
        sun = data_importer::read_monthly_map<ValueGridMap<float>>(sun_fname).at(monthidx);
    }
    validate_maps();
}

abiotic_maps_package::abiotic_maps_package(data_importer::data_dir targetdir, suntype sun, aggr_type aggr)
    : abiotic_maps_package(targetdir.wet_fname, sun == suntype::CANOPY ? targetdir.sun_tree_fname : targetdir.sun_fname, targetdir.temp_fname, targetdir.slope_fname, aggr)
{}

void abiotic_maps_package::validate_maps()
{
    wet.getDimReal(rw, rh);
    wet.getDim(gw, gh);
    if (fabs(rw) < 1e-5f || fabs(rh) < 1e-5f)
    {
        throw std::runtime_error("Moisture abiotic map does not have a step parameter");
    }
    sun.getDimReal(rw, rh);
    if (fabs(rw) < 1e-5f || fabs(rh) < 1e-5f)
    {
        throw std::runtime_error("Sunlight abiotic map does not have a step parameter");
    }
    temp.getDimReal(rw, rh);
    if (fabs(rw) < 1e-5f || fabs(rh) < 1e-5f)
    {
        throw std::runtime_error("Temperature abiotic map does not have a step parameter");
    }
    slope.getDimReal(rw, rh);
    if (fabs(rw) < 1e-5f || fabs(rh) < 1e-5f)
    {
        throw std::runtime_error("Slope abiotic map does not have a step parameter");
    }
    if (!slope.eqdim(temp) || !slope.eqdim(sun) || !slope.eqdim(wet))
    {
        throw std::runtime_error("Some abiotic maps are not equal in terms of grid or real size");
    }
}
