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

#include "ConfigReader.h"
#include <fstream>
#include <sstream>

ConfigReader::ConfigReader(std::string filename)
    : filename(filename), has_read(false)
{
    params.scene_dirname.clear();
    params.clusterdata_filenames.clear();
    params.canopy_filename.clear();
    params.undergrowth_filename.clear();
    params.ctrlmode = ControlMode::VIEW;
    params.render_canopy = false;
    params.render_undergrowth = false;
}

bool ConfigReader::read()
{
    using namespace rapidjson;

    std::string contents;

    std::ifstream ifs(filename);
    std::stringstream sstr;

    if (!ifs.is_open())
    {
        std::cout << "Could not open file at " << filename << std::endl;
        return false;
    }
    else
    {
        sstr << ifs.rdbuf();
        contents = sstr.str();
        std::cout << "json contents: " << std::endl;
        std::cout << contents << std::endl;
        std::cout << "Parsing..." << std::endl;
        jsondoc.Parse(contents.c_str());
        std::cout << "Done parsing json file at " << filename << std::endl;
        std::cout << "Document is object? " << jsondoc.IsObject() << std::endl;

        params.scene_dirname = "";
        Value::ConstMemberIterator iter = jsondoc.FindMember("scene_dirname");
        if (iter != jsondoc.MemberEnd())
            params.scene_dirname = iter->value.GetString();

        std::cout << "Scene dirname: " << params.scene_dirname << std::endl;

        params.clusterdata_filenames.clear();
        iter = jsondoc.FindMember("clusterdata_filenames");
        if (iter != jsondoc.MemberEnd())
        {
            for (const auto &fname : iter->value.GetArray())
            {
                params.clusterdata_filenames.push_back(fname.GetString());
            }
        }

        params.canopy_filename = "";
        iter = jsondoc.FindMember("canopy_filename");
        if (iter != jsondoc.MemberEnd())
        {
            params.canopy_filename = iter->value.GetString();
        }

        params.undergrowth_filename = "";
        iter = jsondoc.FindMember("undergrowth_filename");
        if (iter != jsondoc.MemberEnd())
        {
            params.undergrowth_filename = iter->value.GetString();
        }

        params.ctrlmode = ControlMode::VIEW;
        iter = jsondoc.FindMember("ctrlmode");
        if (iter != jsondoc.MemberEnd())
        {
            if (!strcmp(iter->value.GetString(), "VIEW"))
                params.ctrlmode = ControlMode::VIEW;
            else if (!strcmp(iter->value.GetString(), "PAINTLEARN"))
                params.ctrlmode = ControlMode::PAINTLEARN;
            else if (!strcmp(iter->value.GetString(), "PAINTSPECIES"))
                params.ctrlmode = ControlMode::PAINTSPECIES;
            else if (!strcmp(iter->value.GetString(), "UNDERGROWTH_SYNTH"))
                params.ctrlmode = ControlMode::UNDERGROWTH_SYNTH;
            else if (!strcmp(iter->value.GetString(), "CANOPYTREE_ADD"))
                params.ctrlmode = ControlMode::CANOPYTREE_ADD;
            else if (!strcmp(iter->value.GetString(), "CANOPYTREE_REMOVE"))
                params.ctrlmode = ControlMode::CANOPYTREE_REMOVE;
            else
                params.ctrlmode = ControlMode::VIEW;
        }

        params.render_canopy = false;
        iter = jsondoc.FindMember("render_canopy");
        if (iter != jsondoc.MemberEnd())
        {
            params.render_canopy = iter->value.GetBool();
        }

        params.render_undergrowth = false;
        iter = jsondoc.FindMember("render_undergrowth");
        if (iter != jsondoc.MemberEnd())
        {
            params.render_undergrowth = iter->value.GetBool();
        }
    }

    has_read = true;

    return true;
}

configparams ConfigReader::get_params()
{
    if (!has_read)
        std::cout << "Warning: ConfigReader::get_params() called without successful read!" << std::endl;
    return params;
}
