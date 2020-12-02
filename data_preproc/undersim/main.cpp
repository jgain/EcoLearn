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
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 ********************************************************************************/


#include <UnderSim/sim/sim.h>
#include <data_importer/data_importer.h>
#include <UnderSim/sim/pft.h>

// NOTE: we do not write the code to the SubBiome object. It does not seem to get used anyway and it would require
// some extra effort to import
SubBiome data_sb_to_SubBiome(const data_importer::sub_biome & sb)
{
    SubBiome bme;
    for (auto sp : sb.species)
    {
        if (sp.canopy)
        {
            bme.canopies.push_back(sp.id);
        }
        else
        {
            bme.understorey.push_back(sp.id);
        }
    }
    return bme;
}

PFType species_to_PFType(const data_importer::species &sp)
{
    PFType pft;
    pft.alm_a = sp.a;
    pft.alm_b = sp.b;
    pft.alm_rootmult = sp.alm_rootmult;
    pft.allometry_code = sp.allometry_code;
    pft.alpha = sp.alpha;
    for (int i = 0; i < 4; i++)
        pft.basecol[i] = sp.basecol[i];
    pft.code = sp.name;
    pft.draw_box1 = sp.draw_box1;
    pft.draw_box2 = sp.draw_box2;
    pft.draw_hght = sp.draw_hght;
    pft.draw_radius = sp.draw_radius;
    pft.growth_period = sp.growth_period;
    pft.grow_c1 = sp.grow_c1;
    pft.grow_c2 = sp.grow_c2;
    pft.grow_end = sp.grow_end;
    pft.grow_m = sp.grow_m;
    pft.grow_months = sp.grow_months;
    pft.grow_start = sp.grow_start;
    pft.maxage = sp.maxage;
    pft.maxhght = sp.maxhght;
    pft.shapetype = (TreeShapeType)sp.shapetype;

    data_importer::viability sun = sp.sun, wet = sp.wet, slope = sp.slope, temp = sp.temp;
    pft.sun.setValues(sun.cmin, sun.cmax, sun.c, sun.r);
    pft.wet.setValues(wet.cmin, wet.cmax, wet.c, wet.r);
    pft.slope.setValues(slope.cmin, slope.cmax, slope.c, slope.r);
    pft.temp.setValues(temp.cmin, temp.cmax, temp.c, temp.r);

    return pft;
}

Biome *create_biome_from_commondata(data_importer::common_data common)
{
    Biome *biome = new Biome();
    biome->evaporation = common.soil_info.evap;
    biome->runofflevel = common.soil_info.runofflim;
    biome->slopemax = common.soil_info.slopemax;
    biome->slopethresh = common.soil_info.slopethresh;
    biome->soilsaturation = common.soil_info.soilsat;
    biome->waterlevel = common.soil_info.riverlevel;		// XXX: is this correct???

    int nspecies = common.canopy_and_under_species.size();

    int prev_idx = -1;
    for (auto &sppair : common.canopy_and_under_species)
    {
        assert(sppair.first - prev_idx == 1);		// we need to make sure that indices follow one another contiguously, otherwise we need a different way of indexing for the undergrowth sim
        prev_idx = sppair.first;
    }

    std::cout << "Adding plant functional types..." << std::endl;

    // all indices follow contiguously (if we got to this point) so we can safely find indices in the map and push_back into the vector, and indices will correspond
    for (int i = 0; i < nspecies; i++)
    {
        data_importer::species sp = common.canopy_and_under_species[i];
        biome->add_pftype(species_to_PFType(sp));
    }

    for (auto &subb : common.subbiomes_all_species)
    {
        biome->add_subbiome(data_sb_to_SubBiome(subb.second), subb.first);
    }
}

int main(int argc, char * argv [])
{
    if (argc != 3)
    {
        std::cout << "Usage: undersim <data directory> <sql database filename>" << std::endl;
        return 1;
    }

    data_importer::common_data common(argv[2]);
    data_importer::data_dir ddir(argv[1]);

    std::string elv_filename = ddir.dem_fname;
    Terrain *ter = new Terrain();
    ter->loadElv(elv_filename);		//xxx: set elv_filename first

    Biome *biome = create_biome_from_commondata(common);

    Simulation sim(ter, biome, 20);

    Ecosystem *eco = new EcoSystem(biome);

    std::string seedbank_filename;
    std::string seedchance_filename;

    sim.simulate(eco, seedbank_filename, seedchance_filename, 10);

    delete biome;
    delete ter;

	return 0;
}

