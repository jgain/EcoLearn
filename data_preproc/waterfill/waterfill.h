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


#ifndef WATERFILL_H
#define WATERFILL_H

#include "data_importer/data_importer.h"

#include <vector>
#include <cstdint>
#include <string>
// class Terrain;

class WaterFill
{
public:
	typedef unsigned int uint;

    WaterFill()
        : twidth(0), theight(0), gwidth(0), gheight(0)
    {}
    WaterFill(const ValueMap<float> &tgrid, int twidth, int theight);

    void setTerrain(ValueMap<float> &tgrid, int tw, int th)
    {
        this->tgrid = tgrid;
        //tgrid.getDim(gwidth, gheight);
        this->tgrid.getDim(gwidth, gheight);
        twidth = tw;
        theight = th;
        precipitation = 0.1f; // m per year
        river_width_constant = 0.00178f; //  y^(1/2)m^(-1/2)
    }
    void setAbsorbsion(unsigned int a){absorbsion = a;}

    void compute(uint step = (uint)(-1));

    void expandRivers(float max_moisture_factor, float slope_effect);

    std::vector<uint> getRiverMoisture() {return river_side;}

    bool isFlowingWater(uint x, uint y);
    float riverMoisture(uint x, uint y);

    void reset();
    void addWaterInflow(uint x, uint y, int delta);
    void smartWaterInflow(uint x, uint y);

	float getFlatHeight(int idx);
	float getHeight(int x, int y);
	float getCellArea();
	float getCellExtent();

	std::vector<uint16_t> get_16bit_img_greyscale_data();

#ifdef IMG_WRITE
	void write_to_greyscale_png(std::string filename);
#endif // IMG_WRITE

private:
    std::vector<uint> flow, river_side, inflow;
    std::vector<double> lakes;
    //Terrain* terrain;
    ValueMap<float> tgrid;
	int twidth, theight;
	unsigned int gwidth, gheight;

    uint absorbsion;
    float precipitation;
    float river_width_constant;

	class wf_point
	{
	public:
		wf_point()
			: x_val(0), y_val(0)
		{}

		wf_point(int x, int y)
			: x_val(x), y_val(y)
		{}

		int x() const { return x_val; }
		int y() const { return y_val; }

		wf_point operator + (const wf_point &other) const
		{
			return wf_point(x() + other.x_val, y() + other.y_val);
		}

		bool operator == (const wf_point &other) const
		{
			return other.x_val == this->x_val && other.y_val == this->y_val;
		}

		bool operator != (const wf_point &other) const
		{
            return !(*this == other);
		}
		
	private:
		int x_val, y_val;
	};
};

#endif // WATERFILL_H
