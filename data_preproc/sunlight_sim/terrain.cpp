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


#include "terrain.h"

#include <stdexcept>

terrain::terrain(float *data, int width, int height, glm::vec3 north, float latitude)
    : width(width),
      height(height),
      north(north),
      latitude(latitude),
      extent(std::sqrt(width * width + height * height))
{
    this->data.setDim(width, height);
    memcpy(this->data.data(), data, sizeof(float) * width * height);
}

terrain::terrain(const ValueMap<float> &data, glm::vec3 north, float latitude)
    : data(data),
      north(north),
      latitude(latitude)
{
    this->data.getDim(width, height);
    extent = std::sqrt(width * width + height * height);
}

terrain::terrain(const terrain &ter)
{
    width = ter.width;
    height = ter.height;
    north = ter.north;
    data = ter.data;
    latitude = ter.latitude;
    extent = ter.extent;
}

float terrain::at(int x, int y) const
{
    if (x < 0 || x >= width || y < 0 || y >= height)
    {
        throw std::out_of_range("terrain::at received out of range arguments");
    }
    return data.get(x, y);
}

float *terrain::get_data()
{
    return data.data();
}

int terrain::get_width() const
{
    return width;
}

int terrain::get_height() const
{
    return height;
}

float terrain::get_extent() const
{
    return extent;
}

void terrain::get_size(int &w, int &h) const
{
    w = width;
    h = height;
}

float terrain::get_latitude() const
{
    return latitude;
}

glm::vec3 terrain::get_north() const
{
    return north;
}
