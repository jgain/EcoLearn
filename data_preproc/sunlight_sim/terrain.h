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


#ifndef TERRAIN_H
#define TERRAIN_H

#include "common/basic_types.h"
#include <glm/glm.hpp>

class terrain
{
public:
    terrain(float *data, int width, int height, glm::vec3 north, float latitude);
    terrain(const ValueMap<float> &data, glm::vec3 north, float latitude);

    float at(int x, int y) const;
    float *get_data();

    int get_width() const;
    int get_height() const;
    void get_size(int &w, int &h) const;
    glm::vec3 get_north() const;
    float get_latitude() const;
    float get_extent() const;


    terrain(const terrain &ter);
private:
    //float *data;
    ValueMap<float> data;
    int width;
    int height;
    glm::vec3 north;
    float latitude;
    float extent;
};

#endif // TERRAIN_H
