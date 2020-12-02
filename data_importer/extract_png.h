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


#include <vector>
#include <cstdint>
#include <string>
#include <cstdint>

std::vector<std::vector<uint16_t> > get_image_vals_48bit(const char * img_path_param, int string_len, int &width, int &height);
std::vector<std::vector<uint8_t> > get_image_vals_8bit(const char * img_path_param, int string_len, int &img_width, int &img_height);
int write_png(std::string img_path, const std::vector<uint16_t> &data, int width, int height);
std::vector<std::vector<float> > get_image_data_48bit(std::string img_path, int &img_width, int &img_height);
std::vector<std::vector<float> > get_image_data_8bit(std::string img_path, int &img_width, int &img_height);
std::vector<uint16_t> create_img_data(const float *data, int size);
std::vector<uint16_t> create_img_data(const std::vector<float> &data);
int write_png(std::string img_path, const float *data, int width, int height);
int write_png(std::string img_path, const std::vector<float> &data, int width, int height);
