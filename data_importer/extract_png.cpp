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


#include <libpng16/png.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <cstring>

#include "extract_png.h"

using namespace std;

static int width, height;
static png_bytep *row_pointers;
static png_byte nchannels_byte;
static png_byte color_type_byte;
static png_byte bit_depth_byte;

bool read_png_file(char *filename) {
  FILE *fp = fopen(filename, "rb");

  if (!fp)
  {
	  cout << "Could not open file at " << filename << endl;
	  return false;
  }

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(!png) abort();

  png_infop info = png_create_info_struct(png);
  if(!info) abort();

  if(setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  png_read_info(png, info);

  width      = png_get_image_width(png, info);
  height     = png_get_image_height(png, info);
  color_type_byte = png_get_color_type(png, info);
  bit_depth_byte  = png_get_bit_depth(png, info);
  nchannels_byte  = png_get_channels(png, info);

  cout << "bit depth of image: " << (uint32_t)bit_depth_byte << endl;
  cout << "number of channels: " << (uint32_t)nchannels_byte << endl;
  cout << "Width, height: " << width << ", " << height << endl;

  if(bit_depth_byte == 16)
  {
	png_set_swap(png);
  }

  if(color_type_byte == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png);

  if(png_get_valid(png, info, PNG_INFO_tRNS))
    png_set_tRNS_to_alpha(png);

  switch (color_type_byte)
  {
	  case(PNG_COLOR_TYPE_RGB):
		  cout << "Color type is RGB" << endl;
		  break;
	  default:
		  break;
  }

  png_read_update_info(png, info);

  row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
  for(int y = 0; y < height; y++) {
    row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
  }

  png_read_image(png, row_pointers);
  
  bool not_all_equal = false;
  for (int i = 0; i < width * 6; i += 6)
  {
		uint32_t prev_val = (uint32_t)*(uint16_t *)&row_pointers[0][i];
		for (int row = 1; row < height; row++)
		{
			uint32_t val = (uint32_t)*(uint16_t *)&row_pointers[row][i];
			if (val != prev_val)
			{
				not_all_equal = true;
			}
			prev_val = val;
		}
  }

  fclose(fp);

  return true;
}

uint8_t ** make_row_ptrs(const std::vector<uint16_t> &data, int width, int height)
{
	uint8_t **row_ptrs = (uint8_t**)malloc(height * sizeof(uint8_t*));
	for (int row = 0; row < height; row++)
	{
		row_ptrs[row] = (uint8_t*)((uint16_t*)data.data() + 3*width * row);
	}

	return row_ptrs;
}

int write_png(std::string img_path, const std::vector<uint16_t> &data, int width, int height)
{
	FILE *fp = fopen(img_path.c_str(), "wb");
	if (!fp)
		return 1;

	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (!png_ptr)
		return 1;

	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		return 1;

	if (setjmp(png_jmpbuf(png_ptr)))
		return 1;

	png_init_io(png_ptr, fp);

	png_set_IHDR(
		png_ptr,
		info_ptr, 
		width, height,
		16,
		PNG_COLOR_TYPE_RGB,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
	);

	png_write_info(png_ptr, info_ptr);

	uint8_t ** row_ptrs = make_row_ptrs(data, width, height);
	
	for (int row = 0; row < height; row++)
	{
		for (int i = 0; i < width * 6; i += 2)
		{
			swap(row_ptrs[row][i], row_ptrs[row][i + 1]);
			//uint32_t val = (uint32_t)*(uint16_t *)&row_ptrs[row][i];
		}
	}

	png_write_image(png_ptr, row_ptrs);
	png_write_end(png_ptr, NULL);

	bool not_all_equal = false;
	for (int i = 0; i < width * 6; i += 6)
	{
		  uint32_t prev_val = (uint32_t)*(uint16_t *)&row_ptrs[0][i];
		  for (int row = 1; row < height; row++)
		  {
			  uint32_t val = (uint32_t)*(uint16_t *)&row_ptrs[row][i];
			  if (val != prev_val)
			  {
				  not_all_equal = true;
			  }
			  prev_val = val;
		  }
	}

	free(row_ptrs);

	fclose(fp);

	return 0;
}

int write_png(std::string img_path, const std::vector<float> &data, int width, int height)
{
	vector<uint16_t> img_data = create_img_data(data);
	return write_png(img_path, img_data, width, height);
}

int write_png(std::string img_path, const float *data, int width, int height)
{
    vector<uint16_t> img_data = create_img_data(data, width * height);
    return write_png(img_path, img_data, width, height);
}


vector<vector<uint16_t> > get_image_vals_48bit(const char * img_path_param, int string_len, int &img_width, int &img_height)
{

	char * img_path = (char *)malloc(sizeof(char) * string_len);
	memcpy(img_path, img_path_param, string_len * sizeof(char));

	if (!read_png_file(img_path))
	{
		return {};
	}

	img_width = width;
	img_height = height;
	vector< vector<uint16_t> > vecs((uint32_t)nchannels_byte, vector<uint16_t>(width * height));
	//vector<uint16_t> vals(width * height);
	uint8_t *row_ptr;
	int stride = (uint32_t)nchannels_byte * (uint32_t)bit_depth_byte / 8;	//number of bytes per pixel
	//cout << "stride: " << stride << endl;

	for (int row = 0; row < height; row++)
	{
		row_ptr = row_pointers[row];
		for (int col = 0; col < width; col++)
		{
			uint16_t val;
			for (int channel = 0; channel < (uint32_t)nchannels_byte; channel++)
			{
				uint8_t *val_ptr = (uint8_t*)row_ptr + col * stride + channel * (uint32_t)bit_depth_byte / 8;
				val = *((uint16_t *)val_ptr);
				vecs[channel][width * row + col] = val;
			}
		}
	}

	for(int y = 0; y < height; y++) {
	  free(row_pointers[y]);
	}
	free(row_pointers);
	free(img_path);

	return vecs;
}

vector<vector<uint8_t> > get_image_vals_8bit(const char * img_path_param, int string_len, int &img_width, int &img_height)
{
    char * img_path = (char *)malloc(sizeof(char) * string_len);
    memcpy(img_path, img_path_param, string_len * sizeof(char));

    if (!read_png_file(img_path))
    {
        return {};
    }

    img_width = width;
    img_height = height;
    vector< vector<uint8_t> > vecs((uint32_t)nchannels_byte, vector<uint8_t>(width * height));
    //vector<uint16_t> vals(width * height);
    uint8_t *row_ptr;
    int stride = (uint32_t)nchannels_byte * (uint32_t)bit_depth_byte / 8;	//number of bytes per pixel
    //cout << "stride: " << stride << endl;

    for (int row = 0; row < height; row++)
    {
        row_ptr = row_pointers[row];
        for (int col = 0; col < width; col++)
        {
            uint8_t val;
            for (int channel = 0; channel < (uint32_t)nchannels_byte; channel++)
            {
                uint8_t *val_ptr = (uint8_t*)row_ptr + col * stride + channel * (uint32_t)bit_depth_byte / 8;
                val = *((uint8_t *)val_ptr);
                vecs[channel][width * row + col] = val;
            }
        }
    }

    for(int y = 0; y < height; y++) {
      free(row_pointers[y]);
    }
    free(row_pointers);
    free(img_path);

    return vecs;
}

vector<uint16_t> create_img_data(const vector<float> &data)
{
	vector<uint16_t> img_data(3 * data.size());

	for (int i = 0; i < data.size(); i++)
	{
		for (int channel = 0; channel < 3; channel++)
		{
			img_data[i * 3 + channel] = static_cast<uint16_t>(data[i]);
		}
	}

	return img_data;
}

vector<uint16_t> create_img_data(const float *data, int size)
{
    vector<uint16_t> img_data(3 * size);

    for (int i = 0; i < size; i++)
    {
        for (int channel = 0; channel < 3; channel++)
        {
            img_data[i * 3 + channel] = static_cast<uint16_t>(data[i]);
        }
    }

    return img_data;
}

vector<vector<float> > get_image_data_48bit(string img_path, int &img_width, int &img_height)
{
	vector< vector<uint16_t> > vecs = get_image_vals_48bit(img_path.c_str(), img_path.size() + 1, img_width, img_height);
	if (vecs.size() == 0)
	{
		return {};
	}
	vector<vector<float> > fvecs(vecs.size(), vector<float>(vecs[0].size()));

	auto cast_for_transform = [](const uint16_t &val) { return static_cast<float>(val); };

	for (int i = 0; i < fvecs.size(); i++)
	{
		int zero_count = 0;
		uint32_t value_sum = 0;
		for (auto &v : vecs[i])
		{
			if (v == 0)
				zero_count++;
			value_sum += v;
		}
		transform(vecs[i].begin(), vecs[i].end(), fvecs[i].begin(), cast_for_transform);
	}

	return fvecs;
}

vector<vector<float> > get_image_data_8bit(string img_path, int &img_width, int &img_height)
{
    vector< vector<uint8_t> > vecs = get_image_vals_8bit(img_path.c_str(), img_path.size() + 1, img_width, img_height);
    if (vecs.size() == 0)
    {
        return {};
    }
    vector<vector<float> > fvecs(vecs.size(), vector<float>(vecs[0].size()));

    auto cast_for_transform = [](const uint8_t &val) { return static_cast<float>(val); };

    for (int i = 0; i < fvecs.size(); i++)
    {
        int zero_count = 0;
        uint32_t value_sum = 0;
        for (auto &v : vecs[i])
        {
            if (v == 0)
                zero_count++;
            value_sum += v;
        }
        transform(vecs[i].begin(), vecs[i].end(), fvecs[i].begin(), cast_for_transform);
    }

    return fvecs;
}

/*
int main(int argc, char *argv[]) {

  if(argc != 2) 
  {
	  std::cout << "usage: ./libpng_test <input_file>" << std::endl;
	  exit(1);
  }

  int string_len = 0;

  char * ch = argv[1];
  while (*(ch++) != '\0' && string_len < 512)
  {
	  string_len++;
  }
  if (string_len == 512)
  {
	  std::cout << "string argument must be less than 512 characters" << std::endl;
	  exit(1);
  }
  string_len++;

  vector<uint16_t> vals = get_image_vals_48bit(argv[1], string_len);

  for (int i = 0; i < width / 2; i++)
  {
	  cout << (uint32_t)vals[i] << " ";
  }
  cout << endl;

  return 0;
}
*/
