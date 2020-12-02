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


#include "gpu_procs.h"
#include "extract_png.h"

int main(int argc, char * argv [])
{
    int factor = 8;
    int srcw, srch;
    std::vector<float> data = get_image_data_48bit("/home/konrad/PhDStuff/data/small_vh16/size2561.png", srcw, srch)[0];
    std::vector<float> upsampled(data.size() * factor * factor);

    bilinear_upsample_colmajor_allocate_gpu(data.data(), upsampled.data(), srcw, srch, factor);

    write_png("/home/konrad/upsample_gpu_out.png", upsampled, srcw * factor, srch * factor);

    return 0;
}
