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


#include "species_assign_exp.h"
#include "species_optim/gpu_eval.h"
#include "data_importer/data_importer.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <cassert>
#include <chrono>
#include <random>
#include <functional>
//#include <FastNoise/FastNoise.h>

template<typename T>
void trim(T &val, T min, T max)
{
    if (min > max) std::swap(min, max);
    if (val < min) val = min;
    if (val > max) val = max;
}

template<typename T>
T trim(T val, T min, T max)
{
    if (min > max) std::swap(min, max);
    if (val < min) val = min;
    if (val > max) val = max;
    return val;
}

suit_func::suit_func(float loc, float d)
    : suit_func(loc, d, 1.0f)
{}

suit_func::suit_func(float loc, float d, float mult)
    : d(d), loc(loc), mult(mult)
{}



void suit_func::set_values(float loc, float d)
{
    set_d(d);
    set_loc(loc);
}

void suit_func::set_d(float scale)
{
    if (scale < 0.01)
        d = 0.01;
    else
        this->d = scale;
}

float suit_func::get_d() const
{
    return d;
}

float suit_func::get_loc() const
{
    return loc;
}

void suit_func::set_loc(float loc)
{
    this->loc = loc;
}

float suit_func::operator () (float x) const
{
    float s = log(0.2f) / pow(d, 4.5f);
    float diff_abs = fabs(x - loc);
    float expo = s * pow(diff_abs, 4.5f);
    return std::exp(expo) * mult;
}

inline std::ostream& operator << (std::ostream &ost, const suit_func &sf)
{
    ost << sf.get_d() << ", (" << log(0.001) / (sf.get_d() * sf.get_d()) << "), " << sf.get_loc();
    return ost;
}



species::species(data_importer::species specimport, float maxheight)
{
    auto get_ideal = [] (const data_importer::viability &viab)
    {
        return (viab.cmin + viab.cmax) / 2.0f;
    };

    auto get_tolerance = [] (const data_importer::viability &viab)
    {
        return (viab.cmax - viab.cmin) / 2.0f;
    };

    data_importer::viability tempviab = specimport.temp;
    data_importer::viability slopeviab = specimport.slope;
    data_importer::viability wetviab = specimport.wet;
    data_importer::viability sunviab = specimport.sun;
    std::vector<suit_func> funcs = {
        suit_func(get_ideal(tempviab), get_tolerance(tempviab)),
        suit_func(get_ideal(slopeviab), get_tolerance(slopeviab)),
        suit_func(get_ideal(wetviab), get_tolerance(wetviab)),
        suit_func(get_ideal(sunviab), get_tolerance(sunviab))
    };

    this->adaptations = funcs;
    this->maxheight = maxheight;
}

species::species(const std::vector< suit_func > &adaptations, float maxheight, int seed)
    : adaptations(adaptations), maxheight(maxheight)
{
}

float species::operator () (const std::vector<float> &values)
{
    std::vector<float> adapts = values;
    float minvalue = std::numeric_limits<float>::max();
    int minval_idx = -1;
    for (int i = 0; i < values.size(); i++)
    {
        float adaptvalue = adaptations[i](values[i]);
        if (adaptvalue < 1e-6)
            adaptvalue = 0;
        if (adaptvalue < minvalue)
        {
            minvalue = adaptvalue;
            minval_idx = i;
        }
        adapts[i] = adaptations[i](values[i]);
    }
    return *std::min_element(adapts.begin(), adapts.end());
}

float species::get_loc_for_map(int idx) const
{
    return adaptations[idx].get_loc();
}

float species::get_scale_for_map(int idx) const
{
    return adaptations[idx].get_d();
}

species_assign::species_assign(const ValueMap<float> &chm, const std::vector<ValueMap<float> > &abiotics, const std::vector<species> &species_vec, const std::vector<float> &max_heights)
    : chm(chm), abiotics(abiotics), species_vec(species_vec), nspecies(species_vec.size()), max_heights(max_heights)
{
    assert(nspecies == species_vec.size());
    assert(abiotics.size() > 0);
    chm.getDim(width, height);
    assigned.setDim(width, height);
    assigned.fill(-2);
    for (int i = 0; i < species_vec.size(); i++)
    {
        seeding[i].setDim(width, height);
        seeding[i].fill(0.0f);
        spec_seeding[i].setDim(width, height);
        spec_seeding[i].fill(false);
        noise[i].setDim(width, height);
        noise.at(i).fill(0.0f);
        backup_noise[i].clone(noise.at(i));
        drawing[i].setDim(width, height);
        drawing.at(i).fill(1.0f);
        drawing_indicator[i].setDim(width, height);
        drawing_indicator[i].fill(false);
        raw_adapt_vals[i].setDim(width, height);
    }
    bs_indicator.setDim(width, height);
    bs_indicator.fill(false);
    bs_abiotics.resize(abiotics.size());
    for (int mapidx = 0; mapidx < abiotics.size(); mapidx++)
    {
        for (auto &sp : species_vec)
        {
            species_locs.push_back(sp.get_loc_for_map(mapidx));
            species_scales.push_back(sp.get_scale_for_map(mapidx));
        }
    }
    create_nonzero();
    assign();
}



void species_assign::get_dims(int &w, int &h) const
{
    w = width;
    h = height;
}

void species_assign::set_progress_func(std::function<void (int)> progress_func)
{
    this->progress_func = progress_func;
}

species_assign::~species_assign()
{
    if (gpu_initialized)
        free_cuda_memory(gpu_mem);
}

void species_assign::set_chm(const ValueMap<float> &chm)
{
    this->chm = chm;
    create_nonzero();
}

void species_assign::create_nonzero()
{
    nonzero_idxes.clear();
    nonzero_abiotics.clear();
    nonzero_chm.clear();
    assert(abiotics.size() > 0);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = abiotics[0].flatten(x, y);
            //if (chm.get(x, y) > 0)		// currently, we look at all values, not only CHM values. This is a hack. TODO: do this properly
            if (true)
            {
                nonzero_idxes.push_back(idx);
            }
        }
    }

    nonzero_abiotics.resize(abiotics.size(), std::vector<float>(nonzero_idxes.size()));
    nonzero_chm.resize(nonzero_idxes.size());

    int i = 0;
    for (auto &idx : nonzero_idxes)
    {
        nonzero_chm.at(i) = chm.get(idx);
        for (int abmap_idx = 0; abmap_idx < abiotics.size(); abmap_idx++)
        {
            nonzero_abiotics.at(abmap_idx).at(i) = abiotics.at(abmap_idx).get(idx);
        }
        i++;
    }

    float **maps = new float *[nonzero_abiotics.size()];
    i = 0;
    for (auto &ab : nonzero_abiotics)
    {
        maps[i] = ab.data();
        i++;
    }
    int msize = nonzero_idxes.size();
    int nmaps = nonzero_abiotics.size();
    if (gpu_initialized)
        free_cuda_memory(gpu_mem);
    gpu_mem = create_cuda_memory(maps, chm.data(), msize, nmaps, nspecies, nonzero_idxes.data(), 0, 0, max_heights.data());
    gpu_initialized = true;
    delete [] maps;
}


const ValueMap<int> &species_assign::get_assigned()
{
    return assigned;
}

float species_assign::get_mult_at(int spec_idx, int x, int y)
{
    float mult;
    if (drawing_indicator.at(spec_idx).get(x, y))
    {
        mult = drawing.at(spec_idx).get(x, y);
    }
    else
    {
        mult = 1.0f;
    }
    if (mult < 0.0f)
        mult = 0.0f;

    return mult;
}

void species_assign::clear_brushstroke_data()
{
    for (auto &bsa : bs_abiotics)
        bsa.clear();
    bs_indicator.fill(false);
    bs_nonzero_indices.clear();
    bs_multmaps.clear();
    bs_max_mult = 0.0f;
    bs_npixels = 0;
    bs_nonzero_npixels = 0;
    bs_all_indices.clear();
}

void species_assign::get_mult_maps(std::map<int, ValueMap<float> > &drawmap, std::map<int, ValueMap<bool> > &draw_indicator)
{
    drawmap = this->drawing;
    draw_indicator = this->drawing_indicator;
}

void species_assign::set_mult_maps(const std::map<int, ValueMap<float> > &drawmap, const std::map<int, ValueMap<bool> > &draw_indicator)
{
    this->drawing = drawmap;
    this->drawing_indicator = draw_indicator;
}

void species_assign::write_species_drawing(int specidx, std::string outfile)
{
    data_importer::write_txt<ValueMap<float> >(outfile, &drawing.at(specidx));
}

void species_assign::optimise(int spec_idx,
                              float req_perc,
                              std::vector< std::vector<float> > &abiotics,
                              std::vector<int> &nonzero_idxes,	// indices where CHM is nonzero. Corresponds with abiotics maps
                              float max_mult,	// maximum value by which we can multiply, to get the highest value for the percentage of species spec_idx
                              //int nonzero_npixels,	// number of pixels where species spec_idx has nonzero adaptability value
                              std::vector<int> all_indices,	// all indices of the brushstroke or optimisation area, regardless of whether CHM is nonzero or not
                              ValueMap<float> smoothmap
                              )
{
    float **maps = new float *[abiotics.size()];
    for (int i = 0; i < abiotics.size(); i++)
    {
        maps[i] = abiotics[i].data();
    }

    std::vector<float> mult_map(abiotics.at(0).size() * nspecies, 1.0f);
    for (int i = 0; i < nonzero_idxes.size(); i++)
    {
        int idx = nonzero_idxes[i];
        for (int si = 0; si < nspecies; si++)
        {
            int x, y;
            chm.idx_to_xy(idx, x, y);
            mult_map[i * nspecies + si] = get_mult_at(si, x, y);
        }
    }

    interm_memory mem = create_cuda_memory(maps, nullptr, abiotics.at(0).size(), abiotics.size(), nspecies, nonzero_idxes.data(), 0, 0, max_heights.data());

    int npixels = nonzero_idxes.size();
    float curr_mult = 10.0f;
    float curr_min = 0.0f;
    float curr_max = max_mult;
    //float max_perc = nonzero_npixels / (float)npixels;
    float curr_perc = 0.0f;
    int niters = 0;
    do
    {
        curr_mult = (curr_min + curr_max) / 2.0f;
        for (int i = spec_idx; i < nonzero_idxes.size() * nspecies; i += nspecies)
        {
            int nonz_idx = (i - spec_idx) / nspecies;
            if (smoothmap.get(nonzero_idxes.at(nonz_idx)))
                mult_map[i] = curr_mult * smoothmap.get(nonzero_idxes.at(nonz_idx));
            else
                mult_map[i] = drawing.at(spec_idx).get(nonzero_idxes.at(nonz_idx));
        }

        std::vector<float> perc = evaluate_gpu(species_locs, species_scales, nullptr, nullptr, mem, mult_map, nullptr);
        curr_perc = perc[spec_idx] / npixels;

        if (curr_perc < req_perc)
        {
            curr_min = curr_mult;
        }
        else
        {
            curr_max = curr_mult;
        }

        std::cout << "Percentages for mult " << curr_mult << ": ";
        for (auto &p : perc)
        {
            std::cout << p / nonzero_idxes.size() << " ";
        }
        std::cout << std::endl;
        //curr_mult /= 2.0f;
        niters++;
        if (progress_func)
            progress_func(50 + 50 - std::min(50.0f, abs(curr_perc - req_perc) * 50));
    } while (abs(curr_perc - req_perc) > 0.01f && niters < 50);
    if (progress_func)
        progress_func(100);

    for (auto &idx : all_indices)
    {
        int x, y;
        drawing.at(spec_idx).idx_to_xy(idx, x, y);
        if (smoothmap.get(x, y) > 0.5f)
            drawing.at(spec_idx).set(x, y, curr_mult * smoothmap.get(x, y));
    }

    delete []maps;

    free_cuda_memory(mem);

}



void species_assign::optimise_brushstroke(int spec_idx, float req_perc, std::string outfile)
{
    std::default_random_engine gen;
    std::uniform_real_distribution<float> unif;

    int gw, gh;
    abiotics.at(0).getDim(gw, gh);
    ValueMap<float> smoothmap;
    smoothmap.setDim(gw, gh);
    smoothmap.fill(0.0f);

    for (auto &idx : bs_all_indices)
    {
        smoothmap.set(idx, 1.0f);
    }

    int rad = 50;
    int radsq = rad * rad;

    int nfeathered = 0;

    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            if (smoothmap.get(x, y) > 1e-4f)
            {
                int sx = trim(x - rad, 0, gw - 1);
                int ex = trim(x + rad, 0, gw - 1);
                int sy = trim(y - rad, 0, gh - 1);
                int ey = trim(y + rad, 0, gh - 1);
                float sum = 0.0f;
                int count = 0;
                for (int cx = sx; cx <= ex; cx++)
                {
                    for (int cy = sy; cy <= ey; cy++)
                    {
                        int dx = cx - x;
                        int dy = cy - y;
                        int dsq = dx * dx + dy * dy;
                        if (dsq < radsq)
                        {
                            count++;
                            sum += smoothmap.get(cx, cy);
                        }
                    }
                }
                //smoothmap.set(x, y, pow(sum / count, 1.5f));
                float prob = pow(sum / count, 1.0f);
                if (unif(gen) < prob)
                {
                    smoothmap.set(x, y, 1.0f);
                }
                else
                {
                    smoothmap.set(x, y, 0.0f);
                }
                nfeathered++;
                float ratio = static_cast<float>(nfeathered) / static_cast<float>(bs_all_indices.size());
                int perc = ratio * 50;

                if (progress_func)
                    progress_func(perc);
            }
        }
    }

    if (outfile.size() > 0)
        data_importer::write_txt(outfile, &smoothmap);


    if (bs_abiotics.size() > 0 && bs_abiotics.at(0).size() > 0)
        optimise(spec_idx, req_perc, bs_abiotics, bs_nonzero_indices, bs_max_mult, bs_all_indices, smoothmap);
}

void species_assign::assign_to(int x, int y)
{
    std::vector<float> values(abiotics.size());
    float maxadapt = 0.0f;
    int max_spec = -1;
    for (int i = 0; i < values.size(); i++)
    {
        values[i] = abiotics[i].get(x, y);
    }
    for (int i = 0; i < species_vec.size(); i++)
    {
        float tree_height = chm.get(x, y);
        if (tree_height > max_heights[i])
        {
            continue;
        }
        else if (tree_height < 1e-5)
        {
            assigned.set(x, y, -2);
        }
        float adapt = species_vec[i](values);
        raw_adapt_vals.at(i).set(x, y, adapt);
        float mult = get_mult_at(i, x, y);
        adapt *= mult;
        if (adapt > maxadapt)
        {
            maxadapt = adapt;
            max_spec = i;
        }
    }
    assigned.set(x, y, max_spec);
}

void species_assign::assign()
{
    assign_gpu();
}

void species_assign::assign_gpu()
{

    std::vector<float> mult_maps(nspecies * nonzero_idxes.size());
    int assign_idx = 0;
    for (auto &idx : nonzero_idxes)
    {
        int x, y;
        chm.idx_to_xy(idx, x, y);
        for (int spec_idx = 0; spec_idx < nspecies; spec_idx++)
            mult_maps[assign_idx * nspecies + spec_idx] = get_mult_at(spec_idx, x, y);
        assign_idx++;
    }

    std::vector<int> species_winners(nonzero_idxes.size(), -1);

    species_percs = evaluate_gpu(species_locs, species_scales, nullptr, nullptr, gpu_mem, mult_maps, species_winners.data());

    for (auto &p : species_percs)
        p = p / nonzero_idxes.size();

    for (int i = 0; i < species_winners.size(); i++)
    {
        int x, y;
        assigned.idx_to_xy(nonzero_idxes.at(i), x, y);
        assigned.set(x, y, species_winners[i]);
    }

    std::vector<int> spec_idx_map;
    for (int i = 0; i < nspecies; i++)
        spec_idx_map.push_back(i);
    if (first_eval)
    {
        get_species_minvals(raw_adapt_vals, spec_idx_map, gpu_mem, width, height);
        first_eval = false;
    }
}

int species_assign::get(int x, int y)
{
    return assigned.get(x, y);
}


void species_assign::add_drawn_circle(int x, int y, int radius, float mult, int specie)
{

    if (!(drawing.count(specie)))
    {
        return;
    }

    std::cout << "Adding seeding at " << x << ", " << y << std::endl;

    int sx = trim(x - radius, 0, width - 1);
    int ex = trim(x + radius, 0, width - 1);
    int sy = trim(y - radius, 0, height - 1);
    int ey = trim(y + radius, 0, height - 1);

    int radsq = radius * radius;
    for (int cx = sx; cx <= ex; cx++)
    {
        for (int cy = sy; cy <= ey; cy++)
        {
            int xd = cx - x;
            int yd = cy - y;
            int distsq = xd * xd + yd * yd;
            if (distsq <= radsq)
            {
                if (specie >= 0)
                {
                    drawing_indicator.at(specie).set(cx, cy, true);
                    if (!bs_indicator.get(cx, cy))		// since individual circular regions will heavily overlap when we do the brushstroke,
                                                        // we only add the values when we have not encountered this pixel before
                    {
                        int cidx = abiotics.at(0).flatten(cx, cy);
                        bs_all_indices.push_back(cidx);
                        //if (chm.get(cx, cy) > 1e-5)		// previously, we only added indices where the CHM is nonzero. Now, we optimise over the entire brushstroke.
                        if (true)
                        {
                            bs_npixels++;
                            bs_indicator.set(cx, cy, true);
                            for (int ab_idx = 0; ab_idx < abiotics.size(); ab_idx++)
                            {
                                bs_abiotics[ab_idx].push_back(abiotics[ab_idx].get(cx, cy));
                            }
                            bs_nonzero_indices.push_back(cidx);
                            for (int spec_idx = 0; spec_idx < nspecies; spec_idx++)
                            {
                                if (nspecies > 1 && spec_idx == specie) continue;		// we do not update maximum multiplier based on the species' own multiplier
                                float mult = get_mult_at(spec_idx, cx, cy);
                                bs_multmaps.push_back(mult);		// bs_multmaps not used
                                float maxmult = mult / ADAPT_CUTOFF;		// finds the maximum possible multiplier by checking by what we have to multiply our minimum allowable adaptability value, to get to maximum multiplier
                                if (maxmult > bs_max_mult)
                                {
                                    bs_max_mult = maxmult;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    bs_max_mult += 0.001f;
}
