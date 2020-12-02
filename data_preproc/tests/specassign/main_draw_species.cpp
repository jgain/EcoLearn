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



#include "basic_types.h"
#include "canopy_placement/gpu_procs.h"
#include "species_optim.h"

#include <SDL2/SDL.h>
#include <vector>



/*
template<typename T>
void trim(T &val, T min, T max)
{
    if (val < min) val = min;
    if (val > max) val = max;
}
*/

void conditional_smooth(ValueMap<float> &map, const ValueMap<float> &kernel, const ValueMap<int> &refmap, int targetval)
{
    int kw, kh;
    int mw, mh;
    map.getDim(mw, mh);
    kernel.getDim(kw, kh);
    int radx = kw / 2;
    int rady = kh / 2;

    for (int y = 0; y < mh; y++)
    {
        for (int x = 0; x < mw; x++)
        {
            float sum = 0;
            bool foundref = false;
            for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++)
                {
                    int cx = x - radx + kx;
                    int cy = y - radx + ky;
                    if (cx < mw && cx >= 0 && cy >= 0 && cy < mh)
                    {
                        if (refmap.get(cx, cy) == targetval)
                        {
                            foundref = true;
                            sum += kernel.get(kx, ky) * 1.0f;
                        }
                    }
                }
            if (foundref)
                map.set(x, y, sum);
        }
    }
}

ValueMap<float> create_round_uniform_kernel(int radius)
{
    ValueMap<float> kernel;
    kernel.setDim(radius * 2 +1, radius * 2 + 1);
    kernel.fill(0.0f);
    int nfilled = 0;
    int w = radius * 2 +1;
    int h = w;

    int radsq = radius * radius;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int dx = x - radius;
            int dy = y - radius;
            if (dx * dx + dy * dy <= radsq)
            {
                nfilled++;
            }
        }
    }
    float val = 1 / (float)nfilled;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int dx = x - radius;
            int dy = y - radius;
            if (dx * dx + dy * dy <= radsq)
            {
                kernel.set(x, y, val);
            }
        }
    }
    return kernel;
}

class draw_interface
{
public:
    draw_interface(int width, int height)
        : width(width), height(height)
    {
        init();
    }
    draw_interface(int width, int height, basic_types::MapFloat *chm, std::vector<basic_types::MapFloat *> maps_ptrs)
        : width(width), height(height), chm(chm), maps_ptrs(maps_ptrs)
    {
        init();
    }

    void init()
    {
        SDL_Init(SDL_INIT_VIDEO);
        window = SDL_CreateWindow("Species", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, 0);
        renderer = SDL_CreateRenderer(window, -1, 0);
        sfc = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA8888);
        specmap.setDim(width, height);
        specmap.fill(-1);
        drawmap.setDim(width, height);
        drawmap.fill(0.0f);
        specdraw_map.setDim(width, height);
        specdraw_map.fill(-1);
        set_kernel(create_round_uniform_kernel(cursor_radius));
    }

    ~draw_interface()
    {
        std::cout << "In draw_interface dtor" << std::endl;
        SDL_Quit();
    }

    void modify_surface(const ValueMap<int> &vals)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = sfc->pitch * y + x * sizeof(uint8_t) * 4;
                uint32_t *curr_pos = (uint32_t *)(((uint8_t *)sfc->pixels) + idx);
                *curr_pos = colors[vals.get(x, y) + 1];
            }
        }
    }
    void modify_surface(const ValueMap<float> &vals)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                uint32_t r, g, b, a;
                float val = vals.get(x, y);
                trim(val, 0.0f, 1.0f);
                r = val * 255;
                g = b = r;
                a = 255;
                uint32_t col = to_rgba(r, g, b, a);
                int idx = sfc->pitch * y + x * sizeof(uint8_t) * 4;
                uint32_t *curr_pos = (uint32_t *)(((uint8_t *)sfc->pixels) + idx);
                //*curr_pos = colors[vals.get(x, y) + 1];
                *curr_pos = col;
            }
        }
    }

    void render_surface()
    {
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, sfc);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, tex, NULL, NULL);

        SDL_RenderPresent(renderer);
        SDL_DestroyTexture(tex);
    }
    void modify_and_render_surface(const ValueMap<int> &vals)
    {
        modify_surface(vals);
        render_surface();
    }

    template<typename T, typename U>
    void modify_map_draw(T &map, int x, int y, U setval)
    {
        int sx = x - cursor_radius;
        int ex = x + cursor_radius;
        int sy = y - cursor_radius;
        int ey = y + cursor_radius;

        trim(sx, 0, width - 1);
        trim(ex, 0, width - 1);
        trim(sy, 0, height - 1);
        trim(ey, 0, height - 1);

        float circ_rad_sq = cursor_radius * cursor_radius;

        for (int cy = sy; cy < ey; cy++)
            for (int cx = sx; cx < ex; cx++)
            {
                int dx = x - cx;
                int dy = y - cy;
                float distsq = dx * dx + dy * dy;
                if (distsq < circ_rad_sq)
                {
                    map.set(cx, cy, setval);
                }
            }
    }

    void set_drawmap_to_specdraw_map()
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (specdraw_map.get(x, y) > -1)
                {
                    drawmap.set(x, y, 1.0f);
                }
                else
                {
                    drawmap.set(x, y, 0.0f);
                }
            }
        }
    }

    void create_eval(std::vector<species> solutions)
    {
        std::vector<float> maxheights(solutions.size(), 100.0f);
        evaluator = species_set::create_evaluator(chm, maps_ptrs, solutions, true, maxheights);
        evaluator->evaluate_gpu();
        std::vector<int> specidxes;
        for (int i = 0; i < solutions.size(); i++)
            specidxes.push_back(i);
        specmap = evaluator->create_species_map(chm, specidxes);

        std::vector<int> speccount(3, 0);
        for (int y = 0; y < width; y++)
        {
            for (int x = 0; x < height; x++)
            {
                int idx = specmap.get(x, y);
                if (idx >= 0 && idx < 3)
                {
                    speccount[idx]++;
                }
            }
        }
        for (auto &c : speccount)
        {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }

    void mainloop()
    {
        SDL_Event event;
        bool quit = false;
        while (!quit)
        {
            while (SDL_PollEvent(&event))
            {
                if (event.type == SDL_QUIT)
                {
                    quit = true;
                    break;
                }
                if (event.type == SDL_KEYDOWN)
                {
                    switch (event.key.keysym.sym)
                    {
                        case SDLK_0:
                        case SDLK_1:
                        case SDLK_2:
                            curr_color = event.key.keysym.sym - SDLK_0;
                            update_cursor_color();
                            break;
                        default:
                            break;
                    }
                }
                if (event.type == SDL_MOUSEMOTION)
                {
                    uint32_t mousestate = SDL_GetMouseState(&cursor_x, &cursor_y);
                    if (cursor_x == prev_cursor_x && cursor_y == prev_cursor_y)
                        continue;
                    prev_cursor_x = cursor_x;
                    prev_cursor_y = cursor_y;
                    bool leftdown = mousestate & SDL_BUTTON(SDL_BUTTON_LEFT);
                    bool rightdown = mousestate & SDL_BUTTON(SDL_BUTTON_RIGHT);
                    update_surface_with_cursor();
                    render_surface();
                    if (rightdown)
                    {
                        specdraw_map.fill(-1.0f);
                        set_drawmap_to_specdraw_map();
                        evaluator->set_drawn_map(specdraw_map, drawmap);
                        evaluator->evaluate_gpu();
                        specmap = evaluator->create_species_map(chm, {0, 1, 2});

                    }
                    else if (leftdown)
                    {
                        //drawmap.fill(0.0f);
                        //std::cout << "modifying draw map" << std::endl;
                        modify_map_draw(specdraw_map, cursor_x, cursor_y, curr_color);
                        set_drawmap_to_specdraw_map();
                        //modify_map_draw(drawmap, cursor_x, cursor_y, 1);
                        //conditional_smooth(drawmap, kernel, specdraw_map, curr_color);
                        smooth_uniform_radial(60, drawmap.data(), drawmap.data(), width, height);
                        evaluator->set_drawn_map(specdraw_map, drawmap);
                        evaluator->evaluate_gpu();
                        specmap = evaluator->create_species_map(chm, {0, 1, 2});
                    }
                    modify_surface(specmap);
                    //modify_surface(drawmap);
                    updatecount++;
                    //std::cout << updatecount << std::endl;
                    //std::cout << "mouse at " << cursor_x << ", " << cursor_y << std::endl;
                }
            }
        }
        std::cout << "exiting mainloop function" << std::endl;
    }

    void update_surface_with_cursor()
    {
        uint32_t cr, cg, cb, ca;
        get_rgba(cursor_color, cr, cg, cb, ca);
        float cursor_ratio = ca / 255.0f;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int dx = cursor_x - x;
                int dy = cursor_y - y;
                float distance = sqrt(dx * dx + dy * dy);
                if (distance > cursor_radius) continue;
                uint32_t *pixel = ((uint32_t *)((uint8_t *)sfc->pixels + sfc->pitch * y) + x);
                uint32_t pr, pg, pb, pa;
                get_rgba(*pixel, pr, pg, pb, pa);
                uint32_t nr, ng, nb, na;
                nr = (pr * (1 - cursor_ratio) + cr * cursor_ratio);
                ng = (pg * (1 - cursor_ratio) + cg * cursor_ratio);
                nb = (pb * (1 - cursor_ratio) + cb * cursor_ratio);
                na = 255;

                uint32_t newval = to_rgba(nr, ng, nb, na);
                *pixel = newval;

                /*
                std::cout << nr << ", " << ng << ", " << nb << ", " << na << std::endl;
                std::cout << "cursor ratio: " << cursor_ratio << std::endl;
                std::cout << "cursor color: " << cr << ", " << cg << ", " << cb << ", " << ca << std::endl;
                */
            }
        }
    }

    void get_rgba(uint32_t val, uint32_t &r, uint32_t &g, uint32_t &b, uint32_t &a)
    {
        r = val >> 24;
        g = (val << 8) >> 24;
        b = (val << 16) >> 24;
        a = (val << 24) >> 24;
    }
    uint32_t to_rgba(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
    {
        return ((r << 24) | (g << 16) | (b << 8) | a);
    }

    void update_cursor_color()
    {
        cursor_color = colors[curr_color + 1];
        cursor_color -= 0x000000FF;
        cursor_color += 0x00000080;
    }

    void set_kernel(const ValueMap<float> &kernel)
    {
        this->kernel = kernel;
    }

private:
    int width, height;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Surface *sfc;
    std::vector<uint32_t> colors = {0x000000FF, 0xFF0000FF, 0x00FF00FF, 0x0000FFFF};
    uint32_t cursor_color = 0x00000000;
    int curr_color;
    int cursor_x, cursor_y;
    int prev_cursor_x = -100, prev_cursor_y = -100;
    float cursor_radius = 30;
    ValueMap<int> specmap;
    ValueMap<float> drawmap;
    ValueMap<int> specdraw_map;
    ValueMap<float> kernel;
    std::unique_ptr<species_set> evaluator;
    std::vector<basic_types::MapFloat *> maps_ptrs;
    basic_types::MapFloat *chm;
    int updatecount = 0;
};

int main(int argc, char * argv [])
{
    using namespace basic_types;

    if (argc != 2)
    {
        std::cout << "Usage: draw_species_test <data directory>" << std::endl;
        return 1;
    }

    std::string data_directory = argv[1];
    //data_importer::common_data simdata(database_filename);
    data_importer::data_dir ddir(data_directory, 1);

    /*
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.first << ", " << sp_pair.second.idx << ", " << sp_pair.second.a << ", " << sp_pair.second.b << std::endl;
    }
    */

    int width, height;
    auto month_moisture = data_importer::read_monthly_map< ValueMap<float> >(ddir.wet_fname);
    MapFloat moisturemap = data_importer::average_mmap< MapFloat, ValueMap<float> >(month_moisture);
    MapFloat slopemap = data_importer::load_txt<MapFloat>(ddir.slope_fname);
    //std::vector<MapFloat> month_sun = data_importer::read_monthly_map(sun_filename, width, height);
    auto month_sun = data_importer::read_monthly_map< ValueMap<float> >(ddir.sun_fname);
    auto sunmap = data_importer::average_mmap< MapFloat, ValueMap<float> >(month_sun);
    auto month_temp = data_importer::read_monthly_map< ValueMap<float> > (ddir.temp_fname);
    MapFloat tempmap = data_importer::average_mmap<MapFloat, ValueMap<float> >(month_temp);
    tempmap.getDim(width, height);

    int chmw, chmh;
    MapFloat chmdata = data_importer::load_txt(ddir.chm_fname, chmw, chmh);
    //mask_nonzero_map(chmdata, 800);
    std::vector< MapFloat > abiotic_maps = {moisturemap, slopemap, sunmap, tempmap};
    std::vector<MapFloat *> maps_ptrs;
    for (auto &amap : abiotic_maps)
    {
        maps_ptrs.push_back(&amap);
    }

    /*
    std::cout << "Number of species: " << simdata.all_species.size() << std::endl;
    for (auto &sp_pair : simdata.all_species)
    {
        std::cout << sp_pair.second.sun.c << ", " << sp_pair.second.sun.r << std::endl;
        std::cout << sp_pair.second.temp.c << ", " << sp_pair.second.temp.r << std::endl;
    }
    */

    std::vector<float> species_perc;
    std::vector<float> maxheights(3, 100.0f);
    species_perc.push_back(0.4);
    species_perc.push_back(0.4);
    species_perc.push_back(0.2);

    auto optim_ptr = species_set::create_optim(&chmdata, maps_ptrs, species_perc, true, maxheights);
    optim_ptr->optimise(200);

    auto specvec = optim_ptr->get_species_vec();
    for (auto &sp : specvec)
    {
        for (auto &adapt : sp.get_adaptations())
        {
            std::cout << adapt.get_loc() << ", " << adapt.get_d() << std::endl;
        }
        std::cout << "------------------------" << std::endl;
    }

    draw_interface interf(chmw, chmh, &chmdata, maps_ptrs);
    interf.create_eval(optim_ptr->get_species_vec());
    auto specmap = optim_ptr->create_species_map(&chmdata, {0, 1, 2});
    interf.mainloop();

    return 0;
}
