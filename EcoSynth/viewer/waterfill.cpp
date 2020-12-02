/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za)
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

#include "waterfill.h"
#include "terrain.h"

#include <QPoint>
#include <iostream>
#include <stack>
#include <queue>
#include <set>


void WaterFill::setTerrain(Terrain* t)
{
    terrain = t;
}

bool WaterFill::isFlowingWater(uint x, uint y)
{
    uint w, h;
    terrain->getGridDim(w, h);
    return flow[x+y*w] >absorbsion;
}

float WaterFill::riverMoisture(uint x, uint y)
{
    uint w, h;
    terrain->getGridDim(w, h);
    return float(river_side[x + y*w]) / float(absorbsion);
}

struct Pass
{
    Pass(){}
    Pass (uint rid, uint did, float h) : rcv_id{rid}, donnor_id{did}, height{h}{}
    uint rcv_id; // hight part
    uint donnor_id; // low part (root)
    float height;
};
bool operator < (const Pass& a, const Pass& b) {if(a.height == b.height) return a.donnor_id < b.donnor_id; else return a.height < b.height;}

template<class T, int n>
static void shuffle(T* vec)
{
   for(int i = n-1; i>=1; --i)
   {
       int j = rand() % (i+1);
       std::swap(vec[i], vec[j]);
   }
}

void WaterFill::compute(uint step)
{
    uint dx, dy;
    terrain->getGridDim(dx, dy);

    flow.resize(dx * dy+1, 0);
    lakes.resize(dx * dy, 0.0);

    uint s_alloc = dx * dy;
    std::vector<uint> rcv(s_alloc, s_alloc);
    std::vector<uint> donnor_start(s_alloc+1, 0);
    std::vector<uint> donnor_end(s_alloc+1);
    std::vector<uint> donnors(s_alloc);
    std::vector<uint> parse_order;
    parse_order.reserve(s_alloc);

    std::vector<bool> in_resolve_stack(s_alloc, false);
    std::vector<uint> catchement_root(s_alloc, (uint)(-1));
    std::vector<uint> to_resolve, loc_min;
    std::vector<Pass> catchement_pass(s_alloc);
    std::set<Pass> wait_passes;
    std::vector<std::set<Pass>::iterator> catchement_wait_pass(s_alloc, wait_passes.end());


    QPoint shift [4];
    shift[0] = QPoint(-1, 0);
    shift[1] = QPoint(1, 0);
    shift[2] = QPoint(0, -1);
    shift[3] = QPoint(0, 1);

    for(uint x = 0; x< dx; ++x)
        for(uint y = 0; y< dy; ++y)
        {
            // get lowest neighbor
            QPoint p(x, y);
            QPoint nb = p;
            double this_h = (double) terrain->getHeight((int) x, (int) y);
            double h = this_h;
            shuffle<QPoint, 4>(shift);
            for(int j =0; j<4; ++j)
            {
                QPoint cur = p + shift[j];
                if(cur.x()<0.0 || cur.x() >= (int) dx || cur.y()<0.0 || cur.y() >= (int) dy)
                    continue;

                double ch = (double) terrain->getHeight((int) cur.x(), (int) cur.y());
                if(h > ch)
                {
                    nb = cur;
                    h = ch;
                }
            }
            uint cur_id = x + y * dx;

            if(nb != p)
                rcv[cur_id] =  nb.x() + nb.y() * dx;
            else
            {
                catchement_root[cur_id] = cur_id;
                // add external local min as to resolve
                if(x == 0 || y == 0 || x == dx-1 || y == dy-1)
                {
                    to_resolve.push_back(cur_id);
                    in_resolve_stack[cur_id] = true;
                    catchement_pass[cur_id].height = this_h;
                    catchement_pass[cur_id].donnor_id = cur_id;
                    catchement_pass[cur_id].rcv_id = cur_id;
                }
            }
        }

    // compute donnors
    for(uint id : rcv)
        ++donnor_start[id];
    uint total = 0;
    for(uint& c : donnor_start)
    {
        uint tmp = c;
        c = total;
        total += tmp;
    }
    donnor_end = donnor_start;
    for(uint i = 0; i< s_alloc; ++i)
        donnors[donnor_end[rcv[i]]++] = i;

    // first upstream parse to mark catchments
    std::stack<uint> parse_stack;
    //local min are stored as donnors of the last element
    for(uint i = donnor_start[s_alloc]; i <donnors.size(); ++i)
        parse_stack.push(donnors[i]);

    while(!parse_stack.empty())
    {
        uint cur = parse_stack.top();
        parse_stack.pop();
        uint root = catchement_root[cur];
        for(uint c = donnor_start[cur]; c<donnor_end[cur]; ++c)
        {
            uint d = donnors[c];
            catchement_root[d] = root;
            parse_stack.push(d);
        }
    }

    // add all passes toward the boudary
    std::vector<uint> to_parse(2*dx+2*dy);
    for(uint i =0; i<dx; ++i)
    {
        to_parse.push_back(i);
        to_parse.push_back(dx * (dy-1) + i);
    }
    for(uint i =0; i< dy; ++i)
    {
        to_parse.push_back(i*dx);
        to_parse.push_back((i+1)*dx-1);
    }

    for(uint id : to_parse)
    {
        double ch = (double) terrain->getFlatHeight((int) id);
        uint root = catchement_root[id];

        if(!in_resolve_stack[root])
        {
            std::set<Pass>::iterator pass_it = catchement_wait_pass[root];

            if(pass_it == wait_passes.end())
            {
                catchement_wait_pass[root] = wait_passes.insert(Pass(root, root, ch)).first;
            }
            else if(pass_it->height > ch)
            {
                wait_passes.erase(pass_it);
                catchement_wait_pass[root] = wait_passes.insert(Pass(root, root, ch)).first;
            }
        }
    }

    loc_min = to_resolve;
    uint n_step = 0;
    // loop while there are some un-resolved catchments
    while(true)
    {

        // parse all curent catchments and update passes
        for(uint i = 0; i<to_resolve.size(); ++i)
        {
            uint cur_loc_min  =  to_resolve[i];
            // parse catchment.
            parse_stack.push(cur_loc_min);
            while(!parse_stack.empty())
            {
                uint cur = parse_stack.top();

                // this parse order is supposed to be correct with respect to drainage
                parse_order.push_back(cur);

                parse_stack.pop();
                for(uint c = donnor_start[cur]; c<donnor_end[cur]; ++c)
                {
                    uint d = donnors[c];
                    parse_stack.push(d);
                }
                QPoint p(cur%dx, cur/dx);
                double ch = terrain->getHeight((int) p.x(), (int) p.y());

                shuffle<QPoint, 4>(shift);
                for(int j =0; j<4; ++j)
                {
                    QPoint nb = p + shift[j];
                    if(nb.x()<0.0 || nb.x() >= (int) dx || nb.y()<0.0 || nb.y() >= (int) dy)
                        continue;
                    uint nb_id = nb.x() + nb.y() * dx;

                    uint nb_root = catchement_root[nb_id];


                    if(!in_resolve_stack[nb_root])
                    {
                        std::set<Pass>::iterator pass_it = catchement_wait_pass[nb_root];

                        if(pass_it == wait_passes.end())
                        {
                            catchement_wait_pass[nb_root] = wait_passes.insert(Pass(cur, nb_root, ch)).first;
                        }
                        else if(pass_it->height > ch)
                        {
                            wait_passes.erase(pass_it);
                            catchement_wait_pass[nb_root] = wait_passes.insert(Pass(cur, nb_root, ch)).first;
                        }
                    }
                }
            }

        }
        if(n_step++ == step)
            break;


        // save min pass.
        if(wait_passes.empty())
            break;

        std::set<Pass>::iterator min_pass_it = wait_passes.begin();
        Pass& pass = catchement_pass[min_pass_it->donnor_id];
        pass = *min_pass_it;
        wait_passes.erase(min_pass_it);

        // pass higth is the higher along the path
        pass.height = std::max(pass.height, catchement_pass[catchement_root[pass.rcv_id]].height);

        // add to stack
        in_resolve_stack[pass.donnor_id] = true;
        to_resolve.resize(1);
        to_resolve.front() = pass.donnor_id;
    }

    // compute drainage along reverse parse order
    for(int i = 0; i< (int) flow.size(); ++i)
        flow[i] = inflow[i]+1;

    for(int i = (int)s_alloc-1; i>=0; --i)
    {
        uint id = parse_order[i];
        if(id == catchement_root[id])
        {
            Pass& p = catchement_pass[id];
            if(p.donnor_id != p.rcv_id)
                flow[p.rcv_id]+=flow[id];
        }
        flow[rcv[id]]+=flow[id];
    }
    flow.resize(flow.size()-1);

//    std::vector<uint> dbg_pass;
//    std::vector<uint> dbg_l_min;

//    std::vector<uint> dbg_crests;

    // and lakes
    for(uint i = 0; i< lakes.size(); ++i)
    {
        uint root = catchement_root[i];
        const Pass& p = catchement_pass[root];

//        if(i == root)
//            dbg_l_min.push_back(p.donnor_id);

//        if(p.donnor_id == p.rcv_id)
//            continue;

//        if(i == root)
//        {
//            dbg_pass.push_back(p.rcv_id);

//        }

        double h = terrain->getFlatHeight(i);

        lakes[i] = std::max(0.0, p.height - h);
        if(p.height > h)
           flow[i] = flow[root];

    }
//    for(uint i  : dbg_pass)
//        lakes[i] = -1.0;
//    for(uint i  : dbg_l_min)
//        lakes[i] = -2.0;

//    for(uint i  : dbg_crests)
//        lakes[i] = -3.0;


}

void WaterFill::expandRivers(float max_moisture_factor, float slope_effect)
{
    std::vector<uint> newFlow(flow.size(), 0);
    int dx, dy;

    terrain->getGridDim(dx, dy);

    float cell_width = std::sqrt(terrain->getCellArea());

    for(uint i =0; i<flow.size(); ++i)
    {
        // compute actual drainage
        uint f = flow[i];

        float drainage = float(f) * precipitation * terrain->getCellArea();

        float river_width = std::sqrt(drainage) * river_width_constant;

        int i_width =  std::floor(river_width/2 / cell_width);

        int p_x = ((int) i)%dx;
        int p_y = ((int) i)/dx;

        for(int x = std::max(0, p_x - i_width); x <= std::min(dx-1, p_x+i_width); ++x)
            for(int y = std::max(0, p_y - i_width); y <= std::min(dy-1, p_y+i_width); ++y)
            {
                float tx = float(p_x-x);
                float ty = float(p_y-y);

                if(tx*tx+ty*ty <= float(i_width*i_width))
                {
                    uint& nf = newFlow[x+dx*y];
                    nf = std::max(f, nf);
                }
            }

    }

    flow = newFlow;
    river_side = std::vector<uint>(flow.size(), 0);

    for(uint i =0; i<newFlow.size(); ++i)
    {
        // compute actual drainage
        uint f = newFlow[i];

        if(f<absorbsion)
            continue;

        float drainage = float(f) * precipitation * terrain->getCellArea();

        float river_width = std::sqrt(drainage) * river_width_constant * max_moisture_factor;
        float normalized = river_width/2 / cell_width;

        int i_width =  std::ceil(normalized);

        int p_x = ((int) i)%dx;
        int p_y = ((int) i)/dx;

        float p_h = terrain->getFlatHeight(i);

        for(int x = std::max(0, p_x - i_width); x <= std::min(dx-1, p_x+i_width); ++x)
            for(int y = std::max(0, p_y - i_width); y <= std::min(dy-1, p_y+i_width); ++y)
            {
                float tx = float(p_x-x);
                float ty = float(p_y-y);

                float h = terrain->getHeight(x, y);

                float d = std::sqrt(float(tx*tx+ty*ty));
                // float sdx = dx*terrain->scale().x();
                // float sdy = dy*terrain->scale().y();

                float sdx = tx*terrain->getCellExtent();
                float sdy = ty*terrain->getCellExtent();
                float sd = std::sqrt(float(sdx*sdx+sdy*sdy));
                //float slope = std::max(0.0f, h-p_h) / d;
                float slope = (h-p_h) /sd ;
                float moisture = std::max(0.0f, normalized - d*(1.0f + slope_effect * slope)) / normalized ;

                uint& nf = flow[x+dx*y];
                nf = std::max(uint(moisture * float(absorbsion)) , nf);
                uint& rs = river_side[x+dx*y];
                rs = std::max(uint(moisture * float(absorbsion)) , rs);
            }
    }


}

void WaterFill::reset()
{
    uint dx, dy;
    terrain->getGridDim(dx, dy);
    inflow = std::vector<uint>(dx * dy+1, 0);
}

void WaterFill::addWaterInflow(uint x, uint y, int delta)
{
    uint dx, dy;
    terrain->getGridDim(dx, dy);

    uint& inf = inflow[x+y*dx];
    if(inf == 0 && delta > 0)
        delta = absorbsion+1;
    inf = std::max(0, (int)inf + delta);
}

void WaterFill::smartWaterInflow(uint x, uint y)
{
    uint dx, dy;
    terrain->getGridDim(dx, dy);
    uint& inf = inflow[x+y*dx];
    uint fl = flow[x+y*dx];

    if(inf == 0) inf = absorbsion+1;
    else
    {
        float river_w = std::sqrt(double(inf+fl)*precipitation)*river_width_constant+1.0;
        // std::cerr << "New flow: " << inf << std::endl;
        inf = uint(river_w*river_w / precipitation/river_width_constant/river_width_constant - (float)fl);

    }
    // std::cerr << "New flow: " << inf << std::endl;
}
