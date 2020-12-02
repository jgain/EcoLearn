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

/* file: fill.cpp
   author: (c) James Gain, 2009
   project: ScapeSketch - sketch-based design of procedural landscapes
   notes: scan-line polygon fill
   changes:
*/

#include "fill.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <list>

//////////////////////////////
/// SCAN-LINE POLYGON FILL ///
/// UNATTRIBUTED FROM WEB ////
//////////////////////////////

struct intPoint
{
    int x,y;
};

class intPointArray
{
public:
    std::vector<intPoint> pt;
};

class Node
{
public:
    Node():yUpper(-500),xIntersect(-500.0),dxPerScan(0.0){ };

    int yUpper;
    float xIntersect,dxPerScan;
};

class EdgeTbl
{
public:
    void buildTable (const intPointArray &, int, int);
    int yNext (int, std::vector<intPoint>);
    void makeEdgeRecord (intPoint,intPoint,int);

    std::vector<std::list<Node> > Edges;
};


/// EDGE TABLE METHODS ///

void insertEdge (std::list<Node>& orderedList, const Node& item)
{
    std::list<Node>::iterator curr = orderedList.begin(), stop = orderedList.end();
    while ((curr != stop) && ((*curr).xIntersect < item.xIntersect))
        curr++;
    orderedList.insert(curr,item);
}

int EdgeTbl::yNext (int k, std::vector<intPoint> p)
{
    int j;

    // next subscript in polygon
    if ((k+1) > ((int) p.size()-1))
        j = 0;
    else
        j = k+1;
    while (p[k].y == p[j].y)
        if ((j+1) > ((int) p.size()-1))
            j = 0;
        else
            j++;
    return (p[j].y);
}

void EdgeTbl::makeEdgeRecord (intPoint lower, intPoint upper, int yComp)
{
    Node n;

    n.dxPerScan = (float)(upper.x-lower.x)/(upper.y-lower.y);
    n.xIntersect = lower.x;
    if (upper.y < yComp) // edge shortening for non-extrema
        n.yUpper = upper.y-1;
    else
        n.yUpper = upper.y;

    // clip to lower edge
    if(lower.y < 0)
    {
        n.xIntersect = lower.x + n.dxPerScan * (0-lower.y);
        lower.y = 0;
    }
    insertEdge (Edges[lower.y],n);
}

void EdgeTbl::buildTable (const intPointArray& Poly, int dimx, int dimy)
{
    intPoint v1,v2;
    int i, yPrev;

    // cerr << "poly size = " << (int) Poly.pt.size() << endl;
    // cerr << "dimx = " << dimx << endl;
    // cerr << "dimy = " << dimy << endl;
    yPrev = Poly.pt[Poly.pt.size()-2].y;
    v1.x = Poly.pt[Poly.pt.size()-1].x;
    v1.y = Poly.pt[Poly.pt.size()-1].y;
    for (i = 0; i < (int) Poly.pt.size(); i++)
    {
        // cerr << "i = " << i << endl;
        v2 = Poly.pt[i];
        if (v1.y != v2.y)
        {
            // if(v1.y < 0 ||
            // check to see if outside bounds of map above and below
            if(!((v1.y < 0 && v2.y < 0) || (v1.y > dimy-1 && v2.y > dimy-1)))
            {
                // non horizontal edge
                if (v1.y < v2.y)
                    makeEdgeRecord (v1,v2,yNext(i,Poly.pt)); //up edge
                else
                    makeEdgeRecord (v2,v1,yPrev); // down edge
                yPrev = v1.y;
            }
        }
        v1 = v2;
    }
}

/// AEL ROUTINES ///

void buildAEL (std::list<Node> &AEL, std::list<Node> ET)
{
    std::list<Node>::iterator iter;

    iter = ET.begin();

    // every Edge table list has a "empty" node at front
    iter++;
    while (iter != ET.end())
    {
        insertEdge (AEL,*iter);
        iter++;
    }
}

void fillScan (int y, std::list<Node> L, Terrain * ter, TypeMap * tmap, int brushtype)
{
    int dimx, dimy;

    // want to pull off pairs of x values from adjacent
    // nodes in the list - the y value = scan line
    std::list<Node>::iterator iter1 = L.begin(), iter2;
    int x1, x2, i;

    ter->getGridDim(dimx, dimy);

    while (iter1 != L.end())
    {
        iter2 = iter1;
        iter2++;
        x1 = (int)(*iter1).xIntersect;
        x2 = (int)(*iter2).xIntersect;

        // allow for drawing outside the map bounds to left and right
        if(x1 < 0) x1 = 0;
        if(x2 > dimx - 1) x2 = dimx-1;
        for(i = x1; i <= x2; i++)
            (* tmap->getMap())[y][i] = brushtype;

        // move on to next pair of nodes
        iter1 = iter2;
        iter1++;
    }
}

void fillMaskScan (int y, std::list<Node> L, Terrain * ter, MemMap<bool> * mask)
{
    int dimx, dimy;

    // want to pull off pairs of x values from adjacent
    // nodes in the list - the y value = scan line
    std::list<Node>::iterator iter1 = L.begin(), iter2;
    int x1, x2, i;

    ter->getGridDim(dimx, dimy);

    while (iter1 != L.end())
    {
        iter2 = iter1;
        iter2++;
        x1 = (int)(*iter1).xIntersect;
        x2 = (int)(*iter2).xIntersect;

        // allow for drawing outside the map bounds to left and right
        if(x1 < 0) x1 = 0;
        if(x2 > dimx - 1) x2 = dimx-1;
        for(i = x1; i <= x2; i++)
            (* mask)[y][i] = true;

        // move on to next pair of nodes
        iter1 = iter2;
        iter1++;
    }
}

void updateAEL (int y, std::list<Node>& L)
{
    // delete completed edges
    // update the xIntersect field
    std::list<Node>::iterator iter = L.begin();

    while (iter != L.end())
        if (y >= (*iter).yUpper)
            L.erase(iter++);
        else
        {
            (*iter).xIntersect += (*iter).dxPerScan;
            iter++;
        }
}

void resortAEL (std::list<Node>& L)
{
    Node n;
    std::list<Node> L1;
    std::list<Node>::iterator iter = L.begin();

    // create a new list from the old
    // note that the sort command for a list would
    // need us to overload the comparisons operators in the
    // Node class. This is probably just as simple
    while (iter != L.end())
    {
        insertEdge (L1,*iter);
        L.erase(iter++);
    }
    L = L1;
}

void scanLoopFill(std::vector<vpPoint> * loop, Terrain * ter, TypeMap * tmap, int brushtype)
{
    std::vector<int> x, y;
    int i, j, k;
    EdgeTbl EdgeTable;
    std::list<Node> AEL;
    intPoint pnt, prev;
    intPointArray P;
    bool hori = false;
    std::list<Node> EmptyList;  // an empty list
    Node EmptyNode;  // an empty node
    int dimx, dimy;

    ter->getGridDim(dimx, dimy);
    for(k = 0; k < (int) loop->size(); k++)
    {
        ter->toGrid((* loop)[k], i, j);
        x.push_back(i);
        y.push_back(j);
    }

    for(i = 0; i < (int) x.size(); i++)
    {
        pnt.x = (GLint) x[i];
        pnt.y = (GLint) y[i];

        if(i==0) // no vertices yet
        {
            P.pt.push_back(pnt);
        }
        else
        {
            if(!(pnt.x == P.pt.back().x && pnt.y == P.pt.back().y)) // skip over duplicates
            {
                if(pnt.y == P.pt.back().y) // skip intermediate points on horizontal edge
                {
                    hori = true;
                }
                else
                {
                    if(hori) // go back and push the last point on the horizontal edge
                    {
                        prev.x = (GLint) x[i-1];
                        prev.y = (GLint) y[i-1];
                        P.pt.push_back(prev);
                        hori = false;
                    }
                    P.pt.push_back(pnt);
                }
            }
        }
    }

    EmptyList.push_front(EmptyNode); // an empty list

    // build the edge table - need the window size
    for (i = 0; i < dimy; i++)
        EdgeTable.Edges.push_back(EmptyList);
    EdgeTable.buildTable(P, dimx, dimy);

    for (int scanLine = 0; scanLine < dimy; scanLine++)
    {
        buildAEL (AEL,EdgeTable.Edges[scanLine]);
        if (!AEL.empty())
        {
            fillScan(scanLine,AEL,ter,tmap,brushtype);
            updateAEL (scanLine,AEL);
            resortAEL(AEL);
        }
    }
    // clear memory before exitin
    for(i = 0; i < dimy; i++)
        EdgeTable.Edges[i].clear();
    EdgeTable.Edges.clear();
    AEL.clear();
}

void scanLoopMaskFill(std::vector<vpPoint> * loop, Terrain * ter, MemMap<bool> * mask)
{
    std::vector<int> x, y;
    int i, j, k;
    EdgeTbl EdgeTable;
    std::list<Node> AEL;
    intPoint pnt, prev;
    intPointArray P;
    bool hori = false;
    std::list<Node> EmptyList;  // an empty list
    Node EmptyNode;  // an empty node
    int dimx, dimy;

    ter->getGridDim(dimx, dimy);
    for(k = 0; k < (int) loop->size(); k++)
    {
        ter->toGrid((* loop)[k], i, j);
        x.push_back(i);
        y.push_back(j);
    }

    for(i = 0; i < (int) x.size(); i++)
    {
        pnt.x = (GLint) x[i];
        pnt.y = (GLint) y[i];

        if(i==0) // no vertices yet
        {
            P.pt.push_back(pnt);
        }
        else
        {
            if(!(pnt.x == P.pt.back().x && pnt.y == P.pt.back().y)) // skip over duplicates
            {
                if(pnt.y == P.pt.back().y) // skip intermediate points on horizontal edge
                {
                    hori = true;
                }
                else
                {
                    if(hori) // go back and push the last point on the horizontal edge
                    {
                        prev.x = (GLint) x[i-1];
                        prev.y = (GLint) y[i-1];
                        P.pt.push_back(prev);
                        hori = false;
                    }
                    P.pt.push_back(pnt);
                }
            }
        }
    }

    EmptyList.push_front(EmptyNode); // an empty list

    // build the edge table - need the window size
    for (i = 0; i < dimy; i++)
        EdgeTable.Edges.push_back(EmptyList);
    EdgeTable.buildTable(P, dimx, dimy);

    for (int scanLine = 0; scanLine < dimy; scanLine++)
    {
        buildAEL (AEL,EdgeTable.Edges[scanLine]);
        if (!AEL.empty())
        {
            fillMaskScan(scanLine,AEL,ter,mask);
            updateAEL (scanLine,AEL);
            resortAEL(AEL);
        }
    }
    // clear memory before exitin
    for(i = 0; i < dimy; i++)
        EdgeTable.Edges[i].clear();
    EdgeTable.Edges.clear();
    AEL.clear();
}
