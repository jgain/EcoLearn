
/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp (konrad.p.kapp@gmail.com)
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

// constraint.cpp: various user generated constraints for ultimate terrain synthesis
// author: James Gain
// date: 5 November 2013
//       21 January 2013 - curve constraints

#include <GL/glew.h>

#include <cassert>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "timer.h"
#include "palette.h"
#include "fill.h"
#include "eco.h"

#include <QtWidgets>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "glwidget.h"

/*
GLfloat manipCol[] = {0.325f, 0.235f, 1.0f, 1.0f};
//GLfloat manipCol[] = {0.406f, 0.294f, 1.0f, 1.0f};
GLfloat curveCol[] = {0.243f, 0.176f, 0.75f, 1.0f};
GLfloat blockedCol[] = {0.5f, 0.5f, 0.8f, 1.0f};
//GLfloat blockedCol[] = {0.325f, 0.235f, 1.0f, 1.0f};
*/

using namespace std;

//
// Palette
//


BrushPalette::BrushPalette(TypeMap *typemap, int nentries, QWidget *parent)
    : palette_base(typemap, nentries, parent)
{

}

void BrushPalette::typeSelect()
{
    typeSelectMode(ControlMode::PAINTLEARN);
}

/*
void BrushPalette::typeSelectSpecies()
{
    // TO DO - deal with the case that one of the palette entries is in selection mode
    for(int i = 0; i < nentries; i++)
    {
        if(sender() == selector[i])
        {
            cerr << "index " << i << " selected" << std::endl;
            currSel = i;
            glparent->setCtrlMode(ControlMode::PAINTSPECIES); // activate paint mode
        }
    }
    setActivePalette();
    cerr << "currently selected brush = " << (int) typeSel[currSel] << ", with index " << currSel << endl;
}
*/

void SpeciesPalette::enable_brush(int idx)
{
    if (idx < nentries)
        selector[idx]->setEnabled(true);
}

void SpeciesPalette::disable_brush(int idx)
{
    if (idx < nentries)
        selector[idx]->setEnabled(false);
}

SpeciesPalette::SpeciesPalette(TypeMap *typemap, const std::vector<int> &species_ids, QWidget *parent)
    : palette_base(typemap, species_ids, parent)
{
    idx_to_id = species_ids;
    for (int i = 0; i < species_ids.size(); i++)
    {
        id_to_idx[species_ids.at(i)] = i;
    }
}

int SpeciesPalette::getDrawTypeIndex()
{
    return currSel;
}

void SpeciesPalette::typeSelect()
{
    typeSelectMode(ControlMode::PAINTSPECIES);
}

//
// BrushCursor
//

void BrushCursor::genBrushRing(View * view, Terrain * terrain, float brushradius, bool dashed)
{
    uts::vector<vpPoint> ring;
    int steps, j;
    float a, stepa, tol, tx, ty;
    vpPoint pnt;

   //  shape.clear();
    terrain->getTerrainDim(tx, ty);
    tol = 0.001f * std::max(tx, ty);

    // draw ring to indicate extent of brush stroke
    // generate vertices for ring and drop onto terrain
    if(active)
    {
        a = 0.0f;
        steps = 1000;
        stepa = PI2 / (float) steps;

        for(j = 0; j < steps+1; j++)
        {
            pnt.x = pos.x + cosf(a) * brushradius;
            if(pnt.x >= tx-tolzero) pnt.x = tx-tolzero;
            if(pnt.x <= tolzero) pnt.x = tolzero;
            pnt.y = 1.0f;
            pnt.z = pos.z + sinf(a) * brushradius;
            if(pnt.z >= ty-tolzero) pnt.z = ty-tolzero;
            if(pnt.z <= tolzero) pnt.z = tolzero;
            ring.push_back(pnt);
            a += stepa;
        }
        drapeProject(&ring, &ring, terrain);

        // add height offset to all ring positions
        for(j = 0; j < (int) ring.size(); j++)
            ring[j].y += hghtoffset;

        if(dashed)
            shape.genDashedCylinderCurve(ring, manipradius * 0.5f * view->getScaleFactor(), tol, manipradius * view->getScaleFactor(), 10);
        else
            shape.genCylinderCurve(ring, manipradius * 0.5f * view->getScaleFactor(), tol, 10);
    }
}

void BrushCursor::cursorUpdate(View * view, Terrain * terrain, int x, int y)
{
    vpPoint frompnt, topnt;

    view->projectingPoint(x, y, frompnt);
    if(terrainProject(frompnt, topnt, view, terrain))
    {
        pos = topnt;
        active = true;
    }
    else
    {
        active = false;
    }
}

/// getters and setters for brush radii
void BrushCursor::setRadius(float rad)
{
    radius = rad;
}

//
// BrushPaint
//

BrushPaint::BrushPaint(Terrain * ter, BrushType btype)
{
    terrain = ter;
    brushtype = btype;
    drawing = false;
}

void BrushPaint::paintMap(TypeMap * pmap, float radius)
{
    int dx, dy, si, sj, ei, ej;
    float inr, h, ox, oy, irad;
    vpPoint p;

    terrain->getGridDim(dx, dy);
    pmap->matchDim(dx, dy);

    // apply stroke to type map by setting index values out to a certain radius around the stroke
    irad = terrain->toGrid(radius);
    inr = irad * irad;

    // convert to grid coordinates
    terrain->toGrid(currpnt, ox, oy, h);

    // bound by edge of map
    si = (int) (ox - irad); if(si < 0) si = 0;
    ei = (int) (ox + irad + 0.5f); if(ei >= dx) ei = dx-1;
    sj = (int) (oy - irad); if(sj < 0) sj = 0;
    ej = (int) (oy + irad + 0.5f); if(ej >= dy) ej = dy-1;


    #pragma omp parallel for
    for(int j = sj; j <= ej; j++)
        for(int i = si; i <= ei; i++)
        {
            float cx, cy;

            cx = ox - (float) i; cx *= cx;
            cy = oy - (float) j; cy *= cy;

            if(cx + cy <= inr) // inside region
                (* pmap->getMap())[j][i] = (int) brushtype;
        }
}

void BrushPaint::addMousePnt(View * view, TypeMap * pmap, int x, int y, float radius)
{
    Region reg;
    bool valid;
    vpPoint prjpnt, terpnt;
    int dx, dy;

    // Timer t;
    // t.start();

    // capture mouse point projected onto terrain
    // must capture current and previous point for cases where updates are not immediate and the mouse has travelled some distance
    // instead of drawing spheres onto the terrain, draw capsules in this case

    view->projectingPoint(x, y, prjpnt);
    valid = terrainProject(prjpnt, terpnt, view, terrain);

    if(valid)
    {
        if(!drawing) // first point in the stroke
        {
            prevpnt = terpnt;
            currpnt = terpnt;
            drawing = true;
        }
        else
        {
            prevpnt = currpnt;
            currpnt = terpnt;
        }

        // set bounding region to surround the points and their offset radius
        // bnd.reset();
        bnd.includePnt(currpnt);
        bnd.includePnt(prevpnt);

        BoundRect locbnd;
        locbnd = bnd;
        locbnd.expand(radius);
        terrain->getGridDim(dx, dy);

        // convert to terrain coordinates
        reg.x0 = (int) terrain->toGrid(locbnd.min.x);
        if(reg.x0 < 0) reg.x0 = 0;
        reg.y0 = (int) terrain->toGrid(locbnd.min.z);
        if(reg.y0 < 0) reg.y0 = 0;
        reg.x1 = (int) terrain->toGrid(locbnd.max.x);
        if(reg.x1 > dx) reg.x1 = dx;
        reg.y1 = (int) terrain->toGrid(locbnd.max.z);
        if(reg.y1 > dy) reg.y1 = dy;

        paintMap(pmap, radius); // render to paint map
        pmap->setRegion(reg);

        // t.stop();
        // cerr << "Brush time = " << t.peek() << endl;
    }
}

void BrushPaint::startStroke()
{
    bnd.reset();
}

void BrushPaint::finStroke()
{
    drawing = false;
}
