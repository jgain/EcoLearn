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

// terrain.h: model for terrain. Responsible for storage and display of heightfield terrain data
// author: James Gain
// date: 17 December 2012

#include <GL/glew.h>
#include "terrain.h"
#include <sstream>
#include <common/debug_string.h>
#include <common/terragen.h>
#include <streambuf>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <utility>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "data_importer/extract_png.h"

using namespace std;

void Terrain::toGrid(vpPoint p, float & x, float & y, float & h) const
{
    int gx, gy;
    float tx, ty, convx, convy;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    convx = (float) (gx-1) / tx;
    convy = (float) (gy-1) / ty;
    x = p.x * convx;
    y = p.z * convy;
    h = p.y;
    if(scaleOn)
        h *= scfac;
}


void Terrain::toGrid(vpPoint p, int &x, int &y) const
{
    int gx, gy;
    float tx, ty, convx, convy;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    convx = (float) (gx-1) / tx;
    convy = (float) (gy-1) / ty;
    x = (int) (p.x * convx);
    y = (int) (p.z * convy);
}


float Terrain::toGrid(float wdist) const
{
    int gx, gy;
    float tx, ty, conv;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    conv = (float) (gx-1) / tx;
    return wdist * conv;
}


vpPoint Terrain::toWorld(float x, float y, float h) const
{
    int gx, gy;
    float tx, ty, convx, convy;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    convx = tx / (float) (gx-1);
    convy = ty / (float) (gy-1);

    return vpPoint(x * convx, h, y * convy);
}

vpPoint Terrain::toWorld(int x, int y, float h) const
{
    int gx, gy;
    float tx, ty, convx, convy;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    convx = tx / (float) (gx-1);
    convy = ty / (float) (gy-1);

    return vpPoint((float) x * convx, h, (float) y * convy);
}


float Terrain::toWorld(float gdist) const
{
    int gx, gy;
    float tx, ty, conv;

    getGridDim(gx, gy);
    getTerrainDim(tx, ty);
    conv = tx / (float) (gx-1);

    return gdist * conv;
}


bool Terrain::inWorldBounds(vpPoint p) const
{
    return (p.x >= 0.0f && p.x <= dimx && p.z >= 0.0f && p.z <= dimy);
}

bool Terrain::inSynthBounds(vpPoint p) const
{
    return (p.x >= 0.0f-synthx && p.x <= dimx+synthx && p.z >= 0.0f-synthy && p.z <= dimy+synthy);
}

void Terrain::init(int dx, int dy, float sx, float sy)
{
    grid.allocate(Region(0, 0, dx, dy));
    grid.fill(0.0f);
    setTerrainDim(sx, sy);
    setFocus(vpPoint(sx/2.0f, grid[dy/2-1][dx/2-1], sy/2.0f));
    scfac = 1.0f;

    // init accel structure
    spherestep = 8;
    numspx = (grid.width()-1) / spherestep + 1; numspy = (grid.height()-1) / spherestep + 1;
    for(int i = 0; i < numspx; i++)
    {
        std::vector<AccelSphere> sphrow;
        for(int j = 0; j < numspy; j++)
        {
            AccelSphere sph;
            sphrow.push_back(sph);
        }
        boundspheres.push_back(sphrow);
    }

    bufferState = BufferState::REALLOCATE;
    accelValid = false;
    scaleOn = false;
}

void Terrain::initGrid(int dx, int dy, float sx, float sy)
{
    init(dx, dy, sx, sy);
    grid.fill(0.0f);
}

void Terrain::delGrid()
{
    grid.clear();
    if(boundspheres.size() > 0)
    {
        for(int i = 0; i < (int) boundspheres.size(); i++)
            boundspheres[i].clear();
        boundspheres.clear();
    }

    bufferState = BufferState::REALLOCATE;
    accelValid = false;
}

void Terrain::clipRegion(Region &reg)
{
    if(reg.x0 < 0) reg.x0 = 0;
    if(reg.y0 < 0) reg.y0 = 0;
    if(reg.x1 > grid.width()) reg.x1 = grid.width();
    if(reg.y1 > grid.height()) reg.y1 = grid.height();
}

void Terrain::setMidFocus()
{
    int dx, dy;
    float sx, sy;
    
    getGridDim(dx, dy);
    getTerrainDim(sx, sy);
    if(dx > 0 && dy > 0)
        setFocus(vpPoint(sx/2.0f, grid[dy/2-1][dx/2-1], sy/2.0f));
    else
        setFocus(vpPoint(0.0f, 0.0f, 0.0f));
}

void Terrain::getMidPoint(vpPoint & mid)
{
    int dx, dy;
    float sx, sy;

    getGridDim(dx, dy);
    getTerrainDim(sx, sy);
    if(dx > 0 && dy > 0)
        mid = vpPoint(sx/2.0f, grid[dy/2-1][dx/2-1], sy/2.0f);
    else
        mid = vpPoint(0.0f, 0.0f, 0.0f);

}

void Terrain::getGridDim(int & dx, int & dy) const
{
    dx = grid.width();
    dy = grid.height();
}

void Terrain::getGridDim(uint & dx, uint & dy)
{
    dx = (uint) grid.width();
    dy = (uint) grid.height();
}

void Terrain::getTerrainDim(float &tx, float &ty) const
{
    tx = dimx; ty = dimy;
}

void Terrain::setTerrainDim(float tx, float ty)
{
    int gx, gy;

    getGridDim(gx, gy);
    dimx = tx; dimy = ty;

    // calc allowable synth border
    synthx = (0.5f / (float) (gx-1) + pluszero) * dimx;
    synthy = (0.5f / (float) (gy-1) + pluszero) * dimy;
}

float Terrain::samplingDist()
{
    int dx, dy;
    float tx, ty;
    getGridDim(dx, dy);
    getTerrainDim(tx, ty);
    return (0.5f * std::min(tx, ty)) / (float) (std::max(dx,dy)-1); // separation between vertices, about 2-3 vertices per grid cell
}

float Terrain::smoothingDist()
{
    return 30.0f * samplingDist(); // about 10 grid points per curve segment
}

float Terrain::longEdgeDist()
{
    float tx, ty;
    getTerrainDim(tx, ty);
    return std::max(tx, ty);
}

const float * Terrain::getGridData(int & dx, int & dy)
{
    int i, j;

    getGridDim(dx, dy);
    if(scaleOn)
    {
        scaledgrid.allocate(Region(0, 0, grid.width(), grid.height()));
        for(j = 0; j < grid.height(); j++)
            for(i = 0; i < grid.width(); i++)
                scaledgrid[j][i] = grid[j][i] * scfac;
        return scaledgrid.get();
    }
    else
    {
        return grid.get();
    }
}

float Terrain::getHeight(int x, int y)
{
    return grid[x][y];
}

float Terrain::getFlatHeight(int idx)
{
    int x, y, dx, dy;
    getGridDim(dx, dy);

    x = idx % dx;
    y = idx / dx;
    return grid[x][y];
}

float Terrain::getHeightFromReal(float x, float y)
{
    int gx, gy;
    toGrid(vpPoint(x, 0, y), gx, gy);
    return grid[gx][gy];
}

void Terrain::getNormal(int x, int y, Vector & norm)
{
    vpPoint x1, x2, y1, y2;
    int dx, dy;
    Vector dfdx, dfdy;

    getGridDim(dx, dy);

    // x-positions
    if(x > 0)
        x1 = toWorld(x-1, y, getHeight(x-1, y));
    else
        x1 = toWorld(x, y, getHeight(x, y));

    if(x < dx-1)
        x2 = toWorld(x+1, y, getHeight(x+1, y));
    else
        x2 = toWorld(x, y, getHeight(x, y));

    // y-positions
    if(y > 0)
        y1 = toWorld(x, y-1, getHeight(x, y-1));
    else
        y1 = toWorld(x, y, getHeight(x, y));

    if(y < dy-1)
        y2 = toWorld(x, y+1, getHeight(x, y+1));
    else
        y2 = toWorld(x, y, getHeight(x, y));

    // cross pattern
    dfdx.diff(x1, x2);
    dfdy.diff(y1, y2);
    dfdx.normalize();
    dfdy.normalize();

    norm.cross(dfdx, dfdy);
    norm.mult(-1.0f); // JGBUG - may be wrong direction
    norm.normalize();
}

float Terrain::getCellExtent()
{
    return dimx / (float) grid.width();
}

void Terrain::updateBuffers(PMrender::TRenderer * renderer) const
{
    const int width = grid.width();
    const int height = grid.height();
    float scx, scy;

    getTerrainDim(scx, scy);

    glewExperimental = GL_TRUE;
    if(!glewSetupDone)
      {
         GLenum err = glewInit();
         if (GLEW_OK != err)
         {
            std::cerr<< "GLEW: initialization failed\n\n";
         }
         glewSetupDone = true;
      }

    if (bufferState == BufferState::REALLOCATE || bufferState == BufferState::DIRTY )
        renderer->updateHeightMap(width, height, scx, scy, (GLfloat*)grid.get(), true);
    else
        renderer->updateHeightMap(width, height, scx, scy, (GLfloat*)grid.get());

    bufferState = BufferState::CLEAN;
}

void Terrain::draw(View * view, PMrender::TRenderer *renderer) const
{
    updateBuffers(renderer);

    // call draw function
    renderer->draw(view);
}

void Terrain::buildSphereAccel()
{
    int si, sj, i, j, imin, imax, jmin, jmax;
    float rad, sqlen;
    vpPoint p, c, b1, b2;
    Vector del;

    // cerr << "numspx = " << numspx << ", numspy = " << numspy << endl;
    for(si = 0; si < numspx; si++)
        for(sj = 0; sj < numspy; sj++)
        {
            imin = si*spherestep; imax = std::min(imin+spherestep, grid.width());
            jmin = sj*spherestep; jmax = std::min(jmin+spherestep, grid.height());
            // cerr << "(" << si << ", " << sj << ") = " << "i: " << imin << " - " << imax << " j: " << jmin << " - " << jmax << endl;

            // center point
            b1 = toWorld(imin, jmin, grid[jmin][imin]);
            b2 = toWorld(imax, jmax, grid[jmax-1][imax-1]);
            c.affinecombine(0.5f, b1, 0.5f, b2);

            // update radius
            rad = 0.0f;
            for(j = jmin; j < jmax; j++)
                for(i = imin; i < imax; i++)
                {
                    p = toWorld(i, j, grid[j][i]);
                    del.diff(c, p);
                    sqlen = del.sqrdlength();
                    if(sqlen > rad)
                        rad = sqlen;
                }
            boundspheres[si][sj].center = c;
            boundspheres[si][sj].radius = sqrtf(rad);
        }
    accelValid = true;
}

bool Terrain::rayIntersect(vpPoint start, Vector dirn, vpPoint & p)
{
    int i, j, si, sj, imin, imax, jmin, jmax;
    vpPoint currp;
    float besttval, tval, dist;
    bool found = false;
    float tol = dimx / (float) (grid.width()-1); // set world space detection tolerance to approx half gap between grid points

    besttval = 100000000.0f;

    if(!accelValid)
        buildSphereAccel();

    // bounding sphere accel structure
    for(si = 0; si < numspx; si++)
        for(sj = 0; sj < numspy; sj++)
        {
            rayPointDist(start, dirn, boundspheres[si][sj].center, tval, dist);
            if(dist <= boundspheres[si][sj].radius) // intersects enclosing sphere so test enclosed points
            {
                imin = si*spherestep; imax = std::min(imin+spherestep, grid.width());
                jmin = sj*spherestep; jmax = std::min(jmin+spherestep, grid.height());
                // check ray against grid points
                for(j = jmin; j < jmax; j++)
                    for(i = imin; i < imax; i++)
                    {
                        currp = toWorld(i, j, grid[j][i]);
                        rayPointDist(start, dirn, currp, tval, dist);
                        if(dist < tol)
                        {
                            found = true;
                            if(tval < besttval)
                            {
                                besttval = tval;
                                p = currp;
                            }
                        }
                    }

            }

        }

    return found;
}

bool Terrain::pick(int sx, int sy, View * view, vpPoint & p)
{
    vpPoint start;
    Vector dirn;

    cerr << "sx = " << sx << ", sy = " << sy << endl;

    // find ray params from viewpoint through screen <sx, sy>
    view->projectingRay(sx, sy, start, dirn);

    return rayIntersect(start, dirn, p);
}

bool Terrain::drapePnt(vpPoint pnt, vpPoint & drape)
{
    float x, y, h, drapeh, u, v, h0, h1, ux, uy;
    int cx, cy, dx, dy;

    getGridDim(dx, dy);
    toGrid(pnt, x, y, h); // locate point on base domain

    // test whether point is in bounds
    ux = (float) (dx-1) - pluszero;
    uy = (float) (dy-1) - pluszero;

    if(x < pluszero || y < pluszero || x > ux || y > uy)
        return false;

    // index of grid cell
    cx = (int) floor(x);
    cy = (int) floor(y);

    // get parametric coordinates within grid cell
    u = (x - (float) cx);
    v = (y - (float) cy);

    // bilinear interpolation
    h0 = (1.0f - u) * grid[cy][cx] + u * grid[cy][cx+1];
    h1 = (1.0f - u) * grid[cy+1][cx] + u * grid[cy+1][cx];
    drapeh = (1.0f - v) * h0 + v * h1;
    // this could be implemented using ray-triangle intersection
    // but it would be much less efficient
    drape = toWorld(x, y, drapeh);

    return true;
}

void Terrain::loadTer(const uts::string &filename)
{
    float sx, sy, step;
    int dx, dy;

    grid.read(filename);
    dx = grid.width(); dy = grid.height();
    step = grid.step();
    sx = (float) grid.width() * step;
    sy = (float) grid.height() * step;
    init(dx, dy, sx, sy);
}

void Terrain::loadElv(const uts::string &filename)
{
    float step, lat;
    int dx, dy;

    float val;
    ifstream infile;

    infile.open((char *) filename.c_str(), ios_base::in);
    if(infile.is_open())
    {
        infile >> dx >> dy;
        infile >> step;
        infile >> lat;
        delGrid();
        init(dx, dy, (float) dx * step, (float) dy * step);
        latitude = lat;
        for (int y = 0; y < dy; y++)
        {
            for (int x = 0; x < dx; x++)
            {
                infile >> val;
                grid[x][y] = val * 0.3048f; // convert from feet to metres
            }
        }
        setMidFocus();
        infile.close();
    }
    else
    {
        cerr << "Error Terrain::loadElv:unable to open file " << filename << endl;
    }
}

void Terrain::loadPng(const uts::string &filename)
{
    int width, height;
    auto png_data = get_image_data_48bit(filename, width, height);
    if (png_data.size() == 0)
    {
        cerr << "Error Terrain::loadPng: unable to open file " << filename << endl;
        return;
    }
    // we assume that the heightfield is a grayscale image, so the first channel will contain the necessary height info
    std::vector<float> heights = png_data[0];
    float vert_units = 0.3048f;
    float horiz_units = vert_units * 3;
    delGrid();	// not sure why I have to do this - just saw it is being done in  loadElv (investigate why)
    init(width, height, (float) width * horiz_units, (float) height * horiz_units);
    latitude = 0.0f;	// keeping it like this for now - maybe separate latitude and terrain loading??
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // the png is imported in row-major order
            int idx = y * width + x;
            grid[x][y] = heights[idx] * vert_units;
        }
    }
    setMidFocus();	// again, not really sure why I have to do this, but keep it for now
}

void Terrain::saveTer(const uts::string &filename)
{
    grid.write(filename);
}

void Terrain::saveElv(const uts::string &filename)
{
    float step;
    int gx, gy;

    float val;
    ofstream outfile;

    outfile.open((char *) filename.c_str(), ios_base::out);
    if(outfile.is_open())
    {
        getGridDim(gx, gy);
        step = grid.step();
        std::cout << "Step size: " << step << std::endl;
        outfile << gx << " " << gy << " " << step << " " << latitude << endl;
        for (int x = 0; x < gx; x++)
        {
            for (int y = 0; y < gy; y++)
            {
                outfile << grid[x][y] << " ";
            }
        }
        outfile << endl;
        outfile.close();
    }
    else
    {
        cerr << "Error Terrain::loadElv:unable to open file " << filename << endl;
    }
}

void Terrain::calcMeanHeight()
{
    int i, j, cnt = 0;
    hghtmean = 0.0f;

    for(j = 0; j < grid.height(); j++)
        for(i = 0; i < grid.width(); i++)
        {
            hghtmean += grid[j][i];
            cnt++;
        }
    hghtmean /= (float) cnt;
}

void Terrain::getHeightBounds(float &minh, float &maxh)
{
    int i, j;
    float hght;

    maxh = -10000000.0f;
    minh = 100000000.0;

    for(j = 0; j < grid.height(); j++)
        for(i = 0; i < grid.width(); i++)
        {
            hght = grid[j][i];
            if(hght < minh)
                minh = hght;
            if(hght > maxh)
                maxh = hght;
        }
}
