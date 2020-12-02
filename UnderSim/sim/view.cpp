/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems (Undergrowth simulator)
 * Copyright (C) 2020  J.E. Gain  (jgain@cs.uct.ac.za)
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


/* file: view.cpp
   author: (c) James Gain, 2006
   project: ScapeSketch - sketch-based design of procedural landscapes
   notes: controlling viewpoint changes to support terrain sketching
   changes:
*/

#include "view.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;

// #define VIEW_ASPECT 1.6
#define VIEW_ASPECT 1.333333

float wallAmbient[] = {0.7f, 0.7f, 1.0f, 1.0f};
float wallDiffuse[] = {0.225f, 0.225f, 0.75f, 1.0f};
float wallEdge[] = {0.0f, 0.0f, 0.0f, 0.5f};
float gridEdge[] = {0.3f, 0.3f, 0.3f, 0.5f};

#ifdef FIGURE
#define BASE_OFFSET 0.08
#else
#define BASE_OFFSET 0.0005
#endif

float gestureAmbient[] = {0.5f, 0.5f, 0.5f, 1.0f}; //0.2f};
float gestureDiffuse[] = {0.5f, 0.5f, 0.5f, 1.0f}; //0.2f};
float gestureEdge[] = {0.3f, 0.3f, 0.3f, 1.0f};
float highlightcol[] = {1.0f, 0.0f, 0.0f, 1.0f};

// ---------------------------------------------
// BEGIN : Code from SGI
// ---------------------------------------------

/*
 * Pass the x and y coordinates of the last and current positions of
 * the mouse, scaled so they are from (-1.0 ... 1.0).
 *
 * The resulting rotation is returned as a quaternion rotation in the
 * first paramater.
 */
void trackball(float q[4], float p1x, float p1y, float p2x, float p2y);

void negate_quat(float *q, float *qn);

/*
 * Given two quaternions, add them together to get a third quaternion.
 * Adding quaternions to get a compound rotation is analagous to adding
 * translations to get a compound translation.  When incrementally
 * adding rotations, the first argument here should be the new
 * rotation, the second and third the total rotation (which will be
 * over-written with the resulting new total rotation).
 */
void add_quats(float *q1, float *q2, float *dest);

/*
 * A useful function, builds a rotation matrix in Matrix based on
 * given quaternion.
 */
void build_rotmatrix(float m[4][4], float q[4]);

/*
 * This function computes a quaternion based on an axis (defined by
 * the given vector) and an angle about which to rotate.  The angle is
 * expressed in radians.  The result is put into the third argument.
 */
void axis_to_quat(float a[3], float phi, float q[4]);

// ---------------------------------------------
// END : Code from SGI
// ---------------------------------------------

void View::setAnimFocus(vpPoint pnt)
{
    prevfocus = focus; currfocus = focus; focalstep = 20; focus = pnt;
}

void View::setForcedFocus(vpPoint pnt)
{
    prevfocus = pnt; currfocus = pnt; focalstep = 0; focus = pnt;
}

void View::startSpin()
{
    focalstep = (int) spinsteps;
    time.start();
}

void View::updateDir()
{
    Vector copdir;
    vpPoint origin = vpPoint(0.0f, 0.0f, 0.0f);

    float m[4][4];
    float _x = 0.0f, _y = 0.0f, _z = 1.0f;

    build_rotmatrix(m, curquat);

    dir.i = m[0][0] * _x +  m[0][1] * _y +  m[0][2] * _z;
    dir.j = m[1][0] * _x +  m[1][1] * _y +  m[1][2] * _z;
    dir.k = m[2][0] * _x +  m[2][1] * _y +  m[2][2] * _z;
    dir.normalize();
    copdir = dir; copdir.mult(zoomdist);
    copdir.pntplusvec(origin, &cop);
    cop.x += currfocus.x; cop.y += currfocus.y; cop.z += currfocus.z;
}

void View::startArcRotate(float u, float v)
{
    bu = u;
    bv = v;
}

void View::arcRotate (float u, float v)
{
    trackball(lastquat, bu, bv, u, v);

    bu = u;
    bv = v;
    add_quats (lastquat, curquat, curquat);
    updateDir();
}

void View::sundir(Vector sunvec)
{
    Vector initvec, nsunvec, bivec, revvec;
    float biangle;

    // set initial quaternion to (0, 0, 1) view direction
    trackball(curquat, 0.0f, 0.0f, 0.0f, 0.1f);
    updateDir();

    // set up view rotation
    initvec = Vector(0.0, 0.0, 1.0);
    nsunvec = sunvec;
    nsunvec.normalize();

    // degenerate cases
    if(!(initvec == nsunvec)) // do nothing when sun oriented with initial
    {
        revvec = initvec;
        revvec.mult(-1.0f);
        if(nsunvec == revvec) // rotate by PI around vertical
        {
            bivec = Vector(0.0, 1.0, 0.0);
            biangle = PI;
        }
        else // non-degenerate
        {
            bivec.cross(initvec, sunvec);
            biangle = initvec.angle(sunvec);
        }

        // apply quaternion transform
        float bv[3], biquat[4];
        bv[0] = bivec.i; bv[1] = bivec.j; bv[2] = bivec.k;
        axis_to_quat(bv, biangle, curquat);
        //trackball(curquat, 0.0f, 0.0f, 0.0f, 0.0f);
        //add_quats (curquat, biquat, curquat);
        updateDir();
    }
}

void View::projectingRay(int sx, int sy, vpPoint & start, Vector & dirn)
{
    // opengl3.2 with glm
    glm::vec3 wrld, win;
    glm::vec4 viewport;
    int realx, realy;
    vpPoint pnt = vpPoint(0.0f, 0.0f, 0.0f);

    cerr << "projecting ray" << endl;
    cerr << "sx = " << sx << " sy = " << sy << endl;
    cerr << "screen params: h = " << height << " w = " << width << " startx = " << startx << " starty = " << starty << endl;
    // unproject screen point to derive world coordinates
    realy = height + 2.0f * starty - sy; realx = sx;
    win = glm::vec3((float) realx, (float) realy, 0.5f); // 0.5f
    viewport = glm::vec4(startx, starty, width, height);

    wrld = glm::unProject(win, getViewMtx(), getProjMtx(), viewport);
    pnt = vpPoint(wrld.x, wrld.y, wrld.z);
    start = cop;
    cerr << "cop = " << cop.x << ", " << cop.y << ", " << cop.z << endl;
    dirn.diff(cop, pnt); dirn.normalize();
/*
    // pre opengl3.2

    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    GLint realy, realx;
    GLdouble wx, wy, wz;
    vpPoint pnt;

    // unproject screen point to derive world coordinates
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);
    realy = viewport[3] - (GLint) sy - 1; realx = (GLint) sx;

    gluUnProject((GLdouble) realx, (GLdouble) realy, (GLdouble) 0.3f, mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
    pnt = vpPoint((float) wx, (float) wy, (float) wz);
    start = cop;
    dirn.diff(cop, pnt); dirn.normalize();
*/
}

void View::projectingPoint(int sx, int sy, vpPoint & pnt)
{
    // opengl3.2 with glm
    glm::vec3 wrld, win;
    glm::vec4 viewport;
    int realx, realy;

    // unproject screen point to derive world coordinates
    realy = height + 2.0f * starty - sy; realx = sx;
    win = glm::vec3((float) realx, (float) realy, 0.5f); // 0.5f
    viewport = glm::vec4(startx, starty, width, height);

    wrld = glm::unProject(win, getViewMtx(), getProjMtx(), viewport);
    pnt = vpPoint(wrld.x, wrld.y, wrld.z);

/*
    // pre opengl3.2

    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    GLint realy, realx;
    GLdouble wx, wy, wz;
    
    // unproject screen point to derive world coordinates
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);
    realy = viewport[3] - (GLint) sy - 1; realx = (GLint) sx;
    
    gluUnProject((GLdouble) realx, (GLdouble) realy, (GLdouble) 0.3f, mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
    pnt = vpPoint((float) wx, (float) wy, (float) wz);
*/
}

void View::inscreenPoint(int sx, int sy, vpPoint & pnt)
{
    // opengl3.2 with glm
    int realx, realy;

    // unproject screen point to derive world coordinates
    realy = height + 2.0f * starty - sy; realx = sx;
    pnt = vpPoint(realx / width, realy / height, 0.0f); // FIX?
}

void View::projectOntoManip(vpPoint pick, vpPoint mpnt, Vector mdirn, vpPoint & mpick)
{
    Vector pdirn;
    float t;
    
    pdirn.diff(mpnt, pick);
    t = pdirn.dot(mdirn);
    mdirn.mult(t);
    mdirn.pntplusvec(mpnt, &mpick);
}

void View::projectMove(int ox, int oy, int nx, int ny, vpPoint cp, Vector & del)
{
    glm::vec3 wrld, win;
    glm::vec4 viewport;
    vpPoint npnt, opnt;
    Vector vecs, vecw;
    float dw, ds;
    int realx, realy;

    viewport = glm::vec4(startx, starty, width, height);

    // unproject new point
    // unproject screen point to derive world coordinates
    realy = height + 2.0f * starty - ny; realx = nx;
    win = glm::vec3((float) realx, (float) realy, 0.5f);
    wrld = glm::unProject(win, getViewMtx(), getProjMtx(), viewport);
    npnt = vpPoint(wrld.x, wrld.y, wrld.z);

    // unproject old point
    realy = height + 2.0f * starty - oy; realx = ox;
    win = glm::vec3((float) realx, (float) realy, 0.5f);
    wrld = glm::unProject(win, getViewMtx(), getProjMtx(), viewport);
    opnt = vpPoint(wrld.x, wrld.y, wrld.z);

    del.diff(opnt, npnt); // direction yes, but scale incorrect
    vecs.diff(cop, opnt);
    vecw.diff(cop, cp);
    ds = vecs.length();
    dw = vecw.length();
    del.mult(dw/ds);

/*
    // pre opengl3.2 - deprecated

    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    GLint realy, realx;
    GLdouble wx, wy, wz;
    vpPoint npnt, opnt;
    Vector vecs, vecw;
    float dw, ds;

    // unproject screen point to derive world coordinates

    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvmatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projmatrix);

    // unproject new point
    realy = viewport[3] - (GLint) ny - 1; realx = (GLint) nx;
    gluUnProject((GLdouble) realx, (GLdouble) realy, (GLdouble) 0.3f, mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
    npnt = vpPoint((float) wx, (float) wy, (float) wz);

    // unproject old point
    realy = viewport[3] - (GLint) oy - 1; realx = (GLint) ox;
    gluUnProject((GLdouble) realx, (GLdouble) realy, (GLdouble) 0.3f, mvmatrix, projmatrix, viewport, &wx, &wy, &wz);
    opnt = vpPoint((float) wx, (float) wy, (float) wz);

    del.diff(opnt, npnt); // direction yes, but scale incorrect
    vecs.diff(cop, opnt);
    vecw.diff(cop, cp);
    ds = vecs.length();
    dw = vecw.length();
    del.mult(dw/ds);
*/
}

glm::mat4x4 View::getMatrix()
{
    glm::mat4x4 projMx, viewMx;

    // frustum
    projMx = getProjMtx();
    viewMx = getViewMtx();
    return projMx  * viewMx;
}

glm::mat4x4 View::getProjMtx()
{
    glm::mat4x4 projMx;
    float minx, maxx, minex, maxex, sx, sy, ex, ey, orthostep;
    // frustum

    if(viewtype == ViewState::PERSPECTIVE)
    {
        minx = -8.0f * ACTUAL_ASPECT;
        maxx = 8.0f * ACTUAL_ASPECT;
        projMx = glm::frustum(minx, maxx, -8.0f, 8.0f, 50.0f, 100000.0f);
    }
    else if(viewtype == ViewState::ORTHOGONAL)
    {
        minex = -terextent/2.0f;
        maxex = terextent/2.0f;
        orthostep = terextent / (float) orthodiv;
        sx = minex + (float) ox * orthostep;
        ex = sx + orthostep;
        sy = minex + (float) oy * orthostep;
        ey = sy + orthostep;
        projMx = glm::ortho(sx, ex, sy, ey, 0.0f, terextent * 4.0f);
        zoomdist = terextent/2.0f;
    }

    return projMx;
}

glm::mat4x4 View::getViewMtx()
{
    glm::mat4x4 viewMx, quatMx;
    glm::vec3 trs;
    float mm[4][4];

    // zoom
    viewMx = glm::mat4x4(1.0f);
    trs = glm::vec3(0.0f, 0.0f, -1.0f * zoomdist);
    viewMx = glm::translate(viewMx, trs);

    // quaternion to mult matrix from arcball
    build_rotmatrix(mm, curquat);
    quatMx = glm::make_mat4(&mm[0][0]);
    viewMx = viewMx * quatMx;

    // center of projection
    trs = glm::vec3(-currfocus.x, -currfocus.y, -currfocus.z);
    viewMx = glm::translate(viewMx, trs);

    return viewMx;
}

glm::mat3x3 View::getNormalMtx()
{
    glm::mat3x3 normMx;

    normMx = glm::transpose(glm::inverse(glm::mat3(getViewMtx())));
    return normMx;
}

float View::getScaleFactor()
{
    float scale, tsizemul;

    tsizemul = viewscale / 10000.0f;

    if(getZoom() > 9.0f * viewscale)
        scale = 1.525f * tsizemul;
    else
        scale = std::max(0.1f * tsizemul, getZoom() / 80000.0f + 0.4f * tsizemul);

    //return scale;
    return 2.3f * scale;
}

float View::getScaleConst()
{
    return viewscale / 10000.0f;
}


void View::apply()
{
    /*
    // oldstyle openGL pre 3.2
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glFrustum(-0.05, 0.05, -0.05, 0.05, 0.5, 150.0);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity(); */

    // calculate view direction and center of projection using stored quaternion
    updateDir();
}

void View::varietyview()
{
    float a[3];
    // rotate first by 45 deg
    zoomdist = 3.8 * viewscale;

    a[0] = 0.0f; a[1] = 1.0; a[2] = 0.0;
    axis_to_quat(a, -7.0f * PI2/8.0f, curquat); // -1.0 for join-manip
    trackball(lastquat, 0.0f, 0.0f, 0.0f, -0.3f);
    add_quats (lastquat, curquat, curquat);
    updateDir();
}

bool View::animate()
{
    if(focalstep > 0)
    {
        float t = (float) (focalstep-1) / 20.0f;
        currfocus.affinecombine(t, prevfocus, (1.0f-t), focus);
        //apply();
        updateDir();
        focalstep--;
        return true;
    }
    return false;
}

bool View::spin()
{
    float a[3];
    if(focalstep > 0)
    {
        a[0] = 0.0f; a[1] = 1.0; a[2] = 0.0;
        axis_to_quat(a, (spinsteps - (float) focalstep) * PI2/spinsteps, curquat);
        trackball(lastquat, 0.0f, 0.0f, 0.0f, -0.5f);
        add_quats (lastquat, curquat, curquat);
        updateDir();
        focalstep--;

        if(focalstep == 1)
        {
            time.stop();
            cerr << "spin took " << time.peek() << "s" << endl;
        }
        return true;
    }
    return false;
}

bool View::save(const char * filename)
{
    ofstream outfile;

    outfile.open(filename, ios_base::out);
    if(outfile.is_open())
    {
        outfile << cop.x << " " << cop.y << " " << cop.z << endl;
        outfile << light.x << " " << light.y << " " << light.z << endl;
        outfile.close();
        // don't need to save direction and up, since these can be derived
        return true;
    }
    else
        return false;
}

bool View::load(const char * filename)
{
    ifstream infile;

    infile.open(filename, ios_base::in);
    if(infile.is_open() && !infile.eof())
    {
        infile >> cop.x; infile >> cop.y; infile >> cop.z;
        infile >> light.x; infile >> light.y; infile >> light.z;
        infile.close();
        return true;
    }
    else
    {
        infile.close();
        return false;
    }
}

void View::print()
{
    cerr << "VIEW" << endl;
    cerr << "width = " << width << ", height = " << height << ", startx = " << startx << ", starty = " << starty << endl;
    cerr << "cop = " << cop.x << ", " << cop.y << ", " << cop.z << endl;
    cerr << "dir = " << dir.i << ", " << dir.j << ", " << dir.k << endl;
    cerr << "zoomdist = " << zoomdist << endl;
    cerr << "perspwidth = " << perspwidth << endl;
}

// ---------------------------------------------
// BEGIN : Code from SGI
// ---------------------------------------------

#include <cstdio>
/*
 * (c) Copyright 1993, 1994, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 */
/*
 * Trackball code:
 *
 * Implementation of a virtual trackball.
 * Implemented by Gavin Bell, lots of ideas from Thant Tessman and
 *   the August '88 issue of Siggraph's "Computer Graphics," pp. 121-129.
 *
 * Vector manip code:
 *
 * Original code from:
 * David M. Ciemiewicz, Mark Grossman, Henry Moreton, and Paul Haeberli
 *
 * Much mucking with by:
 * Gavin Bell
 */
#if defined(_WIN32)
#pragma warning (disable:4244)          /* disable bogus conversion warnings */
#endif
#include <cmath>

/*
 * This size should really be based on the distance from the center of
 * rotation to the point on the object underneath the mouse.  That
 * point would then track the mouse as closely as possible.  This is a
 * simple example, though, so that is left as an Exercise for the
 * Programmer.
 */
#define TRACKBALLSIZE  (0.8f)

/*
 * Local function prototypes (not defined in trackball.h)
 */
static float tb_project_to_sphere(float, float, float);
static void normalize_quat(float [4]);

void
vzero(float *v)
{
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
}

void
vset(float *v, float x, float y, float z)
{
    v[0] = x;
    v[1] = y;
    v[2] = z;
}

void
vsub(const float *src1, const float *src2, float *dst)
{
    dst[0] = src1[0] - src2[0];
    dst[1] = src1[1] - src2[1];
    dst[2] = src1[2] - src2[2];
}

void
vcopy(const float *v1, float *v2)
{
    register int i;
    for (i = 0 ; i < 3 ; i++)
        v2[i] = v1[i];
}

void
vcross(const float *v1, const float *v2, float *cross)
{
    float temp[3];

    temp[0] = (v1[1] * v2[2]) - (v1[2] * v2[1]);
    temp[1] = (v1[2] * v2[0]) - (v1[0] * v2[2]);
    temp[2] = (v1[0] * v2[1]) - (v1[1] * v2[0]);
    vcopy(temp, cross);
}

float
vlength(const float *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void
vscale(float *v, float div)
{
    v[0] *= div;
    v[1] *= div;
    v[2] *= div;
}

void
vnormal(float *v)
{
    vscale(v,1.0/vlength(v));
}

float
vdot(const float *v1, const float *v2)
{
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

void
vadd(const float *src1, const float *src2, float *dst)
{
    dst[0] = src1[0] + src2[0];
    dst[1] = src1[1] + src2[1];
    dst[2] = src1[2] + src2[2];
}

/*
 * Ok, simulate a track-ball.  Project the points onto the virtual
 * trackball, then figure out the axis of rotation, which is the cross
 * product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
 * Note:  This is a deformed trackball-- is a trackball in the center,
 * but is deformed into a hyperbolic sheet of rotation away from the
 * center.  This particular function was chosen after trying out
 * several variations.
 *
 * It is assumed that the arguments to this routine are in the range
 * (-1.0 ... 1.0)
 */
void
trackball(float q[4], float p1x, float p1y, float p2x, float p2y)
{
    float a[3]; /* Axis of rotation */
    float phi;  /* how much to rotate about axis */
    float p1[3], p2[3], d[3];
    float t;

    if (p1x == p2x && p1y == p2y) {
        /* Zero rotation */
        vzero(q);
        q[3] = 1.0;
        return;
    }

    /*
     * First, figure out z-coordinates for projection of P1 and P2 to
     * deformed sphere
     */
    vset(p1,p1x,p1y,tb_project_to_sphere(TRACKBALLSIZE,p1x,p1y));
    vset(p2,p2x,p2y,tb_project_to_sphere(TRACKBALLSIZE,p2x,p2y));

    /*
     *  Now, we want the cross product of P1 and P2
     */
    vcross(p2,p1,a);

    /*
     *  Figure out how much to rotate around that axis.
     */
    vsub(p1,p2,d);
    t = vlength(d) / (2.0*TRACKBALLSIZE);

    /*
     * Avoid problems with out-of-control values...
     */
    if (t > 1.0) t = 1.0;
    if (t < -1.0) t = -1.0;
    phi = 2.0 * asin(t);

    axis_to_quat(a,phi,q);
}

/*
 *  Given an axis and angle, compute quaternion.
 */
void
axis_to_quat(float a[3], float phi, float q[4])
{
    vnormal(a);
    vcopy(a,q);
    vscale(q,sin(phi/2.0));
    q[3] = cos(phi/2.0);
}

/*
 * Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
 * if we are away from the center of the sphere.
 */
static float
tb_project_to_sphere(float r, float x, float y)
{
    float d, t, z;

    d = sqrt(x*x + y*y);
    if (d < r * 0.70710678118654752440) {    /* Inside sphere */
        z = sqrt(r*r - d*d);
    } else {           /* On hyperbola */
        t = r / 1.41421356237309504880;
        z = t*t / d;
    }
    return z;
}

/*
 * Given two rotations, e1 and e2, expressed as quaternion rotations,
 * figure out the equivalent single rotation and stuff it into dest.
 *
 * This routine also normalizes the result every RENORMCOUNT times it is
 * called, to keep error from creeping in.
 *
 * NOTE: This routine is written so that q1 or q2 may be the same
 * as dest (or each other).
 */

#define RENORMCOUNT 97

void
negate_quat(float q[4], float nq[4])
{
    nq[0] = -q[0];
    nq[1] = -q[1];
    nq[2] = -q[2];
    nq[3] = q[3];
}

void
add_quats(float q1[4], float q2[4], float dest[4])
{
    static int count=0;
    float t1[4], t2[4], t3[4];
    float tf[4];

#if 0
printf("q1 = %f %f %f %f\n", q1[0], q1[1], q1[2], q1[3]);
printf("q2 = %f %f %f %f\n", q2[0], q2[1], q2[2], q2[3]);
#endif

    vcopy(q1,t1);
    vscale(t1,q2[3]);

    vcopy(q2,t2);
    vscale(t2,q1[3]);

    vcross(q2,q1,t3);
    vadd(t1,t2,tf);
    vadd(t3,tf,tf);
    tf[3] = q1[3] * q2[3] - vdot(q1,q2);

#if 0
printf("tf = %f %f %f %f\n", tf[0], tf[1], tf[2], tf[3]);
#endif

    dest[0] = tf[0];
    dest[1] = tf[1];
    dest[2] = tf[2];
    dest[3] = tf[3];

    if (++count > RENORMCOUNT) {
        count = 0;
        normalize_quat(dest);
    }
}

/*
 * Quaternions always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
 * If they don't add up to 1.0, dividing by their magnitued will
 * renormalize them.
 *
 * Note: See the following for more information on quaternions:
 *
 * - Shoemake, K., Animating rotation with quaternion curves, Computer
 *   Graphics 19, No 3 (Proc. SIGGRAPH'85), 245-254, 1985.
 * - Pletinckx, D., Quaternion calculus as a basic tool in computer
 *   graphics, The Visual Computer 5, 2-13, 1989.
 */
static void
normalize_quat(float q[4])
{
    int i;
    float mag;

    mag = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    for (i = 0; i < 4; i++) q[i] /= mag;
}

/*
 * Build a rotation matrix, given a quaternion rotation.
 *
 */
void
build_rotmatrix(float m[4][4], float q[4])
{
    m[0][0] = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    m[0][1] = 2.0 * (q[0] * q[1] - q[2] * q[3]);
    m[0][2] = 2.0 * (q[2] * q[0] + q[1] * q[3]);
    m[0][3] = 0.0;

    m[1][0] = 2.0 * (q[0] * q[1] + q[2] * q[3]);
    m[1][1]= 1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]);
    m[1][2] = 2.0 * (q[1] * q[2] - q[0] * q[3]);
    m[1][3] = 0.0;

    m[2][0] = 2.0 * (q[2] * q[0] - q[1] * q[3]);
    m[2][1] = 2.0 * (q[1] * q[2] + q[0] * q[3]);
    m[2][2] = 1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]);
    m[2][3] = 0.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
}

// ---------------------------------------------
// BEGIN : Code from SGI
// ---------------------------------------------
