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

// file: vecpnt.cpp
// author: James Gain
// project: Interactive Sculpting (1997+)
// notes: Basic vector and point arithmetic library. Inlined for efficiency.
// changes: included helpful geometry routines (2006)

#include "vecpnt.h"
#include <stdio.h>
#include <iostream>

using namespace std;

void Plane::formPlane(vpPoint pnt, Vector norm)
{
    n = norm;
    n.normalize();
    d = -1.0f * (n.i * pnt.x + n.j * pnt.y + n.k * pnt.z);
}

bool Plane::formPlane(vpPoint * tri)
{
    Vector v1, v2, zero;

    v1.diff(tri[0], tri[1]);
    v2.diff(tri[0], tri[2]);
    v1.normalize();
    v2.normalize();
    n.cross(v1, v2);
    zero = Vector(0.0f, 0.0f, 0.0f);
    if(n == zero)
        return false;
    n.normalize();
    d = -1.0f * (n.i * tri[0].x + n.j * tri[0].y + n.k * tri[0].z);
    return true;
}

bool Plane::rayPlaneIntersect(vpPoint start, Vector dirn, float & tval)
{
    Vector svec;
    double num, den;

    // this approach can be numerically unstable for oblique intersections
    svec.pntconvert(start);

    num = (double) (d + svec.dot(n));
    den = (double) dirn.dot(n);
    if(den == 0.0f) // dirn parallel to plane
        if(num == 0.0f) // ray lies in plane
            tval = 0.0f;
        else
            return false;
    else
        tval = (float) (-1.0f * num/den);
    return true;
}

bool Plane::rayPlaneIntersect(vpPoint start, Vector dirn, vpPoint & intersect)
{
    Vector svec, dvec;
    double num, den;
    float tval;

    svec.pntconvert(start);

    num = (double) (d + svec.dot(n));
    den = (double) dirn.dot(n);
    if(den == 0.0f) // dirn parallel to plane
        if(num == 0.0f) // ray lies in plane
            tval = 0.0f;
        else
            return false;
    else
        tval = (float) (-1.0f * num/den);
    dvec = dirn;
    dvec.mult(tval);
    dvec.pntplusvec(start, &intersect);
    return true;
}

bool Plane::side(vpPoint pnt)
{
    return ((pnt.x * n.i + pnt.y * n.j + pnt.z * n.k + d) >= 0.0f);
}

float Plane::dist(vpPoint pnt)
{
    return fabs(pnt.x * n.i + pnt.y * n.j + pnt.z * n.k + d);
}

float Plane::height(vpPoint pnt)
{
    // B.y = -A.x - C.z - D
    return (n.i * pnt.x + n.k * pnt.z + d) / (n.j * -1.0f);
}

void Plane::projectPnt(vpPoint pnt, vpPoint * proj)
{
    Vector svec;
    float tval;

    svec.pntconvert(pnt);
    tval = -1.0f * (d + svec.dot(n));
    svec = n;
    svec.mult(tval);
    svec.pntplusvec(pnt, proj);
}

/*
void Plane::drawPlane()
{
    float frontCol[] = { 0.0f, 0.0f, 1.0f, 1.0f };
    float backCol[] = { 0.0f, 1.0f, 0.0f, 1.0f };

    // generate origin

    glBegin(GL_POLYGON);


    glEnd();
} */

///
/// BoundRect
///


// optimized method: Williams et al, "An Efficient and Robust RayñBox Intersection Algorithm",
// Journal of Graphics Tools
/*
bool BBox::rayBoxIntersect(Ray r, float t0, float t1)
{
    float tmin, tmax, tymin, tymax, tzmin, tzmax;
    vpPoint bounds[2];

    bounds[0] = min;
    bounds[1] = max;

    tmin = (bounds[r.sign[0]].x - r.origin.x) * r.inv_direction.i;
    tmax = (bounds[1-r.sign[0]].x - r.origin.x) * r.inv_direction.i;
    tymin = (bounds[r.sign[1]].y - r.origin.y) * r.inv_direction.j;
    tymax = (bounds[1-r.sign[1]].y - r.origin.y) * r.inv_direction.j;

    if ( (tmin > tymax) || (tymin > tmax) )
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[r.sign[2]].z - r.origin.z) * r.inv_direction.k;
    tzmax = (bounds[1-r.sign[2]].z - r.origin.z) * r.inv_direction.k;
    if ( (tmin > tzmax) || (tzmin > tmax) )
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    return ( (tmin < t1) && (tmax > t0) );
}*/

float BoundRect::nearest(vpPoint p) const
{
    float dx, dz, near;

    // quadrant based appraoch
    if(p.x < min.x)
    {
        dx = p.x - min.x; dx *= dx;
        if(p.z < min.z)
        {
            dz = p.z - min.z; dz *= dz;
        }
        else
        {
            if(p.z > max.z)
            {
                dz = p.z - max.z; dz *= dz;
            }
            else // between minz and maxz
            {
                dz = 0.0f;
            }
        }
    }
    else
    {
        if(p.x > max.x)
        {
            dx = p.x - max.x; dx *= dx;
            if(p.z < min.z)
            {
                dz = p.z - min.z; dz *= dz;
            }
            else
            {
                if(p.z > max.z)
                {
                    dz = p.z - max.z; dz *= dz;
                }
                else // between minz and maxz
                {
                    dz = 0.0f;
                }
            }
        }
        else // inside
        {
            dx = 0.0f;
            if(p.z < min.z)
            {
                dz = p.z - min.z; dz *= dz;
            }
            else
            {
                if(p.z > max.z)
                {
                    dz = p.z - max.z; dz *= dz;
                }
                else // completely inside
                {
                    dz = 0.0f;
                }
            }

        }
    }

    near = dx + dz;
    return near;
    // return sqrt(near);
}

float BoundRect::farthest(vpPoint p) const
{
    float dminx, dmaxx, dminz, dmaxz, far;

    far = std::numeric_limits<float>::max();

    dminx = min.x - p.x; dminx *= dminx;
    dmaxx = max.x - p.x; dmaxx *= dmaxx;
    dminz = min.z - p.z; dminz *= dminz;
    dmaxz = max.z - p.z; dmaxz *= dmaxz;

    if(dminx > dmaxx)
        far = dminx;
    else
        far = dmaxx;
    if(dminz > dmaxz)
        far += dminz;
    else
        far += dmaxz;

    return far;
    // return sqrt(far);
}

/*
void BoundRect::test()
{
    min.x = 0.0f; min.z = 0.0f;
    max.x = 1.0f; max.z = 1.0f;

    cerr << "Testing BoundRect" << endl;
    cerr << "near (-0.5, 0.0) = " << nearest(vpPoint(-0.5, 1.0, 0.0)) << endl; // expect 0.5
    cerr << "far (-0.5, 0.0) = " << farthest(vpPoint(-0.5, 1.0, 0.0)) << endl; // expect 1.80278

    cerr << "near (0.2, 1.5) = " << nearest(vpPoint(0.2, -1.0, 1.5)) << endl; // expect 0.5
    cerr << "far (0.2, 1.5) = " << farthest(vpPoint(0.2, -1.0, 1.5)) << endl; // expect 1.7

    cerr << "near (1.5, 1.5) = " << nearest(vpPoint(1.5, 0.0, 1.5)) << endl; // expect 0.707107
    cerr << "far (1.5, 1.5) = " << farthest(vpPoint(1.5, 0.0, 1.5)) << endl; // expect 2.12132

    cerr << "near (0.5, 0.5) = " << nearest(vpPoint(0.5, 1.0, 0.5)) << endl; // expect 0.0
    cerr << "far (0.5, 0.5) = " << farthest(vpPoint(0.5, 1.0, 0.5)) << endl; // expect 0.0
}
*/

void rayPointDist(vpPoint start, Vector dirn, vpPoint query, float &tval, float &dist)
{
    float den;
    vpPoint closest;
    Vector closevec;

    den = dirn.sqrdlength();
    if(den == 0.0f) // not a valid line segmen
        dist = -1.0f;
    else
    {
        // get parameter value of closest poin
        tval = dirn.i * (query.x - start.x) + dirn.j * (query.y - start.y) + dirn.k * (query.z - start.z);
        tval /= den;

        // find closest point on line
        closevec = dirn;
        closevec.mult(tval);
        closevec.pntplusvec(start, &closest);
        closevec.diff(query, closest);
        dist = closevec.length();
    }
}

bool lineCrossing(vpPoint * e1, vpPoint * e2)
{
    float a1, b1, c1, a2, b2, c2;

    // form implicit equations for the two lines
    a1 = e1[0].y - e1[1].y; b1 = e1[1].x - e1[0].x; c1 = -1.0f * (a1*e1[0].x + b1*e1[0].y);
    a2 = e2[0].y - e2[1].y; b2 = e2[1].x - e2[0].x; c2 = -1.0f * (a2*e2[0].x + b2*e2[0].y);

    // now test crossing by difference of signs method
    if((a1*e2[0].x+b1*e2[0].y+c1)*(a1*e2[1].x+b1*e2[1].y+c1) < 0.0f)  // on opposite sides of e1
        if((a2*e1[0].x+b2*e1[0].y+c2)*(a2*e1[1].x+b2*e1[1].y+c2) < 0.0f) // on opposite sides of e2
            return true;
    return false;
}

void clamp(float & t)
{
    if(t > 1.0f+minuszero)
        t = 1.0f+minuszero;
    if(t < 0.0f)
        t = 0.0f;
}
