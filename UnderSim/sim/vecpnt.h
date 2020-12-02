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


// file: vecpnt.h
// author: James Gain
// project: Interactive Sculpting (1997+)
// notes: Basic vector and point arithmetic library. Inlined for efficiency.
// changes: included helpful geometry routines (2006)
#ifndef _INC_VECPNT
#define _INC_VECPNT

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <math.h>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#include <limits.h>

#define pluszero 0.000001f
#define minuszero -0.000001f
#define PI 3.14159265
#define PI2 6.2831853
#define Eta 0.000001f
#define Negeta - 0.000001f

inline float sign(float n)
{
    if(n >= 0.0f)
        return 1.0f;
    else
        return -1.0f;
}

class vpPoint2D
{
public:

    float x, y;

    inline vpPoint2D(){ x = 0.0f; y = 0.0f; }
    inline vpPoint2D(float a, float b){ x = a; y = b; }
};

class vpPoint
{
private:
    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & x;
        ar & y;
        ar & z;
    }

public:

    float x, y, z;

    /* Point: default constructor */
    inline vpPoint()
    { x = 0.0;
      y = 0.0;
      z = 0.0;
    }

    /* Point: constructor assigns (a,b,c) to (x,y,z) fields of point */
    inline vpPoint(float a, float b, float c)
    { x = a;
      y = b;
      z = c;
    }

    /* operator=: copy assignment */
    inline vpPoint & operator =(const vpPoint & from)
    {
      x = from.x;
      y = from.y;
      z = from.z;
      return *this;
    }

    /* dist: determine the distance between two points */
    inline double dist(vpPoint p)
    { float dx, dy, dz;

      dx = (p.x - x) * (p.x - x);
      dy = (p.y - y) * (p.y - y);
      dz = (p.z - z) * (p.z - z);
      return sqrt(dx + dy + dz);
    }

    inline void affinecombine(float c1, vpPoint& p1, float c2, vpPoint& p2)
    { x = c1 * p1.x + c2 * p2.x;
      y = c1 * p1.y + c2 * p2.y;
      z = c1 * p1.z + c2 * p2.z;
    }

    inline void raffinecombine(float c1, vpPoint * p1, float c2, vpPoint * p2)
    { x = c1 * p1->x + c2 * p2->x;
      y = c1 * p1->y + c2 * p2->y;
      z = c1 * p1->z + c2 * p2->z;
    }

    /* ==: test the equality of two points within a given tolerance */
    inline bool operator == (vpPoint p)
    { float dx, dy, dz;
      dx = x - p.x;
      dy = y - p.y;
      dz = z - p.z;
      return ((dx < pluszero) && (dx > minuszero) && (dy < pluszero) &&
              (dy > minuszero) && (dz < pluszero) && (dz > minuszero));
    }

    inline bool equal(vpPoint p, float tol)
    {
        double dx, dy, dz;

        dx = fabs(p.x - x);
        dy = fabs(p.y - y);
        dz = fabs(p.z - z);

        return ((dx < tol) && (dy < tol) && (dz < tol));
    }
};

class Vector
{
private:

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & i;
        ar & j;
        ar & k;
    }

public:

    float i, j, k;

    /* Vector: default constructor */
    inline Vector()
    { i = 0.0;
      j = 0.0;
      k = 0.0;
    }

    Vector& operator=(const Vector &v)
    {
        i = v.i;
        j = v.j;
        k = v.k;

        return *this;
    }


    /* Vector: constructor assigns (a,b,c) to (i,j,k) fields of vector */
    inline Vector(float a, float b, float c)
    { i = a;
      j = b;
      k = c;
    }

    /* angle: find the angle (in radians) between two vectors */
    inline float angle(Vector b)
    { Vector a;

      a = (* this);
      a.normalize();
      b.normalize();
      return (float) acos(a.dot(b));
    }

    /* length: return the euclidian length of a vector */
    inline float length()
    { float len;

      len = i * i + j * j + k * k;
      len = sqrt(len);
      return len;
    }

    inline float sqrdlength()
    { float len;

      len = i * i;
      len += j * j;
      len +=  k * k;
      return len;
    }

    /* normalize: transform vector to unit length */
    inline void normalize()
    {
        float len;
        len = (float) sqrt(i*i + j*j + k*k);
        if(len > 0.0f)
            len = 1.0f / len;
        i *= len;
        j *= len;
        k *= len;
    }

    inline float lnormalize()
    {
        float len;
        len = (float) sqrt(i*i + j*j + k*k);
        if(len > 0.0f)
            len = 1.0f / len;
        i *= len;
        j *= len;
        k *= len;
        return len;
    }

    /* vec_mult: scale the vector by a scalar factor c */
    inline void mult (float c)
    { i = i * c;
      j = j * c;
      k = k * c;
    }

    /* mult: component-wise multiplication with another vector v */
    inline void mult (Vector &v)
    {
        i *= v.i;
        j *= v.j;
        k *= v.k;
    }

    /* div: the vector result of dividing vector a by vector b */
    inline void div(Vector& a, Vector& b)
    { i = a.i / b.i;
      j = a.j / b.j;
      k = a.k / b.k;
    }

    /* pntconvert: convert a point p into a vector */
    inline void pntconvert(vpPoint& p)
    { i = p.x;
      j = p.y;
      k = p.z;
    }

    /*
    inline Vector& operator=(Vector& v)
    {
       i = v.i;
       j = v.j;
       k = v.k;
       return *this;
    }
    */

    /* ==: return true if two vectors are identical within a tolerance */
    inline int operator ==(Vector& v)
    { double di, dj, dk;

      di = i - v.i;
      dj = j - v.j;
      dk = k - v.k;

      return ((di < pluszero) && (di > minuszero) && (dj < pluszero) && (dj > minuszero)
               && (dk < pluszero) && (dk > minuszero));
    }

    /* diff: difference of points: create a vector from point p to point q */
    inline void diff(vpPoint p, vpPoint q)
    { i = (q.x - p.x);
      j = (q.y - p.y);
      k = (q.z - p.z);
    }


    /* add: component-wise addition of a vector v to the current vector */
    inline void add(Vector& v)
    { i += v.i;
      j += v.j;
      k += v.k;
    }

    /* add: component-wise subtraction of a vector v fromthe current vector */
    inline void sub(Vector& v)
    { i -= v.i;
      j -= v.j;
      k -= v.k;
    }

    /* cross: generate the cross product of two vectors */
    inline void cross(Vector& x, Vector& y)
    { i = (x.j * y.k) - (x.k * y.j);
      j = (x.k * y.i) - (x.i * y.k);
      k = (x.i * y.j) - (x.j * y.i);
    }

    /* dot product: generate the dot product of two vectors */
    inline float dot(Vector& v)
    { return ((i * v.i) + (j * v.j) + (k * v.k));
    }

    /* pntplusvec: find the point at the head of a vector which
                    is placed with its tail at p */
    inline void pntplusvec(vpPoint& p, vpPoint * r)
    { r->x = p.x + i;
      r->y = p.y + j;
      r->z = p.z + k;
    }

    /* affinecombine: create an affine combination of two vectors v1 and v2, weighted by c1 and c2 */
    inline void affinecombine(float c1, Vector &v1, float c2, Vector &v2)
    { i = c1 * v1.i + c2 * v2.i;
      j = c1 * v1.j + c2 * v2.j;
      k = c1 * v1.k + c2 * v2.k;
    }

    /* interp: interpolate between vectors <s> and <e> using linear parameter <t> */
    inline void interp(Vector s, Vector e, float t)
    {
        i = (1.0f - t)*s.i + t*e.i;
        j = (1.0f - t)*s.j + t*e.j;
        k = (1.0f - t)*s.k + t*e.k;
    }

    /* rotate: turn the vector in the x-y plane by an angle <a> in radians */
    inline void rotate(float a)
    {
        float ni, nj, ca, sa;

        ca = cos(a); sa = sin(a);
        ni = i * ca - j * sa;
        nj = j * ca + i * sa;
        i = ni; j = nj;
    }

    inline void rotateInXZ(float a)
    {
        float ni, nk, ca, sa;

        ca = cos(a); sa = sin(a);
        ni = i * ca - k * sa;
        nk = k * ca + i * sa;
        i = ni; k = nk;
    }
};

class Plane
{ public:
    float d;    // plane offse
    Vector n;   // plane normal

    Plane(){ d = 0.0f; n = Vector(0.0f, 0.0f, 0.0f);}

    // formPlane: create a plane passing through <pnt> with normal <norm>
    void formPlane(vpPoint pnt, Vector norm);

    // formPlane:   find the plane in which the triangle <tri> is embedded
    //              Return <true> if the triangle vertices are not co-linear, <false> otherwise
    bool formPlane(vpPoint * tri);

    // rayPlaneIntersect:   find the parametric intersection point <tval> of a ray represented
    //                      by a <start> position and vector <dirn> with a plane.
    //                      Return <false> if there is no intersection, <true> otherwise.
    bool rayPlaneIntersect(vpPoint start, Vector dirn, float & tval);

    // rayPlaneIntersect:   find the point of intersection <intersect> of a ray represented
    //                      by a <start> position and vector <dirn> with a plane.
    //                      Return <false> if there is no intersection, <true> otherwise.
    bool rayPlaneIntersect(vpPoint start, Vector dirn, vpPoint & intersect);

    // side:    determine on which side of the plane a point <pnt> lies. Return <true> if in the
    //          direction of the normal, <false> otherwise.
    bool side(vpPoint pnt);

    // dist: calculate and return the distance from the point to the plane
    float dist(vpPoint pnt);

    // height: project the point <pnt> vertically onto the plane, returning the y-value of the intercep
    float height(vpPoint pnt);

    // projectPnt: project <pnt> to the closest point on the plane as <proj>
    void projectPnt(vpPoint pnt, vpPoint * proj);

    // drawPlane:   render the plane. Assumes OpenGL viewing state is already se
    //              front of plane is rendered in blue and back in green.
    // void drawPlane();
};

// univariate 1-dimensional Bezier curve
class Bezier
{
private:

    float h[4];

public:

    Bezier(){ for(int i = 0; i < 4; i++) h[i] = 0.0f; }

    Bezier(float p1, float p2, float p3, float p4)
    {
        h[0] = p1; h[1] = p2; h[2] = p3; h[3] = p4;
    }

    // eval: evaluate the Bezier curve at parameter value <t> in [0,1]
    inline float eval(float t)
    {
        float s, s2, s3, t2, t3;

        s = 1.0f - t; s2 = s*s; s3 = s2*s;
        t2 = t*t; t3 = t*t2;

        return h[0]*s3 + 3.0f*h[1]*t*s2 + 3.0f*h[2]*t2*s + h[3]*t3;
    }
};

// optimization structure for repeated ray box tests
class Ray
{
public:

    vpPoint origin;
    Vector direction;
    Vector inv_direction;
    int sign[3];

    Ray()
    {
        origin = vpPoint(0.0f, 0.0f, 0.0f); direction = Vector(0.0f, 0.0f, 0.0f);
        inv_direction = Vector(0.0f, 0.0f, 0.0f); sign[0] = 0; sign[1] = 0; sign[2] = 0;
    }

    Ray(vpPoint &o, Vector &d)
    {
        origin = o;
        direction = d;
        inv_direction = Vector(1/d.i, 1/d.j, 1/d.k);
        sign[0] = (inv_direction.i < 0);
        sign[1] = (inv_direction.j < 0);
        sign[2] = (inv_direction.k < 0);
    }
};

class BoundRect
{
private:

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & min;
        ar & max;
    }

public:

    vpPoint min, max;

    BoundRect(){ reset(); }

    // reset: initialize bounding box to its default settting
    inline void reset()
    {
        float far = std::numeric_limits<float>::max();
        min = vpPoint(far, 0.0f, far);
        max = vpPoint(-1.0*far, 0.0f, -1.0f*far);
    }

    // includePnt: compare point against current bounding box and expand as required
    inline void includePnt(vpPoint pnt)
    {
        if(pnt.x < min.x)
            min.x = pnt.x;
        if(pnt.x > max.x)
            max.x = pnt.x;
        if(pnt.z < min.z)
            min.z = pnt.z;
        if(pnt.z > max.z)
            max.z = pnt.z;
    }

    // daiglen: return the length of the diagonal of the bounding box
    inline float diaglen()
    {
        Vector diag;

        diag.diff(min, max);
        return diag.length();
    }

    /// find the shortest distance to the bounding box
    float nearest(vpPoint p) const;

    /// find the distance to the farthest corner of the bounding box
    float farthest(vpPoint p) const;

    inline bool empty() const
    {
        return (min.x > max.x);
    }

    /// Enlarge the bounding box uniformly in all directions
    inline void expand(float extent)
    {
        max.x += extent; min.x -= extent;
        max.z += extent; min.z -= extent;
    }

    /// test method
    // void test();

    // rayBoxIntersect: find whether a ray <r> intersects the bounding box on the parametric
    //                  interval <t0> to <t1>.
    //                  Return <false> if there is no intersection, <true> otherwise.
    // bool rayBoxIntersect(Ray r, float t0, float t1);
};


////
//// USEFUL GEOMETRY ROUTINES
////

// linePointDist:   Find the shortest distance from a point <query> to a line segment, represented by an origin <start>
//                  and direction <dirn>. Return the shortest distance from the line segment to the point as <dist> and
//                  the parameter value on the line segment of this intersection as <tval>.
//                  This query assumes 2D computations (z coord dropped)
void rayPointDist(vpPoint start, Vector dirn, vpPoint query, float &tval, float &dist);

// linesDist:   Find the shortest distance between two line segments <q> and <t> represented by their endpoints.
//              Return the parameter value <qval> on the first segment of the closest approach and the distance as <mindist>.
//              Also return <maxdist>, the farthest distance between the two line segments.
void linesDist(vpPoint * q, vpPoint * t, float &qval, float &mindist, float &maxdist);

// lineCrossing: return <true> if two 2d lines e1[0]->e1[1] and e2[0]->e2[1] intersect, false otherwise.
bool lineCrossing(vpPoint * e1, vpPoint * e2);

// clamp: ensure that parameter <t> falls in [0,1]
void clamp(float & t);

////
//// USEFUL PORTABILITY ROUTINES
////

/*
// endianSwap: swap endian for short integer
inline unsigned short endianSwap(unsigned short x)
{
    unsigned short y;
    y = (x>>8) |
        (x<<8);
    return y;
}*/

// endianSwap: swap endian for integer
inline unsigned int endianSwapi(unsigned int x)
{
    unsigned int y;
    y = (x>>24) |
        ((x<<8) & 0x00FF0000) |
        ((x>>8) & 0x0000FF00) |
        (x<<24);
    return y;
}

// endianSwap: swap endian for floa
inline float endianSwapf( float f )
{
    union
    {
        float f;
        unsigned char b[4];
    } dat1, dat2;

    dat1.f = f;
    dat2.b[0] = dat1.b[3];
    dat2.b[1] = dat1.b[2];
    dat2.b[2] = dat1.b[1];
    dat2.b[3] = dat1.b[0];
    return dat2.f;
}

#endif
