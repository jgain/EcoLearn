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


/* file: stroke.cpp
   author: (c) James Gain, 2006
   project: ScapeSketch - sketch-based design of procedural landscapes
   notes: Forming 2d mouse input into strokes for sketch and gesture purposes
   changes:
*/

#include "stroke.h"
#include "shape.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <list>
#include <algorithm>


///
/// GENERAL PROJECTION ROUTINES
///

void planeProject(View * view, uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, Plane * projPlane)
{
    uts::vector<vpPoint> copy; // <from> and <to> may be the same so this requires some care
    float tval;
    Vector dirn;
    vpPoint p, cop;

    // project stroke onto plane
    cop = view->getCOP();
    for(int i = 0; i < (int) from->size(); i++)
    {
        dirn.diff(cop, (* from)[i]);
        if(projPlane->rayPlaneIntersect(cop, dirn, tval))
        {
            dirn.mult(tval);
            dirn.pntplusvec(cop, &p);
            copy.push_back(p);
        }
    }
    to->clear();
    (* to) = copy;
}

// assumes <from> and <to> strokes are not the same
void screenProject(View * view, uts::vector<vpPoint> * from, uts::vector<vpPoint> * to)
{
    vpPoint vcop, pcop, proj;
    Vector vdir, pdir;
    Plane vplane;

    to->clear();

    // set up projection plane
    vcop = view->getCOP();
    vdir = view->getDir();
    vdir.mult(0.2f); // not too close for precision reasons
    vdir.pntplusvec(vcop, &pcop);
    vdir.normalize();
    vplane.formPlane(pcop, vdir);

    // apply projection
    for(int i = 0; i < (int) from->size(); i++)
    {
        pdir.diff(vcop, (* from)[i]);
        if(vplane.rayPlaneIntersect(vcop, pdir, proj))
            to->push_back(proj);
    }
}

void drapeProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, Terrain * ter)
{
    int i;
    uts::vector<vpPoint> copy;
    vpPoint dpnt;

    // drape onto the landscape
    for(i = 0; i < (int) from->size(); i++)
    {
        ter->drapePnt((* from)[i], dpnt);
        copy.push_back(dpnt);           // add draped point to list
    }
    to->clear();
    (* to) = copy;
    copy.clear();
}

void dropProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to)
{
    int i;
    vpPoint dpnt;
    uts::vector<vpPoint> copy;

    // zero y-coordinate
    for(i = 0; i < (int) from->size(); i++)
    {
        dpnt = (* from)[i];
        dpnt.y = 0.0f;
        copy.push_back(dpnt);           // add dropped point to list
    }
    to->clear();
    (* to) = copy;
    copy.clear();
}

bool terrainProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, View * view, Terrain * ter)
{
    bool onterrain = false;

    uts::vector<vpPoint> copy; // <from> and <to> may be the same so this requires some care
    Vector dirn;
    vpPoint p, cop;

    // project stroke onto plane
    cop = view->getCOP();
    // cerr << "cop = " << cop.x << ", " << cop.y << ", " << cop.z << endl;

    for(int i = 0; i < (int) from->size(); i++)
    {
        dirn.diff(cop, (* from)[i]);
        if(ter->rayIntersect(cop, dirn, p))
        {
            // cerr << "from " << (* from)[i].x << ", " << (* from)[i].y << ", " << (* from)[i].z;
            // cerr << " to " << p.x << ", " << p.y << ", " << p.z << endl;
            copy.push_back(p);
            onterrain = true;
        }
    }
    to->clear();
    (* to) = copy;
    copy.clear();
    return onterrain;
}

bool terrainProject(vpPoint &fromPnt, vpPoint &toPnt, View * view, Terrain * ter)
{
    Vector dirn;
    vpPoint p, cop;

    // project stroke onto plane
    cop = view->getCOP();
    dirn.diff(cop, fromPnt);
    if(ter->rayIntersect(cop, dirn, p))
    {
        toPnt = p;
        return true;
    }
    else
    {
        return false;
    }

}

bool testFragmentAttach(uts::vector<vpPoint> * from, uts::vector<vpPoint> * into, float diftol, bool bothends, int &inds, int &inde, float &closeness)
{
    Vector sep;
    vpPoint s, e;
    float currmin, vals, vale, mins = 1000000.0f, mine = 1000000.0f;
    bool isattached, ats, ate;
    int i, rangetol;

    // find closest point on <to> stroke to start and end of <from> stroke
    s = from->front(); e = from->back();
    for(i = 0; i < (int) into->size(); i++)
    {
        sep.diff((* into)[i], s);
        currmin = sep.sqrdlength();
        if(currmin < mins)
        {
            mins = currmin;
            inds = i;
        }
        sep.diff((* into)[i], e);
        currmin = sep.sqrdlength();
        if(currmin < mine)
        {
            mine = currmin;
            inde = i;
        }
    }

    // check if these closest points are within tolerance
    vals = sqrt(mins); vale = sqrt(mine);
    ats = (vals < diftol);
    ate = (vale < diftol);

    if(!ats)
        inds = -1;
    if(!ate)
        inde = -1;


    rangetol = (int) (0.01f * (float) into->size()) + 10;

    /*
    cerr << "into size = " << into->size()-1 << endl;
    cerr << "range tol = " << rangetol << endl;
    cerr << "ats = " << ats << ", vals = " << vals << ", inds = " << inds << endl;
    cerr << "ate = " << ate << ", vale = " << vale << ", inde = " << inde << endl;
     */
    if(bothends)
    {
        isattached = ats && ate;
    }
    else
    {
        isattached = ats && ate;
        if(!isattached) // also allow endpoint to endpoint
        {
                if(ats)
                {
                    isattached = (inds < rangetol) || (inds > (int) into->size()-rangetol);
                    if(inds < rangetol)
                        inds = 0;
                    if(inds > (int) into->size()-rangetol)
                        inds = (int) into->size()-1;
                }
                if(ate)
                {
                    isattached = (inde < rangetol) || (inde > (int) into->size()-rangetol);
                    if(inde < rangetol)
                        inde = 0;
                    if(inde > (int) into->size()-rangetol)
                        inde = (int) into->size()-1;
                }
        }
    }
    closeness = vals + vale;
    return isattached;
}

bool inFill(uts::vector<vpPoint> * from, uts::vector<vpPoint> * into, float diftol, bool bothends, bool closed)
{
    uts::vector<vpPoint> copy;
    Vector vfrom, vinto;
    vpPoint s, e;
    float close;
    bool ats, ate, isforward, overseam = false;
    int i, inds, inde, rubstart, rubend;

    if(testFragmentAttach(from, into, diftol, bothends, inds, inde, close))
    {
        ats = inds != -1; ate = inde != -1;
        vinto.diff(into->front(), into->back());
        vinto.normalize();

        if(ats && ate) // replace section of curve
        {
            isforward = (inds < inde);  // determine relative direction of <from> and <into> strokes
            if(isforward)
            {
                rubstart = inds;
                rubend = inde;
            }
            else
            {
                rubstart = inde;
                rubend = inds;
            }
            if(closed) // specific to closed curves
                overseam = ((float) (rubend - rubstart) / (float) into->size() > 0.5f); // always overwrite the smallest section of curve if it is a loop
        }
        else // join end to end
        {
            /*
            // not at all robust
            // determine relative direction of <from> and <into> using vectors from their start to end
            vfrom.diff(from->front(), from->back());
            vfrom.normalize();
            isforward = (vinto.dot(vfrom) >= 0.0f);
             */
            if(ats)
            {
                isforward = (inds != 0);
                if(isforward) // two strokes are aligned so discard from inds on
                {
                    rubstart = inds;
                    rubend = (int) into->size();
                }
                else // two strokes are misaligned so discard upto inds
                {
                    rubstart = -1;
                    rubend = inds;
                }
            }
            else
            {
                isforward = (inde == 0);
                if(isforward) // two strokes are aligned so discard upto inde
                {
                    rubstart = -1;
                    rubend = inde;
                }
                else // two strokes are misaligned so discard from inde on
                {
                    rubstart = inde;
                    rubend = (int) into->size();
                }
            }
        }

        // interleave old curve and new curve sections into copy vector
        if(overseam) // shift the seam and chop off the start and end. Can only be <true> if curve is closed.
        {
            for(i = rubstart; i <= rubend; i++)
                copy.push_back((* into)[i]);

            if(!isforward)
            {
                for(i = 0; i < (int) from->size(); i++)
                    copy.push_back((* from)[i]);
            }
            else
            {
                for(i = (int) from->size() - 1; i >= 0; i--)
                    copy.push_back((* from)[i]);
            }
        }
        else // insert into an existing section of the curve
        {
            for(i = 0; i <= rubstart; i++)
                copy.push_back((* into)[i]);

            if(isforward)
            {
                for(i = 0; i < (int) from->size(); i++)
                    copy.push_back((* from)[i]);
            }
            else
            {
                for(i = (int) from->size() - 1; i >= 0; i--)
                    copy.push_back((* from)[i]);
            }

            for(i = rubend; i < (int) into->size(); i++)
                copy.push_back((* into)[i]);
        }

        into->clear();
        (* into) = copy;
        return true;
    }
    else
        return false;
}

bool locateIntersect(uts::vector<vpPoint> * strk, int &begin, int &end)
{
    int i, j;
    vpPoint e1[2], e2[2];

    for(i = 0; i < (int) strk->size() - 2; i++)
    {
        e1[0] = (* strk)[i]; e1[1] = (* strk)[i+1];
        for(j = i+2; j < (int) strk->size(); j++)
        {
            e2[0] = (* strk)[j];
            if(j == (int) strk->size()-1) // wrap around
                e2[1] = (* strk)[0];
            else
                e2[1] = (* strk)[j+1];
            if(lineCrossing(e1, e2))
            {
                begin = i; end = j;
                return true;
            }
        }
    }
    return false;
}

void excise(uts::vector<vpPoint> * strk, int begin, int end)
{
    uts::vector<vpPoint>::iterator siter, eiter;

    siter = strk->begin(); eiter = strk->begin();
    siter += begin; eiter += (end+1);
    strk->erase(siter, eiter);
    // printf("excised from %d to %d, size before = %d, size after = %d", begin, end, bsize, esize);
}


///
/// ValueCurve
///


float ValueCurve::posOnSegment(int seg, float t) const
{
    float sqt, h0, h1, h2, h3, v;
    float p0, p1, m0, m1;

    if(t < 0.0f-pluszero || t > 1.0f+pluszero || seg < 0 || seg > (int) splinerep.size() - 2)
    {
        // cerr << "Error ValueCurve::posOnSegment: out of bounds. t = " << t << " seg = " << seg << " out of " << (int) splinerep.size()-2 << endl;
        return 0.0f;
    }
    else
    {
        sqt = (1.0f - t) * (1.0f - t);

        // basis functions
        h0 = (1.0f + 2.0f * t) * sqt; // (1+2t)(1-t)^2
        h1 = t * sqt; // t (1-t)^2
        h2 = t * t * (3.0f - 2.0f * t); // t^2 (3-2t)
        h3 = t * t * (t - 1.0f); // t^2 (t-1)

        // control points and tangents
        p0 = splinerep[seg];
        m0 = splinetan[seg];
        p1 = splinerep[seg+1];
        m1 = splinetan[seg+1];

        // calculate point on curve
        v = h0 * p0 + h1 * m0 + h2 * p1 + h3 * m1;
        return v;
    }
}

void ValueCurve::create(uts::vector<float> * tvals, uts::vector<float> * vvals)
{
    sampling = 30; // number of samples between locators

    vals = (* vvals);
    params = (* tvals);
    deriveCurve();
}

void ValueCurve::flatCaps()
{
    if((int) splinetan.size() >= 2)
    {
        splinetan[0] = 0.0f;
        splinetan[(int) splinetan.size() - 1] = 0.0f;
    }
}

float ValueCurve::getPoint(float t) const
{
    int seg;
    float loct;
    bool found = false;

    // determine segment, using a search initially
    seg = 0;
    while(!found && seg < (int) params.size()-1)
    {
        seg++;
        if(params[seg]+pluszero >= t)
        {
            found = true; seg--;
        }
    }
    if(!found) // end of curve
    {
        loct = 1.0f;
        seg = (int) params.size()-2;
    }
    else
    {
        loct = (t - params[seg]) / (params[seg+1] - params[seg]);
        clamp(loct);
    }
    return posOnSegment(seg, loct);
}


void ValueCurve::deriveTangents()
{
    float tan;

    splinetan.clear();
    for(int i = 0; i < (int) splinerep.size(); i++)
    {
        // tan = (cp[i+1] - cp[i]) / 2 + (cp[i] - cp[i-1]) / 2
        // The curve is extended as piecewise constant past the ends
        int a = std::max(i-1, 0);
        int b = std::min(i+1, (int) splinerep.size()-1);
        tan = 0.5f * (splinerep[b] - splinerep[a]);
        splinetan.push_back(tan);
    }
}


void ValueCurve::deriveVerts(int tnum)
{
    float tsep = 1.0f / (float) tnum;
    float t, v;
    BoundRect lbr, hbr;

    vals.clear();
    deriveTangents();
    for(int i = 0; i < (int) splinerep.size()-1; i++)
    {
        // baset = params[i]; delt = params[i+1] - params[i];
        // create points within the current segment
        for(t = 0.0f; t < 1.0f-pluszero; t+= tsep)
        {
            v = posOnSegment(i, t);
            vals.push_back(v);
        }
    }

    // add last point
    v = posOnSegment((int) splinerep.size()-2, 1.0f);
    vals.push_back(v);
}

void ValueCurve::deriveCurve()
{
    Vector sepvec;
    vpPoint p;

    if((int) vals.size() > 0)
        splinerep = vals;

    // now recreate the vertices according to the new control points
    deriveTangents();
    deriveVerts(sampling);
}

///
/// BrushCurve
///


void BrushCurve::create(vpPoint start, View * view, Terrain * ter)
{
    vpPoint pnt;

    sampling = ter->samplingDist();
    if(terrainProject(start, pnt, view, ter))
    {
        pnt.y = 0.0f;
        vertsrep.push_back(pnt);
        updateIndex = 0;
        update.includePnt(pnt);
        enclose.includePnt(pnt);
        created = true;
    }
}

bool BrushCurve::isCreated() const
{
    return created;
}

void BrushCurve::addPoint(View * view, Terrain * ter, vpPoint pnt)
{
    int j, nums;
    vpPoint start, end, pos;
    float t, len, delt;
    Vector del;
    uts::vector<vpPoint> newverts;

    if(terrainProject(pnt, end, view, ter))
    {
        if(!created)
        {
            create(pnt, view, ter);
        }
        else
        {
            end.y = 0.0f;
            start = vertsrep[(int) vertsrep.size()-1];

            // update bounding boxes
            update.reset();
            enclose.includePnt(end);
            update.includePnt(start);
            update.includePnt(end);

            // linearly subsample each segment
            del.diff(start, end);
            len = del.length();
            updateIndex = (int) vertsrep.size();

            if(len > sampling) // longer than expected interval between points
            {
                // number of subsamples
                nums = (int) ceil(len / sampling);
                delt = 1.0f / (float) nums;
                for(j = 1; j < nums; j++) // start point already part of curve
                {
                    t = j * delt;
                    pos.affinecombine(1.0f-t, start, t, end);
                    vertsrep.push_back(pos);
                }
            }
            else
            {
                vertsrep.push_back(end);
            }
        }
    }
}

Region BrushCurve::getBound(BoundRect &bnd, Terrain * ter, float radius)
{
    int dx, dy;
    Region reg;

    bnd.expand(radius);
    ter->getGridDim(dx, dy);

    // convert to terrain coordinates
    reg.x0 = (int) ter->toGrid(bnd.min.x);
    if(reg.x0 < 0) reg.x0 = 0;
    reg.y0 = (int) ter->toGrid(bnd.min.z);
    if(reg.y0 < 0) reg.y0 = 0;
    reg.x1 = (int) ter->toGrid(bnd.max.x);
    if(reg.x1 > dx) reg.x1 = dx;
    reg.y1 = (int) ter->toGrid(bnd.max.z);
    if(reg.y1 > dy) reg.y1 = dy;
    return reg;
}

Region BrushCurve::encloseBound(Terrain * ter, float radius)
{
    BoundRect bnd;

    bnd = enclose;
    return getBound(bnd, ter, radius);
}

Region BrushCurve::updateBound(Terrain * ter, float radius)
{
    BoundRect bnd;

    bnd = update;
    return getBound(bnd, ter, radius);
}

///
/// Curve3D
///


bool Curve3D::posOnSegment(int seg, float t, vpPoint & p) const
{
    float sqt, h0, h1, h2, h3;
    vpPoint p0, p1;
    Vector m0, m1;

    if(t < 0.0f-pluszero || t > 1.0f+pluszero || seg < 0 || seg > (int) splinerep.size() - 2)
    {
        cerr << "posOnSegment error: out of bounds. t = " << t << " seg = " << seg << endl;
        return false;
    }
    else
    {
        sqt = (1.0f - t) * (1.0f - t);

        // basis functions
        h0 = (1.0f + 2.0f * t) * sqt; // (1+2t)(1-t)^2
        h1 = t * sqt; // t (1-t)^2
        h2 = t * t * (3.0f - 2.0f * t); // t^2 (3-2t)
        h3 = t * t * (t - 1.0f); // t^2 (t-1)

        // control points and tangents
        p0 = splinerep[seg];
        m0 = splinetan[seg];
        p1 = splinerep[seg+1];
        m1 = splinetan[seg+1];

        // calculate point on curve
        p.x = h0 * p0.x + h1 * m0.i + h2 * p1.x + h3 * m1.i;
        p.y = h0 * p0.y + h1 * m0.j + h2 * p1.y + h3 * m1.j;
        p.z = h0 * p0.z + h1 * m0.k + h2 * p1.z + h3 * m1.k;
        return true;
    }
}

void Curve3D::subsample(uts::vector<vpPoint> * strk)
{
    uts::vector<vpPoint> tmp;
    Vector del;
    vpPoint prev, curr, pos;
    float len, t, delt;
    int i, j, nums;

    for(i = 0; i < (int) strk->size()-1; i++)
    {
        prev = (* strk)[i]; curr = (* strk)[i+1]; prev.y = 0.0f; curr.y = 0.0f;
        del.diff(prev, curr);

        len = del.length();
        if(len > 0.1f * sep) // longer than expected interval between points
        {
            // number of subsamples
            nums = (int) ceil(len / (0.2f * sep));
            delt = 1.0f / nums;
            for(j = 0; j < nums; j++)
            {
                t = j * delt;
                pos.affinecombine(1.0f-t, (* strk)[i], t, (* strk)[i+1]);
                tmp.push_back(pos);
            }
        }
        else
        {
            tmp.push_back((* strk)[i]);
        }
    }
    tmp.push_back((* strk)[(int) strk->size()-1]);

    strk->clear();
    (* strk) = tmp;
}

bool Curve3D::create(uts::vector<vpPoint> * strk, View * view, Terrain * ter)
{
    float tx, ty;

    sep = ter->smoothingDist(); // separation between control point

    ter->getTerrainDim(tx, ty);
    farbound = (tx*tx+ty*ty)+100.0f;

    sampling = ter->samplingDist();
    highstep = 6;
    leafstep = 25;

    created = terrainProject(strk, &vertsrep, view, ter);

    // for testing
    /*
    vertsrep.clear();
    vertsrep.push_back(vpPoint(-0.3, 0.0, -0.3));
    vertsrep.push_back(vpPoint(0.0, 0.0, 0.0));
    vertsrep.push_back(vpPoint(0.3, 0.0, 0.3));
    drapeProject(&vertsrep, &vertsrep, ter); */

    if((int) vertsrep.size() > 1)
    {
        deriveCurve(ter, true);
        // cerr << "number curve samples = " << (int) vertsrep.size() << endl;
        // cerr << "sep = " << sep << ", sampling = " << sampling << endl;
        drapeProject(&vertsrep, &vertsrep, ter); // project back onto landscape
    }
    return created;
}

bool Curve3D::nonProjCreate(uts::vector<vpPoint> * strk, Terrain * ter)
{
    float tx, ty, errbnd;
    int i;

    sep = 5.0f * ter->smoothingDist(); // separation between control point

    ter->getTerrainDim(tx, ty);
    farbound = (tx*tx+ty*ty)+100.0f;

    sampling = 10.0f * ter->samplingDist();
    errbnd = ter->samplingDist();
    highstep = 6;
    leafstep = 25;

    //cerr << "stroke size = " << (int) strk->size() << endl;
    vertsrep = (* strk); // copy stroke directly to vertices without requiring projection
    //cerr << "vertsrep size = " << (int) vertsrep.size() << endl;
    if((int) vertsrep.size() > 1)
    {
        deriveCurve(ter, false);
        //drapeProject(&vertsrep, &vertsrep, ter); // project back onto landscape
        // for(int i = 0; i < (int) vertsrep.size(); i++)
        //     vertsrep[i].y = 0.0f;

        // clamp to terrain bounds, which may be exceeded due to smoothing
        for(i = 0; i < (int) vertsrep.size(); i++)
        {
            if(vertsrep[i].x < errbnd) vertsrep[i].x = errbnd;
            if(vertsrep[i].z < errbnd) vertsrep[i].z = errbnd;
            if(vertsrep[i].x > tx - errbnd) vertsrep[i].x = tx - errbnd;
            if(vertsrep[i].z > ty - errbnd) vertsrep[i].z = ty - errbnd;
        }
        created = true;
    }
    else
    {
        created = false;
    }
    return created;
}

void Curve3D::recreate(Terrain * ter)
{
    if((int) vertsrep.size() > 1)
    {
        deriveCurve(ter, true);
        drapeProject(&vertsrep, &vertsrep, ter); // project back onto landscape
    }
}

bool Curve3D::isCreated() const
{
    return created;
}


bool Curve3D::mergeStroke(uts::vector<vpPoint> * strk, uts::vector<vpPoint> * prj, View * view, Terrain * ter, bool &merge, float tol)
{
    uts::vector<vpPoint> frag, tmp;
    bool pass;

    // project fragment onto landscape
    merge = false;
    pass = terrainProject(strk, &frag, view, ter);
    if(pass)
    {
        subsample(&frag); // introduce more vertices as needed by linear interpolation

        if(inFill(&frag, &vertsrep, tol, false, false)) // merge fragment with existing curve
        {
            // to do - test and fix foldover and out of bound errors
            deriveCurve(ter, true);
            drapeProject(&vertsrep, &vertsrep, ter); // project back onto landscape
            (* prj) = vertsrep;
            merge = true;
        }
    }
    return pass;
}


void Curve3D::genGL(View * view, Shape * shape, float radius)
{
    shape->genCylinderCurve(vertsrep, radius, 10.0f * sampling, 10);
}


void Curve3D::adjustHeights(Terrain * ter, ValueCurve hcurve)
{
    drapeProject(&vertsrep, &vertsrep, ter);
}


vpPoint Curve3D::getPoint(float t) const
{
    int seg;
    float segt, loct;
    vpPoint pnt;

    if((int) vertsrep.size() > 1)
    {
        // determine segment, using even subdivision of t across all segments
        segt = t; clamp(segt);
        segt *= (float) ((int) (vertsrep.size()-1));
        seg = (int) floor(segt);
        loct = segt - (float) seg; // local param within segment

        // linearly interpolate vertices
        pnt.x = (1.0f - loct) * vertsrep[seg].x + loct * vertsrep[seg+1].x;
        pnt.y = (1.0f - loct) * vertsrep[seg].y + loct * vertsrep[seg+1].y;
        pnt.z = (1.0f - loct) * vertsrep[seg].z + loct * vertsrep[seg+1].z;
    }
    else
    {
        if((int) vertsrep.size() == 1)
            pnt = vertsrep[0];
        else
            pnt = vpPoint(0.0f, 0.0f, 0.0f);
    }
    return pnt;
}

void Curve3D::getSeg(float t, vpPoint &s0, vpPoint &s1)
{
    int seg;

    if((int) vertsrep.size() > 1)
    {
        seg = getSegIdx(t);
        if(seg < (int) vertsrep.size() - 1)
        {
            s0 = vertsrep[seg]; s1 = vertsrep[seg+1];
        }
        else
        {
            s0 = vertsrep[seg]; s1 = vertsrep[seg];
        }
    }
}

int Curve3D::getSegIdx(float t)
{
    int seg = 0;
    float segt;

    if((int) vertsrep.size() > 1)
    {
        // determine segment, using even subdivision of t across all segments
        segt = t; clamp(segt);
        segt *= (float) (numPoints()-1);
        seg = (int) floor(segt);
    }

    return seg;
}

Vector Curve3D::getDirn(float t) const
{
    int seg, last;
    float segt;
    vpPoint p0, p1;
    Vector delv;

    // determine segment, using even subdivision of t across all segments
    if((int) vertsrep.size() > 1)
    {
        segt = t;
        segt *= (float) ((int) (vertsrep.size()-1));
        seg = (int) floor(segt);

        if(seg < (int) vertsrep.size()-1)
        {
            p0 = vertsrep[seg];
            p1 = vertsrep[seg+1];
        }
        else
        {
            last = (int) vertsrep.size()-1;
            p0 = vertsrep[last-1];
            p1 = vertsrep[last];
        }
        delv.diff(p0, p1);
        delv.normalize();
    }
    else
    {
        delv = Vector(0.0f, 0.0f, 0.0f);
    }
    return delv;
}


void Curve3D::closest(vpPoint p, float & t, float & dist, vpPoint &cpnt, Vector &cdirn)  const
{
    float cdist, delt, nearfar, nfc;
    vpPoint currpnt, npnt;
    Vector del;
    float dx, dz;
    int i, j, k, vmin, vmax, lmax, bestk = 0;

    dist = farbound; t = -1.0f;

    if((int) vertsrep.size() > 0)
    {
        delt = 1.0f / (float) ((int) vertsrep.size()-1);

        // find the box whose farthest point is closest
        nearfar = farbound;
        for(i = 0; i < (int) highsegboxes.size(); i++)
        {
            nfc = highsegboxes[i].farthest(p);
            if(nfc < nearfar)
                nearfar = nfc;
        }

        // only check boxes whose nearest point is closer than nearfar
        for(i = 0; i < (int) highsegboxes.size(); i++)
        {
            if(highsegboxes[i].nearest(p) <= nearfar)
            {

                // update nearfar against leaf boxes
                lmax = std::min(highstep*(i+1), (int) leafsegboxes.size());
                // lmax = highstep*(i+1);
                for(j = highstep * i; j < lmax; j++)
                {
                    nfc = leafsegboxes[j].farthest(p);
                    if(nfc < nearfar)
                        nearfar = nfc;
                }

                // now check contents of leaf boxes if within nearfar
                for(j = highstep * i; j < lmax; j++)
                {
                    if(leafsegboxes[j].nearest(p) <= nearfar)
                    {
                        // find range of vertices corresponding to this segment
                        vmin = j * leafstep;
                        vmax = std::min(vmin + leafstep, (int) vertsrep.size());

                        // find closest point on shadow in x-z plane
                        for(k = vmin; k < vmax; k++)
                        {
                            currpnt = vertsrep[k];
                            dx = p.x - currpnt.x;
                            dz = p.z - currpnt.z;
                            cdist = dx * dx + dz * dz;

                            if(cdist < dist)
                            {
                                dist = cdist;
                                bestk = k;
                            }
                        }
                    }
                }
            }
        }

    /*
        // slow alternative - useful for testing
        // find closest point on shadow in x-z plane
        for(j = 0; j < (int) vertsrep.size(); j++)
        {
            currpnt = vertsrep[j];
            dx = p.x - currpnt.x;
            dz = p.z - currpnt.z;
            cdist = dx * dx + dz * dz;

            if(cdist < dist)
            {
                dist = cdist;
                bestk = j;
            }
        }
    */
        t = delt * (float) bestk;
        cpnt = vertsrep[bestk];
        if(bestk < (int) vertsrep.size()-1)
        {
            npnt = vertsrep[bestk+1];
            cdirn.diff(cpnt, npnt);
        }
        else
        {
            npnt = vertsrep[bestk-2];
            cdirn.diff(npnt, cpnt);
        }
        cdirn.normalize();
        dist = sqrt(dist);
    }
}

void Curve3D::closestToRay(vpPoint cop, Vector dirn, float & t, float & dist)  const
{
    float cdist, delt, tval;
    vpPoint currpnt;
    int j, bestj = 0;

    dist = farbound; t = -1.0f;

    if((int) vertsrep.size() > 0)
    {
        delt = 1.0f / (float) ((int) vertsrep.size()-1);

        // slow approach - exhaustive testing
        for(j = 0; j < (int) vertsrep.size(); j++)
        {
            currpnt = vertsrep[j];
            rayPointDist(cop, dirn, currpnt, tval, cdist);

            if(cdist < dist)
            {
                dist = cdist;
                bestj = j;
            }
        }

        t = delt * (float) bestj;
    }
}


bool Curve3D::testIntersect(Curve3D * dstcurve, uts::vector<float> &srct, uts::vector<float> &dstt)
{
    int i, j, dis, die;
    vpPoint pnt, c[2], x[2];
    Vector dirn;
    float dist, st, dt, tol, delt;
    bool cross = false;

    tol = 2.0f * sampling;
    st = 0.0f; delt = 1.0f / (float) (numPoints()-1);
    for(i = 0; i < numPoints()-1; i++)
    {
        // test closeness of approach
        dstcurve->closest(vertsrep[i], dt, dist, pnt, dirn);
        if(dist < tol) // test segment crossing
        {
            // test in surrounding area
            getSeg(st, c[0], c[1]);
            c[0].y = c[0].z; c[1].y = c[1].z;

            dis = dstcurve->getSegIdx(dt-0.05f);
            die = dstcurve->getSegIdx(dt+0.05f);
            for(j = dis; j < die; j++)
            {
                x[0] = (* dstcurve->getVerts())[j];
                x[1] = (* dstcurve->getVerts())[j+1];
                x[0].y = x[0].z; x[1].y = x[1].z;
                if(lineCrossing(c,x)) // intersection detected
                {
                    srct.push_back(st);
                    dstt.push_back(dt);
                    cross = true;
                }
            }
        }
        st += delt;
    }
    return cross;
}


bool Curve3D::testSelfIntersect(uts::vector<float> &srct, uts::vector<float> &dstt)
{
    int i, j, dis, die;
    vpPoint c[2], x[2];
    Vector dirn;
    float sqdist, st, dt, tol, sqtol, delt;
    bool cross = false;
    
    tol = 2.0f * sampling; sqtol = tol*tol;
    st = 0.0f; delt = 1.0f / (float) (numPoints()-1);
    for(i = 0; i < numPoints()-1; i++)
    {
        for(j = i+10; j < numPoints()-1; j++)
        {
            dt = j*delt;
            dirn.diff(vertsrep[i], vertsrep[j]);
            sqdist = dirn.sqrdlength();
            
            if(sqdist < sqtol) // test segment crossing
            {
                // test in surrounding area
                getSeg(st, c[0], c[1]);
                c[0].y = c[0].z; c[1].y = c[1].z;
                
                dis = getSegIdx(dt-0.05f);
                die = getSegIdx(dt+0.05f);
                for(j = dis; j < die; j++)
                {
                    x[0] = vertsrep[j];
                    x[1] = vertsrep[j+1];
                    x[0].y = x[0].z; x[1].y = x[1].z;
                    if(lineCrossing(c,x)) // intersection detected
                    {
                        srct.push_back(st);
                        dstt.push_back(dt);
                        cross = true;
                    }
                }
            }
        }
        st += delt;
    }
    return cross;
}


bool Curve3D::closeApproach(Curve3D * dstcurve, uts::vector<float> &srct, uts::vector<float> &dstt, float tol)
{
    int i;
    vpPoint spnt, dpnt;
    Vector dirn;
    float et[2], st, dt, dist;
    bool close = false;

    et[0] = 0.0f; et[1] = 1.0f;

    // src endpoints
    for(i = 0; i < 2; i++)
    {
        spnt = getPoint(et[i]);
        dstcurve->closest(spnt, dt, dist, dpnt, dirn);
        if(dist < tol)
        {
            srct.push_back(et[i]); dstt.push_back(dt); close = true;
        }
    }

    // dst endpoints
    for(i = 0; i < 2; i++)
    {
        dpnt = dstcurve->getPoint(et[i]);
        closest(dpnt, st, dist, spnt, dirn);
        if(dist < tol)
        {
            srct.push_back(st); dstt.push_back(et[i]); close = true;
        }
    }
    return close;
}


void Curve3D::dragPin(float t, vpPoint tpnt, float trange)
{
    Vector del, trx;
    int i, vstart, vend, off;
    float s;
    //ValueCurve interpcurve;

    off = (int) (trange * (float) numPoints());
    del.diff(getPoint(t), tpnt); del.j = 0.0f; trx = del;

    // ramp up to full translate
    vend = getSegIdx(t);  vend = std::min(numPoints()-1, vend); vstart = std::max(0, vend - off);
    if(vstart == vend) // special case for first point in curve
    {
        trx = del;
        trx.pntplusvec(vertsrep[vstart], &vertsrep[vstart]);
    }
    else
    {
        for(i = vstart; i <= vend; i++)
        {
            s = (float) (i - vstart) / (float) (vend - vstart);
            trx = del; trx.mult(s);
            trx.pntplusvec(vertsrep[i], &vertsrep[i]);
        }
    }

    // ramp down from full translate
    vstart = std::min(numPoints()-1, vend+1); vend = std::min(numPoints()-1, vstart + off);
    for(i = vstart; i <= vend; i++)
    {
        s = 1.0f - (float) (i - vstart) / (float) (vend - vstart + 1);
        trx = del; trx.mult(s);
        trx.pntplusvec(vertsrep[i], &vertsrep[i]);
    }
}

float Curve3D::remap(float t, bool left) const
{
    int seg;
    float segt, loct, val;

    if((int) remapleft.size() > 0 && (int) remapright.size() > 0)
    {
        // determine segment, using even subdivision of t across all segments
        segt = t; clamp(segt);
        segt *= (float) ((int) (vertsrep.size()-1));
        seg = (int) floor(segt);
        loct = segt - (float) seg; // local param within segment

        // linearly interpolate remap elements
        if(left)
            val = (1.0f - loct) * remapleft[seg] + loct * remapleft[seg+1];
        else
            val = (1.0f - loct) * remapright[seg] + loct * remapright[seg+1];
    }
    else
    {
        cerr << "Error Curve3D::ramp: param remap not yet created" << endl;
        val = 0.0f;
    }
    return val;
}

void Curve3D::genParamRemap(ValueCurve * distleft, ValueCurve * distright)
{
    int i, k;
    float theta, kap, slen, slensq, nval, lerp, t, dleft, dright, delt, errt = 1.0f, etol = 0.01f;
    uts::vector<float> newmapleft, newmapright;
    Vector prevseg, nextseg, norm, orthog;

    remapleft.clear(); remapright.clear();

    // remap has initial arclength parametrization
    delt = 1.0f / (float) ((int) vertsrep.size()-1);
    for(i = 0; i < (int) vertsrep.size(); i++)
    {
        t = delt * (float) i;
        remapleft.push_back(t); remapright.push_back(t);
    }

    // iteratively refine remap, until it stops changing or number of allowed iterations is exceeded
    // refine left and right remappings simultaneously, to save curvature computations
    k = 0;
    while(errt > etol && k < 5)
    {
        newmapleft.clear(); newmapright.clear();

        // use curvature to revise param map
        newmapleft.push_back(0.0f); newmapright.push_back(0.0f);
        for(i = 1; i < (int) remapleft.size()-1; i++)
        {
            dleft = distleft->getPoint(remapleft[i]);
            dright = distright->getPoint(remapright[i]);

            // calculate curvature
            prevseg.diff(vertsrep[i-1], vertsrep[i]); prevseg.j = 0.0f;
            nextseg.diff(vertsrep[i], vertsrep[i+1]); nextseg.j = 0.0f;
            // cerr << "seg-1 = " << prevseg.i << ", " << prevseg.j << ", " << prevseg.k;
            // cerr << " seg+1 = " << nextseg.i << ", " << nextseg.j << ", " << nextseg.k;
            slen = nextseg.length();
            // cerr << " len seg-1 = " << slen;
            // slen = prevseg.length();
            // cerr << " len seg+1 = " << slen;
            slensq = slen * slen;

            prevseg.normalize(); nextseg.normalize();
            theta = acosf(prevseg.dot(nextseg));
            // cerr << " theta = " << theta;
            // precision issues
            kap = sqrtf(2.0f * slensq * (1.0f - cosf(theta))) / slen;

            /*
            orthog = Vector(nextseg.i - prevseg.i, 0.0f, nextseg.k - prevseg.k);
            kap = orthog.length() / nextseg.length();*/

            /*
            prevseg.normalize(); nextseg.normalize();
            theta = acosf(prevseg.dot(nextseg));
            if(theta > PI / 4.0f)
                cerr << "Error large direction change" << endl;
            */

            // calculate sidedness
            norm.i = -1.0f * prevseg.k; norm.k = prevseg.i; // rotate by 90 degrees
            if(norm.dot(nextseg) < 0.0f)
                kap *= -1.0f;

            // cerr << dleft << " ";
            // left side of curve
            // cerr << " " << kap;
            if(kap <= 0.0f) // expansion or identity
            {
                newmapleft.push_back(newmapleft[i-1]+delt);
                // cerr << " x ";
            }
            else if(kap < 1.0f / dleft) // contraction
            {
                lerp = 1.0f - kap * dleft;
                newmapleft.push_back(newmapleft[i-1]+lerp*delt);
                // cerr << " rc [" << kap << "]";
            }
            else // vanishing point
            {
                newmapleft.push_back(newmapleft[i-1]);
                // cerr << " rv [" << kap << "]";
            }

            // right side of curve
            kap *= -1.0f;
            if(kap <= 0.0f) // expansion or identity
            {
                newmapright.push_back(newmapright[i-1]+delt);
            }
            else if(kap < 1.0f / dright) // contraction
            {
                lerp = 1.0f - kap * dright;
                newmapright.push_back(newmapright[i-1]+lerp*delt);
                // cerr << " lc [" << kap << "]";

            }
            else // vanishing point
            {
                newmapright.push_back(newmapright[i-1]);
                // cerr << " lv [" << kap << "]";
            }
            // cerr << endl;
        }
        newmapleft.push_back(newmapleft[(int) newmapleft.size()-1]+delt);
        newmapright.push_back(newmapright[(int) newmapright.size()-1]+delt);
        cerr << endl;
        // cerr << "newmapleft end = " << newmapleft[(int) newmapleft.size()-1] << endl;
        // cerr << "newmapright end = " << newmapleft[(int) newmapleft.size()-1] << endl;

        // normalize parameters back to [0,1]
        nval = newmapleft[(int) newmapleft.size()-1];
        for(i = 0; i < (int) newmapleft.size(); i++)
            newmapleft[i] /= nval;
        nval = newmapright[(int) newmapright.size()-1];
        for(i = 0; i < (int) newmapright.size(); i++)
            newmapright[i] /= nval;

        // calculate param differences
        errt = 0.0f;
        for(i = 0; i < (int) remapleft.size(); i++)
            errt += fabs(remapleft[i] - newmapleft[i]);
        for(i = 0; i < (int) remapright.size(); i++)
            errt += fabs(remapright[i] - newmapright[i]);

        cerr << "iteration #" << k << " errt = " << errt << endl;
        // newmaps become remaps
        remapleft = newmapleft;
        remapright = newmapright;

        k++;
    }
    cerr << "iterations = " << k << " errt = " << errt << endl;
}

void Curve3D::redrape(Terrain * ter)
{
    drapeProject(&vertsrep, &vertsrep, ter);
}


BoundRect Curve3D::boundingBox(float t0, float t1) const
{
    BoundRect nbox;
    vpPoint p;
    int i, t0ind, t1ind;
    float tincr;

    if(t0 < pluszero && t1 > 1.0f-pluszero) // use cached bounding box
    {
        return bbox;
    }
    else // derive new bounding box
    {
        // use t values to index correct start and end of vertsrep
        tincr = 1.0f / (float) vertsrep.size();
        t0ind = (int) (t0 * tincr);
        t1ind = (int) (t1 * tincr) + 1;

        for(i = t0ind; i <= t1ind; i++)
        {
            p = vertsrep[i];
            nbox.includePnt(p);
        }
        return nbox;
    }

}


void Curve3D::deriveTangents()
{
    Vector tan1, tan2;

    splinetan.clear();
    for(int i = 0; i < (int) splinerep.size(); i++)
    {
        // first and last cp are special cases
        if(i == 0)
        {
            tan1.diff(splinerep[i], splinerep[i+1]);
            tan1.mult(0.5f);
            splinetan.push_back(tan1);
        }
        else if(i == (int) splinerep.size()-1)
        {
            tan2.diff(splinerep[i-1], splinerep[i]);
            tan2.mult(0.5f);

            splinetan.push_back(tan2);
        }
        else
        {
            // tan = (cp[i+1] - cp[i]) / 2 + (cp[i] - cp[i-1]) / 2
            tan1.diff(splinerep[i], splinerep[i+1]);
            tan1.mult(0.5f);

            tan2.diff(splinerep[i-1], splinerep[i]);
            tan2.mult(0.5f);

            tan1.add(tan2);
            splinetan.push_back(tan1);
        }
    }
}


void Curve3D::deriveVerts(bool extend)
{
    int i;
    float dsep = 0.025f;
    float t;
    vpPoint p;
    BoundRect lbr, hbr;
    uts::vector<vpPoint> denserep;

    // create a densely sampled representation of the curve
    for(int i = 0; i < (int) splinerep.size()-1; i++)
    {
        // create points within the current segment
        for(t = 0.0f; t < 1.0f-pluszero; t+= dsep)
        {
            posOnSegment(i, t, p);
            denserep.push_back(p);
        }
    }

    // add last point
    posOnSegment((int) splinerep.size()-2, 1.0f, p);
    denserep.push_back(p);

    // reparametrize according to a set distance between points, to provide an arc length parametrization
    reparametrize(&denserep, &vertsrep, sampling, extend);

    // build bounding box and hierarchical bounding volume
    leafsegboxes.clear(); highsegboxes.clear();
    bbox.reset();
    for(i = 0; i < (int) vertsrep.size(); i++)
    {
        p = vertsrep[i];
        bbox.includePnt(p); hbr.includePnt(p); lbr.includePnt(p);

        if(i != 0 && i % leafstep == 0)
        {
            if((i / leafstep) % highstep == 0)
            {
                highsegboxes.push_back(hbr);
                hbr.reset();
                hbr.includePnt(p); // new box start where the previous ended
            }

            leafsegboxes.push_back(lbr);
            lbr.reset();
            lbr.includePnt(p); // new box start where the previous ended
        }
    }

    // push final hbv boxes if they have not already been pushed
    if(!hbr.empty())
        highsegboxes.push_back(hbr);
    if(!lbr.empty())
        leafsegboxes.push_back(lbr);

/*
    cerr << "num segments = " << (int) splinerep.size()-1 << endl;
    cerr << "num verts = " << (int) vertsrep.size() << endl;
    cerr << "num high segment boxes = " << (int) highsegboxes.size() << endl;
    cerr << "num leaf segment boxes = " << (int) leafsegboxes.size() << endl;


    for(i = 0; i < (int) highsegboxes.size(); i++)
    {
        cerr << "HIGH SEG BOX " << i << endl;
        cerr << "min = (" << highsegboxes[i].min.x << ", " << highsegboxes[i].min.z << ")" << endl;
        cerr << "max = (" << highsegboxes[i].max.x << ", " << highsegboxes[i].max.z << ")" << endl;
    }

    for(i = 0; i < (int) leafsegboxes.size(); i++)
    {
        cerr << "LEAF SEG BOX " << i << endl;
        cerr << "min = (" << leafsegboxes[i].min.x << ", " << leafsegboxes[i].min.z << ")" << endl;
        cerr << "max = (" << leafsegboxes[i].max.x << ", " << leafsegboxes[i].max.z << ")" << endl;
    }


    // cerr << "curve bbox derived" << endl;
    // cerr << bbox.minx << ", " << bbox.minz << " -> " << bbox.maxx << ", " << bbox.maxz << endl;
*/
    /*
    cerr << "VERTS" << endl;
    p = vertsrep[(int) vertsrep.size() -2];
    cerr << "pen = " << p.x << ", " << p.y << ", " << p.z << endl;
    p = vertsrep[(int) vertsrep.size() -1];
    cerr << "end = " << p.x << ", " << p.y << ", " << p.z << endl;*/
}

vpPoint Curve3D::circSegIntersect(vpPoint c, float r, vpPoint f1, vpPoint f2)
{
    vpPoint p, i1, i2, s1, s2;
    float dx, dy, drsq, det, com, comsq, sgn;
    Vector sepvec, segvec;

    // intersection of segment with circle whose radius is seglen
    // guaranteed to have one solution
    s1 = vpPoint(f1.x - c.x, 0.0f, f1.z - c.z);
    s2 = vpPoint(f2.x - c.x, 0.0f, f2.z - c.z);

    dx = s2.x - s1.x;
    dy = s2.z - s1.z;
    drsq = dx*dx + dy*dy;
    det = s1.x * s2.z - s2.x * s1.z;
    comsq = (r*r*drsq-det*det);
    if(comsq <= 0.0f)
    {
        cerr << "no intersection in Curve3D::reparametrize" << endl;
        p = f2;
    }
    else
    {
        com = sqrt(comsq);

        if(dy < 0.0f)
            sgn = -1.0f;
        else
            sgn = 1.0f;

        i1.x = (det * dy + sgn * dx * com) / drsq + c.x;
        i1.y = 0.0f;
        i1.z = (-1.0f * det * dx + fabs(dy) * com) / drsq + c.z;

        i2.x = (det * dy - sgn * dx * com) / drsq + c.x;
        i2.y = 0.0f;
        i2.z = (-1.0f * det * dx - fabs(dy) * com) / drsq + c.z;

        /*
         cerr << "seglen = " << seglen << ", seplen = " << seplen << endl;
         cerr << "i1 = " << i1.x << ", " << i1.z << endl;
         cerr << "i2 = " << i2.x << ", " << i2.z << endl;
         cerr << "f = " << f.x << ", " << f.z << endl;
         cerr << "f1 = " << f1.x << ", " << f1.z << endl;
         cerr << "f2 = " << f2.x << ", " << f2.z << endl;
         */

        // choose correct intersection
        sepvec.diff(c, i1); sepvec.normalize();
        segvec.diff(f1, f2); segvec.normalize();
        // cerr << "dotprod = " << sepvec.dot(segvec) << endl;
        if(sepvec.dot(segvec) >= 0.0f)
            p = i1;
        else
            p = i2;
    }
    return p;
}

void Curve3D::reparametrize(uts::vector<vpPoint> * in, uts::vector<vpPoint> * out, float vsep, bool extend)
{
    Vector sepvec, segvec;
    vpPoint p, p1, p2, f, f1, f2;
    float arclen = 0.0f, seglen, seplen;

    int i, numseg;

    out->clear();
    if((int) in->size() >= 2)
    {
        // determine current arc length of polyline
        for(i = 1; i < (int) in->size(); i++)
        {
            p1 = (* in)[i-1]; p1.y = 0.0f; p2 = (* in)[i]; p2.y = 0.0f;
            sepvec.diff(p1, p2);
            arclen += sepvec.length();
        }

        // divide polyline into sections of vsep length or less
        numseg = (int) ceil(arclen / vsep);
        seglen = arclen / (float) numseg;

        p = (* in)[0]; out->push_back(p); // cp at start of curve
        for(i = 1; i < (int) in->size(); i++)
        {
            f = p; f.y = 0.0f;
            p1 = (* in)[i-1]; f1 = p1; f1.y = 0.0f;
            p2 = (* in)[i]; f2 = p2; f2.y = 0.0f;
            sepvec.diff(f, f2); seplen = sepvec.length();

            while(seplen >= seglen) // place point on this segment
            {
                p = circSegIntersect(f, seglen, f1, f2);
                out->push_back(p);

                // possibly further subdivision of this segment required
                f = p; f.y = 0.0f; p1 = p; f1 = f;
                sepvec.diff(f, f2); seplen = sepvec.length();
            }
        }
        // extrapolate final cp to ensure segment is of correct length
        if(extend)
            p = circSegIntersect(f, seglen, f1, f2);
        else
            p = (* in)[(int) in->size()-1];
        out->push_back(p); // cp at end of curve

        // test segment lengths and report any that diverge from the ideal
        for(i = 0; i < (int) out->size()-1; i++)
        {
            f1 = (* out)[i]; f1.y = 0.0f; f2 = (* out)[i+1]; f2.y = 0.0f;
            sepvec.diff(f1, f2);
            seplen = sepvec.length();
            if(extend)
                if(fabs(seplen - seglen) > 0.001f)
                {
                    cerr << "seg " << i+1 << " of " << (int) out->size()-1 << " has len = " << seplen << " instead of " << seglen << endl;
                }
        }
    }
}


void Curve3D::deriveCurve(Terrain * ter, bool extend)
{
    if((int) vertsrep.size() > 1)
    {
        reparametrize(&vertsrep, &splinerep, sep, extend);
        drapeProject(&splinerep, &splinerep, ter); // project cp back onto landscape
        deriveTangents();
        deriveVerts(extend);
    }
}

///
/// BrushStroke
///

vpPoint BrushStroke::addMousePnt(View * view, Terrain * ter, int x, int y)
{
    vpPoint pnt;

    view->projectingPoint(x, y, pnt);
    fragment.push_back(pnt);
    shadow.addPoint(view, ter, pnt);

    return pnt;
}

void BrushStroke::clearFragment()
{
    fragment.clear();
    shadow.clear();
}

///
/// Fragment
///

vpPoint Fragment::addMousePnt(View * view, int x, int y)
{
    vpPoint pnt, ppnt, mpnt;
    Vector dirn;

    view->projectingRay(x, y, pnt, dirn);
    dirn.mult(200.0f);
    dirn.pntplusvec(pnt, &ppnt);
    // push along projecting ray into scene
    frag.push_back(ppnt);
    view->inscreenPoint(x, y, mpnt);
    mouse.push_back(mpnt);

    return pnt;
}

void Fragment::genGL(View * view, Shape * s)
{
    float w = 0.2f;
    float col[4] = {0.325f, 0.235f, 1.0f, 1.0f};

    s->setColour(col);
    s->genCurve(frag, view, w, 0.5f, false, false, false);
}


bool Fragment::degenerate()
{
    return ((int) frag.size() <= 2);
}

bool Fragment::testLoop(float tol)
{
    Vector sep;
    bool isclosed;

    sep.diff(frag.front(), frag.back());
    isclosed = (sep.length() < tol);

    return isclosed;
}

void Fragment::screenBounds(uts::vector<vpPoint> * strk)
{
    screenMin = vpPoint(10000000.0f, 10000000.0f, 0.0f);
    screenMax = vpPoint(-10000000.0f, -10000000.0f, 0.0f);
    for(int i = 0; i < (int) strk->size(); i++)
    {
        if((* strk)[i].x < screenMin.x)
            screenMin.x = (* strk)[i].x;
        if((* strk)[i].x > screenMax.x)
            screenMax.x = (* strk)[i].x;
        if((* strk)[i].y < screenMin.y)
            screenMin.y = (* strk)[i].y;
        if((* strk)[i].y > screenMax.y)
            screenMax.y = (* strk)[i].y;
    }
}

float Fragment::screenDiag()
{
    Vector diag;

    screenBounds(&frag);
    diag.diff(screenMin, screenMax);
    return diag.length();
}

///
/// Stroke
///
/*
bool Stroke::crossing(Stroke *cross, int thresh)
{
    bool xoverlap, yoverlap;
    int i, j, numcrossings = 0;
    vpPoint c[2], x[2];
    uts::vector<vpPoint> scratch;

    // naive O(n^2) test of each segment against all segments in the crossing stroke
    // but does use inscreen bounding boxes for acceleration

    // project and bound strokes in the current screen as necessary
    if(!isinscreen)
        setProximity();

    if(!cross->isinscreen)
    {
        // form bounding box in screen space
        screenProject(&currview, &cross->fragment, &scratch);
        cross->screenBounds(&scratch);
        cross->isinscreen = false;
    }

    // check for overlap in x
    xoverlap = true;
    if(screenMin.x < cross->screenMin.x)
    {
        if(cross->screenMin.x > screenMax.x)
            xoverlap = false;
    }
    else
    {
        if(screenMin.x > cross->screenMax.x)
            xoverlap = false;
    }


    yoverlap = true;
    // check for overlap in y
    if(screenMin.y < cross->screenMin.y)
    {
        if(cross->screenMin.y > screenMax.y)
            yoverlap = false;
    }
    else
    {
        if(screenMin.y > cross->screenMax.y)
            yoverlap = false;
    }

    //xoverlap = ((cross->screenMin.x > screenMin.x && cross->screenMin.x < screenMax.x)
    //          || (cross->screenMax.x > screenMin.x && cross->screenMax.x < screenMax.x)); // overlap in x
    //yoverlap = ((cross->screenMin.y > screenMin.y && cross->screenMin.y < screenMax.y)
    //          || (cross->screenMax.y > screenMin.y && cross->screenMax.y < screenMax.y)); // overlap in y

    if(xoverlap && yoverlap)
    {
        // test every part of the two strokes against each other
        for(i = 0; i < (int) scratch.size()-1; i++)
        {
            c[0] = scratch[i]; c[1] = scratch[i+1];
            for(j = 0; j < (int) inscreen.size()-1; j++)
            {
                x[0] = inscreen[j]; x[1] = inscreen[j+1];
                if(lineCrossing(c, x)) // intersection detected
                    numcrossings++;
            }
        }
    }

    return(numcrossings >= thresh);
}
*/

void Stroke::screenBounds(uts::vector<vpPoint> * strk)
{
    screenMin = vpPoint(10000000.0f, 10000000.0f, 0.0f);
    screenMax = vpPoint(-10000000.0f, -10000000.0f, 0.0f);
    for(int i = 0; i < (int) strk->size(); i++)
    {
        if((* strk)[i].x < screenMin.x)
            screenMin.x = (* strk)[i].x;
        if((* strk)[i].x > screenMax.x)
            screenMax.x = (* strk)[i].x;
        if((* strk)[i].y < screenMin.y)
            screenMin.y = (* strk)[i].y;
        if((* strk)[i].y > screenMax.y)
            screenMax.y = (* strk)[i].y;
    }
}

bool Stroke::hasCurve() const
{
    return shadow.isCreated();
}

void Stroke::setProximity()
{
    // project stroke onto the current screen
    // printf("projected stroke size = %d\n", (int) projected.size());

    screenProject(&currview, shadow.getVerts(), &inscreen);

    // form bounding box in screen space
    screenBounds(&inscreen);
    isinscreen = true;
}

void Stroke::genGL(View * view, Shape * s, float radius)
{
    if(hasCurve())
        shadow.genGL(view, s, radius);
}

bool Stroke::mergeShadow(View * view, Terrain * ter, Fragment * frag, bool & mrg, float tol, bool brushstroke)
{
    bool pass;

    mrg = false;
    if(brushstroke)
        shadow.clear();
    
    // check to see that fragment has more than one point
    if((int) frag->getFragVec()->size() == 0)
    {
        // cerr << "Error Stroke::mergeShadow: stroke is empty" << endl;
        return false;
    }
        
    if((int) frag->getFragVec()->size() <= 2)
    {
        if(!brushstroke)
        {
            // cerr << "Error Stroke::mergeShadow: too few mouse points in stroke" << endl;
            return false;
        }
        else // special cases for brush strokes
        {
            pass = shadow.create(frag->getFragVec(), view, ter);
            return pass;
        }
    }
    
    if(!shadow.isCreated()) // new stroke so shadow not created yet
    {
        pass = shadow.create(frag->getFragVec(), view, ter);
        if(!brushstroke)
            frag->clear();
    }
    else // otherwise merge into existing stroke
    {
        // project stroke onto the current screen
        screenProject(&currview, &projected, &inscreen);
        pass = shadow.mergeStroke(frag->getFragVec(), &projected, view, ter, mrg, tol);
    }
    return pass;
}

void Stroke::adjustHeights(Terrain * ter, ValueCurve hcurve)
{
    if(hasCurve())
        shadow.adjustHeights(ter, hcurve);
}

void Stroke::redrape(Terrain * ter)
{
    if(hasCurve())
        shadow.redrape(ter);
}

vpPoint Stroke::getPoint(float t) const
{
    float ct;

    if(hasCurve())
    {
        ct = t;
        clamp(ct);
        return shadow.getPoint(ct);
    }
    else
        return vpPoint(0.0f, 0.0f, 0.0f);
}


float Stroke::getAngle(float t) const
{
    Vector delv;

    delv = getDirn(t);
    return -1.0f * RAD2DEG * atan2(delv.k, delv.i);
}


Vector Stroke::getDirn(float t) const
{
    Vector delv;
    vpPoint p, delp;
    float ct;

    ct = t;
    clamp(ct);
    if(hasCurve())
    {
        delv = shadow.getDirn(ct);
        /*
        p = shadow->getPoint(ct);
        if(ct >= 0.98f)
        {
            delp = shadow->getPoint(ct-0.02f);
            delv.diff(delp, p);
        }
        else
        {
            delp = shadow->getPoint(ct+0.02f);
            delv.diff(p, delp);
        }
        delv.normalize();*/
        return delv;
    }
    else
        return Vector(0.0f, 0.0f, 0.0f);
}


void Stroke::closestToPnt(vpPoint p, float & t, float & dist) const
{
    Vector sdirn, pdirn, norm;
    vpPoint sp, pp;

    if(hasCurve())
    {
        shadow.closest(p, t, dist, sp, sdirn);
        /*
        if(fabs(sdirn.i) <= pluszero && fabs(sdirn.j) <= pluszero && fabs(sdirn.k) <= pluszero)
        {
            cerr << "Error Stroke::closest: zero dirn vector" << endl;
            cerr << "p = " << p.x << ", " << p.y << ", " << p.z << endl;
            cerr << "t = " << t << endl;
            cerr << "sdirn = " << sdirn.i << ", " << sdirn.j << ", " << sdirn.k << endl;
        }*/
        norm.i = -1.0f * sdirn.k; norm.j = 0.0f; norm.k = sdirn.i; // rotate by 90 degrees
        if(fabsf(norm.i) < pluszero && fabsf(norm.k) < pluszero)
            cerr << "error (Stroke::closest): degenerate direction vector" << endl;
        norm.normalize();
        sp.y = 0.0f;
        pp = p; pp.y = 0.0f;
        pdirn.diff(sp, pp); // from stroke point to query point
        pdirn.normalize();
        if(norm.dot(pdirn) >= 0.0f) // right-sided pnt
            dist *= -1.0f;
    }
    else
    {
        t = -1.0f;
        dist = 0.0f;
    }
}

void Stroke::closestToRay(vpPoint cop, Vector dirn, float & t, float & dist) const
{
    if(hasCurve())
    {
        shadow.closestToRay(cop, dirn, t, dist);
    }
    else
    {
        t = -1.0f;
        dist = 0.0f;
    }
}

BoundRect Stroke::boundingBox(float t0, float t1) const
{
    BoundRect cb;

    if(hasCurve())
        return shadow.boundingBox(t0, t1);
    else
    {
        // cerr << "Error (Stroke::boundingBox): curve not properly intialized" << endl;
        cb.min = vpPoint(0.0f, 0.0f, 0.0f);
        cb.max = vpPoint(0.0f, 0.0f, 0.0f);
        return cb;
    }
}

void Stroke::printEndpoints()
{
    vpPoint pnt;
    pnt = getPoint(0.0f);
    cerr << "start = " << pnt.x << ", " << pnt.y << ", " << pnt.z << endl;
    pnt = getPoint(1.0f);
    cerr << "end = " << pnt.x << ", " << pnt.y << ", " << pnt.z << endl;
}
