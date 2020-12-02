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


#ifndef _INC_STROKE
#define _INC_STROKE
/* file: stroke.h
   author: (c) James Gain, 2006
   project: ScapeSketch - sketch-based design of procedural landscapes
   notes: Forming 2d mouse input into strokes for sketch and gesture purposes
   changes:
*/

#include "terrain.h"
#include "shape.h"
#include <common/debug_vector.h>

// general projection and fragment management routines

// planeProject:    project <from> stroke onto <projPlane> from the perspective of the viewpoint <view>.
//                  Return the projected stroke in the <to> vector
void planeProject(View * view, uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, Plane * projPlane);

// screenProject:   take a stroke, <from>, in world coordinates and embed it in the
//                  current screen returning the resulting stroke as <to>, using <view> as the
//                  viewing parameters
void screenProject(View * view, uts::vector<vpPoint> * from, uts::vector<vpPoint> * to);

// dropProject: project <from> stroke, in world coordinates, vertically downwards, returning as <to> stroke
void dropProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to);

// drapeProject:    vertically project a stroke <from> onto the terrain <ter>
//                  to produce a new draped stroke <to>
void drapeProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, Terrain * ter);

// terrainProject:  project a stroke onto the terrain <ter> at smoothness <lvl> from the perspective of
//                  the viewpoint <view>. return the projected stroke in the <to> vector
//                  return true if at least some of the stroke is on the terrain.
bool terrainProject(uts::vector<vpPoint> * from, uts::vector<vpPoint> * to, View * view, Terrain * ter);
bool terrainProject(vpPoint &fromPnt, vpPoint &toPnt, View * view, Terrain * ter);

// testFragmentAttach:  test whether <from> stroke endpoints are within <diftol> of <to> stroke.
//                      Return true if either are (if bothends is not set) or both are (if bothends is set)
//                      Also return the <from> index of the start and end attachments (<inds>, <inde>) and
//                      a closeness value for the quality of the link. <inds> and <inde> are set to -1 if
//                      they are not within diftol
bool testFragmentAttach(uts::vector<vpPoint> * from, uts::vector<vpPoint> * into, float diftol, bool bothends, int &inds, int &inde, float &closeness);

// inFill:  merge <from> stroke into <to> stroke as long as either the beginning or end of <from>
//          are within <diftol> of some portion of the <to> stroke. <to> stroke may have a section
//          overwritten or removed in the splicing process. if <bothends> is true then both the
//          beginning and end of the <from> stroke must attach to the existing <to> stroke.
//          if <closed> is set then allow infilling across the ends.
//          return <true> if merge takes place, otherwise <false> if <from> stroke is discarded
bool inFill(uts::vector<vpPoint> * from, uts::vector<vpPoint> * into, float diftol, bool bothends, bool closed);

// locateIntersect: return <true> and the start and end vertex indices of an overlap region,
//                  if no self-intersection exists then return false
bool locateIntersect(uts::vector<vpPoint> * strk, int &begin, int &end);

// excise: remove points in the fragment between and including the <begin> and <end> indices
void excise(uts::vector<vpPoint> * strk, int begin, int end);

// testIntersect: test to see whether the stroke intersects itself significantly, return <true> if it does
//                  self intersection which occupy a small bounding box with diagonal less than <tol>
//                  and are caused by sketching innacuracy are automatically excised.
bool testIntersect(uts::vector<vpPoint> * strk, float tol);


class ValueCurve
{
private:
    uts::vector<float> vals;  // points for piecewise segmented representation
    uts::vector<float> params;
    uts::vector<float> splinerep; // spline control points
    uts::vector<float> splinetan;  // spline tangets at control points
    int sampling;                   // number of samples per segment

    /**
     * Return a point on the hermite curve localised to a single segment
     * @param seg   point lies within this segment of the curve
     * @param t     parameter value within segment in [0,1]
     * @retval      curve value
     */
    float posOnSegment(int seg, float t) const;

public:

    /// constructor
    ValueCurve(){}

    /// constructor
    ValueCurve(uts::vector<float> * tvals, uts::vector<float> * vvals)
    {
        create(tvals, vvals);
    }

    /// destructor
    ~ValueCurve(){ clear(); }

    /// reset to an empty curve
    void clear(){ vals.clear(); params.clear(); splinerep.clear(); splinetan.clear(); }

    /// create a curve given parameters and values
    void create(uts::vector<float> * tvals, uts::vector<float> * vvals);

    /// Force the tangents at the endpoints to be horizontal
    void flatCaps();

    /**
     * Find the point at parameter value on the curve.
     * @param t parameter value
     */
    float getPoint(float t) const;

    /// Create hermite tangents using a finite difference on the control points
    void deriveTangents();

    /// Clamp tangents at endpoints so that they are horizontal
    void clampEndTangents();

    /**
     * Calculate vertices on curve.
     * @param tnum  number of vertices per segment
     */
    void deriveVerts(int tnum);

    /// Given a sequence of vertices already stored in vertsrep derive the control points by subsampling
    void deriveCurve();
};

class BrushCurve
{

private:
    uts::vector<vpPoint> vertsrep;  // points for piecewise segmented representation
    float sampling;                 // arclength separation between vertices
    bool created;                   // has stroke data been supplied?
    BoundRect enclose, update;      // total bound for curve and bound on latest update
    int updateIndex;                // start index for recent update of curve

    /**
     * Introduce extra interpolating vertices as determined by ideal point separation (sep)
     * @param[out] strk     stroke being subsampled.
     */
    void subsample(uts::vector<vpPoint> * strk);

    /**
     * Calculate vertices on curve separated according to sampling
     */
    void deriveVerts();

    /// Return a region version of a bounding box with a surrounding offset radius
    Region getBound(BoundRect &bnd, Terrain * ter, float radius);

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & vertsrep;
        ar & sampling;
        ar & created;
        ar & enclose; ar & update;
        ar & updateIndex;
    }

public:

    /// constructor
    BrushCurve(){ created = false; }

    /// constructor with actual curve data supplied
    BrushCurve(vpPoint start, View * view, Terrain * ter)
    {
        create(start, view, ter);
    }

    /// create stroke with stroke, view and terrain input
    void create(vpPoint start, View * view, Terrain * ter);

    /// return creation status of curve
    bool isCreated() const;

    /// destructor
    ~BrushCurve(){ clear(); }

    /// reset to an empty curve
    void clear(){ vertsrep.clear(); created = false; }

    /// Return a pointer to the vertices of the curve
    inline uts::vector<vpPoint> * getVerts(){ return &vertsrep; }

    /// Return the index of the start position for recent changes to curve vertices
    inline int getStartIndex(){ return updateIndex; }

    /**
     * Add a point on to the end of the curve
     * @param view      current view state
     * @param ter       terrain that point is projected onto
     * @param pnt       point to be added
     */
    void addPoint(View * view, Terrain * ter, vpPoint pnt);

    /**
     * Get the bounding box for the entire curve
     * @param ter       terrain containing the stroke
     * @param radius    brush stroke radius for expanding the bound
     * @retval          bounding rectangle
     */
    Region encloseBound(Terrain * ter, float radius);

    /**
     * Get the bounding box for the latest update to the curve
     * @param ter       terrain containing the stroke
     * @param radius    brush stroke radius for expanding the bound
     * @retval          bounding rectangle
     */
    Region updateBound(Terrain * ter, float radius);
};

class Curve3D
{

private:
    uts::vector<vpPoint> vertsrep;  // points for piecewise segmented representation
    uts::vector<vpPoint> splinerep; // spline control points
    uts::vector<Vector> splinetan;  // spline tangets at control points
    float sep;                      // arclength separation between curve control points
    float sampling;                   // arclength separation between vertices
    int highstep, leafstep;         // number of segments per box for both high and leaf hierarchy levels
    BoundRect bbox;
    uts::vector<BoundRect> leafsegboxes; // bounding boxes for each segment of the curve
    uts::vector<BoundRect> highsegboxes;
    bool created;                   // has stroke data been supplied?
    float farbound;
    uts::vector<float> remapleft;
    uts::vector<float> remapright;

    /**
     * Return a point on the hermite curve localised to a single segment
     * @param seg   point lies within this segment of the curve
     * @param t     parameter value within segment in [0,1]
     * @param[out] p    point on curve
     * @retval @c true  if input parameters are valid.
     * @retval @c false otherwise.
     */
    bool posOnSegment(int seg, float t, vpPoint & p) const;

    /**
     * Introduce extra interpolating vertices as determined by ideal point separation (sep)
     * @param[out] strk     stroke being subsampled.
     */
    void subsample(uts::vector<vpPoint> * strk);

    /// Create hermite tangents using a finite difference on the control points
    void deriveTangents();

    /**
     * Calculate vertices on curve separated according to sampling
     * @param extend    allow extension of curve during parametrization
     */
    void deriveVerts(bool extend);

    /// find the intersection of a line segment and a circle
    vpPoint circSegIntersect(vpPoint c, float r, vpPoint f1, vpPoint f2);

    /**
     * shift the vertices so that the curve has even arc lengths and a regular parametrisation
     * in the base plane
     * @param in        vector of input vertices to be resampled
     * @param[out] out  vector of output vertices with vsep distance between each point
     * @param vsep      required separation between reparametrised vertices
     * @param extend    allow curve extension to satisfy exact reparametrization
     */
    void reparametrize(uts::vector<vpPoint> * in, uts::vector<vpPoint> * out, float vsep, bool extend);

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & vertsrep;
        ar & splinerep;
        ar & splinetan;
        ar & sep;
        ar & sampling;
        ar & highstep; ar & leafstep;
        ar & bbox;
        ar & leafsegboxes; ar & highsegboxes;
        ar & created;
        ar & farbound;
    }

public:

    /// constructor
    Curve3D(){ created = false; }

    /// constructor with actual curve data supplied
    Curve3D(uts::vector<vpPoint> * strk, View * view, Terrain * ter)
    {
        create(strk, view, ter);
    }

    /**
     * Generate a stroke from point inputs
     * @param strk      input points
     * @param view      current view state
     * @param ter       terrain that stroke lies on
     * @retval @c       true if the creation succeeds
     * @retval @c       false otherwise.
     */
    bool create(uts::vector<vpPoint> * strk, View * view, Terrain * ter);

    /**
     * Generate a stroke from point inputs without requiring view-based projection
     * @param strk      input points
     * @param ter       terrain that stroke lies on
     * @retval @c       true if the creation succeeds
     * @retval @c       false otherwise.
     */
    bool nonProjCreate(uts::vector<vpPoint> * strk, Terrain * ter);

    /**
     * Regenerate a stroke where the vertex representation has been altered
     * @param ter       terrain that stroke lies on
     */
    void recreate(Terrain * ter);

    /// return creation status of curve
    bool isCreated() const;

    /// destructor
    ~Curve3D(){ clear(); }

    /// reset to an empty curve
    void clear(){ vertsrep.clear(); splinerep.clear(); splinetan.clear(); created = false; }

    /// Return a pointer to the vertices of the curve
    inline uts::vector<vpPoint> * getVerts(){ return &vertsrep; }

    /**
     * Merge the stroke fragment with the existing curve on the terrain
     * @param strk          stroke in screen space
     * @param prj           projected result after merging
     * @param view          current view state
     * @param ter           terrain that stroke is projected onto
     * @retval merge[out]   true if the existing stroke is modified
     * @param tol           how far the fragment away the fragment is allowed to begin or end from the existing curve
     * @retval @c           true  if the stroke is valid.
     * @retval @c           false otherwise.
     */
    bool mergeStroke(uts::vector<vpPoint> * strk, uts::vector<vpPoint> * prj, View * view, Terrain * ter, bool & merge, float tol);

    /**
     * Generate geometry for rendering the curve.
     * @param view      current view state
     * @param shape     container for openGL geometry and rendering
     * @param radius    radius of cylinders constituting the curve
     */
    void genGL(View * view, Shape * shape, float radius);

    /**
     * Apply locator height adjustments to the shadow
     * @param ter       terrain associated with shadow stroke
     * @param hcurve    smoothed curve representing changes in height
     */
    void adjustHeights(Terrain * ter, ValueCurve hcurve);

    /// Return the number of points in the sampled representation
    inline int numPoints(){ return (int) vertsrep.size(); }

    /**
     * Find the point at parameter value on the curve.
     * @param t parameter value
     */
    vpPoint getPoint(float t) const;

    /**
     * Get the points belonging to the curve segment beginning at a parameter value on the curve.
     * @param t     parameter value
     * @param[out] s0   start of segment
     * @param[out] s1   end of segment
     */
    void getSeg(float t, vpPoint &s0, vpPoint &s1);

    /**
     * Get the vertsrep index corresponding to a particular parameter value
     * @param t     parameter value
     */
    int getSegIdx(float t);

    /**
     * Find the curve direction at a parameter value on the curve.
     * @param t parameter value
     */
    Vector getDirn(float t) const;

    /**
     * Transform parameter value to compensate for distance field artefacts
     * @param t     parameter value to be remapped
     * @param left  true, if the remapping is for the left side of the curve, otherwise select right
     */
    float remap(float t, bool left) const;

    /**
     * Reparametrise the curve based on curvature to compensate for distance field compression
     * @param distleft      smoothed interpolation curve with distance values for left side
     * @param distright     smoothed interpolation curve with distance values for right side
     */
    void genParamRemap(ValueCurve * distleft, ValueCurve * distright);

    /**
     * Find the parameter value and distance to the nearest point on the stroke to a given point
     * @param p             query point for shortest distance test
     * @param[out] t        parameter value of closest point on curve
     * @param[out] dist     distance to closest point on curve
     * @param[out] cpnt     closest point on curve
     * @param[out] cdirn    direction of curve at closest point
     */
    void closest(vpPoint p, float & t, float & dist, vpPoint &cpnt, Vector &cdirn) const;

    /**
     * Find the parameter value and distance to the nearest point on the stroke to a give ray
     * @param cop       origin of ray
     * @param dirn      direction of ray
     * @param[out] t        parameter value of closest point on curve
     * @param[out] dist     distance to closest point on curve
     */
    void closestToRay(vpPoint cop, Vector dirn, float & t, float & dist) const;

    /**
     * Test strokes against each other to see if they intersect and return intersection parameter(s)
     * if they do
     * @param dstcurve      curve being tested against for intersection
     * @param[out] srct     list of intersection parameters on this curve (possibly empty)
     * @param[out] dstt     list of intersection parameters on dstcurve, corresponding by entry to srct
     * @retval @c           true if the two strokes intersect
     * @retval @c           false otherwise.
     */
    bool testIntersect(Curve3D * dstcurve, uts::vector<float> &srct, uts::vector<float> &dstt);

    /**
     * Test stroke against itself to see if it intersects and return intersection parameter(s)
     * if they do
     * @param srct[out]     list of intersection parameters on this curve (possibly empty)
     * @param dstt[out]     list of intersection parameters on dstcurve, corresponding by entry to srct
     * @retval @c           true if the stroke self-intersects
     * @retval @c           false otherwise.
     */
    bool testSelfIntersect(uts::vector<float> &srct, uts::vector<float> &dstt);

    /**
     * Test strokes against each other to see if their endpoints approach the other curve closely
     * @param dstcurve      curve being tested against for close approach
     * @param srct[out]     list of approach parameters on this curve (possibly empty)
     * @param dstt[out]     list of approach parameters on dstcurve, corresponding by entry to srct
     * @param tol           allowable approach distance
     * @retval @c           true if the any endpoints approach with tol
     * @retval @c           false otherwise.
     */
    bool closeApproach(Curve3D * dstcurve, uts::vector<float> &srct, uts::vector<float> &dstt, float tol);

    /**
     * Drag a section of the curve by interpolation with a linear fall towards a target position
     * @param t     parameter value for center of dragging
     * @param tpnt  target point for dragging
     * @param trange    t parameter range for interpolation on either side of t
     */
    void dragPin(float t, vpPoint tpnt, float trange);

    /**
     * Drape curve onto terrain after synthesis
     * @param ter   terrain associated with stroke
     */
    void redrape(Terrain * ter);

    /**
     * Return a bounding box (possibly a conservative one) for the curve
     * @param t0, t1      Portion of curve to consider.
     */
    BoundRect boundingBox(float t0, float t1) const;

    /**
     * Given a sequence of vertices already stored in vertsrep derive the control points by subsampling
     * @param ter       terrain associated with stroke
     * @param extend    allow extension of the curve to guarantee arc length parametrization
     */
    void deriveCurve(Terrain * ter, bool extend);

    /*
    // bound: move a point <p> that is out of bounds [-0.5, 0.5]^3 back into bounds
    void bound(vpPoint & p);

    // controlPick: return <true> if the ray from the center of projection through screen coordinates <sx, sy>
    //              passes within <tol> of a control point
    bool controlPick(int sx, int sy, View * view, float tol);

    // controlMove: translate the currently picked control point parallel to the image plane using
    //              old position <ox, oy> and new position <nx, ny> in screen coordinates
    void controlMove(int ox, int oy, int nx, int ny, View * view);

    // startDraw: delete previous curve (if any) and start drawing a new curve at <x, y> in screen coordinates
    void startDraw(int x, int y, View * view);
    void endDraw();

    // addDrawpnt: append a point to the curve being drawn, at position <x, y> in screen coordinates
    void addDrawPnt(int x, int y, View * view);

    // test: create test data for a curve (sinusoidal)
    void test();
    */
};

class BrushStroke
{
private:

    uts::vector<vpPoint> fragment;      /// partial stroke (in camera plane)

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & fragment;
        ar & shadow;
    }

public:

    BrushCurve shadow;

    BrushStroke(){}

    ~BrushStroke()
    {
        fragment.clear();
    }

    /**
     * Capture mouse point, convert into world coordinates and add to fragment list.
     * @param view  the current viewpoint
     * @param ter   the terrain for projection purposes
     * @param x     x-position in screen coordinates
     * @param y     y-position in screen coordinates
     * @retval      in-screen position in world coordinates
     */
    vpPoint addMousePnt(View * view, Terrain * ter, int x, int y);

    /// clearFragment: clear current stroke fragment so that a new fragment can be started
    void clearFragment();
};

class Fragment
{
private:

    uts::vector<vpPoint> frag;          // partial stroke (in camera plane)
    uts::vector<vpPoint> mouse;         // normalized mouse coordinates
    vpPoint screenMin, screenMax;       // bounding box corners for stroke in screenspace

    /// calculate and store the screenspace bounding box for the given stroke. It is assumed that the stroke has already been projected into screen coordinates
    void screenBounds(uts::vector<vpPoint> * strk);

public:

    Fragment(){}

    ~Fragment(){ clear(); }

    /// clear current stroke fragment
    void clear(){ frag.clear(); mouse.clear(); }

    /**
     * Capture mouse point, convert into world coordinates and add to fragment list.
     * @param view  the current viewpoint
     * @param x     x-position in screen coordinates
     * @param y     y-position in screen coordinates
     * @retval      in-screen position in world coordinates
     */
    vpPoint addMousePnt(View * view, int x, int y);

    /**
     * Generate geometry for rendering the curve.
     * @param view      current view state
     * @param s         container for openGL geometry and rendering
     */
    void genGL(View * view, Shape * s);

    /// check to see if a stroke fragment is degenerate (return true), potentially representing a click rather than drag
    bool degenerate();

    /// testLoop: test whether the starting an ending points of the fragment are within tol of each other
    ///              in screen coordinates, return true if they are and make the first and last points equal
    bool testLoop(float tol);

    /// screenDiag: return the length of the bounding box diagonal of the current stroke fragmen
    float screenDiag();

    /// getter for fragment stroke point data
    uts::vector<vpPoint> * getFragVec(){ return &frag; }

};

class Stroke
{
private:

    uts::vector<vpPoint> inscreen;      // currently complete stroke (in the camera plane)
    uts::vector<vpPoint> draped;        // projected stroke (on the landscape)
    uts::vector<vpPoint> projected;     // projected stroke (in a plane or ruled surface vertical to the landscape)
    float sampledist;                   // optimal distance between point samples
    View currview;                      // current camera view parameters
    View featureview;                   // stored view parameters for the feature stroke
    vpPoint screenMin, screenMax;       // bounding box corners for stroke in screenspace
    // has the stroke been projected into world spaced, flattened onto the landscape
    bool isdraped, isinscreen;



    // screenBounds:    calculate and store the screenspace bounding box for
    //                  the given <strk>. It is assumed that the stroke has already been projected
    //                  into screen coordinates
    void screenBounds(uts::vector<vpPoint> * strk);

    friend class boost::serialization::access;
    /// Boost serialization
    template<class Archive> void serialize(Archive & ar, const unsigned int version)
    {
        ar & inscreen;
        ar & draped;
        ar & projected;
        ar & sampledist;
        ar & screenMin; ar & screenMax;
        ar & isdraped; ar & isinscreen;
        ar & shadow;
    }

public:

    Curve3D shadow;


    ~Stroke()
    {
        inscreen.clear(); draped.clear();
    }

    Stroke()
    {
        isdraped = false; isinscreen = false;
        sampledist = 1.0f / (float) (DEFAULT_DIMX+2);
    }

    /// clearFragment: clear current stroke fragment so that a new fragment can be started
    // void clearFragment();

    /// Does this stroke have an existing curve component on the landscape
    bool hasCurve() const;

    /*
     * Copy existing drawing framents to the stroke.
     * @param strk  Source for drawing fragment to copy from
     */
    // void copyFragment(Stroke * strk);

    // setProximity:    Prepare stroke for proximity tests.
    //                  This setting is invalidated should the current view point change.
    void setProximity();

    // crossing:    Check to see if <cross> intersects the current stroke at least <thresh>
    //              times. In which case this is a scratch stroke and <true> is returned.
    // bool crossing(Stroke * cross, int thresh);

    // setViewPnt: define the centre of projection
    void setView(View cameraview)
    {
        currview = cameraview;
        isinscreen = false;
    }

    // storeViewPnt: save the viewpoint for the feature stroke
    void storeView(){ featureview = currview; }

    /**
     * Generate geometry for rendering the curve.
     * @param view      current view state
     * @param s         container for openGL geometry and rendering
     * @param radius    radius of cylinders constituting the curve
     */
    void genGL(View * view, Shape * s, float radius);

    // mergeLoop: merge the stroke fragment
    bool mergeLoop(View * view, Terrain * ter, float tol);

    /*
     * Merge the new stroke fragment with the existing shadow
     * @param view          the current viewpoint
     * @param ter           the terrain for projection purposes
     * @param frag          screen fragment to be merged with stroke
     * @param[out] mrg      true, if this stroke merges with an existing stroke
     * @param tol           the inscreen allowable distance to the existing curve for merging purposes
     * @param brushstroke   true if this is being drawn in type painting mode
     * @retval @c           true if this is a valid stroke
     * @retval @c           false otherwise.
     */
    bool mergeShadow(View * view, Terrain * ter, Fragment * frag, bool &mrg, float tol, bool brushstroke);

    /**
     * Apply locator height adjustments to stroke
     * @param ter   terrain associated with stroke
     * @param hcurve    smoothed curve representing changes in height
     */
    void adjustHeights(Terrain * ter, ValueCurve hcurve);

    /**
     * Drape curve onto terrain after synthesis
     * @param ter   terrain associated with stroke
     */
    void redrape(Terrain * ter);

    /*
     * Return the world coordinate position of a point on the curve
     * @param t         parameter of position along curve, in [0,1]
     * @retval pnt      point position returned
     */
    vpPoint getPoint(float t) const;

    /**
     * Find the normal direction in the base plane as an angle in degrees, with the positive z-axis as zero degrees
     * @param t     parametric position on the curve
     * @retval      angle in degrees
     */
    float getAngle(float t) const;

    /**
     * Similar to getAngle except return a vector for the base plane normal direction
     * @param t     parametric position on the curve
     * @retval      vector direction in x-z plane
     */
    Vector getDirn(float t) const;

    /**
     * Find the parameter value and distance to the nearest point on the stroke to a given point
     * @param p         query point for shortest distance test
     * @param[out] t    parameter value of closest point on curve
     * @param[out] dist distance to closest point on curve
     */
    void closestToPnt(vpPoint p, float & t, float & dist) const;

    /**
     * Find the parameter value and distance to the nearest point on the stroke to a given ray
     * @param cop       origin of ray
     * @param dirn      direction of ray
     * @param[out] t    parameter value of closest point on curve
     * @param[out] dist distance to closest point on curve
     */
    void closestToRay(vpPoint cop, Vector dirn, float & t, float & dist) const;

     /**
     * Return a bounding box (possibly a conservative one) for the stroke
     * @param t0, t1      Portion of curve to consider.
     */
     BoundRect boundingBox(float t0, float t1) const;

    /// print out endpoints of the curve for debugging purposes
    void printEndpoints();

    inline uts::vector<vpPoint> * getInscreen(){ return &inscreen; }
};

# endif // _INC_STROKE
