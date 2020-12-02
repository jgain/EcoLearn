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

#ifndef _INC_VIEW
#define _INC_VIEW
/* file: view.h
   author: (c) James Gain, 2006
   project: ScapeSketch - sketch-based design of procedural landscapes
   notes: controlling viewpoint changes to support terrain sketching
   changes: modified for 3D curve project 5 December 2012
*/



#include "vecpnt.h"
#include "timer.h"
#include <common/debug_vector.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

using namespace std;

enum viewType {perspective, frontOrtho, upOrtho, leftOrtho};

// #define FIGURE

#define VIEW_WIDTH 800
#define VIEW_HEIGHT 800

#define RAD2DEG 57.2957795f
#define DEG2RAD 0.0174532925f
// #define ACTUAL_ASPECT 1.25f
#define ACTUAL_ASPECT 1.0f
#define spinsteps 481.0f

void trackball(float q[4], float p1x, float p1y, float p2x, float p2y);
void axis_to_quat(float a[3], float phi, float q[4]);

enum class ViewState {
    PERSPECTIVE, //< perspective viewing transform
    ORTHOGONAL, //< orthogonal viewing transform
    VSEND
};

enum class ViewMode {
    FIRSTPERSON,
    OVERVIEW,
    VMEND
};

#define orthodiv 3

// information structure for view control
class View
{
private:
    vpPoint cop;                // centre of projection
    vpPoint light;              // light position for default diffuse rendering
    Vector dir;                 // forward camera direction for current view
    vpPoint focus, prevfocus, currfocus; // controls for transition of focal point
    int focalstep;              // linear interpolation value for focal transition
    float zoomdist;             // distance to origin
    float perspwidth;           // back plane width of perspective projection
    float bu, bv;                 // arcball position parameters in normalized coordinate image plane
    float lastquat[4], curquat[4]; // quaternions for arcball
    float viewscale;            // scale adaptation to terrain size
    ViewState viewtype;          // {PERSPECTIVE, ORTHOGONAL} form of viewing
    float terextent;            // the diagonal extent of the terrain for sizing orthogonal views to fit
    int ox, oy;               // ortho quadrant in range 0 .. orthodiv-1
    Timer time;
    ViewMode vmode;		// first person, or overview mode


    float bp_u, bp_v;		// parameters for handling panning

    /// Recalculate viewing direction and up vector
    void updateDir();

public:

    float width, height;        // viewport dimensions
    float startx, starty;       // bottom left corner of viewport
    

    View(){ reset(); }

    View(float extent){ reset(extent); }

    ~View(){}

    /// Set view parameters to default
    void reset(float extent = 10000.0f)
    {
        perspwidth = 6.0f * tanf(0.5f * DEG2RAD * 45.0);
        resetLight();
        setDim(0.0f, 0.0f, (float) VIEW_WIDTH, (float) VIEW_HEIGHT);
        focus = vpPoint(0.0f, 0.0f, 0.0f);
        prevfocus = currfocus = focus;
        focalstep = 0;
        viewscale = extent;
        viewtype = ViewState::PERSPECTIVE;
        vmode = ViewMode::OVERVIEW;
        
        zoomdist = 4.5f * viewscale;

        bu = 0.0f;
        bv = 0.0f;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -0.5f);
        cop = vpPoint(0.0f, 0.0f, 1.0f);
        terextent = 1.0f;
        ox = 0; oy = 0;
        updateDir();
    }

    /// Return the center of projection of the view, recalculated as necessary
    inline vpPoint getCOP(){ return cop; }

    inline void setMode(ViewMode vmode) { this->vmode = vmode; }

    /// Change the viewpoint to directly above the terrain
    inline void topdown()
    {
        zoomdist = 4.0f * viewscale;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -1.0f);
    }

    /// Change the viewpoint zoomed in and low on the terrain
    inline void closeview()
    {
        zoomdist = 1.8f * viewscale;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -0.3f);
    }

    /// Change the viewpoint zoomed in and low on the terrain
    inline void farview()
    {
        zoomdist = 3.8 * viewscale;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -0.5f);
    }

    inline void canyonview()
    {
        zoomdist = 3.8 * viewscale;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -0.55f);
    }

    inline void cohview()
    {
        zoomdist = 3.8 * viewscale;
        trackball(curquat, 0.0f, 0.0f, 0.0f, -0.45f);
    }

    /// Align view with sun direction
    void sundir(Vector sunvec);

    void varietyview();

    /// Set the center of rotation to @a pnt, enabling an animated shift in the focal point
    void setAnimFocus(vpPoint pnt);
    
    /// Set the center of rotation to @a pnt, with immediate non-animated transition
    void setForcedFocus(vpPoint pnt);
    
    /// Return the center of rotation
    inline vpPoint getFocus(){ return currfocus; }

    /// Start spinning the viewpoint
    void startSpin();

    /// Set the terrain scaling factor
    inline void setViewScale(float extent)
    {
        zoomdist = zoomdist / viewscale * extent; // adjust zoomdist
        viewscale = extent;
    }

    /// set the extent of the orthogonal view using the terrain dimensions (tx, ty) in metres
    inline void setOrthoViewExtent(float tx, float ty)
    {
        terextent = sqrt(tx*tx + ty*ty);
    }

    /// set the form of viewing
    inline void setViewType(ViewState vs){ viewtype = vs; }
    
    /// Return the view vector from the center of projection to the focal point
    inline Vector getDir()
    {
        updateDir();
        return dir;
    }

    /// Set viewport dimensions @a width and @a height
    inline void setDim(float ox, float oy, float w, float h){ startx = ox, starty = oy; width = w; height = h;}

    /// set orthogonal quadrant for rendering
    void setOrthoQuadrant(int oqx, int oqy){ ox = oqx; oy = oqy; }
    
    /// Increment the current zoom distance by @a delta
    inline void incrZoom(float delta)
    {
        zoomdist += (delta * viewscale / 10000.0f);
        if(zoomdist < 0.2f * viewscale) // closest zoom in
            zoomdist = 0.2f * viewscale;
        if(zoomdist > 14.0f * viewscale) // furthest zoom out
            zoomdist = 14.0f * viewscale;
        apply();
    }

    /// Get the current zoom distance
    inline float getZoom()
    {
        return zoomdist;
    }

    // resetLight, setLight: light position control methods
    inline void resetLight(){ light = vpPoint(0.0f, 0.5f, 1.0f); };
    inline void setLight(vpPoint p){ light = p; }

    /// Begin rotation on button down
    void startArcRotate(float u, float v);

    /// Rotate view according to current normalized windows coordinates @a u, @a v
    void arcRotate(float u, float v);

    /// Find the ray starting at the viewing and intersecting the screen coordinates @a sx, @a sy.
    //                 Return the start position and direction vector @a dirn of the ray
    void projectingRay(int sx, int sy, vpPoint & start, Vector & dirn);

    /**
     * find the world coordinate position corresponding to screen space coordinates.
     * @param sx, sy    Screen space coordinates
     * @param[out] pnt  Screen space point but now in world coordinates
     */
    void projectingPoint(int sx, int sy, vpPoint & pnt);

    /**
     * find the screen coordinate position corresponding to mouse input.
     * @param sx, sy    Screen space coordinates
     * @param[out] pnt  mouse point but now in screen coordinates
     */
    void inscreenPoint(int sx, int sy, vpPoint & pnt);
    
    /**
     * project a the pick intersection point to the closest point on the manipulator line
     * @pre mdirn is a unit vector
     * @param pick      Pick intersection point with manipulator plane
     * @param mpnt      Point on manipulator arm
     * @param mdirn     Direction of manipulator arm
     * @param[out] mpick    Pick point projected onto manipulator arm
     */
    void projectOntoManip(vpPoint pick, vpPoint mpnt, Vector mdirn, vpPoint & mpick);
    
    /// Find a position change vector in world coordinates @a del given a change
    ///              in screen coordinates from old position @a ox, @a oy to new position @a nx, @a ny
    void projectMove(int ox, int oy, int nx, int ny, vpPoint cp, Vector & del);
    
    /// Retrieve composite model-view transformation
    glm::mat4x4 getMatrix();

    /// Retrieve the glm projection matrix
    glm::mat4x4 getProjMtx();

    /// Retrieve glm view transformation matrix
    glm::mat4x4 getViewMtx();

    /// Retrieve inverse transpose of the view transform
    glm::mat3x3 getNormalMtx();

    /// Provide a scaling factor for manipulators, which depends on zoom
    float getScaleFactor();

    /// Provide a scaling factor for brush ring, which depends on terrain scale but not zoom
    float getScaleConst();

    /// Set the current OpenGL viewing state to accord with the stored viewing parameters
    ///          Note - this does not clear the current view.
    void apply();

    /// Draw a vertical line through a pick point on the terrain to indicate the center of arcball rotation
    // void drawPick(vpPoint p);
    
    /// Shift the focal point gradually over a number of frames. Returns true if their is an animated change in focus
    bool animate();

    /// Rotate the view around the focal point over a number of frames. Returns true if spin is active
    bool spin();
    
    /// Save the current view to the file named @a filename
    ///              return true if operation is successful, otherwise false
    bool save(const char * filename);
    
    /// Load the current view from the file named @a filename
    ///              return true if operation is successful, otherwise false
    bool load(const char * filename);

    /// print view stats to cerr
    void print();
    void pan(float u, float v);
    void startPan(float u, float v);
    void setForcedLocation(float x, float y, float z);
    void modLocation(float x, float y, float z);
};
# endif // _INC_VIEW
