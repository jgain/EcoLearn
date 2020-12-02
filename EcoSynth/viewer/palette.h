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

// palette.h: ecosystem painting controls to apply plant distributions onto a terrain
// author: James Gain
// date: 4 July 2016

#ifndef _constraint_h
#define _constraint_h

#include "stroke.h"
#include "typemap.h"
#include "shape.h"
#include <common/debug_vector.h>
#include <common/debug_list.h>
#include <QWidget>
#include <QPushButton>
#include <QImage>
#include <common/map.h>

#include "palette_base.h"

const float manipradius = 75.0f;
const float manipheight = 750.0f;
const float armradius = manipradius / 2.5f;
const float tolzero = 0.01f;

class ConditionsMap;
class GLWidget;

#define PALETTE_ENTRIES 3 // full is 21

class specpalette_window;

class BrushPalette : public palette_base
{
    Q_OBJECT

public:
    BrushPalette(TypeMap *typemap, int nentries, QWidget *parent = 0);

public slots:
    virtual void typeSelect() override;
};

class SpeciesPalette : public palette_base
{
    Q_OBJECT

public:

    friend class specpalette_window;

    SpeciesPalette(TypeMap *typemap, const std::vector<int> &species_ids, QWidget *parent = 0);

    int getDrawTypeIndex();

public slots:
    virtual void typeSelect();

    void enable_brush(int idx);
    void disable_brush(int idx);

private:
    std::vector<int> idx_to_id;
    std::unordered_map<int, int> id_to_idx;
};

/*
class BrushPalette : public QWidget
{
    Q_OBJECT

public:

    friend class specpalette_window;

    BrushPalette(TypeMap * typemap, int nentries, QWidget *parent = 0);
    BrushPalette(TypeMap *typemap, const std::vector<int> &btypes, QWidget *parent = 0);

    ~BrushPalette()
    {
        delete [] selector;
        delete [] typeSel;
    }

    //QSize minimumSizeHint() const;
    QSize sizeHint() const;

    /// Obtain the current active brush type from the palette
    BrushType getDrawType(){ return typeSel[currSel]; }

    /// Set the currently active brush type in the palette
    void setDrawType(BrushType btype);

    /// move the currently active icon to the correct palette entry
    void setActivePalette();
    void deactiveSelection();

    void addBrushes(const std::vector<int> &btypes);

public slots:

    /// palette entry button press for painting neural net input
    void typeSelect();

    /// palette entry button press for painting species over vegetation
    void typeSelectSpecies();

    void enable_brush(int id);
    void disable_brush(int id);

private:
    int nentries;
    GLWidget * glparent;
    TypeMap * tmap;
    QImage activeImg;
    //QPushButton * selector[PALETTE_ENTRIES];
    //BrushType typeSel[PALETTE_ENTRIES];
    QPushButton **selector;
    BrushType *typeSel;
    int currSel;

    std::vector<int> idx_to_id;
    std::unordered_map<int, int> id_to_idx;
};
*/


/// Widget that displays a double torus over the landscape to indicate bounds of a paint operation
class BrushCursor
{
private:
    vpPoint pos;        ///< cursor position while over terrain
    bool active;        ///< determines whether the radius indicator should be displayed
    float radius;       ///< radial effect of painting on terrain
    float hghtoffset;   ///< offset to raise ring so that it clear the tree tops

public:

    Shape shape; ///< geometry for rendering

    BrushCursor(){ active = false; hghtoffset = 0.0f; }

    /// setter for active indicator
    void setActive(bool on){ active = on; }

    /// getter for active indicator
    bool getActive(){ return active; }

    /// getter and setter for brush radii
    void setRadius(float rad);
    float getRadius(){ return radius; }

    /// setter for height offset
    void setHeightOffset(float off){ hghtoffset = off; }

    /// setter for brush colour
    void setBrushColour(GLfloat * col)
    {
        shape.setColour(col);
    }

    /**
     * Create manipulator geometry on update
     * @param view      current view state
     * @param terrain   terrain being synthesized
     * @param brushradius radius of ring
     * @param dashed    whether or not to draw in a dashed style
     */
    void genBrushRing(View * view, Terrain * terrain, float brushradius, bool dashed);

    /**
     * Update the cursor position for rendering the brush radius
     * @param view      current view state
     * @param terrain   terrain being synthesized
     * @param x, y      on-screen mouse position
     * @retval @c true if the cursor is over the terrain
     */
    void cursorUpdate(View * view, Terrain * terrain, int x, int y);

    /**
     * Get terrain type corresponding to current mouse screen coordinates by picking
     * @param view      current view state
     * @param terrain   terrain being synthesized
     * @param tmap      terrain type map
     * @param x, y      on-screen mouse position
     */
    // int pickType(View * view, Terrain * terrain, TypeMap * tmap, int x, int y);
};

class BrushPaint
{
private:
    Terrain * terrain;      ///< to access the terrain
    Region coverage;        ///< bounding box for stroke coverage
    float radius;      ///< radial effect of stroke on terrain
    BrushType brushtype;    ///< action of brush on the ecosystem
    vpPoint currpnt;        ///< most recent mouse position on the terrain
    vpPoint prevpnt;        ///< previous painted mouse position on the terrain
    bool drawing;           ///< is drawing active
    BoundRect bnd;          ///< bounding box for update to paint map

    /**
     * Write the brush stroke col to the Ecosys Paint Map
     * This is also where ecosys updates will happen in due course
     * @param pmap          type map holding all paint colours
     * @param radius   radial effect of the brush on the ecosystem
     */
    void paintMap(TypeMap * pmap, float radius);

public:

    BrushPaint(){ brushtype = BrushType::FREE; drawing = false; }

    /**
     * Create manipulator geometry on update
     * @param view  current view state
     */
    void genManipulator(View * view);

    /**
     * Constructor
     * @param ter        terrain being synthesized
     * @param btype      type associated with brush
     */
    BrushPaint(Terrain * ter, BrushType btype);

    ~BrushPaint(){}

    /// setter for brush type
    void setBrushType(BrushType type){ brushtype = type; }

    /// getter for brush type
    BrushType getBrushType() { return brushtype; }

    /**
     * Add a point in screen coordinates to the current stroke and update the paint map accordingly
     * @param view      current view state
     * @param pmap      paint map
     * @param x, y      on-screen mouse position
     * @param radius radial effect of the brush on the ecosystem
     */
    void addMousePnt(View * view, TypeMap * pmap, int x, int y, float radius);

    /**
     * @brief startStroke Start the current brush stroke. Called on mouse down.
     */
    void startStroke();

    /**
     * Complete the current brush stroke. Called on mouse up.
     */
    void finStroke();
};

#endif
