// Authors: K.P. Kapp and J.E. Gain

/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com) and J.E. Gain (jgain@cs.uct.ac.za)
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

/****************************************************************************
**
** Copyright (C) 2012 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <GL/glew.h>
#include "glheaders.h" // Must be included before QT opengl headers
#include <QGLWidget>
#include <QLabel>
#include <QTimer>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QPushButton>
#include <QWindow>
#include <list>
#include <mutex>
#include <common/debug_vector.h>
#include <common/debug_list.h>

#include "view.h"
#include "sim.h"
#include "ipc.h"
#include "palette.h"
#include "species_optim/species_assign_exp.h"
#include "canopy_placement/canopy_placer.h"
#include "ClusterMatrices.h"

#define PAINTCONTROL

//! [0]


const float seaval = 2000.0f;
const float initmint = 0.0f;
const float initmaxt = 40.0f;

class SunWindow;
class UndergrowthSpacer;
class UndergrowthRefiner;

enum class ControlMode
{
    VIEW,   // free viewing of scene
    PAINTLEARN,  // painting for training
    PAINTSPECIES,	// painting species
    PAINTECO, // painting ecosystems
    UNDERGROWTH_SYNTH,	// undergrowth synthesis
    CMEND
};

class Scene
{
public:

    View * view;
    Terrain * terrain;
    TypeMap * maps[(int) TypeMapType::TMTEND];
    MapFloat * moisture, * illumination, * temperature, * chm, * cdm; //< condition maps
    basic_types::MapFloat *chm_cpl;
    EcoSystem * eco;
    Biome * biome;
    Simulation * sim;
    GrassSim * grass;
    TypeMapType overlay; //< currently active overlay texture: CATEGORY, WATER, SUNLIGHT, TEMPERATURE, etc

    Scene();

    ~Scene();
};

class Window;

class GLWidget : public QGLWidget
{
    Q_OBJECT

public:

    enum class clearplants
    {
        NO,
        YES
    };

public:

    GLWidget(const QGLFormat& format, int scale_size, QWidget *parent = 0);
    ~GLWidget();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    /**
     * capture the framebuffer as an image
     * @param capImg    framebuffer is written to this image
     * @param capSize   the image is scaled by linear interpolation to this size
     */
    void screenCapture(QImage * capImg, QSize capSize);

    /// getters for currently active view, terrain, typemaps, renderer, ecosystem
    View * getView();
    Terrain * getTerrain();
    TypeMap * getTypeMap(TypeMapType purpose);
    PMrender::TRenderer * getRenderer();
    EcoSystem * getEcoSys();
    Simulation * getSim();
    GrassSim * getGrass();
    MapFloat * getSunlight();
    MapFloat * getSunlight(int month);
    MapFloat * getSlope();
    MapFloat * getMoisture();
    MapFloat * getMoisture(int month);
    MapFloat * getTemperature();
    MapFloat * getCanopyHeightModel();
    MapFloat * getCanopyDensityModel();
    Biome * getBiome();
    GLSun * getGLSun();

    SunWindow *sunwindow;

    // initialize simulator for current scene
    void initSceneSim();

    /// setter for paint or view modes
    void setCtrlMode(ControlMode mode);

    /// change control mode
    void setMode(ControlMode mode);

    /// getter and setter for brush radii
    float getRadius();
    void setRadius(float rad);

    const data_importer::common_data &get_cdata();

    /// getter, setter, refresher for overlay texture being displayed
    void refreshOverlay();
    void setOverlay(TypeMapType purpose);
    TypeMapType getOverlay();

    /// getter and setter for palette
    // void setPalette(BrushPalette * pal){ palette = pal; }
    BrushPalette * getPalette(){ return palette; }
    SpeciesPalette * getSpeciesPalette() { return species_palette; }

    QWidget *species_palette_window;

    void set_underspace_viabbase(float viabbase);
    void set_underspace_radmult(float radmult);

    /**
     * Load a sampling database of ecosystems located in the directory specified
     * @param dirprefix     directory path for loading a database
     * @param sRange        start of sampling range to be returned
     * @param eRange        end of sampling range to be returned
     */
    // void loadSampling(std::string dirprefix, SampleCoord &sRange, SampleCoord &eRange);

    /**
     * @brief bandCanopyHeightTexture   Recolour the canopy height texture according to a band of min and max tree heights
     * @param mint  Minimum tree height (below which heights are coloured black)
     * @param maxt  Maximum tree height (above which heights are coloured red)
     */
    void bandCanopyHeightTexture(float mint, float maxt);

    /**
     * Load scene attributes that are located in the directory specified
     * @param dirprefix     directory path and file name prefix combined for loading a scene
     */
    void loadScene(std::string dirprefix);

     /**
      * Save scene attributes to the directory specified
      * @param dirprefix     directory path and file name prefix combined for saving a scene, directory is assumed to exist
      */
     void saveScene(std::string dirprefix);

    /// Reading and writing ecosystem state
    // void readState(std::string statefile);
    // void writeState(std::string statefile);
    void writePlants(std::string plantfile);

    /**
     * @brief writePaintMap Output image file encoding the paint texture layer. Paint codes are converted to greyscale values
     * @param paintfile image file name
     */
    void writePaintMap(std::string paintfile);

    void writeCanopyHeightModel(std::string chmfile);

    /// Add an extra scene with placeholder view, terrain and typemap onto the end of the scene list
    void addScene();

    /// change the scene being displayed
    void setScene(int s);

    /// Prepare decal texture
    void loadDecals();

    /// Load from file to appropriate TypeMap depending on purpose
    int loadTypeMap(MapFloat * map, TypeMapType purpose);

    /// Respond to key press events
    void keyPressEvent(QKeyEvent *event);

    /// set scaling value for all terrains
    void setScales(float sc);

    void initCHM(int w, int h);

    // methods for executing various parts of the pipeline
    void doCanopyPlacement();
    void doSpeciesAssignment();
    void doCanopyPlacementAndSpeciesAssignment();
    void doUndergrowthSynthesisPart(int startrow, int endrow, int startcol, int endcol);
    void doUndergrowthSynthesis();

    // import canopy trees from pdb file at 'pathname'
    void read_pdb_canopy(std::string pathname);
    // import undergrowth plants from pdb file at 'pathname'
    void read_pdb_undergrowth(std::string pathname);

    /// check if a scene has been loaded
    bool hasSceneLoaded();

    void grow_grass();
    MapFloat *get_rocks();

    // update the pretty map based on grass, rocks, moisture on landscape
    void set_pretty_map();

    // getter for backup canopy height model to be used for canopy placement object
    basic_types::MapFloat *getPlacerCanopyHeightModel();

    // @brief get_terscale_down Get the scaling factor for scaling landscape down to req_dim
    // @param req_dim the required dimension to which the landscape should be scaled down to
    int get_terscale_down(int req_dim);

    // reinitialize species assignment pointer
    void reset_specassign_ptr();

    // @brief update_species_brushstroke Update painting for species assignment by applying brushstroke
    // @param x location to apply brushstroke to
    // @param y location to apply brushstroke to
    // @param radius radius of brushstroke
    // @param specie the species for which this brushtroke was applied
    void update_species_brushstroke(int x, int y, float radius, int specie);

    // @brief setSpecPerc Set the percentage that the currently selected species should satisfy in next species optimisation
    // @param perc the percentage the species must satisfy
    void setSpecPerc(float perc);

    // @brief getSpecPerc Get the percentage that the currently selected species should satisfy in next species optimisation
    float getSpecPerc();

    // get and set brush radiuses for species and density painting
    float getSpeciesBrushRadius();
    void setSpeciesBrushRadius(float rad);
    float getLearnBrushRadius();
    void setLearnBrushRadius(float rad);

    // register species 'id' as being added or removed from consideration for species assignment
    void species_added(int id);
    void species_removed(int id);

    // set clustermap used in interface from clustermap computed by undergrowth sampler
    // FIXME: this function will only work if undergrowth has already been sampled. It should
    // 		  be refactored to also work if undergrowth has not been sampled
    void set_clustermap();

    // same as 'set_clustermap' function above, same FIXME also, but just for the cluster density
    void set_clusterdensitymap();

    // @brief loadTypeMap copy data from 'img' into a typemap
    // @param img the image containing the data
    // @param purpose the enum corresponding to the typemap which we will copy the data into
    void loadTypeMap(const QImage &img, TypeMapType purpose);

    // @brief import_drawing create a density drawing from data in a QImage
    // @param img the image containing the draw data
    void import_drawing(const QImage &img);

    // @brief convert_painting convert brushtypes in the painting, e.g., convert all dense areas to sparse areas
    // @param from brushtype from which to convert
    // @param to brushtype to which to convert
    void convert_painting(BrushType from, BrushType to);

    // setter for cluster filenames
    void set_clusterfilenames(std::vector<std::string> cluster_filenames);

    // @brief import_canopyshading import canopy shading from a file
    // @param canopyshading filename from which to import
    void import_canopyshading(std::string canopyshading);

    // @brief report_cudamem report how much GPU memory is in use by CUDA
    // @param msg string that will comprise most of the displayed message, along with used memory
    void report_cudamem(std::string msg) const;

    // set visibility for all plants, canopy trees and undergrowth plants, respectively
    void setPlantsVisibility(bool visible);
    void setCanopyVisibility(bool visible);
    void setUndergrowthVisibility(bool visible);

    // update undergrowth synthesis object with canopyshading map held by interface
    // XXX: only used by undergrowth sampling if it is done by CPU, not GPU.
    // 		Is used by undergrowth refinement
    void update_canopyshading_undersynth();

signals:
    void signalRepaintAllGL();
    void signalRepaintAllFromThread();
    void signalUpdateCanopyPlacementProgress(int val);
    void signalUpdateQuickUndergrowthProgress(int val);
    void signalUpdateUndergrowthProgress(int val);
    void signalUpdateProgress(int val);
    void signalDisableSpecSelect();
    void signalEnableSpecSelect();

    void signalCanopyPlacementStart();
    void signalSpeciesOptimStart();
    void signalUndergrowthSampleStart();
    void signalUndergrowthRefineStart();
    
public slots:
    void repaint();

    void animUpdate(); // animation step for change of focus
    void rotateUpdate(); // animation step for rotating around terrain center

    // set required percentages for each species based on the vector 'perc', where each element corresponds with a species
    void setSpeciesPercentages(const std::vector<float> &perc);

    // @brief redrawPlants redraw plants in interface.
    // @param repaint_here if true, then update interface within this function call. If false, interface will be updated elsewhere, probably in next render loop iteration
    // @param layer Specify layers to do the update on, either canopy, undergrowth, or both
    // @param clearplants specify if plants should be cleared before redrawing
    void redrawPlants(bool repaint_here = true, ClusterMatrices::layerspec layer = ClusterMatrices::layerspec::ALL, clearplants clr = clearplants::YES);

    // launch thread which will do undergrowth refinement of existing sampled undergrowth plants
    void doUndergrowthSynthesisCallback();

    void doFastUndergrowthSampling();

    // send dense/sparse drawing to CGAN for evaluation
    void send_drawing();

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);

    void mousePressEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent * wheel);

    void init_csynth();
    void update_prettypaint(int x, int y, float radius);
    void reset_common_maps();
private:

    int underspace_count = 0;

    std::string prj_src_dir;
    std::string db_pathname;

    struct ipc_scaling_
    {
        int srcw, srch, terw, terh;
        int intscale;
    } ipc_scaling;

    std::unique_ptr<species_assign> specassign_ptr;
    std::unique_ptr<UndergrowthRefiner> undersynth;
    bool undersynth_init = false;
    std::vector<species> allspecs;
    std::vector<int> specidxes;		// mapping from index to real species
    std::unordered_map<int, int> specassign_id_to_idx;

    std::map<int, data_importer::species> species_infomap;
    std::map<int, data_importer::species> all_possible_species;
    std::vector<int> allcanopy_idx_to_id;
    std::unordered_map<int, int> allcanopy_id_to_idx;

    std::vector<float> species_percentages;
    float specperc = 0.5f;		// for the single-species drawing
    float specbrush_rad;
    float learnbrush_rad;

    std::unique_ptr<abiotic_maps_package> amaps_ptr;

    int scale_size;


    // scene control
    uts::vector<Scene *> scenes;
    GLSun * glsun;

    bool sceneloaded = false;

    // inteprocess communication
    IPC * ipc;

    int currscene;
    bool dbloaded, ecoloaded; // set to true once the user has opened a database and ecosystem

    // render variables
    PMrender::TRenderer * renderer;
    bool decalsbound;
    GLuint decalTexture;

    MapFloat *ipc_received_raw;

    std::string base_dirname;

    int ncanopyfiles = 0, nchmfiles = -1, nundergrowthfiles = 0, ngrassfiles = 0, ndrawingfiles = 0;
    int nspecassign = 0;
    int nfast_undergrowth_passes = 0;

    std::string pipeout_dirname;

    std::ofstream logfile_ofs;

    bool undergrowth_sampled = false;
    //std::unique_ptr<ClusterMatrices> clptr;

    // gui variables
    bool viewing;
    bool viewlock;
    bool focuschange;
    bool focusviz;
    bool timeron;
    //bool plantvis[maxpftypes];
    std::vector<bool> plantvis;
    float scf;
    ControlMode cmode;
    int sun_mth; // which month to display in the sunlight texture
    int wet_mth; // which month to display in the moisture texture

    float movespeed = 10.0f;

    std::mutex canopycalc;

    // brush variables
    BrushPalette * palette;
    SpeciesPalette * species_palette;
    BrushCursor brushcursor;
    BrushPaint brush;

    QPoint lastPos;
    QColor qtWhite;
    QTimer * atimer, * rtimer; // timers to control different types of animation
    QLabel * vizpopup;  //< for debug visualisation

    canopy_placer *spacer = nullptr;
    //canopy_synthesizer *csynth = nullptr;
    ClusterMatrices *clptr_temp = nullptr;
    std::unique_ptr<ClusterMatrices> clptr;

    std::vector<basic_tree> underplants;
    std::vector<basic_tree *> canopytrees_ptrs;
    std::vector<basic_tree> canopytrees;
    bool canopytrees_indices = true;

    std::string plant_sqldb_name;
    data_importer::common_data cdata;

    std::vector<species> prev_species_vec;
    int nspecies;
    int assign_times;
    bool show_undergrowth = true;
    bool show_canopy = true;
    const bool synth_undergrowth = true;
    const bool memory_scarce = false;

    bool species_changed = false;

    bool gpusample = true;

    MapFloat pretty_map_painted, pretty_map;
    MapFloat clustermap;
    MapFloat clusterdensitymap;

    ValueMap<int> species_assigned;
    ValueGridMap<float> canopyshading_temp;


    std::vector<std::string> cluster_filenames;

    /**
     * @brief floodSea  set all moisture values for a map below a certain altitude to water
     * @param ter       Terrain that provides altitudes
     * @param sealevel  shore altitude value
     * @param seaval    moisture value that represents sea water
     */
    void floodSea(Terrain * ter, MapFloat * wet, float sealevel, float seaval);

    /**
     * @brief pickInfo  write information about a terrain cell to the console
     * @param x         x-coord on terrain grid
     * @param y         y-coord on terrain grid
     */
    void pickInfo(int x, int y);

    void genOpenglTexturesForTrees();

    void convert_canopytrees_to_real_species();
    void convert_canopytrees_to_indices();
    void write_chmfiles();
    void set_ipc_scaling();

    void set_specassign_chm(MapFloat *chm);
    void send_and_receive_nnet();
    void adapt_species_changed();
    void optimise_species_brushstroke(string outfile = "");

    void reset_filecounts();
    void correct_chm_scaling();
    void calc_fast_canopyshading();

    void sparse_to_dense();
    void dense_to_sparse();
    void void_to_sparse();
    void void_to_dense();
    void dense_to_void();
    void sparse_to_void();
    void init_underspacer();
    void init_undersynth();
};

#endif
