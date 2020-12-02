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
// #include "grass.h"
#include "sim.h"
#include "ipc.h"
#include "palette.h"
//#include "canopy_placement/mosaic_spacing.h"
#include "species_optim/species_assign_exp.h"
#include "canopy_placement/canopy_placer.h"
#include "ClusterMatrices.h"
//#include "species_optim/species_optim.h"

#define PAINTCONTROL

//! [0]


const float seaval = 2000.0f;
const float initmint = 0.0f;
const float initmaxt = 40.0f;

class SunWindow;
class histcomp_window;
class UndergrowthSpacer;
class UndergrowthRefiner;

enum class ControlMode
{
    VIEW,   // free viewing of scene
    PAINTLEARN,  // painting for training
    PAINTSPECIES,	// painting species
    PAINTECO, // painting ecosystems
    SIZEDISTRIB_INSPECT,	// for showing size distributions of picked cluster
    CANOPYUNDER_INSPECT,
    UNDERUNDER_INSPECT,
    UNDERGROWTH_SYNTH,
    CANOPYTREE_ADD,
    CANOPYTREE_REMOVE,
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

    GLWidget(const QGLFormat& format, QWidget *parent = 0);
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
    MapFloat * getSunlight(int month);
    MapFloat * getSlope();
    MapFloat * getMoisture(int month);
    MapFloat * getTemperature();
    MapFloat * getCanopyHeightModel();
    MapFloat * getCanopyDensityModel();
    Biome * getBiome();
    GLSun * getGLSun();

    histcomp_window *get_histcomp() { return histcomp; }

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

    /**
     * @brief writeGrass Output terragen image files related to the grass layer
     * @param grassrootfile  name of root image file, all images use this as the prefix
     */
    void writeGrass(std::string grassrootfile);
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

    void doCanopyPlacement();
    void doSpeciesAssignment();
    void doCanopyPlacementAndSpeciesAssignment();

    void doUndergrowthSynthesisPart(int startrow, int endrow, int startcol, int endcol);
    void doUndergrowthSynthesis();

    int get_speciesid_from_index(int idx);
    int get_index_from_speciesid(int id);
    void read_pdb_canopy(std::string pathname);
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

    void grow_grass();
    MapFloat *get_rocks();
    void set_pretty_map();
    void setSpeciesPercentages(const std::vector<float> &perc);
    bool hasTrees();
    void redrawPlants(bool repaint_here = true, ClusterMatrices::layerspec layer = ClusterMatrices::layerspec::ALL, clearplants clr = clearplants::YES);
    int getnspecies();
    basic_types::MapFloat *getPlacerCanopyHeightModel();
    //void doFastUndergrowthSynthesis();
    int get_terscale_down(int req_dim);
    void save_drawing_to_file();
    void reset_specassign_ptr();
    void update_species_brushstroke(int x, int y, float radius, int specie);
    void setSpecPerc(float perc);
    float getSpecPerc();
    float getSpeciesBrushRadius();
    void setSpeciesBrushRadius(float rad);
    float getLearnBrushRadius();
    void setLearnBrushRadius(float rad);
    void species_added(int id);
    void species_removed(int id);
    void set_clustermap();
    void set_clusterdensitymap();
    void doFastUndergrowthSampling();
    void calcAndReportCanopyMtxDiff();
    void compare_sizedistribs(int cluster, int spec);
    void compare_canopyunder(int cluster);
    void compare_underunder(int cluster);
    void toUnderUnderCmode();
    std::map<int, int> get_species_clustercounts(int species);
    MapFloat *getSunlight();
    MapFloat *getMoisture();
    std::string generate_outfilename(std::string basename, int nunderpass = -1);
    void report_cudamem(std::string msg) const;
public slots:
    void read_pdb_undergrowth(std::string pathname);
    void loadTypeMap(const QImage &img, TypeMapType purpose);
    void import_drawing(const QImage &img);
    void convert_painting(BrushType from, BrushType to);
    void set_clusterfilenames(std::vector<std::string> cluster_filenames);
    void send_drawing();
    void import_canopyshading(std::string canopyshading);
    void setPlantsVisibility(bool visible);
    void setCanopyVisibility(bool visible);
    void setUndergrowthVisibility(bool visible);
    void update_canopyshading_undersynth();
    void doUndergrowthSynthesisCallback();
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
    void compute_sampleprob_map(const std::vector<basic_tree> &trees);
    void reset_common_maps();
private:

    std::ofstream ofs_timings;

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

    const float synth_cellsize = 80.0f;
    histcomp_window *histcomp;
    histcomp_window *canopyunder_comp;
    histcomp_window *underunder_comp;

    std::unique_ptr<abiotic_maps_package> amaps_ptr;


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

    std::string diffresults_filename = "/home/konrad/PhDStuff/mtxdiff.txt";
    std::ofstream diffresults_ofs;

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
    ValueGridMap<float> sampleprob_map;


    std::vector<std::string> cluster_filenames;

    /**
     * @brief floodSea  set all moisture values for a map below a certain altitude to water
     * @param ter       Terrain that provides altitudes
     * @param sealevel  shore altitude value
     * @param seaval    moisture value that represents sea water
     */
    void floodSea(Terrain * ter, MapFloat * wet, float sealevel, float seaval);

    /**
     * @brief fillBeach set all paint values to empty below a certain altitude to represent the beach
     * @param ter       Terrain that provides altitudes
     * @param paintmap  map holding different types of terrain
     * @param beachlevel    beach altitude value
     */

    /**
     * @brief pickInfo  write information about a terrain cell to the console
     * @param x         x-coord on terrain grid
     * @param y         y-coord on terrain grid
     */
    void pickInfo(int x, int y);
    /*
    void fillBeach(Terrain * ter, TypeMap * paintmap, float beachlevel);
    void fillWater(Terrain * ter, MapFloat * wet, TypeMap * paintmap, float waterval);
    void fillRock(Terrain * ter, EcoSystem * ecs, TypeMap * paintmap, float slopeval);*/

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
