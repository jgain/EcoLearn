/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  K.P. Kapp  (konrad.p.kapp@gmail.com)
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


#ifndef CANOPY_PLACER_H
#define CANOPY_PLACER_H

//#include "gpu_procs.h"
//#include "gl_wrapper.h"
//#include "canopy_placement/canopy_placer_gpu.h"
//#include "MapFloat.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <curand_kernel.h>
#include "common/basic_types.h"
#include <glm/glm.hpp>
#include <memory>
#include "data_importer/data_importer.h"

struct specmodels;

class gl_wrapper;


/*
 * canopy_placer class manages the integration of the CUDA computations on the textures obtained by the
 * rendering class (gl_wrapper gl_renderer member variable).
 * CUDA functions are defined in gpu_procs.cu in the same directory
 */

class canopy_placer
{
public:

    /*
     * Struct to help with computation of center of mass of visible portion of each canopy tree
     */
    struct xy_avg
    {
        xy_avg() : count(0), visible_count(0), x(0), y(0)
        {}

        __host__ __device__
        void reset()
        {
            count = visible_count = 0;
            x = y = 0.0f;
        }

        float x, y;
        int count;
        int visible_count;
    };

    enum class spec_convert
    {
        TO_IDX,
        TO_ID
    };

    /*
     * Struct that packages the relevant pointers into GPU memory for use by CUDA.
     * Helps reduce number of arguments passed to CUDA-related function calls
     */
    struct gpumem
    {
        mosaic_tree *d_trees;
        mosaic_tree *d_new_trees;
        mosaic_tree *compaction_temp;
        int *d_ntrees;	// only a single integer
        int *d_nlocs; // only a single integer
        int *d_nsampled; // only a single integer
        int *d_indicator_memspace;
        int *d_species_map;
        float *d_a, *d_b;
        //species_params *d_species_params;
        xy_avg *d_centers;
        glm::mat4 *d_translate_matrices, *d_scale_matrices;
        glm::vec4 *d_color_vecs;
        float *d_chm;
        cudaTextureObject_t d_rendered_chm_texture;
        cudaGraphicsResource *bufres_translation;
        cudaGraphicsResource *bufres_scale;
        cudaGraphicsResource *bufres_color;
        cudaGraphicsResource *texres_chm_rendered;
        curandState *d_rstate_arr;
        specmodels *models;
        float *whratios;
    };

public:
    canopy_placer(basic_types::MapFloat *chm, ValueMap<int> *species, const std::map<int, data_importer::species> &params_species, const data_importer::common_data &cdata);
    ~canopy_placer();

    typedef xy_avg xy_avg_type;

    void optimise(int max_iters);
    void save_to_file(std::string filepath);
private:
    int data_w;
    int data_h;
    gl_wrapper *gl_renderer;
    data_importer::common_data cdata;

private:	// CUDA-specific functions
    // initialization methods

	/*
	 * Completely initializates GPU resources. 
	 * Calls some other GPU initialization functions, some listed below
	 */
    void init_gpu();

    /*
	 * Some initialization and finalization fuctions for CUDA, called by other functions
	 */
    void allocate_cuda_buffers();
    void free_cuda_resources();
    void register_gl_resources();
    void unregister_gl_resources();

	/*
	 * Texture access to an OpenGL resource must be signalled, then also
	 * 'freed', so that OpenGL can access the texture again after CUDA is
	 * finished
	 */
    void ready_cuda_texture_access();
    void finish_cuda_texture_access();

    // canopy placement methods (that utilise the GPU)

    /*
     * sum all x, y locations for each tree's visible pixels, identified by its unique
     * colour. Also find its visible count and count which will act as divisor when
     * computing its center ('find_centers')
     */
    void populate_centers();

    /*
     * computes the center of each tree, based on its pixel colour and mean of
     * each of its pixel's x, y locations (found by 'populate_centers' function)
     */
    void find_centers();

    /*
     * moves each tree towards its computed center (center of mass)
     */
    void move_trees();

    /*
     * remove trees where too few of its pixels are visible in the rendering
     */
    void remove_dominated_trees();

    /*
     * sample new trees. If 'radially' is true, then sample from each existing tree
     * in a radial pattern. Otherwise, sample with a certain probability on each
     * pixel of the CHM where it's nonzero and not covered by an existing tree
     */
    void sample_new_trees(bool radially);

    /*
     * Do a top-down, orthographic rendering of all trees represented by their spheres.
     * Result of rendering is stored on a texture which is used on the GPU by the other
     * canopy placement functions
     */
    void do_rendering();

    /*
     * After adding and/or removing trees, there will be empty locations in the tree array
     * This function does stream compaction so that filled elements of the array will be
     * adjacent. Note: this invalidates previous colours for all trees, so colours are reassigned
     * based on location in array, as the colour of each tree's sphere maps to a location in the
     * array via a simple hashing function
     */
    void compact_trees();

    /*
     * Find local maxima on CHM for initial trees, and initialize those trees
     */
    void find_local_maxima_trees();

    /*
     * Reset computed center of mass for all trees, so that we can recompute them as required in a new iteration
     */
    void reset_centers();
public:
	/*
	 * Prepare initial conditions for canopy placement such as finding local maxima on which to place initial trees
	 */
    void init_optim();
	/*
	 * Initialize some required maps for canopy placement on the GPU, in this case, the CHM and species map
	 */
    void init_maps(basic_types::MapFloat *chm, ValueMap<int> *species, std::map<int, data_importer::species> params_species);

	/*
	 * A single iteration for canopy placement. The implementation of this function in the cpp file is well
	 * commented, so refer to that for details
	 */
    void iteration();

	/*
	 * Any required final adjustments to the result of canopy placement can be placed in this function. It is called
	 * after all iterations are done
	 */
    void final_adjustments_gpu();

	/*
	 * Get the rendered texture as a vector of unsigned 32 bit integers, where each integer consists of 4 8-bit
	 * RGBA channels
	 */
    void get_chm_rendered_texture(std::vector<uint32_t> &result);

	/*
	 * Write rendered texture from canopy trees to file at 'out_file'. Useful for debugging
	 */
    void write_chm_data_to_file(std::string out_file);

	/*
	 * transfer the current trees from the GPU memory space and return in a vector on the CPU
	 */
    std::vector<mosaic_tree> get_trees_from_gpu();

	/*
	 * Get species texture containing internal representation of each species, i.e. a sequential
	 * ordering of species ids 0, 1, 2, ..., nspecies - 1
	 */
    ValueMap<int> get_species_texture_raw();

	/*
	 * Get species texture containing actual species ids
	 */
    ValueMap<int> get_species_texture();

	/*
	 * Save species texture (useful for debugging)
	 */
    void save_species_texture(std::string out_filename);

	/*
	 * Get number of trees currently in canopy placement algorithm
	 */
    int get_ntrees();
	
	/* 
	 * Update the object's trees on the CPU side and return a copy of this vector also
	 */
    std::vector<mosaic_tree> get_trees();

	/*
	 * Create and return a mapping from species index (internal representation by object) to gthe actual
	 * species ids.
	 * XXX: this can be done using a std::vector instead of std::map
	 */
    std::map<int, int> create_idx_to_species();

	/*
	 * Create and return a mapping from actual species id to sequential species idx (see above for further comments)
	 */
    std::map<int, int> create_species_to_idx();

	/*
	 * When obtaining trees from CP algorithm, their species are still specified using sequential species indices,
	 * not real indices. This function converts between sequential and actual tree species and vice versa
	 * (the parameter 'convert' specifies which direction the conversion must occur).
	 * create_idx_to_species and create_species_to_idx are used by this function to achieve this
	 */
    bool convert_trees_species(spec_convert convert);

	/*
	 * Update the canopy trees held by this object on the CPU side
	 */
    void update_treesholder();

	/*
	 * Get trees in terms of basic_tree objects
	 */
    std::vector<basic_tree> get_trees_basic();

	/*
	 * Get trees in terms of basic_tree objects, in real world (not grid) coordinates
	 */
    std::vector<basic_tree> get_trees_basic_rw_coords();

private:	// CUDA-specific variables
    mosaic_tree *d_trees;
    mosaic_tree *d_new_trees;
    mosaic_tree *compaction_temp;
    int *d_ntrees;	// points to memory for only a single integer
    int *d_nlocs; // points to memory for only a single integer
    int *d_nsampled; // points to memory for only a single integer
    int *d_indicator_memspace;
    int *d_species_map;
    float *d_a, *d_b;
    //species_params *d_species_params;
    xy_avg *d_centers;
    glm::mat4 *d_translate_matrices, *d_scale_matrices;
    glm::vec4 *d_color_vecs;
    float *d_chm;
    cudaTextureObject_t d_rendered_chm_texture;
    cudaGraphicsResource *bufres_translation;
    cudaGraphicsResource *bufres_scale;
    cudaGraphicsResource *bufres_color;
    cudaGraphicsResource *texres_chm_rendered;
    curandState *d_rstate_arr;

    float *d_height_to_width;

    gpumem mem;


private:	// general variables
    int niters;
    int ntrees;
    data_struct chm;
    data_struct dem;
    int cell_size;
    std::vector<data_importer::species> all_species;
    int sample_mult = 5;
    std::vector<mosaic_tree> treesholder;	// holding location for trees, if pointers to them are used elsewhere

    std::map<int, data_importer::species> params_species;

    ValueMap<unsigned char> duplmap;

    int el_alloc;

    ValueMap<int> *orig_specmap;

    int nwhratios = 0, nspecmodels = 0;


public:
    static int memorydiv;
    template<typename T>
    std::vector<int> get_duplicate_idxes(const T &trees)
    {
        std::vector<int> dupl_idxes;

        get_trees();

        for (int i = trees.size() - 1; i >= 0; i--)
        {
            const auto &tr = trees.at(i);
            int x = tr.x + 1e-3, y = tr.y + 1e-3;
            if (duplmap.get(x, y))
            {
                dupl_idxes.push_back(i);
            }
            else
            {
                duplmap.set(x, y, 1);
            }
        }

        duplmap.fill((unsigned char)0);		// reset the duplicate map
        return dupl_idxes;
    }

    template<typename T>
    void remove_duplicate_trees(std::vector<T> &trees)
    {
        auto dupl_idxes = get_duplicate_idxes(trees);
        for (auto &idx : dupl_idxes)
        {
            trees.erase(std::next(trees.begin(), idx));
        }
    }

    void save_rendered_texture(std::string out_filename);
    void check_species_outofbounds(std::string locdesc);
    static void erase_duplicates(std::vector<basic_tree> &trees, float rwidth, float rheight);
    void erase_duplicates_fast(std::vector<basic_tree> &trees, float rwidth, float rheight);
};

#endif // CANOPY_PLACER_H
