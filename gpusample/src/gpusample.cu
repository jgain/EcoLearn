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


#include <common/basic_types.h>
#include <data_importer/data_importer.h>
#include <cluster_distribs/src/ClusterMatrices.h>
#include <cluster_distribs/src/AllClusterInfo.h>
#include <cluster_distribs/src/ClusterAssign.h>
#include <cluster_distribs/src/ClusterMaps.h>
#include "kmeans_assign.h"
#include "cudatypes.h"
#include "gpusample.h"

#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#include <curand_kernel.h>



struct map_params
{
    int dupl_gw, dupl_gh;
    int prob_gw, prob_gh;
    float rw, rh;
    float realtog_probmap, realtog_duplmap;
};

struct region_data
{
    int *region_target_counts;
    int *region_synthcount;
    int nregions;
};

/*
 * struct to sample species, given a probability associated with each species.
 * Each region or cluster gets assigned an object of this type
 * XXX, TODO: Could use its own 'sample' function, rather than using its 'samplemap' array directly?
 */
struct species_sampler
{
    species_sampler(std::vector<float> probs, std::vector<int> specids, int res)
    {
        this->res = res;
        std::vector<int> samplemap_cpu(res, 0);

        float initsum = std::accumulate(probs.begin(), probs.end(), 0.0f);

        if (fabs(initsum) > 1e-5f)
        {
            for (auto &p : probs)
                p /= initsum;

            float minprob = 1.0f / res;

            float adjust = 0.0f;
            for (auto &p : probs)
            {
                if (p > 1e-5f && p < minprob)
                {
                    adjust += minprob - p;
                    p = minprob;
                }
            }

            for (auto &p : probs)
            {
                if (p > minprob * 5.0f)
                {
                    if (p - adjust > 3.0f * minprob)
                    {
                        p -= adjust;
                        break;
                    }
                }
            }


            int curridx = 0;

            for (int i = 0; i < probs.size(); i++)
            {
                int nidxes = probs.at(i) * res;
                for (int j = 0; j < nidxes; j++)
                {
                    samplemap_cpu.at(curridx) = specids.at(i);
                    curridx++;
                }
            }
            while (curridx < samplemap_cpu.size())
            {
                samplemap_cpu.at(curridx) = specids.back();
                curridx++;
            }
        }

        gpuErrchk(cudaMalloc(&samplemap, sizeof(int) * res));
        gpuErrchk(cudaMemcpy(samplemap, samplemap_cpu.data(), sizeof(int) * res, cudaMemcpyHostToDevice));
    }

    species_sampler()
    {
        res = -1;
        samplemap = nullptr;
    }

    int res;
    int *samplemap;
};

/*
 * struct to sample sizes for a given cluster.
 */
struct sizesampler_cluster
{
    sizesampler_cluster(ClusterData &cldata)
    {
        res = 0;
        for (const std::pair<int, HistogramDistrib> &sd : cldata.get_sizematrix())
        {
            const HistogramDistrib &d = sd.second;
            const std::vector<float> &samplemap = d.get_rnddistrib();
            if (res != 0 && samplemap.size() > 0 && samplemap.size() != res)
            {
                std::cout << "res: " << res << std::endl;
                std::cout << "samplemap size: " << samplemap.size() << std::endl;
                throw std::invalid_argument("Rnddistrib samplemaps must be same size");
            }
            else if (res == 0 && samplemap.size() > 0)
            {
                res = samplemap.size();
            }
        }

        auto plantids = cldata.get_plant_ids();

        int max_pid = *std::max_element(plantids.begin(), plantids.end());

        nspecies = max_pid;

        gpuErrchk(cudaMalloc(&heights, sizeof(float) * res * nspecies));
        gpuErrchk(cudaMemset(heights, 0, sizeof(float) * res * nspecies));

        for (int pid = 0; pid < max_pid; pid++)
        {
            const std::vector<float> &samplemap = cldata.get_sizedistrib(pid).get_rnddistrib();
            if (samplemap.size() > 0)
            {
                if (samplemap.size() != res)
                {
                    std::cout << "res: " << res << std::endl;
                    std::cout << "samplemap size: " << samplemap.size() << std::endl;
                    throw std::runtime_error("Bug: samplemap.size() != res in sizesample_cluster ctor");
                }
                gpuErrchk(cudaMemcpy(heights + pid * res, samplemap.data(), samplemap.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    sizesampler_cluster()
    {
        heights = nullptr;
        res = -1;
        nspecies = -1;
    }

    /*
     * Sample a height for a plant, given the species
     */
    __device__
    float sample(int species, curandState_t *rstate)
    {
        float r = curand_uniform(rstate);
        return heights[res * species + int(r * res)];
    }

    float *heights;
    int res;
    int nspecies;
};

/*
 * Initialize memory for synthcount for each region, as well as meta info
 * like total number of regions and target counts
 */
static region_data init_region_data(int *targetcounts, int nregions)
{
    region_data cpudata;

    cpudata.nregions = nregions;
    cpudata.region_target_counts = targetcounts;

    gpuErrchk(cudaMalloc(&cpudata.region_synthcount, sizeof(int) * cpudata.nregions));
    gpuErrchk(cudaMemset(cpudata.region_synthcount, 0, sizeof(int) * cpudata.nregions));

    return cpudata;
}

/*
 * Inscribe alpha into 'alphamap' for a single tree, 'plnt'
 */
__device__
void inscribe_alpha_gpu(basic_tree *plnt, float plntalpha, ValueGridMapGPU<float> alphamap)
{
    const xy<int> gridcoords = alphamap.togrid_safe(plnt->x, plnt->y);
    const xy<int> gridstart = alphamap.togrid_safe(plnt->x - plnt->radius, plnt->y - plnt->radius);
    const xy<int> gridend = alphamap.togrid_safe(plnt->x + plnt->radius, plnt->y + plnt->radius);
    const int gridrad = alphamap.togrid_safe(plnt->radius, plnt->radius).x;

    for (int x = gridstart.x; x <= gridend.x; x++)
    {
        for (int y = gridstart.y; y <= gridend.y; y++)
        {
            float griddist = (x - gridcoords.x) * (x - gridcoords.x) + (y - gridcoords.y) * (y - gridcoords.y);
            if (griddist <= ((float) gridrad * gridrad) && alphamap.get(x, y) < plntalpha)
            {
                alphamap.set(x, y, plntalpha);
            }
        }
    }

}

/*
 * Core GPU function to be called for creating alphamap based on a set of trees, in array 'trees'.
 * Stored inside 'alphamap' struct
 */
__global__
void create_alphamap_gpu(basic_tree *trees, int ntrees, ValueGridMapGPU<float> alphamap, float *alpha_species)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < ntrees)
    {
        int species = trees[tidx].species;
        float alpha = alpha_species[species];
        inscribe_alpha_gpu(trees + tidx, alpha, alphamap);
    }
}

/*
 * Smooth alphamap 'orig_alphamap' and store in 'sm_alphamap' (pointer not passed, because pointer is inside
 * 'sm_alphamap' struct)
 */
__global__
void smooth_alphamap_gpu(ValueGridMapGPU<float> orig_alphamap, ValueGridMapGPU<float> sm_alphamap, int smrad)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < orig_alphamap.gw * orig_alphamap.gh)
    {
        int gx = tidx % orig_alphamap.gw;
        int gy = tidx / orig_alphamap.gw;

        int sx = gx - smrad;
        sx = sx < 0 ? 0 : sx;
        int sy = gy - smrad;
        sy = sy < 0 ? 0 : sy;
        int ex = gx + smrad;
        ex = ex >= orig_alphamap.gw ? orig_alphamap.gw - 1 : ex;
        int ey = gy + smrad;
        ey = ey >= orig_alphamap.gh ? orig_alphamap.gh - 1 : ey;

        float sum = 0.0f;
        int count = 0;
        for (int y = sy; y <= ey; y++)
        {
            for (int x = sx; x <= ex; x++)
            {
                int gdist = sqrt(float((y - gy) * (y - gy) + (x - gx) * (x - gx)));
                if (gdist <= smrad)
                {
                    sum += orig_alphamap.get(x, y);
                    count++;
                }
            }
        }
        if (count > 0)
        {
            sum /= count;
        }
        sm_alphamap.set(gx, gy, sum);
    }
}

/*
 * Create map of alpha values based on trees in 'trees' vector. Also smooths it to account for sun movement
 */
static ValueGridMap<float> create_alphamap_hostcall(const std::vector<basic_tree> &trees, const data_importer::common_data &cdata, int gw, int gh, float rw, float rh)
{
    ValueGridMapGPU<float> orig_alphamap, smooth_alphamap;

    orig_alphamap.set_dims(gw, gh, rw, rh);
    smooth_alphamap = orig_alphamap;

    gpuErrchk(cudaMalloc(&orig_alphamap.data, sizeof(float) * gw * gh));
    gpuErrchk(cudaMalloc(&smooth_alphamap.data, sizeof(float) * gw * gh));

    orig_alphamap.fill_hostcall(0.0f);

    basic_tree *d_trees;
    int ntrees = trees.size();
    gpuErrchk(cudaMalloc(&d_trees, sizeof(basic_tree) * ntrees));
    gpuErrchk(cudaMemcpy(d_trees, trees.data(), sizeof(basic_tree) * ntrees, cudaMemcpyHostToDevice));

    float *alpha_species;
    int max_pid = 0;
    for (auto &spec : cdata.canopy_and_under_species)
    {
        if (spec.first > max_pid)
            max_pid = spec.first;
    }
    gpuErrchk(cudaMalloc(&alpha_species, sizeof(float) * (max_pid + 1)));
    std::vector<float> cpualpha;
    for (int i = 0; i <= max_pid; i++)
    {
        if (cdata.canopy_and_under_species.count(i))
        {
            cpualpha.push_back(cdata.canopy_and_under_species.at(i).alpha);
        }
        else
        {
            cpualpha.push_back(0.0f);
        }
    }
    gpuErrchk(cudaMemcpy(alpha_species, cpualpha.data(), sizeof(float) * (max_pid + 1), cudaMemcpyHostToDevice));

    int nthreads = 1024;
    int nblocks = (ntrees - 1) / 1024 + 1;
    create_alphamap_gpu<<<nblocks, nthreads>>>(d_trees, ntrees, orig_alphamap, alpha_species);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    nblocks = (gw * gh - 1) / 1024 + 1;
    smooth_alphamap_gpu<<<nblocks, nthreads>>>(orig_alphamap, smooth_alphamap, 15);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    ValueGridMap<float> retval;
    retval.setDim(gw, gh);
    retval.setDimReal(gw, gh);
    gpuErrchk(cudaMemcpy(retval.data(), smooth_alphamap.data, sizeof(float) * gw * gh, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(orig_alphamap.data));
    gpuErrchk(cudaFree(smooth_alphamap.data));
    gpuErrchk(cudaFree(alpha_species));
    gpuErrchk(cudaFree(d_trees));

    return retval;

}

/*
 * Calculate sunshade from trees in 'trees' vector, based on base sunmap 'average_landsun'.
 */
ValueGridMap<float> calc_adaptsun(const std::vector<basic_tree> &trees, ValueGridMap<float> average_landsun, const data_importer::common_data &cdata, float rw, float rh)
{
    ValueGridMap<float> average_adaptsun;
    average_adaptsun.setDim(average_landsun);
    average_adaptsun.setDimReal(average_landsun);

    int gw, gh;
    average_landsun.getDim(gw, gh);

    float *datbegin = average_adaptsun.data();
    memcpy(datbegin, average_landsun.data(), sizeof(float) * gw * gh);

    auto alphamap = create_alphamap_hostcall(trees, cdata, gw, gh, rw, rh);

    for (int y = 0; y < gh; y++)
    {
        for (int x = 0; x < gw; x++)
        {
            average_adaptsun.set(x, y, average_adaptsun.get(x, y) * (1.0f - alphamap.get(x, y)));
        }
    }
    return average_adaptsun;
}

/*
 * Compute target counts for each region based on regions in 'regionmap', and plant densities in 'model'
 */
static int *compute_region_target_counts_hostcall(ValueGridMapGPU<int> regionmap, const ClusterMatrices &model)
{
    std::vector<float> densities(model.get_nclusters(), 0.0f);

    float *d_ftargetcounts;
    gpuErrchk(cudaMalloc(&d_ftargetcounts, sizeof(float) * model.get_nclusters()));
    gpuErrchk(cudaMemcpy(d_ftargetcounts, densities.data(), sizeof(float) * model.get_nclusters(), cudaMemcpyHostToDevice));	// we use densities' zero vector for initializing the temp float array

    int *d_targetcounts;
    gpuErrchk(cudaMalloc(&d_targetcounts, sizeof(int) * model.get_nclusters()));

    for (int i = 0; i < model.get_nclusters(); i++)
    {
        densities.at(i) = model.get_region_density(i);
    }

    float *d_densities;
    gpuErrchk(cudaMalloc(&d_densities, sizeof(float) * model.get_nclusters()));
    gpuErrchk(cudaMemcpy(d_densities, densities.data(), sizeof(float) * model.get_nclusters(), cudaMemcpyHostToDevice));


    int nthreads = 1024;
    int nblocks = (regionmap.gw * regionmap.gh - 1) / nthreads + 1;
    compute_region_target_counts<<<nblocks, nthreads>>>(regionmap, d_densities, d_ftargetcounts);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    nblocks = (model.get_nclusters() - 1) / nthreads + 1;
    cast_region_target_counts<<<nblocks, nthreads>>>(d_ftargetcounts, d_targetcounts, model.get_nclusters());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_densities));
    gpuErrchk(cudaFree(d_ftargetcounts));

    return d_targetcounts;
}

/*
 * Calculate sunshade from trees in 'canopytrees', based on base sunmap in 'amaps'.
 * Overwrites base sunmap in 'amaps'
 */
static void adapt_sunlight(const std::vector<basic_tree> &canopytrees, const data_importer::common_data &cdata, abiotic_maps_package &amaps)
{
    auto bt = std::chrono::steady_clock::now().time_since_epoch();

    amaps.sun = calc_adaptsun(canopytrees, amaps.sun, cdata, amaps.rw, amaps.rh);

    auto et_adaptsun = std::chrono::steady_clock::now().time_since_epoch();
    std::cout << "Time for computing adaptsun: " << std::chrono::duration_cast<std::chrono::milliseconds>(et_adaptsun - bt).count() << std::endl;
}

/*
 * Assign clusters based on kmeans model info in 'classign', and abiotic maps in 'amaps'. Subbiomes are also
 * taken into account by way of the trees in 'canopytrees'
 */
static ValueGridMapGPU<int> init_clustermap(const ClusterAssign &classign, abiotic_maps_package &amaps, const std::vector<basic_tree> &canopytrees, data_importer::common_data cdata)
{
    auto bt_clmapgpu = std::chrono::steady_clock::now().time_since_epoch();
    ValueGridMapGPU<int> clustermap_gpu = assign_clusteridxes(amaps, classign, canopytrees, cdata);
    auto et_clmapgpu = std::chrono::steady_clock::now().time_since_epoch();
    std::cout << "Computation time for gpu raw clustermap " << std::chrono::duration_cast<std::chrono::milliseconds>(et_clmapgpu - bt_clmapgpu).count() << std::endl;

    return clustermap_gpu;
}

/*
 * Sample plants in a checkerboard pattern. This function would be called four times to sample over the whole
 * landscape/grid (assuming divisible by 4, which we do assume
 */
__global__
void sample_plants(basic_tree *plants, ValueGridMapGPU<unsigned char> duplmap, ValueGridMapGPU<float> probmap, ValueGridMapGPU<int> region_map, region_data regdata, species_sampler *specsamplers, sizesampler_cluster *sizesamplers, curandState_t *rstates, bool xinc, bool yinc, float mult)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int xloc = (idx % probmap.gw) * 2;
    int yloc = (idx / probmap.gw) * 2;
    if (xinc) xloc += 1;
    if (yinc) yloc += 1;
    idx = yloc * probmap.gw + xloc;

    if (xloc < probmap.gw && yloc < probmap.gh)
    {
        float cellw = probmap.rw / probmap.gw;
        float cellh = probmap.rh / probmap.gh;

        int regionidx = region_map.get(xloc, yloc);
        if (regionidx < 0)
            return;

        float r = curand_uniform(rstates + idx);
        float xr = curand_uniform(rstates + idx);
        float yr = curand_uniform(rstates + idx);
        float specr = curand_uniform(rstates + idx);

        int gx = idx % probmap.gw;
        int gy = idx / probmap.gw;

        float x = gx * cellw + xr * cellw;
        float y = gy * cellh + yr * cellh;

        xy<int> dxy = duplmap.togrid_safe(x, y);
        int dx = dxy.x;
        int dy = dxy.y;

        if (r < probmap.get(xloc, yloc) * mult)
            if (!duplmap.get(dx, dy))
            {
                int thiscount = atomicAdd(regdata.region_synthcount + regionidx, 1);
                if (thiscount < regdata.region_target_counts[regionidx])
                {
                    species_sampler *specsampler = specsamplers + regionidx;
                    plants[idx] = basic_tree(x, y, 0.5f, 0.5f);
                    if (specsampler->res > 0)
                        plants[idx].species = specsampler->samplemap[int(specr * specsampler->res)];
                    else
                        plants[idx].species = 0;		// TODO: take care of this case, where no species data is recorded for this cluster
                    float height = sizesamplers[regionidx].sample(plants[idx].species, rstates + idx);
                    plants[idx].height = height < 0.05f ? 0.05f : height;
                    plants[idx].radius = height * 0.5f;
                    // TODO: implement checkerboard method for this to work
                    duplmap.set(dx, dy, 1);
                    int dxs = dx - 1 < 0 ? 0 : dx - 1;
                    int dxe = dx + 1 >= duplmap.gw ? duplmap.gw - 1 : dx + 1;
                    int dys = dy - 1 < 0 ? 0 : dy - 1;
                    int dye = dy + 1 >= duplmap.gh ? duplmap.gh - 1 : dy + 1;
                    for (int cy = dys; cy <= dye; cy++)
                    {
                        for (int cx = dxs; cx <= dxe; cx++)
                        {
                            duplmap.set(cx, cy, 1);
                        }
                    }
                }
                else
                {
                    return;
                }
            }
    }
}

struct isnot_valid
{
    __host__ __device__
    bool operator()(const basic_tree &plant)
    {
        return plant.x < 0.0f || plant.y < 0.0f || plant.radius < 0.0f || plant.height < 0.0f || plant.species < 0;
    }
};

/*
 * Initializes random states for array of random number generators 'cuarr'
 */
__global__
void init_curand(curandState_t *cuarr, int n, int seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        curand_init(seed + idx, 0, 0, cuarr + idx);
}

/*
 * Initializes species_sampler objects for each cluster/region, based on species info/ratios for each cluster in 'model'
 */
static species_sampler *create_specsamplers(ClusterMatrices &model)
{
    int nclusters = model.get_nclusters();

    std::vector<species_sampler> samplers_cpu(nclusters);

    for (int clidx = 0; clidx < nclusters; clidx++)
    {
        auto probs_map = model.get_cluster(clidx).get_species_ratios();
        std::vector<float> probs;
        std::vector<int> specids;
        for (auto &sppair : probs_map)
        {
            specids.push_back(sppair.first);
            probs.push_back(sppair.second);
        }
        samplers_cpu.at(clidx) = species_sampler(probs, specids, 100);
    }

    species_sampler *retarr;

    gpuErrchk(cudaMalloc(&retarr, sizeof(species_sampler) * samplers_cpu.size()));
    gpuErrchk(cudaMemcpy(retarr, samplers_cpu.data(), sizeof(species_sampler) * samplers_cpu.size(), cudaMemcpyHostToDevice));

    return retarr;
}


/*
 * Create plant size samplers for each cluster
 */
static sizesampler_cluster *create_sizesamplers(ClusterMatrices &model)
{
    int nclusters = model.get_nclusters();

    std::vector<sizesampler_cluster> samplers_cpu(nclusters);

    for (int clidx = 0; clidx < nclusters; clidx++)
    {
        samplers_cpu.at(clidx) = sizesampler_cluster(model.get_cluster(clidx));
    }

    sizesampler_cluster *retarr;

    gpuErrchk(cudaMalloc(&retarr, sizeof(sizesampler_cluster) * samplers_cpu.size()));
    gpuErrchk(cudaMemcpy(retarr, samplers_cpu.data(), sizeof(sizesampler_cluster) * samplers_cpu.size(), cudaMemcpyHostToDevice));

    return retarr;
}

/*
 * Free species_sampler objects in array 'specsamplers'
 */
static void free_specsamplers(species_sampler *specsamplers, ClusterMatrices &model)
{
    int nclusters = model.get_nclusters();

    std::vector<species_sampler> samplers_cpu(nclusters);

    gpuErrchk(cudaMemcpy(samplers_cpu.data(), specsamplers, samplers_cpu.size() * sizeof(species_sampler), cudaMemcpyDeviceToHost));

    for (auto &sampler : samplers_cpu)
    {
        gpuErrchk(cudaFree(sampler.samplemap));
    }

    gpuErrchk(cudaFree(specsamplers));
}

/*
 * free sizesampler_cluster objects in array 'samplers'
 */
static void free_sizesamplers(sizesampler_cluster *samplers, ClusterMatrices &model)
{
    int nclusters = model.get_nclusters();

    std::vector<sizesampler_cluster> samplers_cpu(nclusters);

    gpuErrchk(cudaMemcpy(samplers_cpu.data(), samplers, samplers_cpu.size() * sizeof(sizesampler_cluster), cudaMemcpyDeviceToHost));

    for (auto &sampler : samplers_cpu)
    {
        gpuErrchk(cudaFree(sampler.heights));
    }

    gpuErrchk(cudaFree(samplers));
}


struct reduce_count_func
{
    __host__ __device__
    int operator () (const int &lhs, const int &rhs)
    {
        if (rhs >= 0)
            return lhs + rhs;
        else
            return lhs;
    }
};

/*
 * Samples undergrowth plants based on 'probmap', 'clustermap', 'region_target_counts', and the model
 */
static std::vector<basic_tree> sample_plants_hostcall(ValueGridMapGPU<unsigned char> duplmap, ValueGridMapGPU<float> probmap, ValueGridMapGPU<int> clustermap, int *region_target_counts, ClusterMatrices &model, std::function<void(int)> progress_callback)
{
    if (progress_callback)
        progress_callback(0);
    std::vector<basic_tree> plants(probmap.gw * probmap.gh, basic_tree(-1.0f, -1.0f, -1.0f, -1.0f));
    for (auto &p : plants)
        p.species = 0;

    std::vector<basic_tree> allplants;

    basic_tree *d_plants;
    gpuErrchk(cudaMalloc(&d_plants, sizeof(basic_tree) * probmap.gw * probmap.gh));
    gpuErrchk(cudaMemcpy(d_plants, plants.data(), sizeof(basic_tree) * probmap.gw * probmap.gh, cudaMemcpyHostToDevice));

    curandState_t *custates;

    int seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

    gpuErrchk(cudaMalloc(&custates, sizeof(curandState_t) * probmap.gw * probmap.gh));
    int nthreads = 1024;
    int nblocks = (probmap.gw * probmap.gh - 1) / nthreads + 1;
    init_curand<<<nblocks, nthreads>>>(custates, probmap.gw * probmap.gh, seed);

    species_sampler *specsamplers = create_specsamplers(model);
    sizesampler_cluster *sizesamplers = create_sizesamplers(model);

    //int required_count = clmaps_cpu.compute_overall_target_count(model);
    int required_count = thrust::reduce(thrust::device, region_target_counts, region_target_counts + model.get_nclusters(), 0, reduce_count_func());

    region_data regdata = init_region_data(region_target_counts, model.get_nclusters());

    auto bt = std::chrono::steady_clock().now().time_since_epoch();

    int ntrees_zerocount = 0;
    int ntrees_total = 0;

    thrust::device_ptr<float> devptr = thrust::device_pointer_cast(probmap.data);
    float maxprob = *thrust::max_element(devptr, devptr + probmap.gw * probmap.gh);

    float mult = 1.0f / maxprob;

    std::cout << "maxprob: " << maxprob << std::endl;

    for (int i = 0; i < 500; i++)
    {
        nthreads = 1024;
        nblocks = (probmap.gw * probmap.gh - 1) / nthreads + 1;

        std::cout << "Sampling plants " << i + 1 << "th iteration..." << std::endl;
        for (int yinc = 0; yinc <= 1; yinc++)
            for (int xinc = 0; xinc <= 1; xinc++)
            {
                sample_plants<<<nblocks, nthreads>>>(d_plants, duplmap, probmap, clustermap, regdata, specsamplers, sizesamplers, custates, xinc, yinc, mult);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }

        basic_tree *new_end = thrust::remove_if(thrust::device, d_plants, d_plants + probmap.gw * probmap.gh, isnot_valid());
        int ntrees = new_end - d_plants;
        gpuErrchk(cudaMemcpy(plants.data(), d_plants, sizeof(basic_tree) * ntrees, cudaMemcpyDeviceToHost));

        std::cout << "Ntrees: " << ntrees << std::endl;

        if (ntrees < (required_count - ntrees_total) * mult / 2.0f)
        {
            mult *= 1.5f;
            std::cout << "new mult: " << mult << std::endl;
        }

        allplants.insert(allplants.end(), plants.begin(), std::next(plants.begin(), ntrees));

        std::fill(plants.begin(), plants.end(), basic_tree(-1.0f, -1.0f, -1.0f, -1.0f));
        gpuErrchk(cudaMemcpy(d_plants, plants.data(), sizeof(basic_tree) * probmap.gw * probmap.gh, cudaMemcpyHostToDevice));

        ntrees_total += ntrees;

        if (ntrees == 0)
            ntrees_zerocount++;
        else
            ntrees_zerocount = 0;

        progress_callback(int(float(required_count) / ntrees_total * 100));

        if (ntrees_total >= int(required_count * 1.0f) || ntrees_zerocount >= 10)
            break;

    }
    auto et = std::chrono::steady_clock().now().time_since_epoch();

    progress_callback(100);

    std::vector<int> synthcounts_fromgpu(regdata.nregions);
    std::vector<int> targetcounts(regdata.nregions);
    gpuErrchk(cudaMemcpy(synthcounts_fromgpu.data(), regdata.region_synthcount, sizeof(int) * regdata.nregions, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(targetcounts.data(), regdata.region_target_counts, sizeof(int) * regdata.nregions, cudaMemcpyDeviceToHost));

    std::cout << "Total number of trees sampled: " << ntrees_total << std::endl;

    std::cout << "Time for undergrowth sampling: " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << " ms" << std::endl;


    free_specsamplers(specsamplers, model);
    free_sizesamplers(sizesamplers, model);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_plants));
    gpuErrchk(cudaFree(custates));
    gpuErrchk(cudaFree(regdata.region_synthcount));
    gpuErrchk(cudaFree(regdata.region_target_counts));

    return allplants;
}

/*
 * Set each cell of the probability map 'probmap' to zero
 */
__global__
void clear_probs(ValueGridMapGPU<float> probmap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < probmap.gw * probmap.gh)
        probmap.data[idx] = 0.0f;
}

/*
 * Given the 'canopytrees' vector, compute a probability for each cell in 'probmap'
 */
__global__
void assign_probs(basic_tree *canopytrees, int ntrees, ValueGridMapGPU<float> probmap, float radmult, float prob)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ntrees)
    {
        basic_tree tree = canopytrees[idx];

        xy<int> gxy = probmap.togrid_safe(tree.x, tree.y);

        int x = gxy.x;
        int y = gxy.y;
        int gsample_radius = probmap.togrid_safe(tree.radius * radmult, 0.0f).x;
        int gsrad_squared = gsample_radius * gsample_radius;
        float gsrad = sqrt(float(gsrad_squared));

        xy<int> start = probmap.togrid_safe(tree.x - tree.radius * radmult, tree.y - tree.radius * radmult);
        xy<int> end = probmap.togrid_safe(tree.x + tree.radius * radmult, tree.y + tree.radius * radmult);

        int sx = start.x;
        int sy = start.y;
        int ex = end.x;
        int ey = end.y;

        for (int cy = sy; cy <= ey; cy++)
        {
            for (int cx = sx; cx <= ex; cx++)
            {
                int distsq = ((cy - y) * (cy - y) + (cx - x) * (cx - x));
                float dist = sqrt(float(distsq));
                float falloff_r = (dist - gsrad * 0.33f);
                falloff_r = falloff_r < 0.0f ? 0.0f : falloff_r;
                falloff_r /= gsrad * 0.67f;
                falloff_r = 1 - falloff_r;
                if (distsq < gsrad_squared)
                {
                    int mapidx = cy * probmap.gw + cx;
                    atomicAdd(probmap.data + mapidx, prob * falloff_r);
                }
            }
        }
    }
}

/*
 * Initialize the probability map which will be used to determine undergrowth plant densities for each cell
 * on the landscape
 */
static ValueGridMapGPU<float> init_probmap(basic_tree *canopytrees, int ntrees, int gw, int gh, float radmult, float prob, float rw, float rh)
{
    float realtog = gw / rw;

    auto bt = std::chrono::steady_clock().now().time_since_epoch();

    ValueGridMapGPU<float> retval;
    retval.set_dims(gw, gh, rw, rh);
    retval.allocate();

    int nthreads = 1024;
    int nblocks = (gw * gh - 1) / nthreads + 1;
    clear_probs<<<nblocks, nthreads>>>(retval);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    nblocks = (ntrees - 1) / nthreads + 1;
    assign_probs<<<nblocks, nthreads>>>(canopytrees, ntrees, retval, radmult, prob);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    auto et = std::chrono::steady_clock().now().time_since_epoch();
    std::cout << "probmap init time: " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << " ms" << std::endl;

    return retval;
}


/*
 * Initialize duplicate map, used to prevent duplicate undergrowth plant locations,
 * or undergrowth plants too close to each other. Undergrowth plants will never be
 * closer than min(rw/gw, rh/gh) to each other.
 */
static void init_duplmap(int gw, int gh, float rw, float rh, unsigned char **mapptr, std::vector<basic_tree> canopytrees)
{
    auto bt = std::chrono::steady_clock::now().time_since_epoch();

    gpuErrchk(cudaMalloc(mapptr, sizeof(unsigned char) * gw * gh));
	
    ValueGridMap<unsigned char> duplmap_cpu;

	duplmap_cpu.setDim(gw, gh);
	duplmap_cpu.setDimReal(rw, rh);
	duplmap_cpu.fill((unsigned char)0);

	for (auto &tree : canopytrees)
	{
        xy<int> sg = duplmap_cpu.togrid_safe(tree.x - 1.0f, tree.y - 1.0f);
        xy<int> eg = duplmap_cpu.togrid_safe(tree.x + 1.0f, tree.y + 1.0f);

        for (int y = sg.y; y <= eg.y; y++)
        {
            for (int x = sg.x; x <= eg.x; x++)
            {
                duplmap_cpu.set(x, y, 1);
            }
        }
    }

    gpuErrchk(cudaMemcpy(*mapptr, duplmap_cpu.data(), sizeof(unsigned char) * gw * gh, cudaMemcpyHostToDevice));

    auto et = std::chrono::steady_clock::now().time_since_epoch();

    std::cout << "Duplmap init time: " << std::chrono::duration_cast<std::chrono::milliseconds>(et - bt).count() << " ms" << std::endl;
}

/*
 * Main function to be called from the host for sampling undergrowth plants.
 * This function computes adapted sunlight, after which it computes clusters/regions over the whole landscape, along
 * with required plant counts for each region, etc.
 * Then it samples undergrowth plants given the criteria of the regions computed
 */
std::vector<basic_tree> compute_and_sample_plants(ClusterMaps &clmaps, ClusterMatrices &model, const std::vector<basic_tree> &canopytrees, std::function<void(int)> progress_callback)
{
    adapt_sunlight(canopytrees, clmaps.get_cdata(), clmaps.get_maps());

    ValueGridMapGPU<int> regmap = init_clustermap(clmaps.get_classign(),
                                  clmaps.get_maps(),
                                  canopytrees,
                                  clmaps.get_cdata());

    int *d_targetcounts = compute_region_target_counts_hostcall(regmap, model);
    std::vector<int> targetcounts(model.get_nclusters());
    gpuErrchk(cudaMemcpy(targetcounts.data(), d_targetcounts, sizeof(int) * model.get_nclusters(), cudaMemcpyDeviceToHost));

    int targetcount = 0;
    for (auto &tc : targetcounts)
    {
        if (tc >= 0)
            targetcount += tc;
    }

    std::cout << "Undergrowth target count: " << targetcount << std::endl;

    basic_tree *d_trees;
    gpuErrchk(cudaMalloc(&d_trees, sizeof(basic_tree) * canopytrees.size()));
    gpuErrchk(cudaMemcpy(d_trees, canopytrees.data(), sizeof(basic_tree) * canopytrees.size(), cudaMemcpyHostToDevice));

    int duplmult = 10;

    ValueGridMapGPU<unsigned char> duplmap_gpu = create_duplmap(d_trees, canopytrees.size(), regmap.gw * duplmult, regmap.gh * duplmult, regmap.rw, regmap.rh);
    ValueGridMapGPU<float> probmap_gpu = init_probmap(d_trees, canopytrees.size(), regmap.gw, regmap.gh, common_constants::undersim_sample_mult, 0.01f, regmap.rw, regmap.rh);
    auto allplants = sample_plants_hostcall(duplmap_gpu, probmap_gpu, regmap, d_targetcounts, model, progress_callback);

    regmap.free_data();
    duplmap_gpu.free_data();
    probmap_gpu.free_data();

    return allplants;
}
