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


#include "UndergrowthRefiner.h"
#include "ClusterDistribDerivator.h"
#include "ClusterMaps.h"
#include <unordered_map>
#include <set>

UndergrowthRefiner::UndergrowthRefiner(std::vector<string> cluster_filenames,
                                       abiotic_maps_package amaps,
                                       std::vector<basic_tree> canopytrees,
                                       data_importer::common_data cdata)
    : UndergrowthSampler(cluster_filenames, amaps, canopytrees, cdata)
{
}

void UndergrowthRefiner::set_undergrowth(const std::vector<basic_tree> &undergrowth)
{
    this->undergrowth = undergrowth;
}

std::vector<basic_tree> UndergrowthRefiner::get_undergrowth()
{
    return undergrowth;
}

void UndergrowthRefiner::sample_init()
{
    undergrowth = sample_undergrowth();
}

void UndergrowthRefiner::derive_complete()
{
    if (undergrowth.size() == 0)
        throw std::runtime_error("No undergrowth on which to derive complete distribution model");

    ClusterDistribDerivator deriv(clmaps, undergrowth);
    ClusterMatrices tempmodel = deriv.deriveHistogramMatricesSeparate(model.get_cdata()).front();
    derivmodel.reset(new ClusterMatrices(tempmodel));
}

void UndergrowthRefiner::add_plant_effects_for_restore(const basic_tree &plnt, const basic_tree &ignore,
                                                PlantSpatialHashmap &underghash, PlantSpatialHashmap &canopyhash,
                                                float *rembenefit)
{
    // XXX: check this get_relevant_cells function. Floating point issues for bordering plants?
    std::vector< std::vector<basic_tree> *> relcells = underghash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    std::vector< std::vector<basic_tree> *> relcells_canopy = canopyhash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    float width = clmaps.get_width(), height = clmaps.get_height();

    // get cluster index of this plant, as well as benchmark matrix and derived matrix based on cluster index
    int clidx = clmaps.get_cluster_idx(plnt.x, plnt.y);
    HistogramMatrix *benchmtx = &model.get_cluster(clidx).get_locmatrix();
    auto &mtx = derivmodel->get_cluster(clidx).get_locmatrix();

    int nadded_ug = 0;
    int nadded_canopy = 0;

    // remove effects on histograms in mtx based on this plant
    for (auto &cvecptr : relcells)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            auto &oplnt = *piter;

            // we allow an 'ignore' plant to be specified. This is useful if we don't want the plant
            // to compute a distance against itself, or against another plant which we wish to ignore
            if (ignore != oplnt)
            {
                // compute row and column in histogram of this plant's distribution relative to oplnt.
                // The row and column, along with the cluster index, are used to uniquely identify this
                // distribution so that we can keep track of it for purposes of restoring it after adding
                // temporary distances
                int row = mtx.get_undergrowth_idx(plnt.species);
                int col = mtx.get_undergrowth_idx(oplnt.species);
                if (col > row) std::swap(col, row);
                std::vector<common_types::decimal> bins;

                // get cluster index and distribution matrix in case the other plant has lower priority than
                // plnt, in which case we calculate the distance from the viewpoint of oplnt.
                int oclidx = clmaps.get_cluster_idx(oplnt.x, oplnt.y);
                auto &omtx = derivmodel->get_cluster(oclidx).get_locmatrix();
                HistogramMatrix *obenchmtx = &model.get_cluster(oclidx).get_locmatrix();

                // Try to add the distance from plnt to oplnt. If it fails, which means added == false, then
                // it likely means that oplnt is lower priority than plnt.
                bool added = mtx.add_to_bin_undergrowth(plnt, oplnt, width, height, normalize_method::COMPLETE,
                                                        benchmtx, rembenefit, &bins);

                // If we successfully added the distance, we backup this distribution so that we can restore it
                // after we are done with this candidate plant
                if (added)
                {
                    if (bins.size() == 0) throw std::runtime_error("bins should not be empty after distance addition");
                    unsigned encode = encode_distrib(clidx, row, col);
                    if (backup_distribs.count(encode) == 0)
                        backup_distribs[encode] = bins;
                    nadded_ug++;
                }
                bins.clear();

                // Same as above. Note that we still try to add this distance even if we successfully added above,
                // since we need to add from the viewpoint of both plants if they are the same priority
                bool added2 = omtx.add_to_bin_undergrowth(oplnt, plnt, width, height, normalize_method::COMPLETE,
                                                          obenchmtx, rembenefit, &bins);

                // Same as above
                if (added2)
                {
                    if (bins.size() == 0) throw std::runtime_error("bins should not be empty after distance addition");
                    unsigned encode = encode_distrib(oclidx, row, col);
                    if (backup_distribs.count(encode) == 0)
                        backup_distribs[encode] = bins;
                    nadded_ug++;
                }
            }
        }
    }

    // Same as above, except now we add and backup the effects of distances from plnt to canopy trees
    for (auto &cvecptr : relcells_canopy)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            int row = mtx.get_undergrowth_idx(plnt.species);
            int col = mtx.get_canopy_idx(piter->species);
            if (col >= row) throw std::runtime_error("canopy idx cannot be equal or higher than undergrowth idx");
            std::vector<common_types::decimal> bins;

            bool added = mtx.add_to_bin_canopy(plnt, *piter, width, height, normalize_method::COMPLETE,
                                               benchmtx, rembenefit, &bins);
            if (added)
            {
                if (bins.size() == 0) throw std::runtime_error("bins should not be empty after distance addition");
                unsigned encode = encode_distrib(clidx, row, col);
                if (backup_distribs.count(encode) == 0)
                    backup_distribs[encode] = bins;
                nadded_canopy++;
            }
        }
    }
}

void UndergrowthRefiner::restore_distribs()
{
    for (auto &p : backup_distribs)
    {
        unsigned code = p.first;
        int row, col, clidx;
        decode_distrib(code, clidx, row, col);
        derivmodel->get_cluster(clidx).get_locmatrix().get_distrib_rowcol(row, col).setbins(p.second);
    }
}


void UndergrowthRefiner::add_plant_effects(const basic_tree &plnt, const basic_tree &ignore,
                                           PlantSpatialHashmap &underghash, PlantSpatialHashmap &canopyhash,
                                           float *rembenefit)
{
    // XXX: check this get_relevant_cells function. Floating point issues for bordering plants?
    std::vector< std::vector<basic_tree> *> relcells = underghash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    std::vector< std::vector<basic_tree> *> relcells_canopy = canopyhash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    float width = clmaps.get_width(), height = clmaps.get_height();

    int clidx = clmaps.get_cluster_idx(plnt.x, plnt.y);
    HistogramMatrix *benchmtx = &model.get_cluster(clidx).get_locmatrix();
    auto &mtx = derivmodel->get_cluster(clidx).get_locmatrix();

    int nadded_ug = 0;
    int nadded_canopy = 0;

    for (auto &cvecptr : relcells)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            auto &oplnt = *piter;
            if (ignore != oplnt)
            {
                int oclidx = clmaps.get_cluster_idx(oplnt.x, oplnt.y);
                auto &omtx = derivmodel->get_cluster(oclidx).get_locmatrix();
                HistogramMatrix *obenchmtx = &model.get_cluster(oclidx).get_locmatrix();
                bool added = mtx.add_to_bin_undergrowth(plnt, oplnt, width, height, normalize_method::COMPLETE,
                                                        benchmtx, rembenefit, nullptr);
                bool added2 = omtx.add_to_bin_undergrowth(oplnt, plnt, width, height, normalize_method::COMPLETE,
                                                          obenchmtx, rembenefit, nullptr);
                if (added) nadded_ug++;
                if (added2) nadded_ug++;
            }
        }
    }

    for (auto &cvecptr : relcells_canopy)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            bool added = mtx.add_to_bin_canopy(plnt, *piter, width, height, normalize_method::COMPLETE,
                                               benchmtx, rembenefit);
            if (added) nadded_canopy++;
        }
    }
    //std::cout << "Number of canopy distances added: " << nadded_canopy << std::endl;
    //std::cout << "Number of undergrowth distances added: " << nadded_ug << std::endl;
}

void UndergrowthRefiner::remove_plant_effects(const basic_tree &plnt, const basic_tree &ignore,
                                              PlantSpatialHashmap &underghash, PlantSpatialHashmap &canopyhash,
                                              float *rembenefit)
{
    // XXX: check this get_relevant_cells function. Floating point issues for bordering plants?
    std::vector< std::vector<basic_tree> *> relcells = underghash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    std::vector< std::vector<basic_tree> *> relcells_canopy = canopyhash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
    float width = clmaps.get_width(), height = clmaps.get_height();

    // get cluster index of this plant, as well as benchmark matrix and derived matrix based on cluster index
    int clidx = clmaps.get_cluster_idx(plnt.x, plnt.y);
    HistogramMatrix *benchmtx = &model.get_cluster(clidx).get_locmatrix();
    auto &mtx = derivmodel->get_cluster(clidx).get_locmatrix();

    // remove effects on histograms in mtx based on this plant
    for (auto &cvecptr : relcells)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            auto &oplnt = *piter;

            // we allow an 'ignore' plant to be specified. This is useful if we don't want the plant
            // to compute a distance against itself, or even against another temporary plant
            if (ignore != oplnt)
            {
                int oclidx = clmaps.get_cluster_idx(oplnt.x, oplnt.y);
                auto &omtx = derivmodel->get_cluster(oclidx).get_locmatrix();
                HistogramMatrix *obenchmtx = &model.get_cluster(oclidx).get_locmatrix();
                mtx.remove_from_bin_undergrowth(plnt, oplnt, width, height, normalize_method::COMPLETE,
                                                benchmtx, rembenefit);
                omtx.remove_from_bin_undergrowth(oplnt, plnt, width, height, normalize_method::COMPLETE,
                                                 obenchmtx, rembenefit);
            }
        }
    }

    for (auto &cvecptr : relcells_canopy)
    {
        for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
        {
            mtx.remove_from_bin_canopy(plnt, *piter, width, height, normalize_method::COMPLETE, benchmtx, rembenefit);
        }
    }
}

unsigned UndergrowthRefiner::encode_distrib(unsigned clidx, unsigned row, unsigned col)
{
    // cluster index value is located at bits 1 - 16, row value at bits 17 - 24, column value at bits 25 - 32
    unsigned retval = clidx | (row << 16) | (col << 24);
    return retval;
}

void UndergrowthRefiner::decode_distrib(unsigned code, int &clidx, int &row, int &col)
{
    clidx = code & 0x0000FFFF;
    row = (code & 0x00FF0000) >> 16;
    col = (code & 0xFF000000) >> 24;
}

void UndergrowthRefiner::test_encode_decode()
{
    int clidx = 500;
    int row = 12;
    int col = 7;
    unsigned encoded = encode_distrib(clidx, row, col);
    std::cout << "Encoding of clidx " << clidx << ", row " << row << ", col " << col << ": " << encoded << std::endl;
    int clidx_back, row_back, col_back;
    decode_distrib(encoded, clidx_back, row_back, col_back);
    if (clidx_back != clidx || row_back != row || col_back != col)
        throw std::runtime_error("Encoding/decoding faulty");
}

void UndergrowthRefiner::refine()
{
    std::uniform_real_distribution<float> unif;
    std::default_random_engine gen;

    if (progress_callback)
        progress_callback(0);

    float maxpert = 5.0f;

    std::cout << "Deriving initial distributions..." << std::endl;

    std::cout.setstate(std::ios_base::failbit);
    derive_complete();
    derivmodel->normalizeAll(normalize_method::COMPLETE);
    std::cout.clear();


    std::cout << "checking maxdists for derivmodel..." << std::endl;
    derivmodel->check_maxdists();
    std::cout << "checking maxdists for benchmark model..." << std::endl;
    model.check_maxdists();

    PlantSpatialHashmap canopyhash(30.0f, 30.0f, clmaps.get_width(), clmaps.get_height());
    PlantSpatialHashmap underghash(5.0f, 5.0f, clmaps.get_width(), clmaps.get_height());

    float width = clmaps.get_width(), height = clmaps.get_height();

    canopyhash.addplants(clmaps.get_canopytrees());
    underghash.addplants(undergrowth);

    std::unordered_map<int, std::set<std::pair<int, int> > > distribscheck;
    auto backupmodel = *derivmodel;


    HistogramMatrix *benchmtx = nullptr;

    int check_every_n = 100;
    int noverall_iters = 5;

    for (int overall_iter = 0; overall_iter < noverall_iters; overall_iter++)
    {
        int plnti = 0;
        int nchanged = 0;

        for (auto &plnt : undergrowth)
        {
            backup_distribs.clear();
            plnti++;
            int clidx = clmaps.get_cluster_idx(plnt.x, plnt.y);
            benchmtx = &model.get_cluster(clidx).get_locmatrix();
            HistogramMatrix &mtx = derivmodel->get_cluster(clidx).get_locmatrix();

            float rembenefit = 0.0f;

            // XXX: check this get_relevant_cells function. Floating point issues for bordering plants?
            std::vector< std::vector<basic_tree> *> relcells =
                    underghash.get_relevant_cells(plnt.x, plnt.y, 4.99f);
            std::vector< std::vector<basic_tree> *> relcells_canopy =
                    canopyhash.get_relevant_cells(plnt.x, plnt.y, 4.99f);

            std::vector<basic_tree>::iterator thisiter;
            std::vector<basic_tree> *thisvecptr;

            int nrem_ug = 0;
            int nrem_canopy = 0;

            // remove all effects from this plant's interactions with other undergrowth plants
            for (auto &cvecptr : relcells)
            {
                for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
                {
                    auto &oplnt = *piter;
                    if (plnt != oplnt)
                    {
                        int oclidx = clmaps.get_cluster_idx(oplnt.x, oplnt.y);
                        auto &omtx = derivmodel->get_cluster(oclidx).get_locmatrix();
                        HistogramMatrix *obenchmtx = &model.get_cluster(oclidx).get_locmatrix();
                        HistogramDistrib *normd = nullptr;
                        bool removed = mtx.remove_from_bin_undergrowth(plnt, oplnt, width, height,
                                                                       normalize_method::COMPLETE, benchmtx, &rembenefit);
                        bool removed2 = omtx.remove_from_bin_undergrowth(oplnt, plnt, width, height,
                                                                         normalize_method::COMPLETE, obenchmtx, &rembenefit);
                        if (removed)
                        {
                            nrem_ug++;

                            auto rowcol = mtx.get_uspecies_rowcol(plnt.species, oplnt.species);
                            distribscheck[clidx].insert(rowcol);
                        }
                        if (removed2)
                        {
                            nrem_ug++;

                            auto rowcol = omtx.get_uspecies_rowcol(oplnt.species, plnt.species);
                            distribscheck[oclidx].insert(rowcol);
                        }
                    }
                    else
                    {
                        thisiter = piter;
                        thisvecptr = cvecptr;
                    }
                }
            }

            // remove all effects from this plant's interactions with canopy trees
            for (auto &cvecptr : relcells_canopy)
            {
                for (auto piter = cvecptr->begin(); piter != cvecptr->end(); advance(piter, 1))
                {
                    bool removed = mtx.remove_from_bin_canopy(plnt, *piter, width, height,
                                                              normalize_method::COMPLETE, benchmtx, &rembenefit);
                    if (removed)
                    {
                        auto rowcol = mtx.get_cspecies_rowcol(piter->species, plnt.species);
                        distribscheck[clidx].insert(rowcol);
                        nrem_canopy++;
                    }

                }
            }

            float bestbenefit = 0.0f;
            basic_tree newplnt = plnt;
            basic_tree bestplnt = plnt;
            for (int i = 0; i < 5; i++)
            {
                // make the add benefit zero, since we are considering a new candidate plant
                float addbenefit = 0.0f;
                float dir = unif(gen) * 2 * M_PI;
                float dist = unif(gen) * maxpert;
                float dx = dist * cos(dir);
                float dy = dist * sin(dir);
                newplnt.x = plnt.x + dx;
                newplnt.y = plnt.y + dy;
                // candidate plant must be inside landscape
                if (newplnt.x >= width - 0.1f || newplnt.x < 0 || newplnt.y >= height - 0.1f || newplnt.y < 0)
                    continue;
                // only consider the candidate if it's in the same cluster as original plant
                if (clmaps.get_cluster_idx(newplnt.x, newplnt.y) != clidx)
                    continue;

                add_plant_effects_for_restore(newplnt, plnt, underghash, canopyhash, &addbenefit);
                //add_plant_effects(newplnt, plnt, underghash, canopyhash, &addbenefit);

                if (addbenefit + rembenefit > bestbenefit)
                {
                    bestplnt = newplnt;
                    bestbenefit = addbenefit + rembenefit;
                }
                else if (unif(gen) < 0.0f)
                {
                    bestplnt = newplnt;
                    bestbenefit = 1000.0f;
                    restore_distribs();
                    backup_distribs.clear();
                    //remove_plant_effects(newplnt, plnt, underghash, canopyhash, nullptr);
                    break;
                }

                // remove all effects from this plant's interactions with other undergrowth and canopy plants
                //remove_plant_effects(newplnt, plnt, underghash, canopyhash, nullptr);

                // restore distributions to how they were before adding effects of candidate plant
                restore_distribs();
                backup_distribs.clear();

            }

            // if the candidate that contributed most towards bringing the current distributions closer
            // to the benchmark distributions, indeed brought those distributions closer to one another,
            // then replace the original plant with that plant
            if (bestbenefit > 0.0f)
            {
                xy<int> oldgc = underghash.get_gridcoord(plnt.x, plnt.y);
                plnt = bestplnt;
                plnt.r = 50, plnt.g = 100, plnt.b = 150, plnt.a = 200;
                xy<int> newgc = underghash.get_gridcoord(plnt.x, plnt.y);
                if (newgc == oldgc)
                {
                    *thisiter = plnt;
                }
                else
                {
                    thisvecptr->erase(thisiter);
                    auto &cell = underghash.get_cell_direct(newgc.x, newgc.y);
                    cell.push_back(plnt);
                }
                nchanged++;
            }

            // add effects of plant occupying location of previous plant, whether it's
            // the original or a new one
            add_plant_effects(plnt, plnt, underghash, canopyhash, nullptr);

            // Do a check every 'check_every_n' iterations on difference between synthesized
            // distributions and benchmark distribs
            if (plnti % check_every_n == 0 && plnti > 0)
            {
                std::cout << "Done with plant " << plnti << " of " << undergrowth.size() << std::endl;
                float diff = derivmodel->diff_other(&model, distribscheck);
                std::cout << "diff: " << diff << std::endl;
                //ofs << diff << std::endl;
            }

            // Optional callback to report progress to external code, such as a user interface progress bar, for example
            if (progress_callback)
            {
                progress_callback(static_cast<int>(
                                      static_cast<float>(overall_iter * undergrowth.size() + plnti) /
                                      (undergrowth.size() * noverall_iters)
                                      * 100));
            }
        }
        std::cout << "Number of plants changed in overall iteration: "
                  << overall_iter + 1 << ": " << nchanged << std::endl;
        std::cout << "Original model difference: " << backupmodel.diff_other(&model, distribscheck);
        std::cout << "Current model difference: " << derivmodel->diff_other(&model, distribscheck);
    }
    progress_callback(100);

}
