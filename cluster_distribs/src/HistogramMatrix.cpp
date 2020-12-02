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


#include "HistogramMatrix.h"
#include "HistogramDistrib.h"
//#include "MinimalPlant.h"
#include <common/basic_types.h>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <set>


HistogramMatrix::HistogramMatrix(const std::vector<int> &plant_ids, const std::vector<int> &canopy_ids, HistogramDistrib::Metadata under_metadata, HistogramDistrib::Metadata canopy_metadata)
    : plant_ids(plant_ids), nreserved_bins(3), nreal_bins(5), canopy_ids(canopy_ids), canopy_metadata(canopy_metadata),
      under_metadata(under_metadata)
{
    for (int idx = 0; idx < canopy_ids.size(); idx++)
    {
        canopy_id_to_idx.insert({canopy_ids[idx], idx});
    }

    for (int idx = canopy_ids.size(); idx < canopy_ids.size() + plant_ids.size(); idx++)
    {
        plnt_id_to_idx.insert({plant_ids[idx - canopy_ids.size()], idx});
    }

    for (int row = 0; row < canopy_ids.size(); row++)
    {
        active_distribs.push_back(std::vector<bool>());
        distribs.push_back(std::vector<HistogramDistrib>());
        for (int col = 0; col <= row; col++)
        {
            HistogramDistrib::Metadata assign_meta = canopy_metadata;
            active_distribs[row].push_back(false);
            distribs[row].push_back(HistogramDistrib());
            distribs[row].back().init(1e-3f, assign_meta.maxdist, assign_meta.nreal_bins, assign_meta.nreserved_bins);
            // these distributions are empty, they are just placeholders to maintain the triangular matrix
            //distribs[row].back().init(1e-3f, maxdist, nreal_bins, nreserved_bins);
        }
    }

    for (int row = canopy_ids.size(); row < plant_ids.size() + canopy_ids.size(); row++)
    {
        active_distribs.push_back(std::vector<bool>());
        distribs.push_back(std::vector<HistogramDistrib>());
        for (int col = 0; col <= row; col++)
        {
            float maxdist;
            int nreal_bins, nreserved_bins;
            HistogramDistrib::Metadata assign_meta;
            /* check if we have a canopy-undergrowth or undergrowth-undergrowth distribution,
             * and assign appropriate metadata
             */
            if (col < canopy_ids.size())
            {
                // in this case, we have a canopy-undergrowth distribution.
                assign_meta = canopy_metadata;
            }
            else
            {
                // in this case, we have an undergrowth-undergrowth distribution.
                assign_meta = under_metadata;
            }
            active_distribs[row].push_back(false);
            distribs[row].push_back(HistogramDistrib());
            distribs[row].back().init(1e-3f, assign_meta.maxdist, assign_meta.nreal_bins, assign_meta.nreserved_bins);
        }
    }
    validate_distribs();
}

float HistogramMatrix::get_under_maxdist()
{
    return under_metadata.maxdist;
}

float HistogramMatrix::get_canopy_maxdist()
{
    return canopy_metadata.maxdist;
}

bool HistogramMatrix::is_equal(const HistogramMatrix &other) const
{
    if (distribs.size() != other.distribs.size())
        return false;
    if (active_distribs.size() != other.active_distribs.size())
        return false;

    for (int i = 0; i < distribs.size(); i++)
        if (distribs[i].size() != other.distribs[i].size())
            return false;
    for (int i = 0; i < active_distribs.size(); i++)
        if (active_distribs[i].size() != other.active_distribs[i].size())
            return false;

    bool equal = true;
    for (int row = 0; row < distribs.size(); row++)
    {
        for (int col = 0; col < distribs[row].size(); col++)
        {
            if (!distribs[row][col].is_equal(other.distribs[row][col]))
            {
                std::cout << "Distribs not equal at row " << row << ", col " << col << std::endl;
                equal = false;
            }
        }
    }
    return equal;
}

int HistogramMatrix::get_canopy_idx(int id)
{
    return canopy_id_to_idx.at(id);
}

int HistogramMatrix::get_undergrowth_idx(int id)
{
    return plnt_id_to_idx.at(id);
}

HistogramMatrix::HistogramMatrix(const std::vector<int> &plant_ids, const std::vector<int> &canopy_ids, const std::vector< std::vector<bool > > &active_distribs, const std::vector< std::vector< HistogramDistrib > > &matrix)
    : active_distribs(active_distribs), distribs(matrix), canopy_ids(canopy_ids), plant_ids(plant_ids)
{
    for (int idx = 0; idx < canopy_ids.size(); idx++)
    {
        canopy_id_to_idx.insert({canopy_ids[idx], idx});
    }

    for (int idx = canopy_ids.size(); idx < canopy_ids.size() + plant_ids.size(); idx++)
    {
        plnt_id_to_idx.insert({plant_ids[idx - canopy_ids.size()], idx});
    }

    nreserved_bins = matrix.at(0).at(0).getnumreserved();
    nreal_bins = matrix.at(0).at(0).getnumbins();

    validate_distribs();

    for (int rowidx = 0; rowidx < distribs.size(); rowidx++)
    {
        int rowsize = distribs.at(rowidx).size();
        assert(rowsize == rowidx + 1);
        assert(active_distribs.at(rowidx).size() == distribs.at(rowidx).size());
    }
}

void HistogramMatrix::validate_distribs()
{
    bool firstcomp = true;
    HistogramDistrib::Metadata canopymeta;
    HistogramDistrib::Metadata undergrowthmeta;

    for (int &plantid : plant_ids)
    {
        int plntidx = plnt_id_to_idx.at(plantid);
        if (distribs.size() <= plntidx)
        {
            throw std::invalid_argument("distribs in HistogramMatrix does not contain row at index " + std::to_string(plantid));
        }
        if (distribs.at(plntidx).size() != plntidx + 1)
        {
            throw std::invalid_argument("Invalid shape for HistogramMatrix spatial_data. Must be a triangular matrix");
        }

        for (int &canopyid : canopy_ids)
        {
            int canopyidx = canopy_id_to_idx.at(canopyid);
            if (distribs.at(plntidx).size() <= canopyidx)
            {
                throw std::invalid_argument("HistogramMatrix does not contain distrib at row " + std::to_string(plntidx) + ", column " + std::to_string(canopyidx));
            }
            else if (firstcomp)
            {
                canopymeta = distribs.at(plntidx).at(canopyidx).get_metadata();
            }
            else if (distribs.at(plntidx).at(canopyidx).get_metadata() != canopymeta)
            {
                throw std::invalid_argument("HistogramMatrix distribs contain inconsistent metadata");
            }
        }
        for (int &plantid2 : plant_ids)
        {
            int plntidx2 = plnt_id_to_idx.at(plantid2);
            if (plntidx2 > plntidx) continue;		// only consider higher priority species than "plantid" species
            if (distribs.at(plntidx).size() <= plntidx2)
            {
                throw std::invalid_argument("HistogramMatrix does not contain distrib at row " + std::to_string(plntidx) + ", column " + std::to_string(plantid2));
            }
            else if (firstcomp)
            {
                undergrowthmeta = distribs.at(plntidx).at(plntidx2).get_metadata();
                firstcomp = false;
            }
            else if (distribs.at(plntidx).at(plntidx2).get_metadata() != undergrowthmeta)
            {
                throw std::invalid_argument("HistogramMatrix distribs contain inconsistent metadata");
            }
        }
    }

    // if we get to this point, it means all metadata is consistent. We can assign it to the class metadata members
    canopy_metadata = canopymeta;
    under_metadata = undergrowthmeta;
}

void HistogramMatrix::addPlantsCanopy(const basic_tree &ref, const std::vector<basic_tree> &others, 
									  float width, float height, 
									  const HistogramMatrix *benchmatrix)
{
    int nadded = 0;

    int refspec = ref.species;
    float refrad = ref.radius;
    float refx = ref.x;
    float refy = ref.y;
    for (auto &othertree : others)
    {
        float ox = othertree.x, oy = othertree.y;
        float orad = othertree.radius;
        int ospec = othertree.species;

		// the canopy species will always be higher priority than the undergrowth species
        int low = canopy_id_to_idx.at(ospec);		
        int high = plnt_id_to_idx.at(refspec);

        bool added = false;
		// only add to distribution if this relationship is recorded in benchmark distribution
        if (!benchmatrix || benchmatrix->is_active_canopy(ospec, refspec))    
        {
            HistogramDistrib &distrib = distribs.at(high).at(low);
            added = distrib.add_plantdist(ref, othertree, width, height, normalize_method::NONE);
			// if this is an inactive distribution, make it active if plant distance was added to distrib
            if (!active_distribs.at(high).at(low))		
                active_distribs.at(high).at(low) = added;
        }

        if (added)
            nadded++;
    }
}

float HistogramMatrix::canopymtx_diff(const HistogramMatrix &other, int &nactive) const
{
    int ncanopyids = canopy_ids.size();

    float sum = 0.0f;
    //int nactive = 0;
    nactive = 0;

    for (int row = ncanopyids; row < active_distribs.size(); row++)
    {
        for (int col = 0; col <= row && col < ncanopyids; col++)
        {
            if (active_distribs.at(row).at(col) && other.active_distribs.at(row).at(col))
            {
                sum += distribs.at(row).at(col).diff(other.distribs.at(row).at(col));
                nactive++;
            }
        }
    }
    return sum;
}


void HistogramMatrix::addPlantsUndergrowth(const basic_tree &ref, const std::vector<basic_tree> &others, float width,
                                           float height, const HistogramMatrix *benchmtx)
{
    int nadded = 0;
    int refspec = ref.species;
    float refrad = ref.radius;
    float refx = ref.x;
    float refy = ref.y;
    for (const auto &other : others)
    {
        // skip doing analysis on the same plant
        if (other == ref)
        {
            continue;
        }
        float ox = other.x, oy = other.y;
        float orad = other.radius;
        int ospec = other.species;
        // we only add this distance to this matrix if the plant we are looking from, is lower priority (refspec)
        if (isHigherOrEqualPriority(ospec, refspec))
        {
            int low, high;
            low = plnt_id_to_idx.at(ospec);
            high = plnt_id_to_idx.at(refspec);

            // if a synthmatrix, only add to distribution if this relationship is recorded in benchmark distribution
            if (!benchmtx || benchmtx->is_active_undergrowth(ospec, refspec))
            {
                HistogramDistrib &distrib = distribs.at(high).at(low);
                bool added = distrib.add_plantdist(ref, other, width, height, normalize_method::NONE);
				// if this is an inactive distribution, make it active if plant distance was added to distrib
                if (!active_distribs.at(high).at(low))
                    active_distribs.at(high).at(low) = added;
                if (added)
                    nadded++;
            }
        }
    }
}

bool HistogramMatrix::isHigherOrEqualPriority(int left, int right) const
{
    return plnt_id_to_idx.at(left) <= plnt_id_to_idx.at(right);
}

void HistogramMatrix::normalizeAll(normalize_method nmeth)
{
    for (int row = canopy_ids.size(); row < plant_ids.size() + canopy_ids.size(); row++)
    {
        for (int col = 0; col <= row; col++)
        {
            if (active_distribs[row][col])
                distribs[row][col].normalize(nmeth);
        }
    }
}
void HistogramMatrix::unnormalizeAll(normalize_method nmeth)
{
    for (int row = canopy_ids.size(); row < plant_ids.size() + canopy_ids.size(); row++)
    {
        for (int col = 0; col <= row; col++)
        {
            if (active_distribs[row][col])
                distribs[row][col].unnormalize(nmeth);
        }
    }
}


void HistogramMatrix::write_matrix(std::string filename) const
{
    std::ofstream ofs(filename);
    write_matrix(ofs);
}

void HistogramMatrix::write_matrix(std::ofstream &out) const
{
    out << nreserved_bins << " " << nreal_bins << "\n";
    out << distribs.at(0).at(0).getmin() << " " << distribs.at(0).at(0).getmax() << "\n";
    for (auto &id : canopy_ids)
    {
        out << id;
        if (id != canopy_ids.back())
            out << " ";
    }
    out << "\n";
    for (auto &id : plant_ids)
    {
        out << id;
        if (id != plant_ids.back())
            out << " ";
    }
    out << "\n";

    for (int rowidx = 0; rowidx < distribs.size(); rowidx++)
    //for (auto &histrow : distribs)
    {
        for (int histidx = 0; histidx < distribs[rowidx].size(); histidx++)
        //for (auto &hist : histrow)
        {
            for (int i = 0; i < nreserved_bins + nreal_bins; i++)
            {
                const HistogramDistrib &hist = distribs[rowidx][histidx];
                out << hist.getbin(i);
                if (i < nreserved_bins + nreal_bins - 1 || rowidx < distribs.size() - 1 || histidx < distribs[rowidx].size() - 1)
                    out << " ";
            }
        }
    }
    out << "\n";
    for (auto &act_row : active_distribs)
    {
        for (auto act : act_row)
        {
            if (act)
                out << "1";
            else
                out << "0";
        }
    }
    out << "\n";
}

const std::vector<int> &HistogramMatrix::get_plant_ids()
{
    return plant_ids;
}

const std::vector<int> &HistogramMatrix::get_canopy_ids()
{
    return canopy_ids;
}

int HistogramMatrix::get_ntotal_bins() const
{
    return nreserved_bins + nreal_bins;
}

int HistogramMatrix::get_nreal_bins() const
{
    return nreal_bins;
}

int HistogramMatrix::get_nreserved_bins() const
{
    return nreserved_bins;
}

std::vector<std::pair<int, int> > HistogramMatrix::get_zero_active() const
{
    std::vector<std::pair<int, int> > zeroactive;
    for (int row = 0; row < active_distribs.size(); row++)
    {
        for (int col = 0; col < active_distribs.at(row).size(); col++)
        {
            if (active_distribs.at(row).at(col))
            {
                if (distribs.at(row).at(col).bins_eqzero())
                {
                    zeroactive.push_back({row, col});
                }
            }
        }
    }
    return zeroactive;
}

std::vector<std::pair<int, int> > HistogramMatrix::get_noteqone_active() const
{
    std::vector<std::pair<int, int> > noteqone;
    for (int row = 0; row < active_distribs.size(); row++)
    {
        for (int col = 0; col < active_distribs.at(row).size(); col++)
        {
            if (active_distribs.at(row).at(col))
            {
                if (!distribs.at(row).at(col).bins_eqone())
                {
                    noteqone.push_back({row, col});
                }
            }
        }
    }
    return noteqone;
}

HistogramMatrix HistogramMatrix::read_matrix(std::ifstream &ifs) {

    // HELPER LAMBDA FUNCTION
    // get row and column index based on the flattened index of current distribution
    auto get_rowcol = [](int count, int nplant_ids, int &rowidx, int &colidx) {
        for (int totsum = 0, i = 0; i < nplant_ids; ++i, totsum += i) {
            if (count >= totsum && count <= totsum + i) {
                rowidx = i;
                colidx = count - totsum;
                break;
            }
        }
    };

    // TEST LAMBDA FUNCTION:
    // helper function to check that all matrix distributions were indeed assigned
    auto has_zeros = [](std::vector<std::vector<uint8_t> > &tf_mtx) {
        return std::any_of(tf_mtx.begin(), tf_mtx.end(), [](std::vector<uint8_t> &tf_vec) {
            return std::any_of(tf_vec.begin(), tf_vec.end(), [](uint8_t val) {
                return val == 0;
            });
        });
    };

    // read global metadata for all distributions in matrix
    int nreal_bins, nreserved_bins;
    int ntot_bins;
    common_types::decimal min, max;
    std::vector<int> canopy_ids;
    std::vector<int> plant_ids;
    ifs >> nreserved_bins;
    ifs >> nreal_bins;
    ntot_bins = nreal_bins + nreserved_bins;
    ifs >> min >> max;
    ifs.ignore();

    if (!ifs.good())
        throw std::runtime_error("input filestream invalid for histogram read");

    std::string line;
    std::getline(ifs, line);

    // read canopy ids
    std::stringstream sstr(line);
    std::string chstr;
    while (sstr.good())
    {
        chstr.clear();
        std::getline(sstr, chstr, ' ');
        if (chstr.size() == 0)
            continue;
        try
        {
            canopy_ids.push_back(std::stoi(chstr));
        }
        catch (std::invalid_argument &e)
        {
            continue;
        }
    }
    sstr.str("");
    sstr.clear();

    // read plant ids
    line.clear();
    std::getline(ifs, line);
    sstr.str(line);
    while (sstr.good())
    {
        chstr.clear();
        std::getline(sstr, chstr, ' ');
        if (chstr.size() == 0)
            continue;
        try
        {
            plant_ids.push_back(std::stoi(chstr));
        }
        catch (std::invalid_argument &e)
        {
            continue;
        }
    }

    std::vector< std::vector< uint8_t > > assigned(canopy_ids.size() + plant_ids.size());

    // read bin values for each distribution
	// -------------------------------------------------
    // prepare distributions data structures
    std::vector< std::vector<HistogramDistrib > > distribs(canopy_ids.size() + plant_ids.size());
    for (int row_idx = 0; row_idx < distribs.size(); row_idx++)
    {
        distribs[row_idx].resize(row_idx + 1);
        assigned.at(row_idx).resize(row_idx + 1, 0);
    }

    line.clear();
    std::getline(ifs, line);
    sstr.clear();
    sstr.str(line);

    int hist_count = 0;
    std::vector<common_types::decimal> binvals;
    int count = 0;
    chstr.clear();
	// read bin values from stringstream
    while (sstr.good())
    {
        std::getline(sstr, chstr, ' ');
        try
        {
            binvals.push_back(std::stof(chstr));
        }
        catch (std::invalid_argument &e)
        {
            continue;
        }
        if (binvals.size() % ntot_bins == 0 && binvals.size() > 0)
        {
            assert(binvals.size() == ntot_bins);
            int rowidx = -1, colidx = -1;
            get_rowcol(count, canopy_ids.size() + plant_ids.size(), rowidx, colidx);
            assert(colidx != -1 && rowidx != -1);
            distribs.at(rowidx).at(colidx) = HistogramDistrib(nreal_bins, nreserved_bins, min, max, binvals);
            assigned.at(rowidx).at(colidx) = 1;
            binvals.clear();
            count++;
        }
    }

    assert(!has_zeros(assigned));

    // read indicators for active distributions
	// -------------------------------------------------------
    // prepare active distributions data structure
    std::vector< std::vector< bool > > active_distribs(distribs.size());
    for (int i = 0; i < distribs.size(); i++)
    {
        active_distribs.at(i).resize(distribs.at(i).size());
    }

    for (auto &bitvec : assigned)
    {
        std::fill(bitvec.begin(), bitvec.end(), 0);
    }

    char ch = ' ';
    count = 0;
	// read indicators from ifstream
    while (ch != '\n')
    {
        ifs.get(ch);
        if (ch == '\n' || ch == ' ')
            break;
        int rowidx = -1, colidx = -1;
        get_rowcol(count, canopy_ids.size() + plant_ids.size(), rowidx, colidx);
        assert(colidx != -1 && rowidx != -1);

        if (!isprint(ch))
            continue;

        if (ch == '1')
        {
            active_distribs.at(rowidx).at(colidx) = true;
        }
        else if (ch == '0')
        {
            active_distribs.at(rowidx).at(colidx) = false;
        }
        else
        {
            assert(false);
        }
        assigned.at(rowidx).at(colidx) = 1;

        count++;
    }

    assert(!has_zeros(assigned));

    return HistogramMatrix(plant_ids, canopy_ids, active_distribs, distribs);
}

HistogramDistrib &HistogramMatrix::get_distrib_canopy(int canopyspec, int underspec)
{
    int low = canopy_id_to_idx.at(canopyspec);
    int high = plnt_id_to_idx.at(underspec);
    assert(low < high);		// canopy species must be strictly higher priority than understorey species
    return distribs.at(high).at(low);
}

HistogramDistrib &HistogramMatrix::get_distrib_rowcol(int row, int col)
{
    return distribs.at(row).at(col);
}

const HistogramDistrib &HistogramMatrix::get_distrib(int idx) const
{
    int count = 0;
    for (const auto &row : distribs)
    {
        for (const auto &distrib : row)
        {
            if (idx == count)
                return distrib;
            count++;
        }
    }
    throw std::invalid_argument("In HistogramMatrix::get_distrib(int idx), distribution does not exist at cluster idx");
}

int HistogramMatrix::get_distrib_canopy_flatidx(int canopyspec, int underspec) const
{
    int past_canopies = canopy_ids.size() * (canopy_ids.size() + 1) / 2;
    int uidx = plnt_id_to_idx.at(underspec);
    int rowidx = (uidx) * (uidx + 1) / 2;
    int cidx = canopy_id_to_idx.at(canopyspec);
    return rowidx + cidx;
}

int HistogramMatrix::get_distrib_under_flatidx(int spec1, int spec2)
{
    int s1idx = plnt_id_to_idx.at(spec1);
    int s2idx = plnt_id_to_idx.at(spec2);
    int low = s1idx < s2idx ? s1idx : s2idx;
    int high = s2idx < s1idx ? s1idx : s2idx;
    int rowidx = high * (high + 1) / 2;
    return rowidx + low;
}

int HistogramMatrix::get_binwidth() const
{
    return distribs.at(0).at(0).get_binwidth();
}

bool HistogramMatrix::is_active_canopy(int canopyspec, int underspec) const
{
    int low = canopy_id_to_idx.at(canopyspec);
    int high = plnt_id_to_idx.at(underspec);
    assert(low < high);		// canopy species must be strictly higher priority than understorey species
    return active_distribs.at(high).at(low);
}

std::vector<int> HistogramMatrix::get_underid_to_idx() const
{
    std::vector<int> toidx(plnt_id_to_idx.size(), -1);
    for (auto &pair : plnt_id_to_idx)
    {
        if (toidx.size() <= pair.first)
        {
            toidx.resize(pair.first + 1, -1);
        }
        toidx.at(pair.first) = pair.second;
    }
    return toidx;
}

bool HistogramMatrix::has_active_distribs() const
{
    for (auto &row : active_distribs)
    {
        for (bool col : row)
        {
            if (col)
                return true;
        }
    }
    return false;
}

const std::vector<std::vector<HistogramDistrib> > &HistogramMatrix::get_distribs()
{
    return distribs;
}

std::vector<int> HistogramMatrix::get_canopyid_to_idx() const
{
    std::vector<int> toidx(canopy_id_to_idx.size(), -1);
    for (auto &pair : canopy_id_to_idx)
    {
        if (toidx.size() <= pair.first)
        {
            toidx.resize(pair.first + 1, -1);
        }
        toidx.at(pair.first) = pair.second;
    }
    return toidx;

}

HistogramDistrib &HistogramMatrix::get_distrib_undergrowth(int spec1, int spec2) {
    int low, high;
    if (isHigherOrEqualPriority(spec1, spec2))
    {
        low = plnt_id_to_idx[spec1], high = plnt_id_to_idx[spec2];
    }
    else
    {
        low = plnt_id_to_idx[spec2], high = plnt_id_to_idx[spec1];
    }
    assert(low <= high);
    return distribs.at(high).at(low);
}

std::pair<int, int> HistogramMatrix::get_uspecies_rowcol(int spec1, int spec2) const
{
    int low, high;
    if (isHigherOrEqualPriority(spec1, spec2))
    {
        low = plnt_id_to_idx.at(spec1), high = plnt_id_to_idx.at(spec2);
    }
    else
    {
        low = plnt_id_to_idx.at(spec2), high = plnt_id_to_idx.at(spec1);
    }
    assert(low <= high);
    return {high, low};
}

std::pair<int, int> HistogramMatrix::get_cspecies_rowcol(int canopyspec, int underspec) const
{
    int low = canopy_id_to_idx.at(canopyspec);
    int high = plnt_id_to_idx.at(underspec);
    assert(low < high);		// canopy species must be strictly higher priority than understorey species
    return {high, low};
}

void HistogramMatrix::set_low_high(int spec1, int spec2, int &low, int &high) const
{
    if (isHigherOrEqualPriority(spec1, spec2))
    {
        low = spec1, high = spec2;
    }
    else
    {
        low = spec2, high = spec1;
    }
    assert(low <= high);
}

bool HistogramMatrix::is_active_undergrowth(int spec1, int spec2) const
{
    int low, high;
    if (isHigherOrEqualPriority(spec1, spec2))
    {
        low = plnt_id_to_idx.at(spec1), high = plnt_id_to_idx.at(spec2);		// high index = lower priority and vice versa
    }
    else
    {
        low = plnt_id_to_idx.at(spec2), high = plnt_id_to_idx.at(spec1);
    }
    assert(low <= high);
    return active_distribs.at(high).at(low);
}

void HistogramMatrix::set_active_undergrowth(int spec1, int spec2)
{
    int low, high;
    if (isHigherOrEqualPriority(spec1, spec2))
    {
        low = plnt_id_to_idx[spec1], high = plnt_id_to_idx[spec2];				// high index = lower priority and vice versa
    }
    else
    {
        low = plnt_id_to_idx[spec2], high = plnt_id_to_idx[spec1];
    }
    assert(low <= high);
    active_distribs.at(high).at(low) = true;
}

void HistogramMatrix::set_active_canopy(int canopyspec, int underspec)
{
    int canopyidx = canopy_id_to_idx.at(canopyspec);
    int underidx = plnt_id_to_idx.at(underspec);
    assert(canopyidx < underidx);
    active_distribs.at(underidx).at(canopyidx) = true;
}

bool HistogramMatrix::remove_from_bin_undergrowth(const basic_tree &refplnt, const basic_tree &oplnt,
                                                  float width, float height, 
												  normalize_method nmeth, HistogramMatrix *benchmtx, float *benefit)
{

    // if this refplnt's species is higher priority than oplnt's, then skip
    if (refplnt.species != oplnt.species && isHigherOrEqualPriority(refplnt.species, oplnt.species))
    {
        return false;
    }

    // if the distance between these plants' discs is greater than the maximum distance or this distribution is 
	// inactive in the benchmark matrix, then skip
    if (refplnt.discdist(oplnt) > get_under_maxdist() || (benchmtx && !benchmtx->is_active_undergrowth(refplnt.species, oplnt.species)))
    {
        return false;
    }

    // since we are removing a distance, it cannot be that this distribution is inactive (i.e. a distance has never been added to it).
    // It's therefore a bug - report it here.
    if (!is_active_undergrowth(refplnt.species, oplnt.species))
    {
        std::cout << "Error for species " << refplnt.species << ", " << oplnt.species << std::endl;
        std::cout << "refplnt colors: " << refplnt.r << ", " << refplnt.g << ", " << refplnt.b << ", " << refplnt.a << std::endl;
        std::cout << "oplnt colors: " << oplnt.r << ", " << oplnt.g << ", " << oplnt.b << ", " << oplnt.a << std::endl;
        throw std::runtime_error("Cannot remove from inactive distribution");
        return false;
    }

    // get the undergrowth-undergrowth distribution between this reference plant's species
    // and the other plant's species
    auto &distrib = get_distrib_undergrowth(refplnt.species, oplnt.species);

    // if a benchmark matrix has been specified and the benefit pointer also, then add the benefit of
    // removing this plant

    // get difference without removing...  (if benchmtx and benefit specified)
    HistogramDistrib *benchdistrib = nullptr;
    float diff1, diff2;
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_undergrowth(refplnt.species, oplnt.species);
        diff1 = benchdistrib->diff(distrib);
    }

    // remove plant effect...
    bool success = distrib.remove_plantdist(refplnt, oplnt, width, height, nmeth);

    // get difference after removing plant effect and compute benefit
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_undergrowth(refplnt.species, oplnt.species);
        diff2 = benchdistrib->diff(distrib);
        *benefit += diff1 - diff2;
    }

    return success;
}

bool HistogramMatrix::add_to_bin_undergrowth(const basic_tree &refplnt, const basic_tree &oplnt,
                                             float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx, 
											 float *benefit, std::vector<common_types::decimal> *bins)
{
    // if this refplnt's species is higher priority than oplnt's, then skip

	
    // if this refplnt's species is higher priority than oplnt's, then skip
    if (refplnt.species != oplnt.species && isHigherOrEqualPriority(refplnt.species, oplnt.species))
        return false;

    // if the distance between these plants' discs is greater than the maximum distance or this distribution is inactive in the benchmark
    // matrix, then skip
    if (refplnt.discdist(oplnt) > get_under_maxdist() || (benchmtx && !benchmtx->is_active_undergrowth(refplnt.species, oplnt.species)))
        return false;

	// if not active yet, then make active
    if (!is_active_undergrowth(refplnt.species, oplnt.species))
    {
        set_active_undergrowth(refplnt.species, oplnt.species);
    }

    // get the undergrowth-undergrowth distribution between this reference plant's species
    // and the other plant's species
    auto &distrib = get_distrib_undergrowth(refplnt.species, oplnt.species);

    // if a benchmark matrix has been specified and the benefit pointer also, then compute the benefit of
    // adding this plant
	// ------------------------------

    // get difference without adding...  (if benchmtx and benefit specified)
    HistogramDistrib *benchdistrib = nullptr;
    float diff1, diff2;
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_undergrowth(refplnt.species, oplnt.species);
        diff1 = benchdistrib->diff(distrib);
    }

    if (bins)
        *bins = distrib.getbins();

	// add plant effect...
    bool success = distrib.add_plantdist(refplnt, oplnt, width, height, nmeth);

    // get difference after adding plant effect and compute benefit
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_undergrowth(refplnt.species, oplnt.species);
        diff2 = benchdistrib->diff(distrib);
        *benefit += diff1 - diff2;
    }

    return success;
}

bool HistogramMatrix::remove_from_bin_canopy(const basic_tree &uplant,
                                             const basic_tree &cplant,
                                             float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx, float *benefit)
{
    // if the distance between these plants' discs is greater than the maximum distance or this distribution is inactive in the benchmark
    // matrix, then skip
    if (uplant.discdist(cplant) > get_canopy_maxdist() || (benchmtx && !benchmtx->is_active_canopy(cplant.species, uplant.species)))
        return false;

	// if not active yet, then make active
    if (!is_active_canopy(cplant.species, uplant.species))
    {
        throw std::runtime_error("Cannot remove from inactive distribution");
        return false;
    }

    // get the undergrowth-canopytree distribution between this reference plant's species
    // and the tree's species
    auto &distrib = get_distrib_canopy(cplant.species, uplant.species);

    // if a benchmark matrix has been specified and the benefit pointer also, then compute the benefit of
    // removing this plant
	// ------------------------------

    // get difference without removing...  (if benchmtx and benefit specified)
    HistogramDistrib *benchdistrib = nullptr;
    float diff1, diff2;
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_canopy(cplant.species, uplant.species);
        diff1 = benchdistrib->diff(distrib);
    }

	// remove plant effect...
    bool success = distrib.remove_plantdist(uplant, cplant, width, height, nmeth);

    // get difference after removing plant effect and compute benefit
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_canopy(cplant.species, uplant.species);
        diff2 = benchdistrib->diff(distrib);
        *benefit += diff1 - diff2;
    }
    return success;
}

bool HistogramMatrix::add_to_bin_canopy(const basic_tree &uplant, const basic_tree &cplant,
                                           float width, float height, normalize_method nmeth, HistogramMatrix *benchmtx, float *benefit, std::vector<common_types::decimal> *bins)
{
    // if the distance between these plants' discs is greater than the maximum distance or this distribution is inactive in the benchmark
    // matrix, then skip
    if (uplant.discdist(cplant) > get_canopy_maxdist() || (benchmtx && !benchmtx->is_active_canopy(cplant.species, uplant.species)))
        return false;

	// if not active yet, then make active
    if (!is_active_canopy(cplant.species, uplant.species))
    {
        set_active_canopy(cplant.species, uplant.species);
    }

    // get the undergrowth-canopytree distribution between this reference plant's species
    // and the tree's species
    auto &distrib = get_distrib_canopy(cplant.species, uplant.species);

    // if a benchmark matrix has been specified and the benefit pointer also, then compute the benefit of
    // adding this plant
	// ------------------------------

    // get difference without adding...  (if benchmtx and benefit specified)
    HistogramDistrib *benchdistrib = nullptr;
    float diff1, diff2;
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_canopy(cplant.species, uplant.species);
        diff1 = benchdistrib->diff(distrib);
    }

    if (bins)
        *bins = distrib.getbins();

	// add plant effect...
    bool success = distrib.add_plantdist(uplant, cplant, width, height, nmeth);

    // get difference after adding plant effect and compute benefit
    if (benchmtx && benefit)
    {
        benchdistrib = &benchmtx->get_distrib_canopy(cplant.species, uplant.species);
        diff2 = benchdistrib->diff(distrib);
        *benefit += diff1 - diff2;
    }

    return success;
}

float HistogramMatrix::sum() const
{
    float sum = 0.0f;
    for (int row = 0; row < distribs.size(); row++)
    {
        for (int col = 0; col < distribs.at(row).size(); col++)
        {
            float d = distribs.at(row).at(col).sumbins();
            sum += d;
        }
    }
    return sum;
}

float HistogramMatrix::diff(const HistogramMatrix &other, const std::set<std::pair<int, int> > &rowcol_check) const
{
    // TODO: make this check on dimensions optional
    if (distribs.size() != other.distribs.size())
        throw std::runtime_error("Histogram matrices must be of the same dimension when doing differencing");
    else
    {
        for (int i = 0; i < distribs.size(); i++)
        {
            if (distribs.at(i).size() != other.distribs.at(i).size())
            {
                throw std::runtime_error("Histogram matrices must be of the same dimension when doing differencing");
            }
        }
    }

    float totsum = 0.0f;
    for (const auto &rowcol : rowcol_check)
    {
        float d = distribs.at(rowcol.first).at(rowcol.second).diff(other.distribs.at(rowcol.first).at(rowcol.second));
        totsum += d;

    }
    return totsum;
}

float HistogramMatrix::diff(const HistogramMatrix &other, float showdiff) const
{
    // TODO: make this check on dimensions optional
    if (distribs.size() != other.distribs.size())
        throw std::runtime_error("Histogram matrices must be of the same dimension when doing differencing");
    else
    {
        for (int i = 0; i < distribs.size(); i++)
        {
            if (distribs.at(i).size() != other.distribs.at(i).size())
            {
                throw std::runtime_error("Histogram matrices must be of the same dimension when doing differencing");
            }
        }
    }

    float totsum = 0.0f;
    for (int row = 0; row < distribs.size(); row++)
    {
        for (int col = 0; col < distribs.at(row).size(); col++)
        {
            float d = distribs.at(row).at(col).diff(other.distribs.at(row).at(col));

			// if the debug parameter, 'showdiff', is true, then output info on the distributions we are differencing
            if (showdiff && d > 1e-2f)
            {
                std::cout << "diff detected in row, col: " << row << ", " << col << std::endl;
                const auto &b1 = distribs.at(row).at(col).getbins();
                const auto &b2 = other.distribs.at(row).at(col).getbins();
                std::cout << "Bins: " << std::endl;
                for (int i = 0; i < b1.size(); i++)
                {
                    std::cout << b1.at(i) << " ";
                }
                std::cout << std::endl;
                for (int i = 0; i < b2.size(); i++)
                {
                    std::cout << b2.at(i) << " ";
                }
                std::cout << std::endl;
                int row_spec, col_spec;
                if (row >= canopy_ids.size())
                {
                    row_spec = std::find_if(plnt_id_to_idx.begin(),
                                 plnt_id_to_idx.end(),
                                 [&row](const typename std::unordered_map<int, int>::value_type &vt)
                    {
                        return row == vt.second;
                    })->first;
                    std::cout << "row species is undergrowth" << std::endl;
                }
                else
                {
                    row_spec = std::find_if(canopy_id_to_idx.begin(),
                                 canopy_id_to_idx.end(),
                                 [&row](const typename std::unordered_map<int, int>::value_type &vt)
                    {
                        return row == vt.second;
                    })->first;
                    std::cout << "row species is canopy" << std::endl;
                }
                if (col >= canopy_ids.size())
                {
                    col_spec = std::find_if(plnt_id_to_idx.begin(),
                                 plnt_id_to_idx.end(),
                                 [&col](const typename std::unordered_map<int, int>::value_type &vt)
                    {
                        return col == vt.second;
                    })->first;
                    std::cout << "col species is undergrowth" << std::endl;
                }
                else
                {
                    col_spec = std::find_if(canopy_id_to_idx.begin(),
                                 canopy_id_to_idx.end(),
                                 [&col](const typename std::unordered_map<int, int>::value_type &vt)
                    {
                        return col == vt.second;
                    })->first;
                    std::cout << "col species is canopy" << std::endl;
                }
                std::cout << "Row species, col species: " << row_spec << ", " << col_spec << std::endl;
                std::cout << "------------------" << std::endl;

            }
            totsum += d;
        }
    }
    return totsum;
}

void HistogramMatrix::zero_out_histograms()
{
    for (int rowi = 0; rowi < active_distribs.size(); rowi++)
    {
        for (int coli = 0; coli < active_distribs.at(rowi).size(); coli++)
        {
            if (active_distribs.at(rowi).at(coli))
            {
                distribs.at(rowi).at(coli).reset();
            }
        }
    }
}

float HistogramMatrix::get_bin_value_undergrowth(float refrad, float orad, int spec1, int spec2, float sep,
                                                 float raw_distance) {
    if (!is_active_undergrowth(spec1, spec2))
        return -1;
    auto &distrib = get_distrib_undergrowth(spec1, spec2);
    return distrib.getbin(distrib.maptobin_raw(refrad, orad, sep, raw_distance));
}


float HistogramMatrix::get_bin_value_canopy(float refrad, float canopyrad, int refspec, int canopyspec, float sep,
                                            float raw_distance) {
    if (!is_active_canopy(canopyspec, refspec))
    {
        return -1;
    }
    auto &distrib = get_distrib_canopy(canopyspec, refspec);
    return distrib.getbin(distrib.maptobin_raw(refrad, canopyrad, sep, raw_distance));
}

