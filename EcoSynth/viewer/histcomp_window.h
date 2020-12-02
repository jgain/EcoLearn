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

#ifndef HISTCOMP_WINDOW
#define HISTCOMP_WINDOW

#include <QLabel>
#include <QRect>

#include <unordered_map>

class histcomp_window : public QLabel
{
private:
    struct {
        int id;
        int nsynth;
        std::vector<QRect> loc_coords;

        //typedef typeof(*this) thistype;

        template<typename T>
        bool operator < (const T &other) { return this->id < other.id; }
    } spinfo;

    typedef struct {
        int id, nsynth;
    } specinfo;

    typedef struct {
        specinfo spec1, spec2;
        int refcount;
        QRect loc_coords;
    } histinfo;
public:
    enum class CompType
    {
        SIZE,
        CANOPYUNDER,
        UNDERUNDER
    };
public:
    histcomp_window(CompType ctype);
    void mouseReleaseEvent(QMouseEvent *ev) override;
    void set_species_info(int id, int nsynth, const QRect &loc_coords);
    void set_hist_info(int id1, int nsynth1, int id2, int nsynth2, int refcount, QRect loc_coords);
    void set_hist_info(int id, int nsynth, int refcount, QRect loc_coords);
protected:
    void closeEvent(QCloseEvent *event) override;
private:
    QLabel *clicklabel;
    std::unordered_map<int, typeof(spinfo)> speciesinfo;
    std::vector<histinfo> hists;
    int curr_selected_id;

    CompType ctype;
};

#endif // HISTCOMP_WINDOW
