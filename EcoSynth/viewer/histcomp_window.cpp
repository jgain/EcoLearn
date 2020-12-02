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

#include "histcomp_window.h"

#include <iostream>
#include <sstream>

#include <QMouseEvent>
#include <QRect>

histcomp_window::histcomp_window(CompType ctype)
    : QLabel(), clicklabel(new QLabel(this, Qt::Window)), curr_selected_id(-1), ctype(ctype)
{
    clicklabel->setWindowFlag(Qt::ToolTip);
    clicklabel->setHidden(true);
}

void histcomp_window::mouseReleaseEvent(QMouseEvent *ev)
{
    std::cout << "Mouserelease event in histcomp_window" << std::endl;

    QPoint pt(ev->localPos().x(), ev->localPos().y());

    std::string labeltext;
    std::stringstream ss;

    int selected_id;
    int idx = 0;
    for (histcomp_window::histinfo &h : hists)
    {
        if (h.loc_coords.contains(pt))
        {
            selected_id = idx;
            if (curr_selected_id == selected_id)
            {
                selected_id = -1;
                curr_selected_id = -1;
                break;
            }
            else
            {
                if (ctype == CompType::SIZE)
                {
                    ss << "Species: " << h.spec1.id << "\nNumber synthesized: " << h.spec1.nsynth << "\n";
                    curr_selected_id = selected_id;
                }
                else
                {
                    std::string understr = " (undergrowth";
                    std::string canopystr = " (canopy";
                    if (ctype == CompType::UNDERUNDER)
                    {
                        understr = "";
                        canopystr = "";
                    }
                    ss << "Species" << understr << ": " << h.spec1.id << "\nNumber synthesized: " << h.spec1.nsynth << "\n";
                    if (h.spec2.id >= 0 && h.spec2.nsynth >= 0)
                    {
                        ss << "Species" << canopystr <<  ": " << h.spec2.id << "\nNumber synthesized: " << h.spec2.nsynth << "\n";
                    }
                    ss << "Number of elements: " << h.refcount << "\n";
                    //std::cout << p.second.nsynth << " synthesized for species " << p.first << std::endl;
                    curr_selected_id = selected_id;
                    //std::cout << "Appending to string" << std::endl;
                }
            }
            //std::cout << "Click inside species rect" << std::endl;
        }
        else
        {
            //std::cout << "Click outside species rect " << r.x() << ", " << r.y() << ", " << r.width() << ", " << r.height() << std::endl;
        }
        idx++;
    }
    labeltext = ss.str();
    if (curr_selected_id == -1)
    {
        clicklabel->setHidden(true);
    }
    else
    {
        clicklabel->move(ev->globalX(), ev->globalY());
        clicklabel->setText(labeltext.c_str());
        clicklabel->show();
    }
}

/*
void histcomp_window::mouseReleaseEvent(QMouseEvent *ev)
{
    std::cout << "Mouserelease event in histcomp_window" << std::endl;

    QPoint pt(ev->localPos().x(), ev->localPos().y());

    std::string labeltext;
    std::stringstream ss(labeltext);

    int selected_id;
    for (auto &p : speciesinfo)
    {
        if (p.second.loc_coords.contains(pt))
        {
            selected_id = p.first;
            if (curr_selected_id == selected_id)
            {
                selected_id = -1;
                curr_selected_id = -1;
                break;
            }
            else
            {
                ss << "Species: " << curr_selected_id << "\nNumber synthesized: " << p.second.nsynth << "\n";
                std::cout << p.second.nsynth << " synthesized for species " << p.first << std::endl;
                curr_selected_id = selected_id;
            }
            //std::cout << "Click inside species rect" << std::endl;
        }
        else
        {
            QRect &r = p.second.loc_coords;
            //std::cout << "Click outside species rect " << r.x() << ", " << r.y() << ", " << r.width() << ", " << r.height() << std::endl;
        }
    }
    if (curr_selected_id == -1)
    {
        clicklabel->setHidden(true);
    }
    else
    {
        clicklabel->move(ev->globalX(), ev->globalY());
        clicklabel->setText(labeltext.c_str());
        clicklabel->show();
    }
}
*/

void histcomp_window::set_species_info(int id, int nsynth, const QRect &loc_coords)
{
    if (!speciesinfo.count(id))
    {
        speciesinfo[id] = {id, nsynth, {loc_coords} };
    }
    else
    {
        speciesinfo.at(id).loc_coords.push_back(loc_coords);
    }
}

void histcomp_window::set_hist_info(int id1, int nsynth1, int id2, int nsynth2, int refcount, QRect loc_coords)
{
    histinfo h = { {id1, nsynth1}, {id2, nsynth2}, refcount, loc_coords };
    hists.push_back(h);
}

void histcomp_window::set_hist_info(int id, int nsynth, int refcount, QRect loc_coords)
{
    histinfo h = { {id, nsynth}, {-1, -1}, refcount, loc_coords };
    hists.push_back(h);
}

void histcomp_window::closeEvent(QCloseEvent *event)
{
    if (clicklabel->isVisible())
    {
        clicklabel->setHidden(true);
    }
    QLabel::closeEvent(event);

    hists.clear();
    speciesinfo.clear();
}
