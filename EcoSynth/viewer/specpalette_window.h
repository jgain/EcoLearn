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

#ifndef SPECPALETTE_WINDOW_H
#define SPECPALETTE_WINDOW_H

#include <QWidget>
#include "palette.h"

class specpalette_window : public QWidget
{
    Q_OBJECT

public:
    specpalette_window(QWidget *parent, SpeciesPalette *specpal);
    SpeciesPalette *specpal;
    void add_widget(QWidget *w);
    ~specpalette_window () {std::cerr << "Deleting specpalette window" << std::endl; }

public slots:
    void enable_species(int id);
    void disable_species(int id);
    int idx_to_id(int idx);
    int id_to_idx(int id);
};

#endif // SPECPALETTE_WINDOW_H
