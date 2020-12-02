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

#include "specpalette_window.h"
#include "palette.h"
#include <QVBoxLayout>

specpalette_window::specpalette_window(QWidget *parent, SpeciesPalette *specpal)
    : QWidget(parent, Qt::Window), specpal(specpal)
{
    QVBoxLayout *specpal_layout = new QVBoxLayout;
    specpal_layout->addWidget(specpal);
    show();
    setLayout(specpal_layout);
}

void specpalette_window::add_widget(QWidget *w)
{
    layout()->addWidget(w);
}

void specpalette_window::enable_species(int id)
{
    specpal->enable_brush(id);
}

void specpalette_window::disable_species(int id)
{
    specpal->disable_brush(id);
}

int specpalette_window::id_to_idx(int id)
{
    return specpal->id_to_idx.at(id);
}

int specpalette_window::idx_to_id(int idx)
{
    return specpal->idx_to_id.at(idx);
}
