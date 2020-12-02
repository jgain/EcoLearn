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

#include "speciesColoursWindow.h"

#include <QLabel>
#include <QPixmap>
#include <QGridLayout>

SpeciesColoursWindow::SpeciesColoursWindow(QWidget *parent, const data_importer::common_data &cdata)
    : QWidget(parent, Qt::Window)
{
    QGridLayout *gridLayout = new QGridLayout;

    int rowidx = 0;
    for (auto &specpair : cdata.canopy_and_under_species)
    {
        const data_importer::species &spec = specpair.second;
        int id = specpair.first;
        std::string name = spec.name;
        float r = spec.basecol[0];
        float g = spec.basecol[1];
        float b = spec.basecol[2];

        QLabel *idlabel = new QLabel(std::to_string(id).c_str());
        QLabel *namelabel = new QLabel(name.c_str());
        QLabel *colourLabel = new QLabel;

        QPixmap colimage(30, 30);
        colimage.fill(QColor(r * 255, g * 255, b * 255));

        colourLabel->setPixmap(colimage);

        gridLayout->addWidget(idlabel, rowidx, 0);
        gridLayout->addWidget(namelabel, rowidx, 1);
        gridLayout->addWidget(colourLabel, rowidx, 2);

        rowidx++;

    }

    setLayout(gridLayout);
}

void SpeciesColoursWindow::display()
{
    this->show();
}

void SpeciesColoursWindow::hide()
{
    this->hide();
}
