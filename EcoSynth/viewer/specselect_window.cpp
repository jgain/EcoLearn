
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

#include "specselect_window.h"
#include "window.h"
#include <QVBoxLayout>

specselect_window::specselect_window(data_importer::common_data cdata, Window *parent)
    : QWidget(parent, Qt::Window)
{
    auto allspecs = cdata.all_species;

    QVBoxLayout *lyout = new QVBoxLayout();
    for (auto &p : allspecs)
    {
        int specid = p.first;
        cboxes[specid] = new QCheckBox(QString::fromStdString(std::to_string(specid)));
        cboxes[specid]->setChecked(true);
        connect(cboxes[specid], &QCheckBox::stateChanged, this, &specselect_window::statechanged);
        lyout->addWidget(cboxes[specid]);
    }

    this->setLayout(lyout);

    connect(this, &specselect_window::species_added, parent, &Window::species_added);
    connect(this, &specselect_window::species_removed, parent, &Window::species_removed);
}

specselect_window::specselect_window(string dbname, Window *parent)
    : specselect_window(data_importer::common_data(dbname), parent)
{

}

void specselect_window::add_widget(QWidget *w)
{
    this->layout()->addWidget(w);
}

void specselect_window::statechanged(int state)
{
    QCheckBox *sender = dynamic_cast<QCheckBox *>(QObject::sender());
    assert(sender);
    Qt::CheckState chst = (Qt::CheckState)state;
    if (chst == Qt::Checked)
    {
        for (auto &p : cboxes)
        {
            int id = p.first;
            if (p.second == sender)
            {
                species_added(id);
                return;
            }
        }
    }
    else if (chst == Qt::Unchecked)
    {
        for (auto &p : cboxes)
        {
            int id = p.first;
            if (p.second == sender)
            {
                species_removed(id);
                return;
            }
        }
    }
    else
    {
        assert(false);
    }
}

void specselect_window::disable()
{
    for (auto &p : cboxes)
    {
        QCheckBox *b = p.second;
        b->setEnabled(false);
    }
}

void specselect_window::enable()
{
    for (auto &p : cboxes)
    {
        QCheckBox *b = p.second;
        b->setEnabled(true);
    }

}
