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

#include "convertpaintingdialog.h"
#include "ui_convertpaintingdialog.h"

ConvertPaintingDialog::ConvertPaintingDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ConvertPaintingDialog)
{
    ui->setupUi(this);
}

void ConvertPaintingDialog::get_values(int &from, int &to)
{
    std::string fromstr = ui->comboBoxFrom->currentText().toStdString();
    std::string tostr = ui->comboBoxTo->currentText().toStdString();

    from = -1, to = -1;

    if (fromstr == "Void")
        from = 0;
    else if (fromstr == "Sparse")
        from = 1;
    else if (fromstr == "Dense")
        from = 2;

    if (tostr == "Void")
        to = 0;
    else if (tostr == "Sparse")
        to = 1;
    else if (tostr == "Dense")
        to = 2;
}

ConvertPaintingDialog::~ConvertPaintingDialog()
{
    delete ui;
}
