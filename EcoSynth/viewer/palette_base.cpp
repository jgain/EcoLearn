/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020  J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp (konrad.p.kapp@gmail.com)
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

#include "palette_base.h"
#include "glwidget.h"

#include <QDir>
#include <QGridLayout>


palette_base::palette_base(TypeMap *typemap, const std::vector<int> &btypes, QWidget *parent)
    : QWidget(parent), /*nentries((int)*std::max_element(btypes.begin(), btypes.end()) + 1)*/ nentries(btypes.size()), selector(new QPushButton * [btypes.size()]),
      typeSel(new BrushType [btypes.size()])
{
    glparent = (GLWidget *) parent;

    setAttribute(Qt::WA_StaticContents);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

    tmap = typemap;

    addBrushes(btypes);
}

palette_base::palette_base(TypeMap *typemap, int nentries, QWidget *parent)
    : QWidget(parent), nentries(nentries), selector(new QPushButton * [nentries]),
      typeSel(new BrushType [nentries])
{

    QDir basedir = QString(SRC_BASEDIR);

    glparent = (GLWidget *) parent;

    setAttribute(Qt::WA_StaticContents);
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

    tmap = typemap;

    //if(!activeImg.load(QCoreApplication::applicationDirPath() + "/../../viewer/Icons/activeIcon.png"))
    //    cerr << QCoreApplication::applicationDirPath().toUtf8().constData() << "/../../viewer/Icons/activeIcon.png" << " not found" << endl;
    if(!activeImg.load(basedir.filePath("Icons/activeIcon.png")))
        cerr << basedir.filePath("Icons/activeIcon.png").toStdString() << " not found" << endl;

    // create as many colour buttons as needed up to limit of PALETTE_ENTRIES

    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->setColumnStretch(0, 0);
    mainLayout->setColumnStretch(1, 0);
    // mainLayout->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    int row, col;

    for(int i = 0; i < nentries; i++)
    {
        selector[i] = new QPushButton(this);

        // which column should palette entry be placed in
        switch(i)
        {
        case 0: row = 0; col = 0;
            break;
        case 1: row = 1; col = 0;
            break;
        case 2: row = 2; col = 0;
            break;
        default:
            break;
        }

        currSel = i;
        setDrawType((BrushType)i);
        selector[i]->setIconSize(activeImg.size());
        selector[i]->setFixedSize(activeImg.size());
        // selector[i]->setMaximumSize(activeImg.size());
        // selector[i]->setMinimumSize(activeImg.size());

        // cerr << "image size = " << (int) activeImg.size().height() << " X " << (int) activeImg.size().width() << endl;
        selector[i]->setIconSize(activeImg.size()*1.25f);
        selector[i]->setFixedSize(activeImg.size()*1.25f);
        selector[i]->setFocusPolicy(Qt::NoFocus);
        connect(selector[i], &QPushButton::clicked, this, &palette_base::typeSelect);
        mainLayout->addWidget(selector[i], row, col, Qt::AlignCenter);
    }
    currSel = 0; setActivePalette();

    QLabel myLabel;
    myLabel.setPixmap(QPixmap::fromImage(activeImg));
    myLabel.show();
    mainLayout->addWidget(&myLabel, nentries/2+1, 0);

    setLayout(mainLayout);
    setFocusPolicy(Qt::StrongFocus);
}

QSize palette_base::sizeHint() const
{
    return QSize(80, 200);
}

void palette_base::setDrawType(BrushType btype)
{
    GLfloat * col;
    int r, g, b;
    QString qss, qss2;

    typeSel[currSel] = btype;
    // set colour
    col = tmap->getColour((int) btype);
    r = (int) (col[0] * 255.0f);
    g = (int) (col[1] * 255.0f);
    b = (int) (col[2] * 255.0f);
    qss = QString("* { background-color: rgb(%1,%2,%3) }").arg(r).arg(g).arg(b);
    //qss2 = QString("QPushButton:disabled { background-color:gray }");
    qss2 = QString("QPushButton:disabled { border-image: url(../Icons/activeEraseIcon.png) 0 0 0 0 stretch stretch; }");
    qss += qss2;
    selector[currSel]->setStyleSheet(qss);
    selector[currSel]->show();
}

void palette_base::setActivePalette()
{
    for(int i = 0; i < nentries; i++)
        if (selector[i])
        {
            if(i == currSel)
                selector[currSel]->setIcon(QPixmap::fromImage(activeImg));
            else
                selector[i]->setIcon(QIcon());
        }
}

void palette_base::deactiveSelection()
{
    for(int i = 0; i < nentries; i++)
        if (selector[i])
        {
            selector[i]->setIcon(QIcon());
        }
}

void palette_base::addBrushes(const std::vector<int> &btypes)
{
    QDir basedir = QString(SRC_BASEDIR);

    if(!activeImg.load(basedir.filePath("Icons/activeIcon.png")))
    {
        cerr << basedir.filePath("Icons/activeIcon.png").toStdString() << " not found" << endl;
    }

    QGridLayout *mainLayout = new QGridLayout;
    mainLayout->setColumnStretch(0, 0);
    mainLayout->setColumnStretch(1, 0);
    // mainLayout->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    int row, col;
    row = 0;
    col = 0;

    /*
    for (int i = 0; i < nentries; i++)
    {
        selector[i] = nullptr;
    }
    */

    for (int i = 0; i < nentries; i++)
    {
        selector[i] = nullptr;
        //int i = (int)bt;
        selector[i] = new QPushButton(this);

        /*
        // which column should palette entry be placed in
        switch(i)
        {
        case 0: row = 0; col = 0;
            break;
        case 1: row = 1; col = 0;
            break;
        case 2: row = 2; col = 0;
            break;
        }
        */

        currSel = i;
        // FIXME: this statement below is done differently than in the case of the other palette (the one for painting dense/sparse veg regions...)
        //		  Find a more consistent way of assigning an identity to the draw type
        //setDrawType((BrushType) (i - (int)BrushType::SPEC1));

        // we assign straight from the 0 to nentries index here. This works because the typemap colour table is also initialized according to this indexing scheme (check usage of the tmap member).
        setDrawType((BrushType)btypes.at(i));
        selector[i]->setIconSize(activeImg.size());
        selector[i]->setFixedSize(activeImg.size());
        // selector[i]->setMaximumSize(activeImg.size());
        // selector[i]->setMinimumSize(activeImg.size());

        // cerr << "image size = " << (int) activeImg.size().height() << " X " << (int) activeImg.size().width() << endl;
        selector[i]->setIconSize(activeImg.size()*1.25f);
        selector[i]->setFixedSize(activeImg.size()*1.25f);
        selector[i]->setFocusPolicy(Qt::NoFocus);
        connect(selector[i], &QPushButton::clicked, this, &palette_base::typeSelect);
        mainLayout->addWidget(selector[i], row, col, Qt::AlignCenter);

        row++;
        if (row > 3)
        {
            row = 0;
            col++;
        }
    }
    currSel = 0; setActivePalette();

    QLabel myLabel;
    myLabel.setPixmap(QPixmap::fromImage(activeImg));
    myLabel.show();
    mainLayout->addWidget(&myLabel, nentries/2+1, 0);

    setLayout(mainLayout);
    setFocusPolicy(Qt::StrongFocus);
}

void palette_base::typeSelectMode(ControlMode mode)
{
    // TO DO - deal with the case that one of the palette entries is in selection mode
    for(int i = 0; i < nentries; i++)
    {
        if(sender() == selector[i])
        {
            currSel = i;
            glparent->setCtrlMode(mode); // activate paint mode
        }
    }
    setActivePalette();
    cerr << "currently selected brush = " << (int) typeSel[currSel] << endl;
}
