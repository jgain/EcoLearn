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

#ifndef PALETTE_BASE_H
#define PALETTE_BASE_H

#include <QWidget>
#include <QPushButton>
#include <unordered_map>

class GLWidget;
class TypeMap;

enum class BrushType
{
    FREE,                  //< bare ground
    SPARSESHRUB,           //< scattered
    SPARSEMED,
    SPARSETALL,
    DENSESHRUB,            //< fully established
    DENSEMED,
    DENSETALL,
    SPEC1,
    BTEND = BrushType::SPEC1 + 32
};
const std::array<BrushType, 7> all_brushtypes = {BrushType::FREE, BrushType::SPARSESHRUB, BrushType::SPARSEMED, BrushType::SPARSETALL,
                                                 BrushType::DENSESHRUB, BrushType::DENSEMED, BrushType::DENSETALL}; // to allow iteration over the brushtypes

enum class ControlMode;

class palette_base : public QWidget
{
    Q_OBJECT

public:

    palette_base(TypeMap * typemap, int nentries, QWidget *parent = 0);
    palette_base(TypeMap *typemap, const std::vector<int> &btypes, QWidget *parent = 0);

    virtual ~palette_base()
    {
        delete [] selector;
        delete [] typeSel;
    }

    //QSize minimumSizeHint() const;
    virtual QSize sizeHint() const;

    /// Obtain the current active brush type from the palette
    BrushType getDrawType(){ return typeSel[currSel]; }

    /// Set the currently active brush type in the palette
    void setDrawType(BrushType btype);

    /// move the currently active icon to the correct palette entry
    void setActivePalette();
    void deactiveSelection();

    void addBrushes(const std::vector<int> &btypes);

    void typeSelectMode(ControlMode mode);

public slots:

    /// this pure virtual function can be implemented to call typeSelect(ControlMode) with the appropriate control mode for which the
    /// derived class is implemented
    virtual void typeSelect() = 0;

protected:
    int nentries;
    GLWidget * glparent;
    TypeMap * tmap;
    QImage activeImg;
    //QPushButton * selector[PALETTE_ENTRIES];
    //BrushType typeSel[PALETTE_ENTRIES];
    QPushButton **selector;
    BrushType *typeSel;
    int currSel;
};

#endif // PALETTE_BASE_H
