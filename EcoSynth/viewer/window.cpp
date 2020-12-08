/****************************************************************************
**
** Copyright (C) 2012 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

/*******************************************************************************
 *
 * EcoSynth - Data-driven Authoring of Large-Scale Ecosystems
 * Copyright (C) 2020 J.E. Gain (jgain@cs.uct.ac.za) and K.P. Kapp  (konrad.p.kapp@gmail.com)
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

#include "ConfigReader.h"
#include "glwidget.h"
#include "window.h"
#include "vecpnt.h"
#include "common/str.h"
#include "specselect_window.h"
#include "convertpaintingdialog.h"

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <functional>

#include <QProgressBar>
#include <QImage>

using namespace std;

////
// SunWindow
///


QSize SunWindow::sizeHint() const
{
    return QSize(800, 800);
}

void SunWindow::setOrthoView(GLWidget * ortho)
{
    QWidget *mainWidget = new QWidget;
    QGridLayout *mainLayout = new QGridLayout;

    orthoView = ortho;

    setCentralWidget(mainWidget);
    mainLayout->setColumnStretch(0, 1);

    // signal to slot connections
    connect(orthoView->getGLSun(), SIGNAL(signalRepaintAllGL()), this, SLOT(repaintAllGL()));

    mainLayout->addWidget(orthoView->getGLSun(), 0, 0);
    mainWidget->setLayout(mainLayout);
    setWindowTitle(tr("EcoSun"));
}

void SunWindow::repaintAllGL()
{
    orthoView->getGLSun()->repaint();
}

////
// Window
///

void Window::addSpecRadSlider(QVBoxLayout *layout, const QString &label,
                          float startValue, int scale, float low, float high)
{
    const float defaultValue = startValue;
    const int defaultScaled = int(std::round(defaultValue * (float) scale));

    radlabel = new QLabel(label);
    species_palette_window->add_widget(radlabel);

    specradslider = new QSlider(Qt::Horizontal);
    specradslider->setMinimum(int(std::ceil(low * scale)));
    specradslider->setMaximum(int(std::floor(high * scale)));
    specradslider->setPageStep(200);
    specradslider->setSingleStep(1);
    specradslider->setTracking(true);
    specradslider->setValue(defaultScaled);

    const float invScale = 1.0f / scale;
    connect(specradslider, &QSlider::valueChanged, [this, invScale] (int newValue)
            {
                perspectiveView->setSpeciesBrushRadius(newValue * invScale);
                repaintAllGL();
            });
    species_palette_window->add_widget(specradslider);
}


void Window::addLearnRadSlider(QVBoxLayout *layout, const QString &label,
                          float startValue, int scale, float low, float high)
{
    const float defaultValue = startValue;
    //const int defaultScaled = int(std::round(defaultValue * (float) scale) / perspectiveView->getView()->getScaleConst());
    const int defaultScaled = int(std::round(defaultValue * (float) scale));

    radlabel = new QLabel(label);
    layout->addWidget(radlabel);

    radslider = new QSlider(Qt::Horizontal);
    radslider->setMinimum(int(std::ceil(low * scale)));
    radslider->setMaximum(int(std::floor(high * scale)));
    radslider->setPageStep(200);
    radslider->setSingleStep(1);
    radslider->setTracking(true);
    radslider->setValue(defaultScaled);

    const float invScale = 1.0f / scale;
    connect(radslider, &QSlider::valueChanged, [this, invScale] (int newValue)
            {
                perspectiveView->setLearnBrushRadius(newValue * invScale);
                //perspectiveView->setRadius(newValue * invScale);
                //radslider->setValue(perspectiveView->getRadius() / invScale);	// seriously?
                repaintAllGL();
            });
    layout->addWidget(radslider);
}

void Window::addPercSlider(QVBoxLayout *layout, const QString &label, int startval)
{
    perclabel = new QLabel(label);
    //layout->addWidget(perclabel);
    species_palette_window->add_widget(perclabel);

    percslider = new QSlider(Qt::Horizontal);
    percslider->setMinimum(0);
    percslider->setMaximum(100);
    percslider->setPageStep(10);
    percslider->setSingleStep(1);
    percslider->setTracking(true);
    percslider->setValue(startval);
    perspectiveView->setSpecPerc(startval / 100.0f);

    connect(percslider, &QSlider::valueChanged, [this] (int val) {
        perspectiveView->setSpecPerc(val / 100.0f);
        repaintAllGL();
    });
    //layout->addWidget(percslider);
    species_palette_window->add_widget(percslider);
}

QSlider *Window::addSpeciesSlider(QVBoxLayout *layout, const QString &label, float startValue, int scale, float low, float high)
{
    const float defaultValue = startValue;
    const int defaultScaled = int(std::round(defaultValue * (float) scale));

    QLabel *newlabel = new QLabel(label);
    layout->addWidget(newlabel);

    QHBoxLayout *horiz_layout = new QHBoxLayout();

    QCheckBox *checkbox = new QCheckBox();
    horiz_layout->addWidget(checkbox);

    QSlider *newslider = new QSlider(Qt::Horizontal);
    newslider->setMinimum(int(std::ceil(low * scale)));
    newslider->setMaximum(int(std::floor(high * scale)));
    newslider->setPageStep(200);
    newslider->setSingleStep(1);
    newslider->setTracking(true);
    newslider->setValue(defaultScaled);

    const float invScale = 1.0f / scale;
    connect(newslider, &QSlider::sliderReleased, this, &Window::speciesSliderCallback);
    horiz_layout->addWidget(newslider);

    layout->addLayout(horiz_layout);

    speciesSliders.push_back(newslider);
    speciesCheckBoxes.push_back(checkbox);


    return newslider;
}

void Window::speciesSliderCallback()
{
        QSlider *sender_slider = (QSlider *)sender();

        assert(sender());

        int target;

        std::vector<QSlider *> checked_sliders;

        bool sender_checked = false;

        for (int i = 0; i < speciesSliders.size(); i++)
        {
            auto sl_ptr = speciesSliders[i];
            auto check_ptr = speciesCheckBoxes[i];
            if (check_ptr->isChecked())
            {
                if (sl_ptr == sender_slider)
                {
                    sender_checked = true;
                }
                checked_sliders.push_back(sl_ptr);
            }
        }
        if (checked_sliders.size() == 0 || checked_sliders.size() == speciesSliders.size())	// change all other sliders if none, or all are checked
        {
            changeAllSpeciesSliders(sender_slider);
            return;
        }
        else
        {
            if (checked_sliders.size() == 1 && sender_checked)	// only one checked, and this is the slider we changed (change all others)
            {
                changeAllSpeciesSliders(sender_slider);
                return;
            }
            else if (!sender_checked)	// one or more sliders are checked, but one we changing is not checked
            {
                changeSpeciesSliders(checked_sliders, sender_slider);
            }
            else	// if the sender has been checked, and one or more others have been checked
            {
                for (int i = 0; i < checked_sliders.size(); i++)
                {
                    if (checked_sliders[i] == sender_slider)
                    {
                        checked_sliders.erase(std::next(checked_sliders.begin(), i));
                        break;
                    }
                }
                changeSpeciesSliders(checked_sliders, sender_slider);
            }
        }

        int newsum = sender_slider->value();
        for (int i = 0; i < speciesSliders.size(); i++)
        {
            auto sl = speciesSliders[i];
            if (sl == sender_slider)
            {
                newsum = sl->value();
                target = (i + 1) % speciesSliders.size();
                break;
            }
        }
}

void Window::species_added(int id)
{
    perspectiveView->species_added(id);
    //int idx = perspectiveView->get_index_from_speciesid(id);
    int idx = species_palette_window->id_to_idx(id);
    species_palette_window->enable_species(idx);
}

void Window::species_removed(int id)
{
    perspectiveView->species_removed(id);
    int idx = species_palette_window->id_to_idx(id);
    species_palette_window->disable_species(idx);
}


void Window::changeSpeciesSliders(std::vector<QSlider *> other_sliders, QSlider *sender_slider)
{
        int newsum = 0;

        //int newsum = sender_slider->value();
        for (auto &sl : speciesSliders)
        {
            newsum += sl->value();
        }
        int diff = 100 - newsum;

        float capacity = 0.0f;
        for (auto other_sl : other_sliders)
        {
            if (diff > 0)
                capacity += 100 - other_sl->value();
            else if (diff < 0)
                capacity += other_sl->value();
        }
        if (capacity < abs(diff))
        {
            float adjustment = sign(diff) * (abs(diff) - capacity);
            sender_slider->setValue(sender_slider->value() + adjustment);
            diff = sign(diff) * capacity;
        }

        while (newsum != 100)
        {
            std::vector<int> rm_idxes;
            newsum = 0;
            for (auto &sl : speciesSliders)
            {
                newsum += sl->value();
            }
            diff = 100 - newsum;
            int ndiv = other_sliders.size();
            if (ndiv == 0)
            {
                int new_senderval = sender_slider->value() + diff;
                assert(new_senderval <= 100 && new_senderval >= 0);
                sender_slider->setValue(new_senderval);
                return;
            }
            int modulo = diff % ndiv;
            int change = diff / ndiv;

            for (int i = other_sliders.size() - 1; i >= 0; i--)
            {
                int newval;
                if (i < abs(modulo))
                {
                    newval = other_sliders[i]->value() + change + sign(modulo);
                }
                else
                {
                    newval = other_sliders[i]->value() + change;
                }

                if (newval < 0)
                {
                    other_sliders[i]->setValue(0);
                    rm_idxes.push_back(i);
                }
                else if (newval > 100)
                {
                    other_sliders[i]->setValue(100);
                    rm_idxes.push_back(i);
                }
                else
                    other_sliders[i]->setValue(newval);
            }
            for (auto idx : rm_idxes)
            {
                other_sliders.erase(std::next(other_sliders.begin(), idx));
            }
        }

        std::vector<float> species_percentages;
        for (auto &spsl : speciesSliders)
        {
            species_percentages.push_back(spsl->value() / 100.0f);
        }

        perspectiveView->setSpeciesPercentages(species_percentages);
}

void Window::changeAllSpeciesSliders(QSlider *sender_slider)
{
        std::vector<QSlider *> other_sliders;

        for (auto &sl : speciesSliders)
        {
            if (sl != sender_slider)
                other_sliders.push_back(sl);
        }

        changeSpeciesSliders(other_sliders, sender_slider);
}

void Window::speciesSliderCallbackChangeAll()
{
        QSlider *sender_slider = (QSlider *)sender();

        assert(sender());

        changeAllSpeciesSliders(sender_slider);
}

QSize Window::sizeHint() const
{
    return QSize(1000, 800);
}

Window::Window(int scale_size)
{
    mainWidget = new QWidget;
    mainLayout = new QGridLayout;
    int dx, dy;
    float sx, sy;

    // default rendering parameters, set using text entry
    // mirrors TRenderer settings
    // grid params
    gridIntensity = 0.8f; // 80% of base colour
    gridSepX = 2500.0f; // separation of grid lines, depends on how input data is scaled
    gridSepZ = 2500.0f; //
    gridWidth = 1.5f; // in pixels?
    
    // contour params
    contourSep = 25.f; // separation (Y direction) depends on how input data is normalized
    numContours = 1.0f / contourSep;
    contourWidth = 1.0f; // in pixels ?
    contourIntensity = 1.2f; // 130% of base colour

    // radiance scaling parameters
    radianceTransition = 0.2f;
    radianceEnhance = 3.0f;

    setCentralWidget(mainWidget);
    mainLayout->setColumnStretch(0, 0);
    mainLayout->setColumnStretch(1, 1);
    mainLayout->setColumnStretch(2, 0);

    // render panel
    renderPanel = new QWidget;
    renderLayout = new QVBoxLayout;

    // Grid Line Widgets
    QGroupBox *gridGroup = new QGroupBox(tr("Grid Lines"));
    QCheckBox * checkGridLines = new QCheckBox(tr("Show Grid Lines"));
    checkGridLines->setChecked(false);
    QLabel *gridSepXLabel = new QLabel(tr("Grid Sep X:"));
    gridSepXEdit = new QLineEdit;
    // gridSepXEdit->setValidator(new QDoubleValidator(0.0, 500000.0, 2, gridSepXEdit));
    gridSepXEdit->setInputMask("0000.0");
    QLabel *gridSepZLabel = new QLabel(tr("Grid Sep Z:"));
    gridSepZEdit = new QLineEdit;
    // gridSepZEdit->setValidator(new QDoubleValidator(0.0, 500000.0, 2, gridSepZEdit));
    gridSepZEdit->setInputMask("0000.0");
    QLabel *gridWidthLabel = new QLabel(tr("Grid Line Width:"));
    gridWidthEdit = new QLineEdit;
    // gridWidthEdit->setValidator(new QDoubleValidator(0.0, 10.0, 2, gridWidthEdit));
    gridWidthEdit->setInputMask("0.0");
    QLabel *gridIntensityLabel = new QLabel(tr("Grid Intensity:"));
    gridIntensityEdit = new QLineEdit;
    // gridIntensityEdit->setValidator(new QDoubleValidator(0.0, 2.0, 2, gridIntensityEdit));
    gridIntensityEdit->setInputMask("0.0");

    // set initial grid values
    gridSepXEdit->setText(QString::number(gridSepX, 'g', 2));
    gridSepZEdit->setText(QString::number(gridSepZ, 'g', 2));
    gridWidthEdit->setText(QString::number(gridWidth, 'g', 2));
    gridIntensityEdit->setText(QString::number(gridIntensity, 'g', 2));

    QGridLayout *gridLayout = new QGridLayout;
    gridLayout->addWidget(checkGridLines, 0, 0);
    gridLayout->addWidget(gridSepXLabel, 1, 0);
    gridLayout->addWidget(gridSepXEdit, 1, 1);
    gridLayout->addWidget(gridSepZLabel, 2, 0);
    gridLayout->addWidget(gridSepZEdit, 2, 1);
    gridLayout->addWidget(gridWidthLabel, 3, 0);
    gridLayout->addWidget(gridWidthEdit, 3, 1);
    gridLayout->addWidget(gridIntensityLabel, 4, 0);
    gridLayout->addWidget(gridIntensityEdit, 4, 1);
    gridGroup->setLayout(gridLayout);

    // Contour Widgets
    QGroupBox *contourGroup = new QGroupBox(tr("Contours"));
    QCheckBox * checkContours = new QCheckBox(tr("Show Contours"));
    checkContours->setChecked(false);
    QLabel *contourSepLabel = new QLabel(tr("Contour Sep:"));
    contourSepEdit = new QLineEdit;
    //contourSepEdit->setValidator(new QDoubleValidator(0.0, 10000.0, 2, contourSepEdit));
    contourSepEdit->setInputMask("000.0");
    QLabel *contourWidthLabel = new QLabel(tr("Contour Line Width:"));
    contourWidthEdit = new QLineEdit;
    // contourWidthEdit->setValidator(new QDoubleValidator(0.0, 10.0, 2, contourWidthEdit));
    contourWidthEdit->setInputMask("0.0");
    QLabel *contourIntensityLabel = new QLabel(tr("Contour Intensity:"));
    contourIntensityEdit = new QLineEdit;
    contourIntensityEdit->setInputMask("0.0");

    // set initial contour values
    contourSepEdit->setText(QString::number(contourSep, 'g', 2));
    contourWidthEdit->setText(QString::number(contourWidth, 'g', 2));
    contourIntensityEdit->setText(QString::number(contourIntensity, 'g', 2));

    QGridLayout *contourLayout = new QGridLayout;
    contourLayout->addWidget(checkContours, 0, 0);
    contourLayout->addWidget(contourSepLabel, 1, 0);
    contourLayout->addWidget(contourSepEdit, 1, 1);
    contourLayout->addWidget(contourWidthLabel, 2, 0);
    contourLayout->addWidget(contourWidthEdit, 2, 1);
    contourLayout->addWidget(contourIntensityLabel, 3, 0);
    contourLayout->addWidget(contourIntensityEdit, 3, 1);
    contourGroup->setLayout(contourLayout);

    // Radiance
    QGroupBox *radianceGroup = new QGroupBox(tr("Radiance"));
    QLabel *radianceEnhanceLabel = new QLabel(tr("Radiance Enhancement:"));
    radianceEnhanceEdit = new QLineEdit;
    radianceEnhanceEdit->setInputMask("0.0");

    // set initial radiance values
    radianceEnhanceEdit->setText(QString::number(radianceEnhance, 'g', 2));

    QGridLayout *radianceLayout = new QGridLayout;
    radianceLayout->addWidget(radianceEnhanceLabel, 0, 0);
    radianceLayout->addWidget(radianceEnhanceEdit, 0, 1);
    radianceGroup->setLayout(radianceLayout);

    renderLayout->addWidget(gridGroup);
    renderLayout->addWidget(contourGroup);
    renderLayout->addWidget(radianceGroup);

    // right-hand panel for pallete
    palLayout = new QVBoxLayout;
    specpalLayout = new QVBoxLayout;

    // OpenGL widget
    // Specify an OpenGL 3.2 format.

    QGLFormat glFormat;
    glFormat.setVersion( 4, 1 );
    glFormat.setProfile( QGLFormat::CoreProfile );
    glFormat.setSampleBuffers( false );

    perspectiveView = new GLWidget(glFormat, scale_size);
    getView().setForcedFocus(getTerrain().getFocus());
    getView().setViewScale(getTerrain().longEdgeDist());

    getTerrain().getGridDim(dx, dy);
    getTerrain().getTerrainDim(sx, sy);

    perspectiveView->getGLSun()->setScene(&getTerrain(), NULL, NULL);

    std::cerr << "done" << std::endl;

    numGridX = 1.0f / gridSepX;
    numGridZ = 1.0f / gridSepZ;

    // Palette Widget

    palLayout->addWidget(perspectiveView->getPalette());
    SpeciesPalette * specpal = perspectiveView->getSpeciesPalette();
    specpalLayout->addWidget(perspectiveView->getSpeciesPalette());

    species_palette_window = new specpalette_window(this, specpal);

    progress_bar_window = new QWidget(this, Qt::Window);

    QVBoxLayout *progress_layout = new QVBoxLayout();

    QVBoxLayout *cplace_layout = new QVBoxLayout();
    QVBoxLayout *qundergrowth_layout = new QVBoxLayout();
    QVBoxLayout *undersynth_layout = new QVBoxLayout();

    QLabel *cplace_label = new QLabel("Canopy Placement");
    cplace_layout->addWidget(cplace_label);
    QLabel *qundergrowth_label = new QLabel("Quick Undergrowth Synthesis");
    qundergrowth_layout->addWidget(qundergrowth_label);
    QLabel *undersynth_label = new QLabel("Undergrowth Synthesis");
    undersynth_layout->addWidget(undersynth_label);

    canopy_placement_progress = new QProgressBar();
    //canopy_placement_progress->setWindowFlags(Qt::Window);
    canopy_placement_progress->setMinimum(0);
    canopy_placement_progress->setMaximum(5);
    canopy_placement_progress->setFixedWidth(200);
    //canopy_placement_progress->show();
    cplace_layout->addWidget(canopy_placement_progress);
    cplace_layout->setSpacing(10);

    quick_undergrowth_progress = new QProgressBar();
    quick_undergrowth_progress->setMinimum(0);
    quick_undergrowth_progress->setMaximum(100);
    quick_undergrowth_progress->setFixedWidth(200);
    qundergrowth_layout->addWidget(quick_undergrowth_progress);
    qundergrowth_layout->setSpacing(10);

    undersynth_progress = new QProgressBar();
    undersynth_progress->setMinimum(0);
    undersynth_progress->setMaximum(100);
    undersynth_progress->setFixedWidth(200);
    undersynth_layout->addWidget(undersynth_progress);
    undersynth_layout->setSpacing(10);

    currsynth_label = new QLabel("Canopy placement");

    synth_progress = new QProgressBar();
    synth_progress->setMinimum(0);
    synth_progress->setMaximum(100);
    synth_progress->setFixedWidth(200);
    progress_layout->addWidget(currsynth_label);
    progress_layout->addWidget(synth_progress);

    progress_bar_window->setLayout(progress_layout);
    progress_bar_window->show();


    connect(perspectiveView, SIGNAL(signalCanopyPlacementStart()), this, SLOT(set_canopy_label()));
    connect(perspectiveView, SIGNAL(signalSpeciesOptimStart()), this, SLOT(set_species_optim_label()));
    connect(perspectiveView, SIGNAL(signalUndergrowthSampleStart()), this, SLOT(set_undersample_label()));
    connect(perspectiveView, SIGNAL(signalUndergrowthRefineStart()), this, SLOT(set_underrefine_label()));

    // signal to slot connections
    connect(perspectiveView, SIGNAL(signalRepaintAllGL()), this, SLOT(repaintAllGL()));
    connect(gridSepXEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(gridSepZEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(gridWidthEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(gridIntensityEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(contourSepEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(contourWidthEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(contourIntensityEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(radianceEnhanceEdit, SIGNAL(editingFinished()), this, SLOT(lineEditChange()));
    connect(radianceEnhanceEdit, SIGNAL(returnPressed()), this, SLOT(lineEditChange()));

    connect(perspectiveView, SIGNAL(signalUpdateCanopyPlacementProgress(int)), canopy_placement_progress, SLOT(setValue(int)));
    connect(perspectiveView, SIGNAL(signalUpdateQuickUndergrowthProgress(int)), quick_undergrowth_progress, SLOT(setValue(int)));
    connect(perspectiveView, SIGNAL(signalUpdateUndergrowthProgress(int)), undersynth_progress, SLOT(setValue(int)));
    connect(perspectiveView, SIGNAL(signalUpdateProgress(int)), synth_progress, SLOT(setValue(int)));

    //connect(minTreeEdit, SIGNAL(editingFinished()), this, SLOT(treeEditChange()));
    //connect(maxTreeEdit, SIGNAL(editingFinished()), this, SLOT(treeEditChange()));

    // display switches
    connect(checkContours, SIGNAL(stateChanged(int)), this, SLOT(showContours(int)));
    connect(checkGridLines, SIGNAL(stateChanged(int)), this, SLOT(showGridLines(int)));

    renderPanel->setLayout(renderLayout);

    mainLayout->addWidget(renderPanel, 0, 0, Qt::AlignTop);
    mainLayout->addWidget(perspectiveView, 0, 1, 10, 1);
    mainLayout->addLayout(palLayout, 0, 2, Qt::AlignTop);

    //QSpacerItem *spaceitem = new QSpacerItem(10, 100, QSizePolicy::Minimum, QSizePolicy::Minimum);
    procdrawingButton = new QPushButton("Process drawing");
    //mainLayout->addItem(spaceitem, 1, 2, Qt::AlignTop);
    mainLayout->addWidget(procdrawingButton, 2, 2, Qt::AlignTop);

    createActions();
    createMenus();

    mainWidget->setLayout(mainLayout);
    setWindowTitle(tr("EcoLearn"));
    mainWidget->setMouseTracking(true);
    setMouseTracking(true);

    renderPanel->hide();

    perspectiveView->getRenderer()->setGridParams(numGridX, numGridZ, gridWidth, gridIntensity);
    perspectiveView->getRenderer()->setContourParams(numContours, contourWidth, contourIntensity);
    perspectiveView->getRenderer()->setRadianceScalingParams(radianceEnhance);

    specwindow = new specselect_window(std::string(PRJ_SRC_DIR) + "/ecodata/sonoma.db", this);
    specwindow->show();

    connect(perspectiveView, &GLWidget::signalDisableSpecSelect, specwindow, &specselect_window::disable);
    connect(perspectiveView, &GLWidget::signalEnableSpecSelect, specwindow, &specselect_window::enable);

    connect(procdrawingButton, SIGNAL(clicked()), perspectiveView, SLOT(send_drawing()));

    procdrawingButton->setEnabled(false);

    std::cerr << "Window construction done" << std::endl;
    //connect(QApplication::instance(), SIGNAL(aboutToQuit()), this, SLOT(cleanup()));
}

void Window::scaleRenderParams(float scale)
{
    gridSepX = scale / 5.0f; // separation of grid lines, depends on how input data is scaled
    gridSepZ = scale / 5.0f;
    numGridX = 1.0f / gridSepX;
    numGridZ = 1.0f / gridSepZ;
    gridSepXEdit->setText(QString::number(gridSepX, 'g', 2));
    gridSepZEdit->setText(QString::number(gridSepZ, 'g', 2));

    contourSep = scale / 100.f; // separation (Y direction) depends on how input data is normalized
    numContours = 1.0f / contourSep;
    contourSepEdit->setText(QString::number(contourSep, 'g', 2));

    perspectiveView->getRenderer()->setGridParams(numGridX, numGridZ, gridWidth, gridIntensity);
    perspectiveView->getRenderer()->setContourParams(numContours, contourWidth, contourIntensity);
    perspectiveView->getRenderer()->setRadianceScalingParams(radianceEnhance);
}

void Window::keyPressEvent(QKeyEvent *e)
{
    /*
    if (e->key() == Qt::Key_Escape)
        close();
    else

        QWidget::keyPressEvent(e);
     */
    
    // pass to render window
    perspectiveView->keyPressEvent(e);
}

void Window::mouseMoveEvent(QMouseEvent *event)
{
    QWidget *child=childAt(event->pos());
    QGLWidget *glwidget = qobject_cast<QGLWidget *>(child);
    if(glwidget) {
        QMouseEvent *glevent=new QMouseEvent(event->type(),glwidget->mapFromGlobal(event->globalPos()),event->button(),event->buttons(),event->modifiers());
        QCoreApplication::postEvent(glwidget,glevent);
    }
}

void Window::repaintAllGL()
{
    perspectiveView->repaint();
}

void Window::set_canopy_label()
{
    currsynth_label->setText("Canopy placement");
    progress_bar_window->repaint();
}

void Window::set_species_optim_label()
{
    currsynth_label->setText("Species optimisation");
    progress_bar_window->repaint();
}

void Window::set_undersample_label()
{
    currsynth_label->setText("Undergrowth sampling");
    progress_bar_window->repaint();
}

void Window::set_underrefine_label()
{
    currsynth_label->setText("Undergrowth refinement");
    progress_bar_window->repaint();
}

void Window::cleanup()
{
}

void Window::openTerrain()
{
    std::string valid_files = "16-bit PNG image (*.png);;Terragen File (*.ter);;Ascii Elevation File (*.elv)";
    //valid_files += ";;16-bit PNG image (*.png)";
    bool valid = false;
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Terrain File"),
                                                    "~/",
                                                    tr(valid_files.c_str()));
    if (!fileName.isEmpty())
    {
        std::string infile = fileName.toUtf8().constData();

        // use file extension to determine action
        if(endsWith(infile, ".ter")) // terragen file format
        {
            getTerrain().loadTer(infile); valid = true;
        }
        else if(endsWith(infile, ".elv")) // simple ascii heightfield
        {
            getTerrain().loadElv(infile); valid = true;
        }
        else if (endsWith(infile, ".png"))	// load terrain from 16-bit PNG image
        {
            getTerrain().loadPng(infile); valid = true;
            int w, h;
            getTerrain().getGridDim(w, h);
            getGLWidget()->initCHM(w, h);
        }

        if(valid)
        {
            getView().setForcedFocus(getTerrain().getFocus());
            getView().setViewScale(getTerrain().longEdgeDist());
            getTerrain().calcMeanHeight();
            getTerrain().updateBuffers(perspectiveView->getRenderer()); // NB - sets width and height for terrain, otherwise crash
            repaintAllGL();
        }
        else
        {
            cerr << "Error Window::open: attempt to open unrecognized file format" << endl;
        }
    }
}

void Window::importDrawing()
{
    QString pathname = QFileDialog::getOpenFileName(this, tr("Import drawing"), QString(), "*.png");

    if (!pathname.isEmpty())
    {
        QImage img(pathname);
        perspectiveView->import_drawing(img);
    }
}

void Window::openScene()
{
    // Ecosystem files are bundled in a directory, which the user specifies.
    QString dirName = QFileDialog::getExistingDirectory(this, tr("Open Landscape Scene"),
                                                    QString(),
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);

    openScene(dirName.toStdString(), true);
}

void Window::openScene(std::string dirName, bool import_cluster_dialog)
{
    if (dirName.size() > 0)
    {
        QDir indir(dirName.c_str());
        QString lastseg = indir.dirName();
        std::string dirstr = dirName;
        std::string segstr = lastseg.toUtf8().constData();
        // use last component of directory structure as ecosystem name
        scenedirname = dirstr + "/" + segstr;
#ifndef PAINTCONTROL
        sunwindow->show();
        sunwindow->repaint();
        sunwindow->hide();
#endif
        bool firstscene = !perspectiveView->hasSceneLoaded();
        perspectiveView->sunwindow = sunwindow;
        perspectiveView->loadScene(scenedirname);
        auto ter = perspectiveView->getTerrain();
        float tx, ty;
        ter->getTerrainDim(tx, ty);
        float tm = std::min(tx, ty);

        float startval = tm / 40.0f;
        float endval = tm / 4.0f;

        perspectiveView->setRadius((startval + endval) / 2.0f);
        perspectiveView->setLearnBrushRadius(perspectiveView->getRadius());
        perspectiveView->setSpeciesBrushRadius(perspectiveView->getRadius());
        if (firstscene)
        {
            addLearnRadSlider(palLayout, tr("Active Brush Size"), perspectiveView->getRadius(), 1, tm / 40.0f, tm / 4.0f);
            addSpecRadSlider(specpalLayout, tr("Active Brush Size"), perspectiveView->getRadius(), 1, tm / 40.0f, tm / 4.0f);
            addPercSlider(specpalLayout, tr("Required species percentage"), 50);
        }

        if (import_cluster_dialog)
            showImportClusterFiles();

        procdrawingButton->setEnabled(true);

        repaintAllGL();
    }

}

void Window::loadConfig(std::string configfilename)
{
    ConfigReader reader(configfilename);
    bool success = reader.read();
    if (!success)
        return;
    loadConfig(reader.get_params());
}

void Window::loadConfig(configparams params)
{
    if (params.scene_dirname.size() == 0)
        return;
    else
        openScene(params.scene_dirname, false);

    if (params.clusterdata_filenames.size() > 0)
        importClusterFiles(params.clusterdata_filenames);

    if (params.canopy_filename.size() > 0)
        perspectiveView->read_pdb_canopy(params.canopy_filename);

    if (params.undergrowth_filename.size() > 0)
        perspectiveView->read_pdb_undergrowth(params.undergrowth_filename);

    switch (params.ctrlmode)
    {
        case ControlMode::CMEND:
            params.ctrlmode = ControlMode::VIEW;
            // don't call break here, we want to this go on to the default case
        default:
            perspectiveView->setCtrlMode(params.ctrlmode);
    }

    perspectiveView->setPlantsVisibility(true);
    perspectiveView->setCanopyVisibility(params.render_canopy);
    perspectiveView->setUndergrowthVisibility(params.render_undergrowth);
}

void Window::saveScene()
{
    if(!scenedirname.empty()) // save directly if we already have a file name
    {
        perspectiveView->saveScene(scenedirname);
    }
    else
        saveAsScene();
}

void Window::saveAsScene()
{
    if (!perspectiveView->hasSceneLoaded())
    {
        QMessageBox mbox;
        mbox.setText("Cannot save scene - no scene loaded yet");
        mbox.exec();
        return;
    }
    QFileDialog::Options options;
    QString selectedFilter;
    // use file open dialog but convert to a directory
    QString scenedir = QFileDialog::getSaveFileName(this,
                                    tr("Save Scene"),
                                    "~/",
                                    tr(""),
                                    &selectedFilter,
                                    options);
    if (!scenedir.isEmpty())
    {
        scenedirname = scenedir.toUtf8().constData();
        QDir dir(scenedir);
        if (!dir.exists()) // create directory if it doesn't already exist
        {
            dir.mkpath(".");
        }
        scenedirname += "/" + dir.dirName().toStdString();

        saveScene();
    }
}

void Window::saveAsPaint()
{

    QFileDialog::Options options;
    QString selectedFilter;
    QString paintfile = QFileDialog::getSaveFileName(this,
                                    tr("Save PaintMap As"),
                                    "~/",
                                    tr("Image Files (*.png)"),
                                    &selectedFilter,
                                    options);
    if (!paintfile.isEmpty())
    {
        std::string paintfilename = paintfile.toStdString();

        if(endsWith(paintfilename, ".png"))
            perspectiveView->writePaintMap(paintfilename);
        else
            cerr << "Error Window::saveAsPaint: file extension is incorrect, should be PNG" << endl;
    }
}

void Window::saveAsCHM()
{
    QFileDialog::Options options;
    QString selectedFilter;
    QString chmfile = QFileDialog::getSaveFileName(this,
                                    tr("Save CHM As"),
                                    "~/",
                                    tr("Image Files (*.png)"),
                                    &selectedFilter,
                                    options);
    if (!chmfile.isEmpty())
    {
        std::string chmfilename = chmfile.toStdString();

        if(endsWith(chmfilename, ".png"))
            perspectiveView->writeCanopyHeightModel(chmfilename);
        else
            cerr << "Error Window::saveAsPaint: file extension is incorrect, should be PNG" << endl;
    }

}

void Window::showRenderOptions()
{
    renderPanel->setVisible(showRenderAct->isChecked());
}

void Window::showContours(int show)
{
    perspectiveView->getRenderer()->drawContours(show == Qt::Checked);
    repaintAllGL();
}

void Window::showGridLines(int show)
{
    perspectiveView->getRenderer()->drawGridlines(show == Qt::Checked);
    repaintAllGL();
}

void Window::lineEditChange()
{
    bool ok;
    float val;
    float tx, ty, hr;
    //QLineEdit* sender = dynamic_cast<QLineEdit*> sender();

    //getTerrain().getTerrainDim(tx, ty);
    tx = 1.0f; ty = 1.0f; // to fix when scale added
    hr = 1.0f;

    if(sender() == gridSepXEdit)
    {
        val = gridSepXEdit->text().toFloat(&ok);
        if(ok)
        {
            gridSepX = val;
            numGridX = tx / gridSepX; // convert separation to num grid lines
        }
    }
    if(sender() == gridSepZEdit)
    {
        val = gridSepZEdit->text().toFloat(&ok);
        if(ok)
        {
            gridSepZ = val;
            numGridZ = ty / gridSepZ;
        }
    }
    if(sender() == gridWidthEdit)
    {
        val = gridWidthEdit->text().toFloat(&ok);
        if(ok)
        {
            gridWidth = val;
        }
    }
    if(sender() == gridIntensityEdit)
    {
        val = gridIntensityEdit->text().toFloat(&ok);
        if(ok)
        {
            gridIntensity = val;
        }
    }
    if(sender() == contourSepEdit)
    {
        val = contourSepEdit->text().toFloat(&ok);
        if(ok)
        {
            contourSep = val;
            numContours = hr / contourSep;
        }
    }
    if(sender() == contourWidthEdit)
    {
        val = contourWidthEdit->text().toFloat(&ok);
        if(ok)
        {
            contourWidth = val;
        }
    }
    if(sender() == contourIntensityEdit)
    {
        val = contourIntensityEdit->text().toFloat(&ok);
        if(ok)
        {
            contourIntensity = val;
        }
    }
    if(sender() == radianceEnhanceEdit)
    {
        val = radianceEnhanceEdit->text().toFloat(&ok);
        if(ok)
        {
            radianceEnhance = val;
        }
    }
    cerr << "val entered " << val << endl;

    // without this the renderer defaults back to factory settings at certain stages - very wierd bug
    perspectiveView->getRenderer()->setGridParams(numGridX, numGridZ, gridWidth, gridIntensity);
    perspectiveView->getRenderer()->setContourParams(numContours, contourWidth, contourIntensity);
    perspectiveView->getRenderer()->setRadianceScalingParams(radianceEnhance);
    repaintAllGL();
}

void Window::treeEditChange()
{
    bool ok;
    float val;

    if(sender() == minTreeEdit)
    {
        val = minTreeEdit->text().toFloat(&ok);
        if(ok)
        {
            minTree = val;
        }
    }
    if(sender() == maxTreeEdit)
    {
        val = maxTreeEdit->text().toFloat(&ok);
        if(ok)
        {
            maxTree = val;
        }
    }
    cerr << "val entered " << val << endl;

    // adjust canopy height texture render
    perspectiveView->bandCanopyHeightTexture(minTree, maxTree);
    repaintAllGL();
}

void Window::createActions()
{
    openSceneAct = new QAction(tr("&OpenScene"), this);
    openSceneAct->setShortcuts(QKeySequence::Open);
    openSceneAct->setStatusTip(tr("Open an ecosystem scene directory"));
    connect(openSceneAct, SIGNAL(triggered()), this, SLOT(openScene()));

    /*
     * // Removing import of only terrain, not scene. Will make interface more complicated.
     * // Can be added at a later stage if necessary
    openTerrainAct = new QAction(tr("OpenTerrain"), this);
    openTerrainAct->setStatusTip(tr("Open an existing terrain file"));
    connect(openTerrainAct, SIGNAL(triggered()), this, SLOT(openTerrain()));
    */

    /*
     * // Removing normal save without specifying directory, because it can accidentally overwrite
     * // an existing scene easily. Safer to use "Save Scene as", which creates a new directory or overwrites
     * // if explicitly specified
    saveSceneAct = new QAction(tr("&Save Scene"), this);
    saveSceneAct->setShortcuts(QKeySequence::Save);
    saveSceneAct->setStatusTip(tr("Save the ecosystem scene"));
    connect(saveSceneAct, SIGNAL(triggered()), this, SLOT(saveScene()));
    */

    saveSceneAsAct = new QAction(tr("Save Scene as"), this);
    saveSceneAsAct->setStatusTip(tr("Save the ecosystem scene under a new name"));
    connect(saveSceneAsAct, SIGNAL(triggered()), this, SLOT(saveAsScene()));

    savePaintAsAct = new QAction(tr("Save PaintMap as"), this);
    savePaintAsAct->setStatusTip(tr("Save the paint map under a new name"));
    connect(savePaintAsAct, SIGNAL(triggered()), this, SLOT(saveAsPaint()));

    saveCHMAsAct = new QAction(tr("Save CHM as"), this);
    saveCHMAsAct->setStatusTip(tr("Save the CHM under a new name"));
    connect(saveCHMAsAct, SIGNAL(triggered()), this, SLOT(saveAsCHM()));

    showRenderAct = new QAction(tr("Show Render Options"), this);
    showRenderAct->setCheckable(true);
    showRenderAct->setChecked(false);
    showRenderAct->setStatusTip(tr("Hide/Show Rendering Options"));
    connect(showRenderAct, SIGNAL(triggered()), this, SLOT(showRenderOptions()));

    showRenderAct = new QAction(tr("Show Render Options"), this);
    showRenderAct->setCheckable(true);
    showRenderAct->setChecked(false);
    showRenderAct->setStatusTip(tr("Hide/Show Rendering Options"));
    connect(showRenderAct, SIGNAL(triggered()), this, SLOT(showRenderOptions()));

    importCanopyAct = new QAction(tr("Import canopy"), this);
    connect(importCanopyAct, SIGNAL(triggered()), this, SLOT(showImportCanopy()));

    importUndergrowthAct = new QAction(tr("Import undergrowth"), this);
    connect(importUndergrowthAct, SIGNAL(triggered()), this, SLOT(showImportUndergrowth()));

    importClusterfilesAct = new QAction(tr("Import cluster files"), this);
    connect(importClusterfilesAct, SIGNAL(triggered()), this, SLOT(showImportClusterFiles()));

    importCanopyshadingAct = new QAction(tr("Import canopy shading"), this);
    connect(importCanopyshadingAct, SIGNAL(triggered()), this, SLOT(showImportCanopyshading()));

    sampleUndergrowthAct = new QAction(tr("Sample undergrowth"), this);
    connect(sampleUndergrowthAct, SIGNAL(triggered()), perspectiveView, SLOT(doFastUndergrowthSampling()));

    showClusterCountsAct = new QAction(tr("Show cluster counts for species..."), this);
    connect(showClusterCountsAct, SIGNAL(triggered()), this, SLOT(showClusterCounts()));

    doCompleteUndergrowthSynthesisAct = new QAction(tr("Do complete undergrowth synthesis"), this);
    connect(doCompleteUndergrowthSynthesisAct, SIGNAL(triggered()), perspectiveView, SLOT(doUndergrowthSynthesisCallback()));

    importDrawingAct = new QAction(tr("Import drawing..."), this);
    connect(importDrawingAct, SIGNAL(triggered()), this, SLOT(importDrawing()));

    convertPaintingAct = new QAction(tr("Convert painting..."), this);
    connect(convertPaintingAct, SIGNAL(triggered()), this, SLOT(convertPainting()));

    processDrawingAct = new QAction(tr("Process drawing"), this);
    connect(processDrawingAct, SIGNAL(triggered()), perspectiveView, SLOT(send_drawing()));

    viewSpeciesColoursAct = new QAction(tr("Species colours"), this);
    connect(viewSpeciesColoursAct, SIGNAL(triggered()), this, SLOT(showSpeciesColours()));
}

void Window::showImportCanopyshading()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Import canopy shading from file"), scenedirname.c_str(), tr("*.txt"));

    if (!filename.isEmpty())
    {
        perspectiveView->import_canopyshading(filename.toStdString());
    }
}

void Window::showImportClusterFiles()
{
    //QStringList fname_list_temp = QFileDialog::getOpenFileNames(this, tr("Import cluster files"), scenedirname.c_str(), tr("*.clm"));
    QStringList fname_list_temp = QFileDialog::getOpenFileNames(this, tr("Import cluster files"), QString(), tr("*.clm"));

    auto fname_list_qstr = fname_list_temp.toStdList();
    std::vector<std::string> fname_list;

    for (auto &qstr : fname_list_qstr)
    {
        fname_list.push_back(qstr.toStdString());
    }

    importClusterFiles(fname_list);
}

void Window::importClusterFiles(std::vector<std::string> fname_list)
{
    std::vector<std::string> clusterfiles;

    for (auto &fname : fname_list)
    {
        QFileInfo finfo(fname.c_str());
        if (finfo.isDir())
        {
            QDir dir(fname.c_str());
            QStringList dirlist_temp = dir.entryList(QStringList() << "*.clm", QDir::Files);
            auto dirlist = dirlist_temp.toStdList();
            for (auto &clfile : dirlist)
            {
                clusterfiles.push_back(clfile.toStdString());
            }
        }
        else
        {
            clusterfiles.push_back(fname);
        }
    }

    std::cout << "Imported cluster files: " << std::endl;
    for (auto &fname : clusterfiles)
    {
        std::cout << fname << std::endl;
    }

    perspectiveView->set_clusterfilenames(clusterfiles);

}

void Window::showImportCanopy()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Import canopy from PDB file"), scenedirname.c_str(), tr("*.pdb"));

    if (!filename.isEmpty())
    {
        perspectiveView->read_pdb_canopy(filename.toStdString());
    }
}

void Window::convertPainting()
{
    ConvertPaintingDialog d;
    if (d.exec() == QDialog::Accepted)
    {
        int from, to;
        d.get_values(from, to);
        BrushType tp_from, tp_to;
        switch (from)
        {
            case 0:
                tp_from = BrushType::FREE;
                break;
            case 1:
                tp_from = BrushType::SPARSETALL;
                break;
            case 2:
                tp_from = BrushType::DENSETALL;
                break;
            default:
                return;
        }
        switch (to)
        {
            case 0:
                tp_to = BrushType::FREE;
                break;
            case 1:
                tp_to = BrushType::SPARSETALL;
                break;
            case 2:
                tp_to = BrushType::DENSETALL;
                break;
            default:
                return;
        }
        perspectiveView->convert_painting(tp_from, tp_to);
    }
    else
    {
    }
}

void Window::hide_all_ctrlwindows()
{
}

void Window::showImportUndergrowth()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Import undergrowth from PDB file"), scenedirname.c_str(), tr("*.pdb"));

    if (!filename.isEmpty())
    {
        perspectiveView->read_pdb_undergrowth(filename.toStdString());
    }
}

void Window::showSpeciesColours()
{

    const auto &cdata = perspectiveView->get_cdata();

    if (specColoursWindow)
        delete specColoursWindow;

    specColoursWindow = new SpeciesColoursWindow(this, cdata);
    specColoursWindow->display();
}

void Window::showClusterCounts()
{
    bool ok;
    int species = QInputDialog::getInt(this, tr("Species count"), tr("Species: "), 1, 1, 16, 1, &ok);
    if (!ok)
        return;
    auto counts = perspectiveView->get_species_clustercounts(species);
    for (auto &clpair : counts)
    {
        std::cout << "Cluster " << clpair.first << ": " << clpair.second << std::endl;
    }
}

void Window::createMenus()
{
    // File menu
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(openSceneAct);
    fileMenu->addAction(saveSceneAsAct);
    fileMenu->addAction(savePaintAsAct);
    fileMenu->addAction(saveCHMAsAct);

    // View menu
    viewMenu = menuBar()->addMenu(tr("&View"));
    viewMenu->addAction(showRenderAct);
    viewMenu->addAction(viewSpeciesColoursAct);

    // Import menu
    importMenu = menuBar()->addMenu(tr("&Import"));
    importMenu->addAction(importCanopyAct);
    importMenu->addAction(importUndergrowthAct);
    importMenu->addAction(importDrawingAct);
    importMenu->addAction(importClusterfilesAct);
    importMenu->addAction(importCanopyshadingAct);

    // Actions menu
    actionMenu = menuBar()->addMenu(tr("&Actions"));
    actionMenu->addAction(sampleUndergrowthAct);
    actionMenu->addAction(showClusterCountsAct);
    actionMenu->addAction(convertPaintingAct);
    actionMenu->addAction(processDrawingAct);
    actionMenu->addAction(doCompleteUndergrowthSynthesisAct);
}
