/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include "gstar/AnnotationToolBarWidget.h"

#include <QAction>
#include <QCheckBox>
#include <QLabel>
#include <QToolBar>

using namespace gstar;

/*---------------------------------------------------------------------------*/

AnnotationToolBarWidget::AnnotationToolBarWidget(QWidget* parent) :
   QWidget(parent)
{

   // Background color
   //QPalette pal = this->palette();
//   pal.setColor(this->backgroundRole(), Qt::white);
//   this->setPalette(pal);
//   setAutoFillBackground(true);

   m_toolbar = new QToolBar();
   //m_toolbar->setPalette(pal);

   m_rulerAction = new QAction(QIcon(":images/ruler.png"),
                               "Ruler",
                               nullptr);

   m_intensityLineAction = new QAction(QIcon(":images/intensity.png"),
                               "Line Out Intensity",
                               nullptr);

   m_intensityPieAction = new QAction(QIcon(":images/arc.png"),
                               "Arc Intensity",
                               nullptr);

   m_markerAction = new QAction(QIcon(":images/marker.png"),
                               "Marker",
                               nullptr);

   m_crossHairAction = new QAction(QIcon(":images/crosshair.png"),
                                  "Cross Hair",
                                  nullptr);


   connect(m_rulerAction,
           SIGNAL(triggered()),
           this,
           SIGNAL(clickRuler()));

   connect(m_intensityLineAction,
           SIGNAL(triggered()),
           this,
           SIGNAL(clickIntensityLine()));

   connect(m_intensityPieAction,
           SIGNAL(triggered()),
           this,
           SIGNAL(clickIntensityPie()));

   connect(m_markerAction,
           SIGNAL(triggered()),
           this,
           SIGNAL(clickMarker()));

   connect(m_crossHairAction,
           SIGNAL(triggered()),
           this,
           SIGNAL(clickCrossHair()));

   QLabel* enableLable = new QLabel("Enable:");
   m_chkSetVisible = new QCheckBox();
   m_chkSetVisible->setChecked(true);
   connect(m_chkSetVisible,
           SIGNAL(stateChanged(int)),
           this,
           SLOT(setActionsEnabled(int)));

   m_toolbar->addWidget(enableLable);
   m_toolbar->addWidget(m_chkSetVisible);
   m_toolbar->addAction(m_rulerAction);
//   m_toolbar->addAction(m_intensityLineAction);
//   m_toolbar->addAction(m_intensityPieAction);
   m_toolbar->addAction(m_markerAction);
//   m_toolbar->addAction(m_crossHairAction);

}

AnnotationToolBarWidget::~AnnotationToolBarWidget()
{

   if(m_rulerAction != nullptr)
   {
      delete m_rulerAction;
      m_rulerAction = nullptr;
   }

   if(m_intensityLineAction != nullptr)
   {
      delete m_intensityLineAction;
      m_intensityLineAction = nullptr;
   }

   if(m_intensityPieAction != nullptr)
   {
      delete m_intensityPieAction;
      m_intensityPieAction = nullptr;
   }

   if(m_markerAction != nullptr)
   {
      delete m_markerAction;
      m_markerAction = nullptr;
   }

   if(m_crossHairAction != nullptr)
   {
      delete m_crossHairAction;
      m_crossHairAction = nullptr;
   }

}

/*---------------------------------------------------------------------------*/

QWidget* AnnotationToolBarWidget::getToolBar()
{

   return m_toolbar;

}

/*---------------------------------------------------------------------------*/

void AnnotationToolBarWidget::setActionsEnabled(int state)
{

   bool enabled = false;
   if (state == Qt::Checked)
   {
      enabled = true;
   }

   m_rulerAction->setEnabled(enabled);
   m_markerAction->setEnabled(enabled);

   emit enabledStateChanged(enabled);

}

/*---------------------------------------------------------------------------*/
