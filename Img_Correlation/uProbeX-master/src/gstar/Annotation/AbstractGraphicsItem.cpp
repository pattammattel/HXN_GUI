/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include "gstar/Annotation/AbstractGraphicsItem.h"

#include <QGraphicsScene>
#include <QGraphicsView>
#include <typeinfo>

using namespace gstar;

/*---------------------------------------------------------------------------*/

AbstractGraphicsItem::AbstractGraphicsItem(AbstractGraphicsItem* parent) :
   QGraphicsObject(parent)
{

   m_parent = parent;
   //m_decimalPreci = 2;

   initializePropertyName();

   setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsScenePositionChanges);

   connect(this,
           SIGNAL(xChanged()),
           this,
           SLOT(viewChanged()));

   connect(this,
           SIGNAL(yChanged()),
           this,
           SLOT(viewChanged()));

   connect(this,
           SIGNAL(zChanged()),
           this,
           SLOT(viewChanged()));

}

/*---------------------------------------------------------------------------*/

AbstractGraphicsItem::~AbstractGraphicsItem()
{

   if (m_children.count() > 0)
   {
      qDeleteAll(m_children);
   }
   m_children.clear();
   clearProperties();

   m_parent = nullptr;

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::appendChild(AbstractGraphicsItem* child)
{

   m_children.append(child);

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::appendProperty(AnnotationProperty* prop)
{

   m_data.push_back(prop);

}

/*---------------------------------------------------------------------------*/

AbstractGraphicsItem* AbstractGraphicsItem::child(int row)
{

   return m_children.value(row);

}

/*---------------------------------------------------------------------------*/

QString AbstractGraphicsItem::classId()
{

   return QString(typeid(*this).name());

}

/*---------------------------------------------------------------------------*/

QList<AbstractGraphicsItem*> AbstractGraphicsItem::childList() const
{

   return m_children;

}

/*---------------------------------------------------------------------------*/

int AbstractGraphicsItem::childCount() const
{

   return m_children.count();

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::clearProperties()
{

   if (m_data.count() > 0)
   {
      qDeleteAll(m_data);
   }
   m_data.clear();

}

/*---------------------------------------------------------------------------*/

int AbstractGraphicsItem::columnCount() const
{

   return m_data.count();

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::connectAllProperties()
{

   foreach (AnnotationProperty* prop, m_data)
   {
      connect(prop,
              SIGNAL(valueChanged()),
              this,
              SLOT(modelChanged()));
   }

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::connectAllViewItems()
{

   connect(this,
           SIGNAL(xChanged()),
           this,
           SLOT(viewChanged()));

   connect(this,
           SIGNAL(yChanged()),
           this,
           SLOT(viewChanged()));

   connect(this,
           SIGNAL(zChanged()),
           this,
           SLOT(viewChanged()));

   foreach (AbstractGraphicsItem* item, m_children)
   {
      connect(item,
              SIGNAL(xChanged()),
              this,
              SLOT(viewChanged()));

      connect(item,
              SIGNAL(yChanged()),
              this,
              SLOT(viewChanged()));

      connect(item,
              SIGNAL(zChanged()),
              this,
              SLOT(viewChanged()));
   }

}

/*---------------------------------------------------------------------------*/

QVariant AbstractGraphicsItem::data(int row, int column) const
{

   Q_UNUSED(row);

   // Invalid column
   if (column < 0 || column > columnCount())
   {
      return QVariant();
   }

   if (m_data.count() <= column)
   {
      return QVariant();
   }


   AnnotationProperty* data = m_data.value(column);

   QVariant::Type valueType = data->getValue().type();

   QPoint point;
   QPointF pointf;

   switch (valueType)
   {
   case QVariant::Point:
      point = data->getValue().toPoint();
      return QString("%1, %2").arg(point.x()).arg(point.y());
   case QVariant::PointF:
      pointf = data->getValue().toPointF();
      return QString("%1, %2").arg(pointf.x()).arg(pointf.y());
   case QVariant::StringList:
       return data->getValue().toStringList()[0];
   default:
      return data->getValue();
   }

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::disconnectAllProperties()
{

   foreach (AnnotationProperty* prop, m_data)
   {
      disconnect(prop,
                 SIGNAL(valueChanged()),
                 this,
                 SLOT(modelChanged()));
   }

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::disconnectAllViewItems()
{

   disconnect(this,
              SIGNAL(xChanged()),
              this,
              SLOT(viewChanged()));

   disconnect(this,
              SIGNAL(yChanged()),
              this,
              SLOT(viewChanged()));

   disconnect(this,
              SIGNAL(zChanged()),
              this,
              SLOT(viewChanged()));

   foreach (AbstractGraphicsItem* item, m_children)
   {
      disconnect(item,
                 SIGNAL(xChanged()),
                 this,
                 SLOT(viewChanged()));

      disconnect(item,
                 SIGNAL(yChanged()),
                 this,
                 SLOT(viewChanged()));

      disconnect(item,
                 SIGNAL(zChanged()),
                 this,
                 SLOT(viewChanged()));
   }

}

/*---------------------------------------------------------------------------*/

Qt::ItemFlags AbstractGraphicsItem::displayFlags(int row, int column) const
{

   Q_UNUSED(row);

    Qt::ItemFlags flags = Qt::NoItemFlags;

   if (column < 0 || column > m_data.count())
   {
      return flags;
   }

   AnnotationProperty* prop = m_data.value(column);
   if (prop == nullptr)
   {
      return flags;
   }

   QVariant::Type t = prop->getValue().type();
   switch(t)
   {
   case QVariant::Color:
      flags = Qt::ItemIsSelectable |  Qt::ItemIsEnabled;
       break;
   case QVariant::Bool:
      flags = Qt::ItemIsSelectable | Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsUserCheckable;
       break;
   default:
      flags = Qt::ItemIsSelectable |  Qt::ItemIsEditable | Qt::ItemIsEnabled;
       break;
   }

   return flags;
}

/*---------------------------------------------------------------------------*/

bool AbstractGraphicsItem::hasChild(AbstractGraphicsItem* child)
{

   foreach (AbstractGraphicsItem* item, m_children)
   {
      if (item == child || true == item->hasChild(child))
         return true;
   }

   return false;

}

/*---------------------------------------------------------------------------*/

QTransform AbstractGraphicsItem::getFirstViewTransform() const
{

   QTransform trans;

   QGraphicsScene *s = scene();
   if (s != nullptr)
   {
      QList<QGraphicsView*> v = s->views();
      if (v.length() > 0)
      {
         QGraphicsView* fv = v.first();
         trans = fv->transform();
      }
   }

   return trans;

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::initializePropertyName()
{

   //const QString UPROBECOLOR = "Color";

}

/*---------------------------------------------------------------------------*/

QVariant AbstractGraphicsItem::itemChange(GraphicsItemChange change,
                                          const QVariant& value)
{

  if (change == ItemPositionChange && scene())
  {
      // value is the new position.
      QPointF newPos = value.toPointF();

      if (m_parent != nullptr)
      {
         newPos += m_parent->pos();
      }

      double dScale = scale();
//      QRectF bRect = boundingRect();
      QRectF bRect = boundingRectMarker();

      QRectF rect = scene()->sceneRect();

      double halfrW = bRect.right()*dScale;
      double halfrH = bRect.bottom()*dScale;
      QPointF halfP(halfrW, halfrH);

      double halflW = -bRect.left()*dScale;
      double halflH = -bRect.top()*dScale;
      QPointF halfl(halflW, halflH);

      if (!rect.contains(newPos+halfP) ||
          !rect.contains(newPos - halfl))
      {
         double xp = newPos.x();
         if ((xp + halfrW) > rect.right())
         {
            xp = rect.right() - halfrW;
         }
         else if ((xp -halflW) < rect.left())
         {
            xp = rect.left() + halflW;
         }
         newPos.setX(xp);

         double yp = newPos.y();
         if ((yp + halfrH) > rect.bottom())
         {
            yp = rect.bottom() - halfrH;
         }
         else if ((yp - halflH) < rect.top())
         {
            yp = rect.top() + halflH;
         }
         newPos.setY(yp);

         // Keep the item inside the scene rect.
         //newPos.setX(qMin(rect.right(), qMax(newPos.x(), rect.left())));
         //newPos.setY(qMin(rect.bottom(), qMax(newPos.y(), rect.top())));

         if (m_parent != nullptr)
         {
            newPos -= m_parent->pos();
         }
         return newPos;
      }
   }
   else if (change == QGraphicsItem::ItemSelectedHasChanged)
   {
     bool selected = value.toBool();
     if (selected)
     {
        setZValue( 999 );
     }
     else
     {
        setZValue( 0 );
     }
   }
   return QGraphicsItem::itemChange(change, value);

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::modelChanged()
{

   disconnectAllViewItems();
   disconnectAllProperties();
   updateView();
   connectAllViewItems();
   connectAllProperties();

}

/*---------------------------------------------------------------------------*/

AbstractGraphicsItem* AbstractGraphicsItem::parent() const
{

   return m_parent;

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::prependProperty(AnnotationProperty* prop)
{

   m_data.push_front(prop);

}

/*---------------------------------------------------------------------------*/

QList<AnnotationProperty*> AbstractGraphicsItem::properties() const
{

   return m_data;

}

/*---------------------------------------------------------------------------*/

QVariant AbstractGraphicsItem::propertyValue(QString name) const
{

   foreach(AnnotationProperty* prop, m_data)
   {
      if(prop->getName() == name)
         return prop->getValue();
   }

   return QVariant();

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::setPropertyValue(QString name, QVariant value)
{

   foreach(AnnotationProperty* prop, m_data)
   {
      if(prop->getName() == name)
         prop->setValue(value);
   }

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::removeChild(AbstractGraphicsItem* item)
{

   m_children.removeOne(item);

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::removeChildAt(int row)
{

   m_children.removeAt(row);

}

/*---------------------------------------------------------------------------*/

int AbstractGraphicsItem::row() const
{

   if (m_parent)
      return m_parent->
            m_children.indexOf(const_cast<AbstractGraphicsItem*>(this));

   return 0;

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::setAllChildrenSelected(bool select)
{

   foreach (AbstractGraphicsItem* item , m_children)
   {
      item->setSelected(select);
   }

}

/*---------------------------------------------------------------------------*/

//void AbstractGraphicsItem::setCoordinatePrecision(int number)
//{

//   m_decimalPreci = number;

//}

/*---------------------------------------------------------------------------*/

bool AbstractGraphicsItem::setData(const QModelIndex& index,
                               const QVariant& value)
{

   // Get desired index
   int c = index.column();
//   int r = index.row();

/*
   // Invalid row
   if (r < 0 || r >= rowCount(QModelIndex()))
   {
      return false;
   }
*/
   // Invalid column
   if (c < 0 || c >= columnCount())
   {
      return false;
   }


   AnnotationProperty* data = m_data.value(c);

   QVariant::Type valueType = data->getValue().type();

   switch (valueType)
   {
   case QVariant::Double:
      if (value.type() == QVariant::String)
      {
         QString sVal = value.toString();
         bool ok = false;
         double dVal = sVal.toDouble(&ok);
         if (ok)
         {
            data->setValue(dVal);
         }
         return ok;
      }
      else if (value.type() == QVariant::Double ||
               value.type() == QVariant::Int)
      {
         data->setValue(value.toDouble());
      }
      break;
   case QVariant::PointF:
   case QVariant::Point:
      if (value.type() == QVariant::String)
      {
         QString sVal = value.toString();
         if(sVal.contains(","))
         {
            QStringList sList = sVal.split(",");
            if (sList.count() > 1)
            {
               int x = sList[0].toInt();
               int y = sList[1].toInt();
               data->setValue(QPoint(x,y));
            }
            else
               return false;
         }
         else
            return false;
      }
      else if (value.type() == QVariant::Point ||
               value.type() == QVariant::PointF )
      {
         data->setValue(value);
      }
      break;
   case QVariant::Color:
      if (value.type() == QVariant::String)
      {
         data->setValue(QColor(value.toString()));
      }
      else
      {
         data->setValue(value);
      }
      break;
   default:
      data->setValue(value);
      break;
   }

   return true;

}

/*---------------------------------------------------------------------------*/
/*
void AbstractGraphicsItem::setSelected(bool selected)
{

    QGraphicsItem::setSelected(selected);

}
*/
/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::setParent(AbstractGraphicsItem* parent)
{

   m_parent = parent;

}

/*---------------------------------------------------------------------------*/

void AbstractGraphicsItem::viewChanged()
{

   disconnectAllViewItems();
   disconnectAllProperties();
   updateModel();
   connectAllViewItems();
   connectAllProperties();

   emit viewUpdated(this);

}

/*---------------------------------------------------------------------------*/
