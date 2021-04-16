﻿/*-----------------------------------------------------------------------------
 * Copyright (c) 2013, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#include "gstar/AnnotationTreeModel.h"

#include "gstar/Annotation/AbstractGraphicsItem.h"
#include "gstar/Annotation/EmptyGraphicsItem.h"
#include "AnnotationProperty.h"

#include <QFont>
#include <QItemSelectionModel>
#include <QPointF>
#include <QStringList>

using namespace gstar;

/*---------------------------------------------------------------------------*/

AnnotationTreeModel::AnnotationTreeModel(QObject* parent) : QAbstractItemModel(parent)
{

   m_root = new EmptyGraphicsItem();
   m_root->appendProperty(new AnnotationProperty("", "Items"));

}

/*---------------------------------------------------------------------------*/

AnnotationTreeModel::~AnnotationTreeModel()
{  
   if (m_root != nullptr)
   {
      delete m_root;
      m_root = nullptr;
   }

   m_groups.clear();
   m_groupsCnt.clear();

}

/*---------------------------------------------------------------------------*/

QModelIndex AnnotationTreeModel::appendNode(AbstractGraphicsItem* item)
{

   //int row = rowCount();

   connect(item,
           SIGNAL(viewUpdated(AbstractGraphicsItem*)),
           this,
           SLOT(refreshModel(AbstractGraphicsItem*)));

   QString displayName = item->displayName();
   if (m_groups.contains(item->classId()) == false)
   {
      createGroup(item);
   }
   AbstractGraphicsItem* groupRoot = m_groups[item->classId()];

   m_groupsCnt[item->classId()]++;
   QString dName = QString("%1%2")
                   .arg(displayName)
         .arg(m_groupsCnt[item->classId()]);
   item->prependProperty(new AnnotationProperty(DEF_STR_DISPLAY_NAME, dName));
   item->setParent(groupRoot);

   int row = groupRoot->childCount();
  // QModelIndex rIndex = index(0, 0, QModelIndex());
   QModelIndex gIndex = index(groupRoot->row(), 0, QModelIndex());

   beginInsertRows(gIndex, row, row);

   groupRoot->appendChild(item);

   endInsertRows();

   return gIndex;

}

/*---------------------------------------------------------------------------*/

void AnnotationTreeModel::calculate()
{

   recursiveCalculate(m_root);

}

/*---------------------------------------------------------------------------*/

int AnnotationTreeModel::columnCount(const QModelIndex& parent) const
{

   AbstractGraphicsItem* pItem;

   if(parent.isValid())
   {
      pItem = static_cast<AbstractGraphicsItem*>(parent.internalPointer());
   }
   else
   {
      pItem = m_root;
   }

   return pItem->columnCount();

}

/*---------------------------------------------------------------------------*/

AbstractGraphicsItem* AnnotationTreeModel::createGroup(AbstractGraphicsItem* item)
{

   QList<AnnotationProperty*> pList = item->properties();

   AbstractGraphicsItem* groupRoot = new EmptyGraphicsItem();
   groupRoot->setParent(m_root);
   groupRoot->appendProperty(new AnnotationProperty("", item->displayName()));

   foreach (AnnotationProperty* prop, pList)
   {
      groupRoot->appendProperty(new AnnotationProperty("", prop->getName()));
   }

   //root index
   int row = m_root->childCount();
   //QModelIndex rIndex = index(0, 0, QModelIndex());
   beginInsertRows(QModelIndex(), row, row);

   m_groupsCnt[item->classId()] = 0;
   m_groups[item->classId()] = groupRoot;
   m_root->appendChild(groupRoot);

   endInsertRows();

   int rootCount = m_root->properties().count();
   int pCount = pList.count() + 1;
   if (pCount > rootCount)
   {
      int nDiff = pCount - rootCount;
      beginInsertColumns(QModelIndex(), 0, nDiff - 1);
      for (int i = 0; i < nDiff; i++)
      {
         m_root->appendProperty(new AnnotationProperty("",""));
      }
      endInsertColumns();
   }

   return groupRoot;

}

/*---------------------------------------------------------------------------*/

QVariant AnnotationTreeModel::data(const QModelIndex& index, int role) const
{

    if (!index.isValid())
    {
        return QVariant();
    }

    AbstractGraphicsItem* item = static_cast<AbstractGraphicsItem*>(index.internalPointer());

    QVariant var = item->data(index.row(), index.column());

    if(role == Qt::FontRole)
    {
        if (item->parent() == m_root)
        {
            QFont font;
            font.setBold(true);
            return font;
        }
    }
    else if (role == Qt::DecorationRole)
    {
        if (var.type() == QVariant::Color)
        {
            return QColor(var.toString());
        }
    }
    else if ( role == Qt::CheckStateRole  )
    {
        if (var.type() == QVariant::Bool)
        {
            Qt::CheckState eChkState = ( item->data(index.row(), index.column() ).toBool() ) ? Qt::Checked : Qt::Unchecked;
            return eChkState;
        }
    }
    else if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        //this stops the color variants from displaying the color as
        // a hex value in the tree
        QVariant var = item->data(index.row(), index.column());
        if (var.type() == QVariant::Color || var.type() == QVariant::Bool)
        {
            return QVariant();
        }

        return item->data(index.row(), index.column());
    }

    return QVariant();

}

/*---------------------------------------------------------------------------*/

 bool AnnotationTreeModel::setData(const QModelIndex& index, const QVariant& value, int role)
 {
    bool changed  = false;
   if (value.type() == QVariant::String)
   {
      QString sValue = value.toString();
      if (sValue.length() < 1)
      {
         return false;
      }
   }

   if (!index.isValid())
   {
      return false;
   }

   AbstractGraphicsItem* item = static_cast<AbstractGraphicsItem*>(index.internalPointer());

   QVariant var = item->data(index.row(), index.column());

   if (item->parent() == m_root)
   {
      return false;
   }

    if ( role == Qt::CheckStateRole && var.type() == QVariant::Bool)
    {
        Qt::CheckState eChecked = static_cast< Qt::CheckState >( value.toInt() );
        bool bNewValue = eChecked == Qt::Checked;
        changed = item->setData( index, bNewValue );
    }
    else if(role == Qt::EditRole)
    {
        changed = item->setData(index, value);
    }
    if (changed)
        emit dataChanged(index, index);

    return changed;

}

/*---------------------------------------------------------------------------*/

QModelIndex AnnotationTreeModel::duplicateNode(const QModelIndex& index)
{

   if (index.isValid() == false)
      return QModelIndex();

   AbstractGraphicsItem* item = static_cast<AbstractGraphicsItem*>
         (index.internalPointer());

   if (item == nullptr)
      return QModelIndex();

   AbstractGraphicsItem* pItem = item->parent();

   if (pItem == m_root)
   {
      return QModelIndex();
   }


   AbstractGraphicsItem* newItem = item->duplicate();
   newItem->updateView();
   QModelIndex pIndex = appendNode(newItem);
   if (pIndex.isValid())
   {
      QModelIndex cIndex = pIndex.child(newItem->row(), 0);
      return cIndex;
   }

   return QModelIndex();
}

/*---------------------------------------------------------------------------*/

Qt::ItemFlags AnnotationTreeModel::flags(const QModelIndex& index) const
{

   // Check for valid index
   if (!index.isValid())
      return 0;

   // Get desired index
   int c = index.column();

   // Invalid column
   if (c < 0 || c >= columnCount(QModelIndex()))
   {
      return 0;
   }

   AbstractGraphicsItem* item = static_cast<AbstractGraphicsItem*>(index.internalPointer());

   if (item->parent() == m_root || item == m_root)
   {
      return Qt::ItemIsSelectable | Qt::ItemIsEnabled;
   }

   return item->displayFlags(index.row(), index.column());

}

/*---------------------------------------------------------------------------*/

QVariant AnnotationTreeModel::headerData(int section,
                                   Qt::Orientation orientation,
                                   int role) const
{

   Q_UNUSED(section);

   // Check this is DisplayRole
   if (role != Qt::DisplayRole)
      return QVariant();

   // Horizontal headers
   if (orientation == Qt::Horizontal)
   {
      /*
      // Headers
      switch (section)
      {
      case AbstractGraphicsItem::NAME:
         return "Dispaly Name";
         break;
      case AbstractGraphicsItem::VALUE:
         return "Value";
         break;
      default:
         return "";
         break;
      }
      */
      return "";
   }

   // Return empty variant
   return QVariant();

}

/*---------------------------------------------------------------------------*/

QModelIndex AnnotationTreeModel::index(int row,
                                 int column,
                                 const QModelIndex& parent) const
{

   //if (!hasIndex(row, column, parent))
   //   return QModelIndex();

   AbstractGraphicsItem* parentItem;

   if (m_root->childCount() < 1)
   {
      return QModelIndex();
   }

   if(!parent.isValid())
   {
      parentItem = m_root;
   }
   else
   {
      parentItem = static_cast<AbstractGraphicsItem*>(parent.internalPointer());
   }

   if (parentItem == nullptr)
   {
      return QModelIndex();
   }

   if (parentItem->childCount() < row)
   {
      return QModelIndex();
   }

   AbstractGraphicsItem* childItem = parentItem->child(row);

   if (childItem)
   {
      return createIndex(row, column, childItem);
   }
   else
   {
      return QModelIndex();
   }

}

/*---------------------------------------------------------------------------*/

bool AnnotationTreeModel::insertRows(int row,
                               int count,
                               const QModelIndex& parent)
{

   // Mark unused
   Q_UNUSED(row);
   Q_UNUSED(parent);
   Q_UNUSED(count);

   // Check bounds

   // Indicate beginning of insert
//   beginInsertRows(QModelIndex(), row, row + count - 1);

//   AbstractGraphicsItem* newItem = new AbstractGraphicsItem(m_root);
//   m_root->appendChild(newItem);

   // Indicate end of insert
//   endInsertRows();

   // Return true
   return true;

}

/*---------------------------------------------------------------------------*/

QModelIndex AnnotationTreeModel::parent(const QModelIndex& index)const
{

   if (!index.isValid())
   {
      return QModelIndex();
   }

   AbstractGraphicsItem* childItem =
         static_cast<AbstractGraphicsItem*>(index.internalPointer());

   if (childItem == m_root || childItem == nullptr)
   {
      return QModelIndex();
   }

   if (m_root->hasChild(childItem) == false)
   {
      return QModelIndex();
   }

   AbstractGraphicsItem* parentItem = childItem->parent();

   if (parentItem == m_root || parentItem == nullptr)
   {
      return QModelIndex();
   }

   return createIndex(parentItem->row(), 0, parentItem);

}

/*---------------------------------------------------------------------------*/

void AnnotationTreeModel::recursiveCalculate(AbstractGraphicsItem* pItem)
{

   foreach(AbstractGraphicsItem* item , pItem->childList())
   {
      item->calculate();
      recursiveCalculate(item);
   }

}

/*---------------------------------------------------------------------------*/

void AnnotationTreeModel::refreshModel(AbstractGraphicsItem* item)
{

   QModelIndex left = createIndex(item->row(), 0, item);
   QModelIndex right = createIndex(item->row(), item->columnCount(), item);
   emit dataChanged(left, right);

}

/*---------------------------------------------------------------------------*/

bool AnnotationTreeModel::removeRow(int row,
                              const QModelIndex& parent)
{

   int count = 1;

   if (m_root->childCount() < 1)
      return false;

   if (parent.isValid() == false)
      return false;

   AbstractGraphicsItem* item =
           static_cast<AbstractGraphicsItem*>(parent.internalPointer());

   if (item == nullptr)
   {
       return false;
   }

   if (m_root->hasChild(item) == false)
      return false;

   AbstractGraphicsItem* pItem = item->parent();

   if (pItem == nullptr)
   {
       return false;
   }

   //check if group is being deleted
   if (pItem == m_root)
   {
      //this is a group item, delete all the children under it
      //count = item->childCount();

      beginRemoveRows(QModelIndex(), row, row + count - 1);
      //beginRemoveRows(parent, row, row + count - 1);

      QString classIdName;
      for (int i = count - 1; i >= 0; i--)
      {
         AbstractGraphicsItem* cItem = item->child(i);
         item->removeChildAt(i);
         classIdName = cItem->classId();
         delete cItem;
      }
      m_root->removeChildAt(item->row());
      m_groups.remove(classIdName);
      delete item;
      endRemoveRows();
   }
   else
   {
      row = item->row();

      if(row < 0 || row > (pItem->childCount() - 1))
      {
         return false;
      }

      // Indicate beginning of removal
      //beginRemoveRows(QModelIndex(), row, row + count - 1);
      //QModelIndex parentIndex = createIndex(pItem->row(), 0, pItem);
      QModelIndex parentIndex = index(pItem->row(), 0, QModelIndex());
      beginRemoveRows(parentIndex, row, row + count - 1);

      pItem->removeChildAt(row);
      QString classIdName = item->classId();
      delete item;

      if (pItem->childCount() == 0)
      {
         m_root->removeChildAt(pItem->row());
         m_groups.remove(classIdName);
         delete pItem;
      }

      // Indicate end of removal
      endRemoveRows();
   }
   return true;

}

/*---------------------------------------------------------------------------*/

bool AnnotationTreeModel::removeRows(int row,
                               int count,
                               const QModelIndex& parent)
{
   if (parent.isValid() == false)
      return false;

   AbstractGraphicsItem* pItem =
           static_cast<AbstractGraphicsItem*>(parent.internalPointer());

   if (pItem == nullptr)
   {
       return false;
   }
   if (row < 0 || (row + count) > pItem->childCount())
   {
      return false;
   }

   // Indicate beginning of removal
   beginRemoveRows(QModelIndex(), row, row + count - 1);

   for (int i = row + count - 1; i >= row; i--)
   {
       AbstractGraphicsItem* dItem = pItem->child(row);
       pItem->removeChildAt(row);
       delete dItem;
   }

   // Indicate end of removal
   endRemoveRows();

   return true;

}

/*---------------------------------------------------------------------------*/

int AnnotationTreeModel::rowCount(const QModelIndex& parent) const
{

   AbstractGraphicsItem *parentItem;
   if (parent.column() > 0)
      return 0;

   if (!parent.isValid())
   {
      parentItem = m_root;
   }
   else
   {
      parentItem = static_cast<AbstractGraphicsItem*>(parent.internalPointer());
   }

   return parentItem->childCount();

}

/*---------------------------------------------------------------------------*/

QList<AbstractGraphicsItem*> AnnotationTreeModel::get_all_of_type(const QString type_name)
{
    QList<AbstractGraphicsItem*> na;
    if(m_groups.count(type_name) > 0)
    {
        return m_groups[type_name]->childList();
    }
    return na;
}

/*---------------------------------------------------------------------------*/
