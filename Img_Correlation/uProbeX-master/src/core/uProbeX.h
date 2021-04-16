/*-----------------------------------------------------------------------------
 * Copyright (c) 2014, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#ifndef uProbeX_H
#define uProbeX_H

/*---------------------------------------------------------------------------*/

#include <QMainWindow>
#include <QDesktopWidget>
#include <QApplication>
#include <QMap>
#include <QUuid>
#include <preferences/Preferences.h>
#include <mvc/MapsH5Model.h>
#include <mvc/LiveMapsElementsWidget.h>
#include <mvc/MapsWorkspaceController.h>


class SubWindow;
class AbstractWindowController;

class QAction;
class QActionGroup;
class QMenu;
class QMenuBar;
class QToolBar;
class QMdiArea;
class QMdiSubWindow;
class SWSModel;
class RAW_Model;

/*---------------------------------------------------------------------------*/

/**
 * @brief The main uProbeX user interface window. Main application class
 */
class uProbeX
: public QMainWindow
{

   Q_OBJECT

public:

   /**
    * Constructor. See Qt documentation for QMainWindow.
    */
    uProbeX(QWidget* parent, Qt::WindowFlags flags = 0);

   /**
    * Destructor
    */
   ~uProbeX();

   //Static log windows 
   static QTextEdit * log_textedit;

signals:

   void loadMapsWorkspace(MapsWorkspaceController*, QString);

private slots:

   /**
    * @brief Reimplemented from QMainWindow. See Qt documentation.
    * @param event
    */
   void closeEvent(QCloseEvent* event);

   /**
    * @brief Handle application closing task. Triggered from the exit menu as well.
    */
   void exitApplication();

   /**
    * @brief openLiveStreamViewer
    */
   void openLiveStreamViewer();

   /**
    * @brief makeSWSWindow
    * @param path
    * @param newWindow
    */
   void makeSWSWindow(QString path, bool newWindow = false);

   /**
    * @brief makeSWSWindow
    * @param swsModel
    */
   void makeSWSWindow(SWSModel* swsModel);

   /**
    * @brief laodMapsWorkspace
    * @param path
    */
   void onLoadMapsWorkspace(MapsWorkspaceController* controller, QString path);

   /**
    * @brief makeHDFWindow
    * @param path
    * @param newWindow
    */
   void makeHDFWindow(QString path);

   /**
    * @brief makeHDFWindow
    * @param model
    */
   void makeHDFWindow(MapsH5Model* model);

   /**
    * @brief makeMDAWindow
    * @param model
    */
   void makeMDAWindow(RAW_Model* model);

   /**
    * @brief Open the SWS workspace. Triggered from the menu.
    */
   void openSWSFile();

   /**
    * @brief openMAPSWorkspace
    */
   void openMapsWorkspace();

   /**
    * @brief openHDFFile
    */
   void openHDFFile();

   /**
    * @brief Save the screen shot. Triggered from the menu.
    */
   void saveScreenShot();

   /**
    * @brief saveActivatedXML
    */
   void saveActivatedXML();

   /**
    * @brief saveAllXML
    */
   void saveAllXML();

   /**
    * @brief Show about dialog.Triggered from the menu.
    */
   void showAbout();

   /**
    * @brief Show the preferences dialog. Triggered from the menu.
    */
   void showPreferences();

   /**
    * @brief processPreferencesUpdate
    */
   void processPreferencesUpdate();

   /**
    * @brief Handle event when a child MDI window is closed.
    * @param subWindow
    */
   void subWindowClosed(SubWindow* subWindow);

   /**
    * @brief updatePreference
    */
   void updatePreference();

   /**
    * @brief menuBarEnable
    */
   void menuBarEnable();

   /**
    * @brief perform autosave for the open datasets.
    */
   void performAutoSave();

   /**
    * @brief cleanUpAutoSafeData
    */
   void cleanUpAutoSafeData();
   /*
   void savePreferencesXMLData();

   void loadPreferencesXMLData();
   */
   void mapsControllerClosed(MapsWorkspaceController*);

private:

   /**
    * @brief adjustAutoSaveSettings
    */
   //void adjustAutoSaveSettings();

   /**
    * @briefAdjust displays settings (Font type, size etc).
    *
    * The settings are read from the preferences object (if  exists)
    */
   void adjustDisplaySettings();      

   /**
    * @brief checkSameFileWindow
    * @param filePath
    * @return ture if two window open same image
    */
   bool checkSameFileWindow(QString& filePath);

   /**
    * @brief Create menu bar
    */
   void createMenuBar();

   /**
    * @brief Default initialization of variables such as setting pointers to nullptr
    */
   void initialize();
   
   /**
    * @brief updateContextMenus
    */
   void updateContextMenus();

   /**
    * @brief saveAllXML
    */
   void saveAllXML(bool verifyWithUser);

   /**
    * @brief saveAllXMLRequired
    * @return
    */
   bool saveAllXMLRequired();

   /**
    * @brief saveActiveXmlRequired
    * @return
    */
   bool saveActivatedXmlRequired();

private:

   /**
    * @brief Menu bar
    */
   QMenuBar* m_menu;

   /**
    * @brief File menu
    */
   QMenu* m_menuFile;

   /**
    * @brief Stream menu
    */
   QMenu* m_menuStream;

   /**
    * @brief Help menu
    */
   QMenu* m_menuHelp;

   /**
    * @brief MDI area that serves as a central widget
    */
   QMdiArea* m_mdiArea;

   /**
    * @brief Keeps track of all open MDI child windows for hdf5 play-back.
    */
   QMap<QUuid, SubWindow*> m_subWindows;

   /**
    * @brief timer used to execute autosafe.
    */
   //QTimer* m_autosaveTimer;

   /**
    * @brief _liveMapsViewer
    */
   LiveMapsElementsWidget*  _liveMapsViewer;

   // log dock
   QDockWidget *_log_dock;

};

/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
