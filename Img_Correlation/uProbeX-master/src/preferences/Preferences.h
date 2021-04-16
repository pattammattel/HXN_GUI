/*-----------------------------------------------------------------------------
 * Copyright (c) 2020, UChicago Argonne, LLC
 * See LICENSE file.
 *---------------------------------------------------------------------------*/

#ifndef PREFERENCES_H
#define PREFERENCES_H

/*---------------------------------------------------------------------------*/

#include <QString>
#include <QVariant>
#include <unordered_map>
#include <mutex>

/*---------------------------------------------------------------------------*/

#define STR_PRF_NMCoefficient "NMCoefficient"
#define STR_PRF_NMOptions "NMOptions"
#define STR_PRF_ProfileList "ProfileList"
#define STR_PRF_ProfilePythonSolverName "ProfilePythonSolverName"
#define STR_PRF_ProfilePythonCoefficient "ProfilePythonCoefficient"
#define STR_PRF_ProfilePythonOptions "ProfilePythonOptions"
#define STR_PRF_ProfileIndex "ProfileIndex"
#define STR_PRF_PythonSolverName "PythonSolverName"
#define STR_PRF_PythonCoefficient "PythonCoefficient"
#define STR_PRF_PythonOptions "PythonOptions"
#define STR_PRF_SolverCheckedID "SolverCheckedID"
#define STR_PRF_FontSize "FontSize"
#define STR_PRF_RulerUnitLabel "RulerUnitLabel"
#define STR_PRF_RulerUnitX "RulerUnitX"
#define STR_PRF_RulerUnitY "RulerUnitY"
#define STR_PRF_RegionMenuList "RegionMenuList"
#define STR_PRF_MainWindowSavedWidth "MainWindowSavedWidth"
#define STR_PRF_MainWindowSavedHeight "MainWindowSavedHeight"
#define STR_PRF_WindowTitle "WindowTitle"
#define STR_PRF_DecimalPrecision "DecimalPrecision"
#define STR_PRF_MicroProbeXPv "MicroProbeXPv"
#define STR_PRF_MicroProbeYPv "MicroProbeYPv"
#define STR_PRF_AutoSaveRecoveryEnable "AutoSaveRecoveryEnable"
#define STR_PRF_AutoSaveRecoveryEveryMiliseconds "AutoSaveRecoveryEveryMiliseconds"
#define STR_PRF_AutoSaveOnExit "AutoSaveOnExit"
#define STR_PRF_ExportZoomToRegion "ExportZoomToRegion"
#define STR_PRF_ExportDirectoryInDataset "ExportDirectoryInDataset"
#define STR_PRF_ExportPrintNameOnExportedImage "ExportPrintNameOnExportedImage"
#define STR_PRF_ExportPrintPxPyOnExportedImage "ExportPrintPxPyOnExportedImage"
#define STR_PRF_ExportPrintWidthHeightOnExportedImage "ExportPrintWidthHeightOnExportedImage"
#define STR_PRF_ExportSelectedXmlOption "ExportSelectedXmlOption"
#define STR_PRF_LastIP "LastIP"
#define STR_PRF_LastPort "LastPort"
#define STR_PFR_LOG_10 "Log10YAxis"
#define STR_PFR_DETAILED_FIT_SPEC "DetailedFitSpectra"

/**
 * @brief Read and save preferences between application restarts, the vaule key
 * is divided into PVList, EnergyFlux, MirrorEnery, HISTBins, which is
 * corresponding to the preference dialog. Preferences settings are recorded
 * using QSetting class.
 */
class Preferences 
{

public:


   static Preferences* inst();

   ~Preferences();

   /**
    * Read a ValueKey from persistent storage.
    *
    * @param key - the ValueKey enum for the key to read
    * @return value for given key as a QVariant
    */
   QVariant getValue(QString key) const;  

   /**
    * Save a ValueKey to persistent storage.
    *
    * @param key - the ValueKey enum for the key to read
    * @param value - the value to store as a QVariant
    */
   void setValue(QString key, QVariant value);

   /**
   * load from file
   */
   void load();

   /**
   * save to file
   */
   void save();


private:

    /**
     * Constructor
     */
    Preferences();

    static Preferences* _this_inst;

    static std::mutex _mutex;

    std::unordered_map<std::string, QVariant> _pref_map;
};

/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
