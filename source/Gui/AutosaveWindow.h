#pragma once

#include <deque>

#include "PersisterInterface/PersisterFacade.h"
#include "PersisterInterface/SavepointTable.h"

#include "Definitions.h"
#include "AlienWindow.h"

class AutosaveWindow : public AlienWindow<SimulationFacade, PersisterFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(AutosaveWindow);

private:
    AutosaveWindow();

    void initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void processToolbar();
    void processHeader();
    void processTable();
    void processSettings();
    void processStatusBar();

    void onCreateSavepoint();
    void onDeleteSavepoint(SavepointEntry const& entry);

    void scheduleDeleteNonPersistentSavepoint(std::vector<SavepointEntry> const& entries);
    void processDeleteNonPersistentSavepoint();

    void scheduleCleanup();
    void processCleanup();

    void processAutomaticSavepoints();

    void updateSavepoint(int row);

    void updateSavepointTableFromFile();
    std::string getSavepointFilename() const;
    void validationAndCorrection();

    SimulationFacade _simulationFacade; 
    PersisterFacade _persisterFacade;

    bool _settingsOpen = false;
    float _settingsHeight = 130.0f;
    std::string _origDirectory;
    std::string _directory;

    bool _autosaveEnabled = false;
    int _origAutosaveInterval = 40;
    int _autosaveInterval = _origAutosaveInterval;

    using SaveMode = int;
    enum SaveMode_
    {
        SaveMode_Circular,
        SaveMode_Unlimited
    };
    SaveMode _origSaveMode = SaveMode_Circular;
    SaveMode _saveMode = _origSaveMode;
    int _origNumberOfFiles = 20;
    int _numberOfFiles = _origNumberOfFiles;

    std::optional<SavepointTable> _savepointTable;
    SavepointEntry _selectedEntry;
    std::vector<SavepointEntry> _savepointsInProgressToDelete;

    bool _scheduleCleanup = false;

    std::optional<std::chrono::steady_clock::time_point> _lastAutosaveTimepoint;

    using CatchPeak = int;
    enum CatchPeak_
    {
        CatchPeak_None,
        CatchPeak_Variance
    };
    CatchPeak _origCatchPeak = CatchPeak_None;
    CatchPeak _catchPeak = _origCatchPeak;
    std::optional<std::chrono::steady_clock::time_point> _lastCatchPeakTimepoint;
    TaskProcessor _catchPeakProcessor;
};
