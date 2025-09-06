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

    void onCreateSavepoint(bool usePeakSimulation);
    void onDeleteSavepoint(SavepointEntry const& entry);
    void onLoadSavepoint(SavepointEntry const& entry);

    void processStateUpdates();

    void scheduleDeleteNonPersistentSavepoint(std::vector<SavepointEntry> const& entries);
    void processDeleteNonPersistentSavepoint();

    void scheduleCleanup();
    void processCleanup();

    void processAutomaticSavepoints();

    void updateSavepoint(int row);

    void updateSavepointTableFromFile();
    std::string getSavepointFilename() const;
    void validateAndCorrect();

    SimulationFacade _simulationFacade; 
    PersisterFacade _persisterFacade;

    bool _settingsOpen = false;
    float _settingsHeight = scale(130.0f);
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

    std::chrono::steady_clock::time_point _lastAutosaveTimepoint;

    using CatchPeaks = int;
    enum CatchPeaks_
    {
        CatchPeaks_None,
        CatchPeaks_Variance
    };
    CatchPeaks _origCatchPeaks = CatchPeaks_None;
    CatchPeaks _catchPeaks = _origCatchPeaks;
    std::chrono::steady_clock::time_point _lastPeakTimepoint;
    TaskProcessor _peakProcessor;
    SharedDeserializedSimulation _peakDeserializedSimulation;
    std::optional<int> _lastSessionId;
};
