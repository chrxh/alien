#pragma once

#include <chrono>
#include <string>
#include <vector>

#include <Base/Singleton.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/SavepointTable.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class AutosaveController : public MainLoopEntity
{
    MAKE_SINGLETON(AutosaveController);

public:
    void process() override;

    using SaveMode = int;
    enum SaveMode_
    {
        SaveMode_Circular,
        SaveMode_Unlimited
    };

    using CatchPeaks = int;
    enum CatchPeaks_
    {
        CatchPeaks_None,
        CatchPeaks_Variance
    };

    bool isAutosaveEnabled() const;
    void setAutosaveEnabled(bool value);

    int getAutosaveInterval() const;
    void setAutosaveInterval(int value);

    std::string const& getDirectory() const;
    void setDirectory(std::string const& value);

    SaveMode getSaveMode() const;
    void setSaveMode(SaveMode value);

    int getNumberOfFiles() const;
    void setNumberOfFiles(int value);

    std::optional<SavepointTable> const& getSavepointTable() const;
    std::chrono::seconds getDurationUntilNextAutosave() const;

    std::optional<SavepointEntry> getPersistedSavepoint();

    void onCreateSavepoint(bool usePeakSimulation);
    void onDeleteSavepoint(SavepointEntry const& entry);
    void scheduleCleanup();

private:
    void init() override;
    void shutdown() override;

    void processStateUpdates();
    void processDeleteNonPersistentSavepoint();
    void processCleanup();
    void processAutomaticSavepoints();

    void scheduleDeleteNonPersistentSavepoint(std::vector<SavepointEntry> const& entries);
    void updateSavepoint(int row);
    void updateSavepointTableFromFile();
    std::string getSavepointFilename() const;

    std::string _directory;
    bool _autosaveEnabled = false;
    int _autosaveInterval = 40;
    SaveMode _saveMode = SaveMode_Circular;
    int _numberOfFiles = 20;
    CatchPeaks _catchPeaks = CatchPeaks_None;

    std::optional<SavepointTable> _savepointTable;
    std::optional<SavepointEntry> _persistedSavepoint;
    std::vector<SavepointEntry> _savepointsInProgressToDelete;
    bool _scheduleCleanup = false;

    std::chrono::steady_clock::time_point _lastAutosaveTimepoint;
    std::chrono::steady_clock::time_point _lastPeakTimepoint;
    TaskProcessor _peakProcessor;
    SharedDeserializedSimulation _peakDeserializedSimulation;
    std::optional<int> _lastSessionId;
};
