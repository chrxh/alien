#pragma once

#include <deque>

#include "Definitions.h"
#include "AlienWindow.h"
#include "PersisterInterface/PersisterFacade.h"

enum class SavepointState
{
    InQueue,
    InProgress,
    Persisted,
    Error
};

struct SavepointEntry
{
    SavepointState state = SavepointState::InQueue;
    std::string id;
    std::string timestamp;
    std::string name;
    uint64_t timestep;
};

class AutosaveWindow : public AlienWindow<SimulationFacade, PersisterFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(AutosaveWindow);

private:
    AutosaveWindow();

    void initIntern(SimulationFacade simulationFacade, PersisterFacade persisterFacade) override;
    void shutdownIntern() override;
    void processIntern() override;

    void processToolbar();
    void processHeader();
    void processTable();
    void processSettings();

    void createSavepoint();
    void updateSavepoint(SavepointEntry& savepoint);

    void validationAndCorrection();

    SimulationFacade _simulationFacade; 
    PersisterFacade _persisterFacade;

    bool _settingsOpen = false;
    float _settingsHeight = 130.0f;
    std::string _origLocation;
    std::string _location;

    bool _autosaveEnabled = false;
    int _origAutosaveInterval = 40;
    int _autosaveInterval = 40;

    using SaveMode = int;
    enum SaveMode_
    {
        SaveMode_Circular,
        SaveMode_Unlimited
    };
    SaveMode _origSaveMode = SaveMode_Circular;
    SaveMode _saveMode = SaveMode_Circular;
    int _origNumberOfFiles = 20;
    int _numberOfFiles = 20;

    std::deque<SavepointEntry> _savePoints;
};
