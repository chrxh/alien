#pragma once

#include "EngineInterface/PersisterWorker.h"

#include "AlienWindow.h"
#include "Definitions.h"

class _AutosaveWindow : public _AlienWindow
{
public:
    _AutosaveWindow(SimulationController const& simController);
    ~_AutosaveWindow();

private:
    void processIntern() override;

    void processToolbar();
    void processHeader();
    void processTable();
    void processSettings();

    void onCreateSave();

    void validationAndCorrection();

    SimulationController _simController;

    bool _settingsOpen = false;
    float _settingsHeight = 100.0f;
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

    struct SavePointEntry
    {
        bool transient = true;
        int sequenceNumber = 0;
        std::string id;
        std::string timestamp;
        std::string name;
        uint64_t timestep;
    };
    std::vector<SavePointEntry> _savePoints;

    PersisterWorker _worker;
    std::thread* _thread = nullptr;
};
