#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"

#include "Definitions.h"
#include "AlienWindow.h"

class TemporalControlWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TemporalControlWindow);

public:
    void init(SimulationFacade const& simulationFacade);

    void onSnapshot();

private:
    TemporalControlWindow();

    struct Snapshot
    {
        uint64_t timestep;
        std::chrono::milliseconds realTime;
        SimulationParameters parameters;
        DataDescription data;
    };

    void processIntern();

    void processTpsInfo();
    void processTotalTimestepsInfo();
    void processRealTimeInfo();
    void processTpsRestriction();

    void processRunButton();
    void processPauseButton();
    void processStepBackwardButton();
    void processStepForwardButton();
    void processCreateFlashbackButton();
    void processLoadFlashbackButton();

    Snapshot createSnapshot();
    void applySnapshot(Snapshot const& snapshot);

    template <typename MovedObjectType>
    void restorePosition(MovedObjectType& movedObject, MovedObjectType const& origMovedObject, uint64_t origTimestep);
    
    SimulationFacade _simulationFacade; 

    std::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;

    bool _slowDown = false;
    int _tpsRestriction = 30;
};

