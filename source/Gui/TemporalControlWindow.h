#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _TemporalControlWindow : public _AlienWindow
{
public:
    _TemporalControlWindow(SimulationFacade const& simulationFacade, StatisticsWindow const& statisticsWindow);

    void onSnapshot();

private:
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
    StatisticsWindow _statisticsWindow;

    std::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;

    bool _slowDown = false;
    int _tpsRestriction = 30;
};

