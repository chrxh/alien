#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationParameters.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _TemporalControlWindow : public _AlienWindow
{
public:
    _TemporalControlWindow(SimulationController const& simController, StatisticsWindow const& statisticsWindow);

    void onSnapshot();

private:
    struct Snapshot
    {
        uint64_t timestep;
        SimulationParameters parameters;
        DataDescription data;
    };

    void processIntern();

    void processTpsInfo();
    void processTotalTimestepsInfo();
    void processTpsRestriction();

    void processRunButton();
    void processPauseButton();
    void processStepBackwardButton();
    void processStepForwardButton();
    void processSnapshotButton();
    void processRestoreButton();

    Snapshot createSnapshot();
    void applySnapshot(Snapshot const& snapshot);

    SimulationController _simController; 
    StatisticsWindow _statisticsWindow;

    std::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;

    bool _slowDown = false;
    int _tpsRestriction = 30;
};