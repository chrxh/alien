#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _TemporalControlWindow : public _AlienWindow
{
public:
    _TemporalControlWindow(SimulationController const& simController, StatisticsWindow const& statisticsWindow);

private:
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

    SimulationController _simController; 
    StatisticsWindow _statisticsWindow;

    struct Snapshot
    {
        uint64_t timestep;
        DataDescription data;
    };
    std::optional<Snapshot> _snapshot;

    std::vector<Snapshot> _history;

    bool _slowDown = false;
    int _tpsRestriction = 30;
};