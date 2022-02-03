#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _OpenSimulationDialog
{
public:
    _OpenSimulationDialog(
        SimulationController const& simController,
        TemporalControlWindow const& temporalControlWindow,
        StatisticsWindow const& statisticsWindow,
        Viewport const& viewport);
    ~_OpenSimulationDialog();

    void process();

    void show();

private:
    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;
    StatisticsWindow _statisticsWindow;
    Viewport _viewport;

    std::string _startingPath;
};

